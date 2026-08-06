"""Dose-response test of the J-space / chain-of-thought trade-off.

For each problem we supply the first f fraction of a *reference* solution (byte-identical
across ablation conditions) and measure the log probability the model assigns to the
correct answer. Sweeping f gives a dose-response curve. Because the supplied text does
not change with condition, the ablation cannot act by changing what got written -- the
confound that a generate-and-grade design cannot separate.

Conditions per batch, all from one clean forward plus one per ablation:
  clean, jspace, ctrl_random (norm-matched), ctrl_rank (ranks 1000-1010)

Usage:
  python src/run_scoring.py --lens logit --per-tier 24 --fractions 0.0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse, json, os, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
from datasets import load_dataset

from jspace.ablate import AblationConfig, JSpaceAblator
from jspace.generate import build_full_to_sub
from jspace.lens import lens_vectors
from model_utils import band_layers, load_model

TIER_LABELS = {1: "GSM8K", 2: "MATH L1-2", 3: "MATH L3",
               4: "MATH L4", 5: "MATH L5", 6: "AIME 2025"}

INSTRUCTION = ("Solve the problem step by step. "
               "Then give the final answer in the form: The answer is <answer>.")


# ----------------------------------------------------------------------------- data
def _strip_answer_gsm8k(sol: str) -> str:
    sol = sol.split("####")[0]
    return re.sub(r"<<[^>]*>>", "", sol).strip()


def _strip_answer_math(sol: str) -> str:
    """Truncate at the first \\boxed{...} so the final answer is never leaked."""
    i = sol.find("\\boxed")
    return (sol[:i] if i != -1 else sol).strip()


def build_tiers(per_tier: int, seed: int = 0):
    """Return {tier: [{'id','problem','answer','solution'}]}. Tier 6 has no solutions."""
    tiers: dict[int, list] = {}

    gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    tiers[1] = [{"id": f"gsm8k-{i}", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip(),
                 "solution": _strip_answer_gsm8k(r["answer"])}
                for i, r in enumerate(gsm.select(range(min(per_tier, len(gsm)))))]

    math = load_dataset("HuggingFaceH4/MATH-500", split="test").shuffle(seed=seed)
    buckets = {2: ["1", "2"], 3: ["3"], 4: ["4"], 5: ["5"]}
    for tier, levels in buckets.items():
        rows = [r for r in math if str(r["level"]) in levels][:per_tier]
        tiers[tier] = [{"id": f"math{r['level']}-{i}", "problem": r["problem"],
                        "answer": str(r["answer"]).strip(),
                        "solution": _strip_answer_math(r["solution"])}
                       for i, r in enumerate(rows)]

    # AIME 2025: the `solution` field in this release is the bare answer, not a worked
    # derivation, so this tier can only be evaluated at f=0.
    aime = load_dataset("yentinglin/aime_2025", split="train")
    tiers[6] = [{"id": f"aime-{r['id']}", "problem": r["problem"],
                 "answer": str(r["answer"]).strip(), "solution": None}
                for r in list(aime)[:per_tier]]
    return tiers


def build_context(tok, problem: str, solution: str | None, f: float) -> str:
    """Chat prompt + the first f fraction of the reference solution + the answer lead-in."""
    msgs = [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    partial = ""
    if solution and f > 0:
        ids = tok(solution, add_special_tokens=False)["input_ids"]
        cut = max(1, int(round(f * len(ids))))
        partial = tok.decode(ids[:cut]) + "\n"
    return head + partial + "The answer is "


# ------------------------------------------------------------------------- scoring
@torch.no_grad()
def batch_logprobs(model, tok, contexts, answers, ablators, device, exclude_k, f2s):
    """Mean log-prob of each answer under clean + each ablation condition.

    Returns {condition: [mean_logprob per example]}.
    """
    ctx_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in contexts]
    ans_ids = [tok(a, add_special_tokens=False)["input_ids"] for a in answers]
    full = [c + a for c, a in zip(ctx_ids, ans_ids)]
    L = max(len(x) for x in full)
    pad = tok.pad_token_id or tok.eos_token_id

    inp = torch.full((len(full), L), pad, dtype=torch.long)
    att = torch.zeros((len(full), L), dtype=torch.long)
    for j, x in enumerate(full):                       # left-pad
        inp[j, L - len(x):] = torch.tensor(x)
        att[j, L - len(x):] = 1
    inp, att = inp.to(device), att.to(device)

    def score(logits):
        lp = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for j, a in enumerate(ans_ids):
            n = len(a)
            # answer tokens occupy [L-n, L); predicted by logits at [L-n-1, L-1)
            pos = torch.arange(L - n - 1, L - 1, device=device)
            tgt = torch.tensor(a, device=device)
            out.append(lp[j, pos, tgt].mean().item())
        return out

    results = {}
    for a in ablators.values():
        a.enabled = False
    clean_out = model(inp, attention_mask=att, use_cache=False)
    results["clean"] = score(clean_out.logits)

    excluded = f2s.to(device)[clean_out.logits.topk(exclude_k, dim=-1).indices]

    for name, abl in ablators.items():
        abl.set_excluded(excluded)
        abl.enabled = True
        out = model(inp, attention_mask=att, use_cache=False)
        abl.enabled = False
        results[name] = score(out.logits)
    return results


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lens", choices=["logit", "jacobian"], default="logit")
    ap.add_argument("--lens-ckpt", default="results_local/lens.pt")
    ap.add_argument("--band", default="medium")
    ap.add_argument("--per-tier", type=int, default=24)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--conditions", nargs="+",
                    default=["clean", "jspace", "ctrl_random", "ctrl_rank"])
    ap.add_argument("--dict-size", type=int, default=20000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/scoring.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    outp = pathlib.Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and outp.exists():
        for line in outp.open():
            try:
                r = json.loads(line)
                done.add((r["id"], r["f"], r["condition"]))
            except Exception:
                pass
        log(f"resume: {len(done)} cells already present")

    model, tok = load_model(args.model, args.device)
    n_layers = model.config.num_hidden_layers
    band = band_layers(n_layers, args.band)

    # ---- dictionaries ----
    W_U = model.get_output_embeddings().weight.detach().cpu()
    if args.lens == "jacobian":
        blob = torch.load(args.lens_ckpt, weights_only=False)
        Js, freq = blob["J"], blob["freq"]
        band = [l for l in band if l in Js]
        if not band:
            sys.exit(f"no fitted Jacobians in band; have layers {sorted(Js)}")
        subset = torch.tensor([t for t, _ in freq.most_common(args.dict_size)])
        dicts = {li: lens_vectors(Js[li], W_U, subset).to(args.device) for li in band}
    else:
        # logit lens: J = I, so the lens vectors are just the unembedding rows.
        subset = torch.arange(min(args.dict_size, W_U.shape[0]))
        V = W_U[subset].float()
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        V = V.to(args.device)
        dicts = {li: V for li in band}

    f2s = build_full_to_sub(model.config.vocab_size, subset)
    log(f"{args.model}: {n_layers}L, band={args.band} -> {band[0]}..{band[-1]} "
        f"({len(band)}L), lens={args.lens}, dict={len(subset)}")

    ablators = {c: JSpaceAblator(model, dicts, AblationConfig(layers=band, k=args.k, mode=c))
                for c in args.conditions if c != "clean"}

    tiers = build_tiers(args.per_tier)
    for t, probs in tiers.items():
        log(f"  tier {t} ({TIER_LABELS[t]}): {len(probs)} problems"
            f"{' [no reference solutions -> f=0 only]' if t == 6 else ''}")

    fout = outp.open("a")
    for tier, problems in tiers.items():
        for f in args.fractions:
            if tier == 6 and f > 0:
                continue                     # no reference solutions available
            todo = [p for p in problems
                    if (p["id"], f, "clean") not in done]
            if not todo:
                continue
            t1 = time.time()
            for i in range(0, len(todo), args.batch_size):
                chunk = todo[i:i + args.batch_size]
                ctxs = [build_context(tok, p["problem"], p["solution"], f) for p in chunk]
                anss = [p["answer"] for p in chunk]
                try:
                    res = batch_logprobs(model, tok, ctxs, anss, ablators,
                                         args.device, 10, f2s)
                except Exception as e:
                    log(f"    !! batch failed ({type(e).__name__}: {e}); skipping")
                    continue
                for cond, vals in res.items():
                    for p, v in zip(chunk, vals):
                        fout.write(json.dumps({
                            "id": p["id"], "tier": tier, "tier_label": TIER_LABELS[tier],
                            "f": f, "condition": cond, "logprob_mean": v,
                            "answer": p["answer"], "lens": args.lens,
                        }) + "\n")
                fout.flush()
            log(f"  tier {tier} f={f}: {len(todo)} problems x {len(res)} conditions "
                f"({time.time()-t1:.0f}s)")
    fout.close()
    log(f"done -> {outp}")


if __name__ == "__main__":
    main()
