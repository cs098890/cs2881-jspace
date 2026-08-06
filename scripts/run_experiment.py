"""Main experiment: J-space ablation vs chain of thought across three difficulty tiers.

Conditions: {clean, jspace, random} x {cot, direct} on {gsm8k, math500, aime2025}.

Usage:
  uv run python scripts/run_experiment.py --device cuda --n-gsm8k 40 --n-math 40 --n-aime 30
"""

from __future__ import annotations

import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from jspace.lens import LensConfig, fit_jacobian, lens_vectors
from jspace.ablate import AblationConfig, JSpaceAblator
from jspace.generate import generate_paired, build_full_to_sub
from jspace.experiment import load_problems, build_prompt, extract_answer, is_correct

MODEL = "Qwen/Qwen3-4B"


def fit_corpus(tokenizer, n_seq: int, seq_len: int, batch_size: int):
    """Chunks of WikiText-103 as a pretraining-like corpus, plus token frequencies."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    buf, batches, freq = [], [], Counter()
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:
            continue
        ids = tokenizer(t, return_tensors=None)["input_ids"]
        buf.extend(ids)
        freq.update(ids)
        while len(buf) >= seq_len * batch_size:
            chunk = buf[: seq_len * batch_size]
            buf = buf[seq_len * batch_size:]
            batches.append(torch.tensor(chunk).view(batch_size, seq_len))
        if len(batches) * batch_size >= n_seq:
            break
    return batches, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-gsm8k", type=int, default=40)
    ap.add_argument("--n-math", type=int, default=40)
    ap.add_argument("--n-aime", type=int, default=30)
    ap.add_argument("--n-probes", type=int, default=4096)
    ap.add_argument("--dict-size", type=int, default=20000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--band-start", type=int, default=12)
    ap.add_argument("--band-end", type=int, default=20)   # exclusive
    ap.add_argument("--lens-cache", default="results/lens.pt")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(args.device).eval()
    band = list(range(args.band_start, args.band_end))
    log(f"loaded {MODEL} ({model.config.num_hidden_layers}L, d={model.config.hidden_size}); band={band}")

    # ---- 1. fit or load the Jacobian lens ----
    cache = pathlib.Path(args.lens_cache)
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        Js, freq = blob["J"], blob["freq"]
        log(f"loaded cached lens from {cache}")
    else:
        n_seq = max(args.n_probes, 1)
        batches, freq = fit_corpus(tok, n_seq, 128, args.batch_size)
        log(f"fit corpus: {len(batches)} batches x {args.batch_size} seqs")
        cfg = LensConfig(layers=band, n_probes=args.n_probes, batch_size=args.batch_size)
        Js = fit_jacobian(model, batches, cfg, device=args.device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"J": Js, "freq": freq}, cache)
        log(f"fitted Jacobian ({args.n_probes} probes) -> {cache}")

    # ---- 2. dictionaries over the most frequent tokens ----
    subset = torch.tensor([t for t, _ in freq.most_common(args.dict_size)], dtype=torch.long)
    W_U = model.get_output_embeddings().weight.detach().cpu()
    dicts = {li: lens_vectors(Js[li], W_U, subset).to(args.device) for li in band}
    f2s = build_full_to_sub(model.config.vocab_size, subset)
    log(f"dictionaries: {len(dicts)} layers x {dicts[band[0]].shape}")

    # ---- 3. run the condition grid ----
    # Generation budgets: Qwen3-4B writes verbose markdown CoT even with thinking off;
    # too short a budget truncates the clean baseline and inflates apparent retention.
    datasets = [("gsm8k", args.n_gsm8k, 512), ("math500", args.n_math, 640),
                ("aime2025", args.n_aime, 768)]
    results_path = outdir / "raw_generations.jsonl"
    fout = results_path.open("a")

    loaded = {}
    for ds_name, n, cot_tokens in datasets:
        try:
            loaded[ds_name] = (load_problems(ds_name, n), cot_tokens)
            log(f"{ds_name}: {len(loaded[ds_name][0])} problems")
        except Exception as e:
            log(f"!! could not load {ds_name}: {e}")

    # Cheap `direct` cells first, across all datasets, so that running out of wall clock
    # costs us CoT cells on the hardest tier rather than an entire difficulty tier.
    for condition in ["direct", "cot"]:
        for ds_name, (problems, cot_tokens) in loaded.items():
            max_new = cot_tokens if condition == "cot" else 24
            prompts = [build_prompt(tok, p["problem"], condition) for p in problems]

            for mode in ["none", "jspace", "random"]:
                abl_cfg = AblationConfig(layers=band, k=args.k, mode=mode)
                abl = JSpaceAblator(model, dicts, abl_cfg)
                t1 = time.time()
                completions = []
                for i in range(0, len(prompts), args.batch_size):
                    completions += generate_paired(
                        model, tok, prompts[i:i + args.batch_size], abl, f2s,
                        max_new_tokens=max_new, device=args.device)
                abl.remove()

                n_ok = 0
                for p, c in zip(problems, completions):
                    pred = extract_answer(c)
                    ok = is_correct(pred, p["answer"])
                    n_ok += ok
                    fout.write(json.dumps({
                        "dataset": ds_name, "condition": condition, "mode": mode,
                        "problem": p["problem"], "gold": p["answer"],
                        "completion": c, "pred": pred, "correct": bool(ok),
                        "n_completion_chars": len(c),
                    }) + "\n")
                fout.flush()
                log(f"  {ds_name:9s} {condition:6s} {mode:6s} "
                    f"acc={n_ok}/{len(problems)}={n_ok/len(problems):.3f} "
                    f"({time.time()-t1:.0f}s)")

    fout.close()
    log(f"done -> {results_path}")


if __name__ == "__main__":
    main()
