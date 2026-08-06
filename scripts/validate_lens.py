"""Sanity-check the fitted Jacobian lens before trusting any ablation result.

A J-lens readout at an intermediate layer should be *interpretable*: it should name
concepts the model is working with, and should beat the logit lens (which assumes the
same coordinates at every layer) at those layers. If the fitted J is noise, its readouts
will be junk and the ablation experiment is meaningless.

Usage: uv run python scripts/validate_lens.py --lens results_local/lens.pt
"""

from __future__ import annotations

import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-4B"

PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is",
    "Q: If Sarah has 5 apples and buys 3 more boxes of 4 apples each, how many apples does she have? A: Let me think step by step.",
    "The Eiffel Tower is located in the city of",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="results_local/lens.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(args.device).eval()
    blob = torch.load(args.lens, weights_only=False)
    Js = blob["J"]
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    final_norm = model.model.norm

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        print("\n" + "=" * 78)
        print(f"PROMPT: {prompt!r}")
        print(f"model's actual next token: {tok.decode(out.logits[0, -1].argmax())!r}")
        for li in sorted(Js):
            h = out.hidden_states[li + 1][0, -1].float()          # last position, layer li out
            with torch.no_grad():
                # J-lens:  W_U norm(J h)
                jl = (final_norm(( h.to(args.device) @ Js[li].to(args.device).T.float())
                                 ).cpu() @ W_U.T)
                # logit lens: W_U norm(h)   -- the baseline the J-lens is meant to beat
                ll = (final_norm(h).cpu() @ W_U.T)
            j_top = [tok.decode([t]) for t in jl.topk(args.topk).indices]
            l_top = [tok.decode([t]) for t in ll.topk(args.topk).indices]
            print(f"  L{li:02d} J-lens : {j_top}")
            print(f"      logit-lens: {l_top}")


if __name__ == "__main__":
    main()
