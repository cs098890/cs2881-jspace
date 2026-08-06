"""Model loading and layer-band mapping.

The paper specifies ablation bands as percentages of network depth. We map those onto
whatever model is loaded, so the band means the same thing across model sizes.

Usage: python src/model_utils.py Qwen/Qwen3-4B
"""

from __future__ import annotations

import sys

import torch

# Percent-of-depth bands from the paper.
BANDS = {"light": (0.20, 0.38), "medium": (0.38, 0.70), "heavy": (0.38, 0.90)}


def band_layers(n_layers: int, band: str = "medium") -> list[int]:
    lo, hi = BANDS[band]
    return list(range(int(round(lo * n_layers)), int(round(hi * n_layers))))


def load_model(model_name: str, device: str = "mps", dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return model, tok


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    n = cfg.num_hidden_layers
    print(f"{name}: {n} layers, d_model={cfg.hidden_size}, vocab={cfg.vocab_size}")
    for b in BANDS:
        ls = band_layers(n, b)
        print(f"  {b:7s} {BANDS[b][0]:.0%}-{BANDS[b][1]:.0%} -> layers {ls[0]}..{ls[-1]} ({len(ls)} layers)")


if __name__ == "__main__":
    main()
