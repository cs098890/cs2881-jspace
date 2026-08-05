"""End-to-end smoke test on a small model: fit a lens, ablate, generate.

Validates plumbing only (shapes, hooks, cache handling) -- not scientific results.
Run: uv run python scripts/smoke_test.py
"""

import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jspace.lens import LensConfig, fit_jacobian, lens_vectors
from jspace.ablate import AblationConfig, JSpaceAblator
from jspace.generate import generate_paired, build_full_to_sub

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"[{time.time()-t0:.0f}s] loaded {MODEL}: {n_layers} layers, d={d}, dev={DEV}")

    band = list(range(int(n_layers * 0.4), int(n_layers * 0.6)))
    print(f"test band: {band}")

    # --- fit lens on a handful of random-ish text batches ---
    texts = ["The capital of France is Paris, and the capital of Germany is Berlin. " * 8,
             "To solve 17 + 25, first add the tens, then the ones, giving 42. " * 8,
             "Water boils at 100 degrees Celsius under standard atmospheric pressure. " * 8,
             "In 1969 humans first walked on the surface of the Moon during Apollo 11. " * 8]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    batches = [enc.input_ids for _ in range(4)]   # 16 probes total; tiny, just for plumbing

    cfg = LensConfig(layers=band, n_probes=16, batch_size=4, seq_len=64)
    Js = fit_jacobian(model, batches, cfg, device=DEV)
    print(f"[{time.time()-t0:.0f}s] fitted J for {len(Js)} layers, shape {Js[band[0]].shape}")

    # --- build dictionaries over a token subset ---
    V_sub = 4000
    subset = torch.arange(V_sub)
    W_U = model.get_output_embeddings().weight.detach().cpu()
    dicts = {li: lens_vectors(Js[li], W_U, subset).to(DEV) for li in band}
    print(f"[{time.time()-t0:.0f}s] dictionaries: {dicts[band[0]].shape}")

    f2s = build_full_to_sub(model.config.vocab_size, subset)
    prompts = ["Question: What is 12 + 15? Answer:", "Question: What is the capital of Japan? Answer:"]

    for mode in ["none", "jspace", "random"]:
        abl = JSpaceAblator(model, dicts, AblationConfig(layers=band, k=10, mode=mode))
        t1 = time.time()
        outs = generate_paired(model, tok, prompts, abl, f2s, max_new_tokens=24, device=DEV)
        abl.remove()
        print(f"\n--- mode={mode} ({time.time()-t1:.1f}s, {abl.n_calls} hook calls) ---")
        for p, o in zip(prompts, outs):
            print(f"  {p!r} -> {o[:90]!r}")

    print(f"\n[{time.time()-t0:.0f}s] SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
