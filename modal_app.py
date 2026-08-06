"""Modal launcher for the J-space experiment.

The science lives in src/jspace/ and scripts/run_experiment.py, which are provider-
agnostic and run unchanged on a local GPU or MPS. This file only ships them to a GPU.

  modal setup                       # once, browser auth
  modal run modal_app.py            # full run
  modal run modal_app.py --n-gsm8k 8 --n-math 8 --n-aime 8   # quick check
  modal volume get jspace-results /results ./results         # pull artifacts back
"""

import modal

app = modal.App("jspace-cot")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.51", "datasets", "accelerate", "numpy", "pandas")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
)

hf_cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("jspace-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",          # L40S also works for a 4B model; A100 gives bigger batches
    volumes={"/cache": hf_cache, "/results": results_vol},
    timeout=60 * 60 * 3,
)
def run(n_gsm8k: int = 40, n_math: int = 40, n_aime: int = 30,
        n_probes: int = 4096, batch_size: int = 16, dict_size: int = 20000):
    import subprocess, sys
    cmd = [
        sys.executable, "/root/scripts/run_experiment.py",
        "--device", "cuda", "--out", "/results",
        "--lens-cache", "/results/lens.pt",
        "--n-gsm8k", str(n_gsm8k), "--n-math", str(n_math), "--n-aime", str(n_aime),
        "--n-probes", str(n_probes), "--batch-size", str(batch_size),
        "--dict-size", str(dict_size),
    ]
    subprocess.run(cmd, check=True)
    results_vol.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(n_gsm8k: int = 40, n_math: int = 40, n_aime: int = 30,
         n_probes: int = 4096, batch_size: int = 16, dict_size: int = 20000):
    run.remote(n_gsm8k, n_math, n_aime, n_probes, batch_size, dict_size)
