#!/usr/bin/env bash
# Apple Silicon, 48 GB. Qwen3-4B in bfloat16 on MPS.
# Forward passes only, no generation. Expect roughly 20 to 100 minutes.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-4B}
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=${PY:-"uv run python"}

echo "== layer bands =="
$PY src/model_utils.py "$MODEL"

echo "== smoke test: 2 problems, 2 fractions (about 2 min) =="
$PY src/run_scoring.py --model "$MODEL" --lens logit \
  --per-tier 2 --fractions 0.0 1.0 \
  --out results/smoke.jsonl
$PY src/analyze_scoring.py --results results/smoke.jsonl \
  --out-dir results/figures_smoke

echo
echo ">>> Check the smoke output before continuing."
echo ">>> Expect clean logprob_mean to rise from f=0 to f=1 (the page helps),"
echo ">>> and the jspace gap at f=0 to be clearly positive."
read -p "Continue to the main run? [y/N] " ok
[ "$ok" = "y" ] || exit 0

echo "== main run =="
$PY src/run_scoring.py --model "$MODEL" --lens logit \
  --per-tier "${PER_TIER:-24}" \
  --fractions 0.0 0.25 0.5 0.75 1.0 \
  --conditions clean jspace ctrl_random ctrl_rank \
  --out results/scoring.jsonl --resume

echo "== analysis =="
$PY src/analyze_scoring.py --results results/scoring.jsonl \
  --out-dir results/figures
