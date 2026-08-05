# J-space ablation vs. external chain of thought (CS 2881 Homework Zero)

Does written chain of thought keep substituting for the model's internal workspace as
problems get harder?

We replicate the J-space ablation of [Gurnee et al. (2026), *Verbalizable Representations
Form a Global Workspace in Language Models*](https://arxiv.org/abs/2607.15495) on
**Qwen3-4B**, and measure how much accuracy survives ablation with vs. without chain of
thought across three difficulty tiers.

- **Report:** [`report.pdf`](report.pdf)
- **Pre-registered hypothesis:** [`HYPOTHESIS.md`](HYPOTHESIS.md) — written and committed
  *before* any experiment was run (see git history for the timestamp).

## What the experiment does

| Axis | Levels |
|---|---|
| Dataset | GSM8K, MATH-500, AIME 2025 |
| Prompt condition | `cot` (step-by-step) vs. `direct` (final answer only) |
| Ablation mode | `none` (clean) / `jspace` / `random` (control) |

The `random` control removes the same number of directions, from the same dictionary, at
the same layers and positions — so any J-space-specific effect must show up as a gap
between `jspace` and `random`, not merely as a drop from `none`.

### Method summary

1. **Fit the lens.** Estimate the averaged Jacobian `J_l = E[∂h_final,t'/∂h_l,t]` on
   WikiText-103 chunks. We use a randomized estimator: backpropagating
   `s = Σ_t' u·h_final,t'` for a random probe `u` gives `J_l,t^T u` at every layer and
   position in a single backward pass, and `E[u (J^T u)^T] = J`. See `src/jspace/lens.py`.
2. **Build the dictionary.** J-lens vectors are the rows of `W_U J_l`, unit-normalized,
   restricted to the 20k most frequent tokens.
3. **Ablate.** At every token position across the workspace layer band, select the top-k
   (k=10) J-lens vectors by non-negative correlation and remove the residual stream's
   least-squares projection onto their span. Tokens in the clean forward pass's top-10 are
   protected, which requires a paired clean/ablated decode (`src/jspace/generate.py`).
4. **Score.** Exact-match on the extracted final answer, with Wilson 95% intervals.

### Documented deviations from the paper

Forced by compute budget; all are discussed in the report.

- One-shot top-k selection + exact least-squares refit, instead of gradient pursuit.
- Dictionary restricted to 20k frequent tokens, not the full 151,936-token vocabulary.
- `enable_thinking=False` — Qwen3-4B thinking traces run 10k+ tokens, out of budget.
- Small n per dataset, so intervals are wide.

Deviations 1 and 2 both *weaken* the ablation, biasing toward a null result rather than
toward confirming the hypothesis.

## Repository layout

```
HYPOTHESIS.md              pre-registration (committed before running)
report.pdf                 the report
modal_app.py               GPU launcher (Modal); science code is provider-agnostic
src/jspace/
  lens.py                  averaged-Jacobian estimation, J-lens vectors
  ablate.py                top-k selection + span removal, forward hooks
  generate.py              paired clean/ablated batched decoding
  experiment.py            datasets, prompts, answer extraction, scoring
scripts/
  run_experiment.py        main entry point: fits lens, runs the condition grid
  analyze.py               scoring, CIs, figures
  smoke_test.py            fast end-to-end plumbing check on Qwen3-0.6B
results/                   raw_generations.jsonl, accuracy.csv, retention.csv, figures
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
export HF_XET_HIGH_PERFORMANCE=1     # much faster model download
```

## Reproducing

Plumbing check (~2 min, CPU/MPS, small model):

```bash
uv run python scripts/smoke_test.py
```

Full run on a GPU:

```bash
uv run python scripts/run_experiment.py --device cuda \
    --n-gsm8k 40 --n-math 40 --n-aime 30 --n-probes 4096 --batch-size 16
uv run python scripts/analyze.py --results results
```

On Apple Silicon substitute `--device mps --batch-size 8`; expect it to be roughly
10-20x slower.

Via Modal (no local GPU needed):

```bash
uv run modal setup
uv run modal run modal_app.py --n-gsm8k 40 --n-math 40 --n-aime 30
uv run modal volume get jspace-results /results ./results
```

### Regenerating the report's tables and figures

`scripts/analyze.py` reads `results/raw_generations.jsonl` and writes every number and
figure used in the report:

| Artifact | File |
|---|---|
| Accuracy table (all 18 cells, Wilson CIs) | `results/accuracy.csv` |
| Retention ratios | `results/retention.csv` |
| CoT protection margin | `results/protection.csv` |
| Figure 1 (accuracy grid) | `results/fig1_accuracy.png` |
| Figure 2 (protection vs. difficulty) | `results/fig2_protection.png` |

## Exact data sources

| Tier | Dataset | HF repo | Split | n |
|---|---|---|---|---|
| Easy | GSM8K | `openai/gsm8k` (`main`) | `test` | see report |
| Medium | MATH-500 | `HuggingFaceH4/MATH-500` | `test` | see report |
| Hard | **AIME 2025** (AIME I + II, 30 problems) | `yentinglin/aime_2025` | `train` | 30 |

AIME **2025** was chosen over 2024 because Qwen3's pretraining window makes 2024 a
contamination risk.

Lens fitting corpus: `Salesforce/wikitext`, `wikitext-103-raw-v1`, `train` split,
128-token chunks.

## Model

`Qwen/Qwen3-4B` — 36 layers, d_model 2560, vocab 151,936.
