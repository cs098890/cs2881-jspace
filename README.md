# What does written reasoning actually replace? A dose-response test of the J-space trade-off

CS 2881 Homework Zero — **Qwen3-4B**, Apple Silicon, forward passes only.

Gurnee et al. (2026), [*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html), find that GSM8K
solved with explicit chain of thought is far more robust to J-space ablation than the same
problems answered directly. This repo asks whether that interchangeability between the
internal workspace and the written page survives as problems get harder.

- **Report:** [`report.pdf`](report.pdf)
- **Pre-registration:** [`PREREGISTRATION.md`](PREREGISTRATION.md) — committed before the
  main run, including the amendment that replaced the original design and why.
- **Original (superseded) hypothesis:** [`HYPOTHESIS.md`](HYPOTHESIS.md), kept for the
  record.

## The design, and why it is not the obvious one

The obvious design generates a chain of thought under each ablation condition and grades
the answer. We started there and abandoned it. **Ablating the J-space changes what the
model writes**, so an accuracy drop is ambiguous between "the model needed its workspace to
reason over the page" and "the model wrote a worse page and then reasoned over that page
perfectly well." The paper's GSM8K result carries the same ambiguity.

Instead we **supply** the chain of thought. For each problem we take the reference
solution, strip the stated answer, give the model the first `f` fraction of it, and measure
the log probability assigned to the correct answer. Sweeping `f` from 0 to 1 gives a
dose-response curve. Since the supplied text is byte-identical across conditions, the
ablation cannot act by changing what got written — the confound is closed by construction.

| Axis | Levels |
|---|---|
| Difficulty tier | GSM8K, MATH L1-2, MATH L3, MATH L4, MATH L5, AIME 2025 |
| Fraction supplied `f` | 0.0, 0.25, 0.5, 0.75, 1.0 |
| Condition | `clean`, `jspace`, `ctrl_random`, `ctrl_rank` |

**Controls.** `ctrl_random` removes k random directions rescaled to the *same displacement
norm* as the jspace removal at that position — separating the workspace from generic damage
of equal size. `ctrl_rank` removes lens vectors at ranks 1000–1010: same vector family, same
geometry, wrong contents. Only a result where `jspace` separates from **both** supports the
workspace reading.

## Repository layout

```
PREREGISTRATION.md         pre-registration + amendment (committed before the main run)
HYPOTHESIS.md              original superseded design, kept for the record
report.pdf / report.md     the report
run_mac.sh                 smoke test -> confirm -> main run -> analysis
src/
  model_utils.py           percent-of-depth layer band mapping, model loading
  run_scoring.py           the dose-response experiment
  analyze_scoring.py       paired bootstrap, closure_f, figures
  jspace/
    lens.py                averaged-Jacobian estimation, J-lens vectors
    ablate.py              top-k selection, span removal, the two controls
    generate.py            paired clean/ablated decoding (used by the superseded design)
    experiment.py          datasets, answer extraction, scoring (superseded design)
scripts/
  validate_lens.py         J-lens vs logit-lens readout comparison (validity check)
  run_experiment.py        superseded generate-and-grade experiment
  smoke_test.py            fast plumbing check on Qwen3-0.6B
  make_pdf.py              report.md -> report.pdf
results/
  scoring.jsonl            raw per-cell log probabilities
  figures/                 tables + figures used in the report
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
export HF_XET_HIGH_PERFORMANCE=1     # much faster model download
```

## Reproducing

```bash
./run_mac.sh                          # smoke test, prompts, then the main run
```

Or directly:

```bash
uv run python src/model_utils.py Qwen/Qwen3-4B          # show the layer bands

uv run python src/run_scoring.py --lens logit --per-tier 12 \
    --fractions 0.0 0.25 0.5 0.75 1.0 \
    --conditions clean jspace ctrl_random ctrl_rank \
    --out results/scoring.jsonl --resume --device mps --batch-size 4

uv run python src/analyze_scoring.py --results results/scoring.jsonl \
    --out-dir results/figures
```

`--resume` is keyed on `(id, f, condition)`, so raising `--per-tier` later tops up the run
rather than repeating it.

### Regenerating the report's tables and figures

`src/analyze_scoring.py` writes every number and figure in the report:

| Artifact | File |
|---|---|
| Gap curve, paired bootstrap CIs | `results/figures/table1_gap_curve.csv` |
| `closure_f` per tier | `results/figures/table2_closure.csv` |
| Figure 1 — dose-response by tier | `results/figures/fig1_dose_response.png` |
| Figure 2 — ablation vs. matched controls | `results/figures/fig2_controls.png` |
| Per-control curves | `results/figures/curve_ctrl_*.csv` |

## The lens caveat

The main run uses the **logit lens** (`J = I` in the paper's formulation). Computing true
averaged Jacobians needs many backward passes per layer and was out of reach at full scale
here. **This is the single largest caveat on every result below.**

Code to fit real Jacobians is in `src/jspace/lens.py` and was run at reduced scale (2048
probes for d_model = 2560, layers 12–19); the checkpoint is reusable via
`--lens jacobian --lens-ckpt results_local/lens.pt`. `scripts/validate_lens.py` compares
J-lens against logit-lens readouts. At that probe count the estimate is rank-deficient and
its readouts were **not** clearly interpretable, which is why the logit lens is used for the
headline numbers. See the report.

## Exact data sources

| Tier | Dataset | HF repo | Notes |
|---|---|---|---|
| 1 | GSM8K | `openai/gsm8k` (`main`, test) | `<<...>>` annotations stripped; solution truncated at `####` |
| 2–5 | MATH-500 by level | `HuggingFaceH4/MATH-500` (test) | solution truncated at the first `\boxed` so the answer never leaks |
| 6 | **AIME 2025** (AIME I + II, 30 problems) | `yentinglin/aime_2025` (train) | **`solution` field is the bare answer, not a derivation — so AIME appears at `f = 0` only** |

AIME **2025** rather than 2024, because Qwen3's pretraining window makes 2024 a
contamination risk.

## Model

`Qwen/Qwen3-4B` — 36 layers, d_model 2560, vocab 151,936. Medium band (38–70% of depth)
maps to layers **14–24**.
