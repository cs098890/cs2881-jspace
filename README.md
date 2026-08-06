
CS 2881 Homework Zero Attempt, **Qwen3-4B**

Gurnee et al. (2026) find that GSM8K solved with explicit chain of thought is far more
robust to J-space ablation than the same problems answered directly. This repo asks whether
that interchangeability between the internal workspace and the written page survives as
problems get harder. It does not, in this setup, and this report is explicit about how much
of that is a limitation of the proxy lens used rather than a claim about the model.

## Links

| What | Where |
|---|---|
| This repository | https://github.com/cs098890/cs2881-jspace |
| Paper (Transformer Circuits) | https://transformer-circuits.pub/2026/workspace/index.html |
| Paper (arXiv) | https://arxiv.org/abs/2607.15495 |
| Anthropic's reference J-lens implementation | https://github.com/anthropics/jacobian-lens |
| Model | https://huggingface.co/Qwen/Qwen3-4B |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k |
| MATH-500 | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| AIME 2025 | https://huggingface.co/datasets/yentinglin/aime_2025 |
| WikiText-103 (lens fitting corpus) | https://huggingface.co/datasets/Salesforce/wikitext |
| uv (package manager) | https://astral.sh/uv |

## Start here

- **[`report.pdf`](report.pdf)** in the repository root.
- **[`PREREGISTRATION.md`](PREREGISTRATION.md)**, committed before the main run, including
  the amendment that replaced the original design and why.
- **[`HYPOTHESIS.md`](HYPOTHESIS.md)**, the original superseded design, kept for the record.

Both pre-registration files were committed before results existed; `git log` is the record.

## Design
The obvious design generates a chain of thought under each ablation condition and grades
the answer. We started there and attempted an alternative approach. **Ablating the J-space changes what the
model writes**, so an accuracy drop is ambiguous between "the model needed its workspace to
reason over the page" and "the model wrote a worse page and then reasoned over that page
perfectly well." The paper's GSM8K result carries the same ambiguity.

Instead we **supply** the chain of thought. For each problem we take the reference solution,
strip the stated answer, give the model the first `f` fraction of it, and measure the log
probability assigned to the correct answer. Sweeping `f` from 0 to 1 gives a dose-response
curve. Since the supplied text is byte-identical across conditions, the ablation cannot act
by changing what got written, so the confound is closed by construction.

| Axis | Levels |
|---|---|
| Difficulty tier | GSM8K, MATH L1-2, MATH L3, MATH L4, MATH L5, AIME 2025 |
| Fraction supplied `f` | 0.0, 0.25, 0.5, 0.75, 1.0 |
| Condition | `clean`, `jspace`, `ctrl_random`, `ctrl_rank` |

**Controls.** `ctrl_random` removes k random directions rescaled to the *same displacement
norm* as the jspace removal at that position, separating the workspace from generic damage
of equal size. `ctrl_rank` removes lens vectors at ranks 1000-1010: same vector family, same
geometry, wrong contents. Only a result where `jspace` separates from **both** supports the
workspace reading.

## Repository structure

```
report.pdf                   the report (root, as required)
report.md                    its source; scripts/make_pdf.py renders it
PREREGISTRATION.md           pre-registration + amendment, committed before the main run
HYPOTHESIS.md                original superseded design, kept for the record
run_mac.sh                   smoke test -> confirm -> main run -> analysis
pyproject.toml / uv.lock     pinned dependencies
modal_app.py                 optional GPU launcher (unused for the reported results)

src/
  run_scoring.py             THE EXPERIMENT: tiers, prompts, f-sweep, log-prob scoring
  analyze_scoring.py         paired bootstrap, closure_f, all tables and figures
  model_utils.py             percent-of-depth layer band mapping, model loading
  jspace/
    ablate.py                top-k selection, span removal, both controls
    lens.py                  averaged-Jacobian estimation, J-lens vectors
    generate.py              paired clean/ablated decoding (superseded design only)
    experiment.py            datasets, answer extraction (superseded design only)

scripts/
  validate_lens.py           J-lens vs logit-lens readout comparison (validity check)
  make_pdf.py                report.md -> report.pdf
  run_experiment.py          superseded generate-and-grade experiment, kept for the record
  analyze.py                 its analysis, kept for the record
  smoke_test.py              fast plumbing check on Qwen3-0.6B

results/
  scoring.jsonl              raw per-cell log probabilities (1,248 rows) <- all results
  smoke.jsonl                smoke-test output
  figures/                   every table and figure in the report
scoring.log                  full stdout of the reported run, including per-cell timings
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
export HF_XET_HIGH_PERFORMANCE=1     # much faster model download
export PYTORCH_ENABLE_MPS_FALLBACK=1 # Apple Silicon only
```

Qwen3-4B (~8 GB) downloads from Hugging Face on first run. No token needed.

## Reproducing the reported results

The exact command that produced `results/scoring.jsonl`:

```bash
uv run python src/run_scoring.py --lens logit --per-tier 12 \
    --fractions 0.0 0.25 0.5 0.75 1.0 \
    --conditions clean jspace ctrl_random ctrl_rank \
    --out results/scoring.jsonl --resume --device mps --batch-size 4
```

Runtime was 28 minutes on an M5 Pro (48 GB). Then:

```bash
uv run python src/analyze_scoring.py --results results/scoring.jsonl \
    --out-dir results/figures
uv run python scripts/make_pdf.py          # rebuilds report.pdf from report.md
```

`run_mac.sh` wraps all of this with a smoke test and a confirmation prompt.
`--resume` is keyed on `(id, f, condition)`, so raising `--per-tier` tops up an existing run
rather than repeating it. On CUDA substitute `--device cuda` and a larger `--batch-size`.

### Regenerating the report's principal tables and figures

Every number in the report comes from `src/analyze_scoring.py`, which reads only
`results/scoring.jsonl`:

| Report artifact | Generated file |
|---|---|
| Gap curve per tier and `f`, with paired bootstrap CIs (results table 2) | `results/figures/table1_gap_curve.csv` |
| `closure_f` per tier | `results/figures/table2_closure.csv` |
| Pooled condition comparison (results table 1) | printed to stdout; per-condition curves in `results/figures/curve_*.csv` |
| Figure 1, dose-response by tier | `results/figures/fig1_dose_response.png` |
| Figure 2, ablation vs matched controls | `results/figures/fig2_controls.png` |

The pooled `jspace − ctrl_random` and `jspace − ctrl_rank` contrasts quoted in the report are
printed by the same script (the `[specificity]` line) and recomputed from
`paired_gaps()` + `boot_ci()` in `src/analyze_scoring.py`.

## Prompts

Defined in `src/run_scoring.py`. The user turn is the problem followed by:

```
Solve the problem step by step. Then give the final answer in the form: The answer is <answer>.
```

This is wrapped with `tokenizer.apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False)`. The assistant turn is then pre-filled with the first `f` fraction
of the reference solution followed by the literal lead-in `The answer is `, and we score the
log probability of the gold answer tokens at that position. `enable_thinking=False` because
Qwen3 thinking traces run to 10k+ tokens.

## Experimental settings

| Setting | Value | Flag |
|---|---|---|
| Model | `Qwen/Qwen3-4B`, bfloat16 | `--model` |
| Lens | logit lens (`J = I`) | `--lens logit` |
| Ablation band | medium, 38-70% of depth = layers 14-24 | `--band medium` |
| k (vectors removed per position) | 10 | `--k` |
| Clean-pass exemption | top-10 tokens protected | hard-coded, `exclude_clean_topk` |
| `ctrl_rank` offset | ranks 1000-1010 | `AblationConfig.rank_offset` |
| Dictionary size | 20,000 tokens | `--dict-size` |
| Problems per tier | 12 | `--per-tier` |
| Fractions | 0.0, 0.25, 0.5, 0.75, 1.0 | `--fractions` |
| Batch size | 4 | `--batch-size` |
| Bootstrap resamples | 4,000, paired over problems | `boot_ci()` |

## Artifacts not stored in this repository

**The fitted Jacobian lens checkpoint.** ~200 MB, over GitHub's 100 MB file limit, so it is
not committed. It is **not needed to reproduce any reported result** (the headline run uses
the logit lens). To regenerate it:

```bash
uv run python scripts/run_experiment.py --device mps --n-gsm8k 8 --n-math 8 --n-aime 8 \
    --n-probes 2048 --batch-size 8 --out results_local --lens-cache results_local/lens.pt
```

This writes `results_local/lens.pt` (the averaged Jacobians for layers 12-19 plus token
frequencies).Then
inspect it against the logit lens:

```bash
uv run python scripts/validate_lens.py --lens results_local/lens.pt --device mps
```

and, if you want to run the whole experiment on it instead of the logit lens:

```bash
uv run python src/run_scoring.py --lens jacobian --lens-ckpt results_local/lens.pt ...
```

**Note the finding that motivated not doing this:** at 2048 probes for `d_model` = 2560 the
estimate is rank-deficient and its readouts were not interpretable, which is why the logit
lens carries the headline numbers. Raising `--n-probes` past 2560 is the single highest-value
follow-up. See the report's Analysis section.

**Model weights.** Downloaded from Hugging Face on first run.
## The lens caveat

The reported run uses the **logit lens** (`J = I` in the paper's formulation) as a proxy for
the Jacobian lens. **This is the single largest caveat on every result**, and the report
treats the outcome as a null on a proxy rather than as evidence against the paper. The
machinery to do it properly is in `src/jspace/lens.py`; see above.

## Exact data sources

| Tier | Dataset | HF repo | Split | Preprocessing |
|---|---|---|---|---|
| 1 | GSM8K | `openai/gsm8k` (`main`) | test | `<<...>>` annotations stripped, solution truncated at `####` |
| 2-5 | MATH-500 by level | `HuggingFaceH4/MATH-500` | test | split by `level`; solution truncated at the first `\boxed` so the answer never leaks |
| 6 | **AIME 2025** (AIME I + II, 30 problems) | `yentinglin/aime_2025` | train | **the `solution` field is the bare answer, not a derivation, so AIME appears at `f = 0` only** |

AIME **2025** rather than 2024, because Qwen3's pretraining window makes 2024 a
contamination risk. 12 problems per tier are drawn with a fixed shuffle seed of 0.

The lens-fitting corpus (only relevant if you regenerate the Jacobian) is
`Salesforce/wikitext`, `wikitext-103-raw-v1`, train split, 128-token chunks.

## Model

`Qwen/Qwen3-4B`, 36 layers, d_model 2560, vocab 151,936. Medium band (38-70% of depth) maps
to layers **14-24**.
