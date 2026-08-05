# J-space ablation and external chain of thought: does the trade-off survive difficulty?

**CS 2881 Homework Zero — Qwen3-4B**
Code: https://github.com/cs098890/cs2881-jspace

## 1. Hypothesis

*(Pre-registered in `HYPOTHESIS.md`, committed before any experiment was run.)*

Gurnee et al. (2026) find that ablating the J-space — zeroing the residual stream's
projection onto the top-k active J-lens vectors, at every token position across a band of
intermediate layers — leaves parsing, classification and one-step recall largely intact
but collapses internal multi-step reasoning. GSM8K solved with explicit chain of thought
is far more robust to this ablation than the same problems answered directly, which they
read as the model externalizing onto the page what it would otherwise hold internally.

**H1 (primary).** The protective effect of chain of thought against J-space ablation
*decreases* as problem difficulty increases. Chain of thought can substitute for the
internal workspace only insofar as the reasoning state is fully externalizable as text.
On GSM8K each written line ("48 / 2 = 24") carries essentially the whole state forward, so
the internal workspace is nearly redundant. On harder problems much of the work is not
executing a step but *choosing* one — search over strategies, holding candidate approaches
in parallel, recognizing structure, backtracking — and that deliberation is precisely the
flexible cognition the J-space is claimed to mediate, and precisely what does *not* get
written down. The page records the output of a selection, not the selection.

**H2 (competing).** Protection is constant or *increases* with difficulty: harder problems
carry more intermediate state, so writing it down should help more, not less.

**H3 (null).** The ablation is generic damage, and the CoT advantage is an artifact of
longer generations having more room to recover.

PLACEHOLDER_PREDICTIONS

## 2. Experiment design

A 3 x 2 x 3 grid: {GSM8K, MATH-500, AIME 2025} x {chain of thought, direct answer} x
{clean, J-space ablated, random-direction control}.

**Metric.** Exact-match accuracy on the extracted final answer, with 95% Wilson intervals
(which stay well-behaved at zero and at small n — both expected here). From these:

- *retention* = acc(ablated) / acc(clean), computed within a condition, so each condition
  is normalized by its own unablated baseline;
- *protection margin* = retention(CoT) − retention(direct), the quantity H1 predicts
  should shrink with difficulty.

**Why this design can disconfirm.** Three separate ways the result could come out against
H1, all visible:

1. *The random-direction control.* It removes the same number of directions, drawn from the
   same J-lens dictionary, at the same layers and token positions, using the identical
   span-removal operation. If it degrades accuracy as much as J-space ablation, the effect
   is generic perturbation, not the workspace — H3. This is the control that separates
   "we broke the J-space" from "we broke the model."
2. *Difficulty ordering fixed in advance.* Protection is measured on a common scale per
   dataset with the tiers ordered before seeing data, so a flat or rising trend (H2) reads
   as clearly as a falling one.
3. *Clean baselines reported alongside.* If direct-answer accuracy is unaffected by
   ablation, the premise of the original result fails to replicate at this scale, and we
   would report that rather than the trend.

**Anticipated confounds, recorded in advance.**

- *Floor effects.* AIME direct accuracy for a 4B model in non-thinking mode may be ~0 even
  clean, making retention undefined. We therefore report absolute accuracies and clean
  baselines alongside every ratio, and treat a floor as "untestable at this tier" rather
  than reading a trend into noise.
- *Length confound.* CoT conditions emit more tokens, giving the ablation more
  opportunities to bite but also more opportunity to self-correct. Intrinsic to the
  comparison; we report mean completion length per cell so the reader can see it.
- *Power.* With n of 30-40, Wilson intervals are roughly ±15pp; differences under ~20pp are
  not resolvable and are reported as inconclusive.

## 3. Experimental details

**Model.** `Qwen/Qwen3-4B` (36 layers, d_model 2560, vocab 151,936), bfloat16, greedy
decoding, `enable_thinking=False`.

**Fitting the lens.** J_l = E[∂h_final,t' / ∂h_l,t] over source position t, all subsequent
positions t' and a corpus of WikiText-103 chunks (128 tokens). Computing a full 2560x2560
Jacobian by exact backpropagation would need 2560 backward passes per prompt, so we use a
randomized estimator: for a random probe u, backpropagating s = Σ_t' u·h_final,t' yields
g_l,t = J_l,t^T u at every layer and position in a *single* backward pass (causal masking
supplies the t' ≥ t restriction automatically), and E[u g^T] = E[u u^T J] = J. One backward
pass therefore supplies every layer at once. PLACEHOLDER_PROBES

**J-lens vectors and the dictionary.** The J-lens vectors are the rows of W_U J_l,
unit-normalized. We restrict the dictionary to the 20,000 most frequent tokens in the
fitting corpus.

**Ablation.** At every token position, across layers PLACEHOLDER_BAND, we score all
dictionary vectors by correlation with the residual stream, keep the top k=10 with
non-negative correlation, and remove the exact least-squares projection of the residual
stream onto their span. Following the paper, tokens appearing in the top-10 of a clean
forward pass are protected; because the ablated run's context diverges from the clean one,
we maintain two KV caches over the same token sequence — one advanced with ablation off to
read the clean top-10, one with ablation on that actually generates.

**Deviations from the paper, and their direction.** (i) The paper solves for a sparse
non-negative combination by gradient pursuit; we use one-shot top-k selection with an exact
refit on the selected vectors. (ii) The dictionary is 20k tokens, not the full 151,936 —
full-vocabulary pursuit at every token, layer and decode step is roughly 150x our entire
compute budget. Both make the ablation *weaker* than the paper's, so both bias toward a
null result rather than toward confirming H1. (iii) Thinking mode is disabled.

**Data.** GSM8K (`openai/gsm8k`, `main`, test); MATH-500 (`HuggingFaceH4/MATH-500`, test);
**AIME 2025** (AIME I + II, 30 problems, `yentinglin/aime_2025`). AIME 2025 rather than 2024
because Qwen3's pretraining window makes 2024 a contamination risk.

PLACEHOLDER_COMPUTE

## 4. Results

PLACEHOLDER_RESULTS

## 5. Analysis

PLACEHOLDER_ANALYSIS
