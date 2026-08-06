# What does written reasoning actually replace? A dose-response test of the J-space trade-off

**Homework Zero, Fall 2026 — Qwen3-4B**
Code: https://github.com/cs098890/cs2881-jspace

## Scope

This was done alongside a full-time internship and without access to a GPU. Given that, I
spent the time on understanding the paper and on the question of what separates a weak
experimental design from a strong one, and I deliberately kept the empirical component
small enough to run honestly on a laptop rather than large enough to look impressive. The
results below are underpowered, and where they are inconclusive I have said so rather than
reaching. The design reasoning, the controls, and the pre-registration are the parts I
would want read most carefully.

## Hypothesis

Gurnee et al. (2026) find that GSM8K solved with explicit chain of thought is far more
robust to J-space ablation than the same problems answered directly, and read this as the
model writing down what it would otherwise hold in its internal workspace. The two
scratchpads look interchangeable. GSM8K is one point on a difficulty axis and an easy one,
so the question here is whether that interchangeability survives as problems get harder.

Writing things down substitutes for internal workspace only for the part of the work that
is bookkeeping. It does not substitute for the part that decides what to write next. Doing
long division on paper barely taxes working memory, because the paper holds every
intermediate and the procedure tells you where to look. Doing a proof on paper is
different: the paper holds the steps already taken, but choosing the next step still
happens in your head. Easy arithmetic is long division. Competition maths is the proof.

**H1'.** The gap between clean and ablated answer log-probability is largest when no chain
of thought is supplied, and closes as more of it is supplied. The value of `f` at which it
closes rises with difficulty.

Recorded in `PREREGISTRATION.md` before the run, along with the three results that would
falsify it and the outcome I expected.

## Experiment design

The obvious design generates a chain of thought under each condition and grades the
answer. I started there and abandoned it, for a reason that is not about compute. Ablating
the J-space changes what the model writes. So a drop in accuracy is ambiguous between two
claims: the model needed its workspace to reason over the page, or the model wrote a worse
page and then reasoned over that page perfectly well. The design cannot separate them, and
the paper's GSM8K result carries the same ambiguity.

The design used instead **supplies** the chain of thought rather than generating it. For
each problem I take the reference solution, strip the stated answer off the end, give the
model the first `f` fraction of it, and measure the log probability it assigns to the
correct answer. Sweeping `f` from 0 to 1 produces a dose-response curve. At `f = 0` the
model has no external scratchpad. At `f = 1` the whole derivation is on the page but the
answer is not.

Because the supplied text is byte-identical across conditions, the ablation cannot produce
an effect by changing what got written. That confound is closed by construction rather than
argued away. The design also turns a binary correct-or-wrong into a continuous measure,
which is what makes a run of this size worth anything at all.

Conditions are `clean`, `jspace` (the paper's ablation), and two controls. `ctrl_random`
removes k random directions and rescales the removal to have the same norm as the `jspace`
removal at that position, which separates the workspace from generic damage of equal size.
`ctrl_rank` removes lens vectors at ranks 1000 to 1010: same vector family, same geometry,
wrong contents. The first tests whether removing directions hurts. The second tests whether
lens-shaped directions are load-bearing. Only a result where `jspace` separates from both
supports the workspace reading.

Disconfirming evidence appears in the same plot as confirming evidence. A flat gap in `f`
means the page does not substitute at all. A gap closing at the same `f` on every tier means
substitution works but its cost does not depend on difficulty. No measurable gap at `f = 0`
means the comparison is uninformative rather than that the hypothesis failed, and is
reported as a null.

Difficulty is a six-point ladder rather than three datasets. MATH-500 is split by its own
level labels, which puts resolution in the middle of the range instead of only at the ends.

## Experimental details

**Model.** `Qwen/Qwen3-4B`, bfloat16, on Apple Silicon via MPS. Forward passes only, no
generation.

**Lens.** The logit lens, which is `J = I` in the paper's formulation. The paper reports it
captures much of the same workspace structure with somewhat lower reliability, particularly
in early layers. Computing the true averaged Jacobians needs many backward passes per layer
and was out of reach at full scale here. Code to compute them is in `src/jspace/lens.py`;
it was run at reduced scale (2048 probes for `d_model` = 2560, layers 12–19) and the
resulting readouts were inspected against the logit lens in `scripts/validate_lens.py`. At
that probe count the estimate is rank-deficient and its readouts were not clearly
interpretable — top tokens at layers 17–19 were dominated by punctuation and fragments
rather than concepts — which is why the logit lens carries the headline numbers rather than
a Jacobian we could not validate. **This is the single largest caveat on everything below.**

**Ablation.** At every position, in every layer of the medium band, rank lens vectors by
correlation against the residual stream, drop any token in the clean pass top-10, take the
top `k = 10`, and remove the residual stream's least-squares projection onto their span.
The paper's percent-of-depth bands (38 to 70 for medium) are mapped onto Qwen3-4B's layer
count at load time, giving **layers 14–24**. The clean pass is also the baseline, so the
exemption costs nothing extra.

**Sample sizes.** PLACEHOLDER_SIZES

## Experimental results

PLACEHOLDER_RESULTS

## Analysis of results

PLACEHOLDER_ANALYSIS

## References

Gurnee, W. et al. (2026). *Verbalizable Representations Form a Global Workspace in Language
Models.* Transformer Circuits Thread, Anthropic.
https://transformer-circuits.pub/2026/workspace/index.html
