# Pre-registration

Written and committed before the main experiments were run. Nothing below was edited after results came in; corrections and
surprises go in the report instead.

## The claim being tested

Gurnee et al. (2026) report that GSM8K solved with explicit chain of thought is
substantially more robust to J-space ablation than the same problems answered directly,
and read this as the model externalising onto the page what it would otherwise hold in
its internal workspace.

That reading treats the two scratchpads as interchangeable. GSM8K is one point on a
difficulty axis, and it is an easy one. The question here is whether the
interchangeability survives when problems get harder.

## Hypothesis

Writing things down substitutes for internal workspace only for the part of the work that
is bookkeeping. It does not substitute for the part that decides what to write next.

An analogy. Doing long division on paper barely taxes working memory, because the paper
holds every intermediate and the procedure tells you where to look. Doing a proof on paper
is different: the paper holds the steps you have already taken, but choosing the next step
still happens in your head, and no amount of writing offloads that choice. Easy arithmetic
is long division. Competition maths is the proof.

So the prediction is:

**H1.** Chain of thought protects against J-space ablation, and the protection shrinks as
problems get harder. Formally, with `retention = accuracy(ablated) / accuracy(clean)`, the
protection gap `retention(CoT) - retention(direct)` is positive on GSM8K and decreases
monotonically across the difficulty tiers.

Secondary predictions, each of which could fail independently of H1:

**H2.** Under ablation the model writes longer chains of thought on the tiers where CoT
still protects it. If the page is standing in for the workspace, then losing the workspace
should push more content onto the page.

**H3.** The effect is specific to the workspace contents. Removing k random directions of
the same magnitude, and removing J-lens vectors from rank 1001 to 1010 instead of the top
10, both leave accuracy far closer to clean than the real ablation does.

**H4.** Perplexity, sentiment classification, grammatical acceptability, and verbatim
copying are close to unaffected under every ablation condition, reproducing the paper's
selectivity result on a much smaller model.

## What would falsify H1

The design has to make the opposite result visible, so both directions are named in
advance.

1. **Protection grows with difficulty.** Plausible mechanism: harder problems need more
   intermediate steps, so there is more to externalise and the page does proportionally
   more of the work. This would be a clean disconfirmation and is a real possibility.
2. **Protection is flat.** The two scratchpads are interchangeable across the whole range
   and the GSM8K result generalises.
3. **Protection is negative anywhere.** CoT would be a liability under ablation, perhaps
   because a long generation gives a damaged model more opportunities to go wrong.

## Design Risks

Qwen3-4B answering AIME directly, with no reasoning allowed, will be at or near zero
accuracy before anything is ablated. Retention is undefined when the denominator is zero,
and near-zero denominators make it wildly unstable. A downward trend in protection would
then be an artefact of the floor rather than evidence for H1, and it would look exactly
like a confirmation.

Three things guard against this:

1. Difficulty is a six-point ladder, not three datasets. MATH-500 is split by its own level
   labels, so there are intermediate tiers where both conditions are off the floor.
2. Clean accuracy per tier and per mode is reported next to every retention number, and any
   cell with clean accuracy below 5% is flagged and excluded from the trend claim rather
   than plotted as if it were informative.
3. If the trend is only visible in the floor-affected tiers, the pre-registered conclusion
   is that the result is inconclusive, not that H1 is supported.

## Hypothesis: 

I expect H1 to hold directionally but weakly, and I expect the floor problem to eat most of
the top of the difficulty range. My guess before running is that the informative comparison
will turn out to be GSM8K against MATH levels 3 and 4, and that AIME will produce a number
that cannot be interpreted. If that is how it comes out, the report says so.

---

## Amendment:

The original design generated chains of thought under each condition and graded the final
answer. It was replaced before any main run was executed, for one methodological reason and
one practical one. Both are recorded here rather than presented in the report as if they
had been the plan all along.

**Methodological.** Ablating the J-space changes what the model writes. A drop in accuracy
under the generation design is therefore ambiguous between two claims: the model needed its
workspace to reason over the page, or the model wrote a worse page and then reasoned over
that page correctly. The design could not separate them, and neither can the GSM8K result
in the paper.

**Practical.** No GPU was available. Sequential decoding of long chains of thought was not
feasible; forward passes were.

The replacement supplies the chain of thought instead of generating it, holding it
byte-identical across conditions, and sweeps how much of it is supplied. This removes the
confound and costs one forward pass per cell.

### Revised hypothesis

**H1'.** The gap between clean and ablated answer log-probability is largest when no
external scratchpad is supplied, and closes as more of the reference solution is supplied.
The value of f at which it closes, call it `closure_f`, rises with problem difficulty.

The original H1 and H1' make the same underlying claim. H1' measures it on a continuous
scale, which is what makes a run this small worth anything at all.

### What would falsify H1'

1. **The gap is flat in f.** The page does not substitute for the workspace, and the
   paper's interpretation of the GSM8K result does not generalise.
2. **The gap closes at the same f on every tier.** Substitution works, but its cost does
   not depend on difficulty, which is the more interesting half of H1.
3. **There is no measurable gap even at f = 0.** On a 4B model this is a live possibility,
   and it would mean the whole comparison is uninformative rather than that the hypothesis
   is disconfirmed.

### Potential expected outcome: 

I expect the f = 0 gap to be real but small, the curves to be noisy at n = 24 per tier, and
at least one tier to come back with an interval that includes zero. I do not expect to be
able to distinguish `closure_f` across adjacent tiers at this sample size. The most likely
honest outcome is a clear result on the pooled f = 0 comparison, a visible downward slope
in f, and an inconclusive verdict on the difficulty trend that H1 is actually about.
