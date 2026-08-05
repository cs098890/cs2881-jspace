# Pre-registered hypothesis

**Recorded 2026-08-05, 18:33 ET, before any experiment was run.**
Git history of this file is the timestamp of record.

## Background

Gurnee et al. (2026) ablate the J-space (zeroing the residual stream's projection onto
the top-k active J-lens vectors, across a band of intermediate layers, at every token
position) and find that most capabilities survive but internal multi-step reasoning
collapses. Critically, GSM8K solved with explicit chain of thought is far more robust to
the ablation than the same problems answered directly. Their interpretation: written CoT
and the internal J-space are partially interchangeable scratchpads.

## H1 (primary)

**The protective effect of chain of thought against J-space ablation decreases as problem
difficulty increases.**

Concretely, define the *CoT protection margin* on a dataset as

    protection = (relative accuracy retained under ablation, with CoT)
               - (relative accuracy retained under ablation, direct)

where relative accuracy retained = acc_ablated / acc_clean within that condition.

I predict protection is largest on GSM8K, smaller on MATH-500, and smallest (possibly
zero or unmeasurable) on AIME.

### Why

CoT can substitute for the internal workspace only to the extent that the reasoning state
is *fully externalizable as text*. On GSM8K, each step is arithmetically simple and the
written line ("48 / 2 = 24") captures essentially the entire state carried forward; the
internal workspace is close to redundant, so removing it costs little.

On harder problems, much of the work is not executing a step but *choosing* one: search
over strategies, holding several candidate approaches in parallel, noticing structure,
backtracking. This is exactly the flexible, non-automatic cognition the paper argues the
J-space mediates, and it is largely *not* written down — the CoT records the result of a
selection, not the deliberation that produced it. So the external scratchpad should
substitute less well as difficulty rises.

## H2 (competing hypothesis, must remain visible)

**Protection is constant or increases with difficulty.** Harder problems require holding
*more* intermediate state, so externalizing it could matter more, not less. If the J-space
is mainly a working-memory buffer for intermediate values (rather than for strategy
selection), then writing those values down should help more on harder problems, not less.

The experiment is designed so that H2 would be clearly visible if true: protection is
measured per-dataset on a common scale, and the difficulty ordering is fixed in advance,
so a flat or increasing trend is as readable as a decreasing one.

## H3 (null)

**J-space ablation degrades the model roughly uniformly**, and the apparent CoT advantage
is an artifact of CoT conditions simply having more tokens over which to recover, or of
the ablation being a generic perturbation. The random-direction control exists to detect
this.

## Predictions recorded in advance

Numbers are guesses stated for the record, not fitted to anything:

| Dataset | clean CoT acc | clean direct acc | expected CoT retention | expected direct retention |
|---|---|---|---|---|
| GSM8K | 0.70-0.85 | 0.20-0.40 | 0.75-0.95 | 0.25-0.50 |
| MATH-500 | 0.40-0.60 | 0.10-0.25 | 0.55-0.80 | 0.20-0.45 |
| AIME 2025 | 0.03-0.15 | 0.00-0.05 | unmeasurable (floor) | unmeasurable (floor) |

I expect the random-direction control to sit near the clean baseline in all conditions.

## Anticipated problems, recorded in advance

1. **Floor effects.** AIME direct-answer accuracy with a 4B model in non-thinking mode may
   be at or near zero even without ablation, making retention ratios undefined or
   dominated by noise. Mitigation: report absolute accuracies and clean baselines
   alongside ratios; additionally report *clean-conditional accuracy* (accuracy under
   ablation restricted to the problems the model solved when clean), which keeps headroom
   visible even when the overall rate is low. If AIME is at floor in every condition, H1
   is simply untestable at that difficulty tier and I will say so rather than reading a
   trend into noise.

2. **Ceiling / length confound.** CoT conditions produce more tokens than direct
   conditions. More tokens means more opportunities for the ablation to bite, but also
   more opportunity to self-correct. This is intrinsic to the comparison and cannot be
   fully removed; noted as a limitation.

3. **Small samples.** n is 30-40 per dataset, so binomial confidence intervals will be
   wide (roughly +/-15pp). Differences smaller than about 20pp will not be resolvable, and
   I will report them as inconclusive rather than as trends.

## What would falsify H1

- CoT retention on MATH-500 and AIME within confidence intervals of GSM8K retention
  (flat protection) -> H1 not supported, H2 favored.
- Random-direction control degrading accuracy as much as J-space ablation -> H3, the
  effect is generic damage and nothing here is about the J-space specifically.
- Direct-condition accuracy unaffected by ablation -> the premise of the original result
  does not replicate at this model scale.
