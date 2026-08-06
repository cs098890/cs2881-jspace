"""
Analysis for the scoring experiment.

Central quantity
----------------
    gap(tier, f, condition) = logprob_mean(clean) - logprob_mean(condition)

measured paired within problem, so each problem contributes its own difference
and the bootstrap resamples problems rather than cells.

The hypothesis is about the shape of gap versus f:
  - gap large at f = 0 and falling toward zero as f rises means the written
    page substitutes for the internal workspace.
  - gap flat in f means the page does not substitute at all.
  - closure_f, the smallest f at which the paired interval for the gap includes
    zero, is the summary number. Earlier closure means cheaper substitution.

The difficulty prediction is that closure_f rises with tier, or fails to exist
on the hard tiers.

Reporting rules built in
------------------------
Intervals that include zero are reported as including zero, not rounded toward
the hypothesis. A tier where the ablation produces no measurable gap even at
f = 0 is reported as a null result for that tier, and is excluded from the
closure analysis because there was nothing to close.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIER_LABELS = {1: "GSM8K", 2: "MATH L1-2", 3: "MATH L3",
               4: "MATH L4", 5: "MATH L5", 6: "AIME"}


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def paired_gaps(df, condition):
    """One row per (problem, f): clean minus condition, on mean logprob."""
    piv = df.pivot_table(index=["id", "tier", "f"], columns="condition",
                         values="logprob_mean")
    if "clean" not in piv or condition not in piv:
        return pd.DataFrame()
    out = piv[["clean", condition]].dropna().reset_index()
    out["gap"] = out["clean"] - out[condition]
    return out


def boot_ci(values, n_boot=4000, seed=0):
    if len(values) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def curve_table(df, condition):
    g = paired_gaps(df, condition)
    rows = []
    for (tier, f), sub in g.groupby(["tier", "f"]):
        lo, hi = boot_ci(sub.gap.values)
        rows.append({"tier": tier, "tier_label": TIER_LABELS.get(tier, tier),
                     "f": f, "condition": condition, "n": len(sub),
                     "gap": sub.gap.mean(), "lo": lo, "hi": hi,
                     "includes_zero": bool(lo <= 0 <= hi) if not np.isnan(lo) else None,
                     "clean_logprob": sub["clean"].mean()})
    return pd.DataFrame(rows).sort_values(["tier", "f"])


def closure(curve):
    """Smallest f whose interval includes zero, per tier."""
    rows = []
    for tier, sub in curve.groupby("tier"):
        sub = sub.sort_values("f")
        at_zero = sub[sub.f == 0.0]
        if len(at_zero) and at_zero.iloc[0]["includes_zero"]:
            rows.append({"tier": tier, "tier_label": TIER_LABELS.get(tier, tier),
                         "closure_f": None,
                         "note": "no measurable gap even at f=0; null result "
                                 "for this tier, excluded from the trend"})
            continue
        closed = sub[sub.includes_zero == True]  # noqa: E712
        rows.append({"tier": tier, "tier_label": TIER_LABELS.get(tier, tier),
                     "closure_f": float(closed.iloc[0].f) if len(closed) else None,
                     "note": "" if len(closed) else
                             "gap never closes within the tested range of f"})
    return pd.DataFrame(rows)


def plot_curves(curve, out_path, title):
    tiers = sorted(curve.tier.unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(tiers):
        d = curve[curve.tier == t].sort_values("f")
        c = cmap(i / max(len(tiers) - 1, 1))
        ax.plot(d.f, d.gap, marker="o", color=c, label=TIER_LABELS.get(t, t))
        ax.fill_between(d.f, d.lo, d.hi, color=c, alpha=0.15)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("fraction of the reference solution supplied (f)")
    ax.set_ylabel("clean minus ablated, mean logprob of the answer")
    ax.set_title(title)
    ax.legend(fontsize=8, title="difficulty")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_condition_compare(df, out_path, conditions):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for c in conditions:
        g = paired_gaps(df, c)
        if g.empty:
            continue
        m = g.groupby("f").gap.mean()
        cis = [boot_ci(g[g.f == f].gap.values) for f in m.index]
        lo = [x[0] for x in cis]
        hi = [x[1] for x in cis]
        ax.plot(m.index, m.values, marker="o", label=c)
        ax.fill_between(m.index, lo, hi, alpha=0.15)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("fraction of the reference solution supplied (f)")
    ax.set_ylabel("clean minus condition, mean logprob")
    ax.set_title("Ablation against matched controls, pooled over tiers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/scoring.jsonl")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args.results)
    print(f"{len(df)} rows, {df.id.nunique()} problems, "
          f"conditions: {sorted(df.condition.unique())}")

    conds = [c for c in ["jspace", "ctrl_random", "ctrl_rank"]
             if c in set(df.condition)]

    main_curve = curve_table(df, "jspace")
    main_curve.to_csv(f"{args.out_dir}/table1_gap_curve.csv", index=False)
    clo = closure(main_curve)
    clo.to_csv(f"{args.out_dir}/table2_closure.csv", index=False)

    plot_curves(main_curve, f"{args.out_dir}/fig1_dose_response.png",
                "Does written reasoning substitute for the ablated workspace?\n"
                "gap falling to zero = yes; flat = no")
    plot_condition_compare(df, f"{args.out_dir}/fig2_controls.png", conds)

    for c in conds:
        curve_table(df, c).to_csv(f"{args.out_dir}/curve_{c}.csv", index=False)

    print("\n=== Gap curve (jspace) ===")
    print(main_curve.to_string(index=False))
    print("\n=== Closure ===")
    print(clo.to_string(index=False))

    nulls = main_curve[(main_curve.f == 0.0) & (main_curve.includes_zero == True)]  # noqa: E712
    if len(nulls):
        print("\n[null] no measurable ablation effect at f=0 in these tiers. "
              "State this plainly in the report rather than pooling it away:")
        print(nulls[["tier_label", "gap", "lo", "hi", "n"]].to_string(index=False))

    if "ctrl_random" in conds:
        j = paired_gaps(df, "jspace").groupby("f").gap.mean()
        r = paired_gaps(df, "ctrl_random").groupby("f").gap.mean()
        ratio = (r / j.replace(0, np.nan)).mean()
        print(f"\n[specificity] matched random control recovers "
              f"{ratio:.0%} of the jspace gap on average. "
              "Near 100% means the effect is generic damage, not the workspace.")


if __name__ == "__main__":
    main()
