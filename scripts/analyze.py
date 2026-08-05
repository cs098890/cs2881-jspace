"""Score raw generations, compute retention / protection with CIs, and make figures.

Usage: uv run python scripts/analyze.py --results results
"""

from __future__ import annotations

import argparse, json, math, pathlib, sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["gsm8k", "math500", "aime2025"]
LABELS = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime2025": "AIME 2025"}
MODES = ["none", "jspace", "random"]
MODE_LABEL = {"none": "clean", "jspace": "J-space ablated", "random": "random-direction control"}
COLORS = {"none": "#4C6EF5", "jspace": "#E8590C", "random": "#868E96"}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval; behaves sensibly at 0 and at small n."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)

    rows = [json.loads(l) for l in (rdir / "raw_generations.jsonl").open()]
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no results")
    # de-duplicate in case of re-runs: keep the last record per cell+problem
    df = df.drop_duplicates(subset=["dataset", "condition", "mode", "problem"], keep="last")

    # ---- accuracy table with Wilson CIs ----
    recs = []
    for (ds, cond, mode), g in df.groupby(["dataset", "condition", "mode"]):
        p, lo, hi = wilson(int(g.correct.sum()), len(g))
        recs.append(dict(dataset=ds, condition=cond, mode=mode, n=len(g),
                         n_correct=int(g.correct.sum()), acc=p, lo=lo, hi=hi,
                         mean_chars=g.n_completion_chars.mean()))
    acc = pd.DataFrame(recs)
    acc.to_csv(rdir / "accuracy.csv", index=False)

    # ---- retention and protection ----
    ret = []
    for ds in df.dataset.unique():
        for cond in ["cot", "direct"]:
            sub = acc[(acc.dataset == ds) & (acc.condition == cond)]
            base = sub[sub["mode"] == "none"].acc
            if base.empty or base.iloc[0] == 0:
                r_j = r_r = float("nan")
            else:
                b = base.iloc[0]
                jj = sub[sub["mode"] == "jspace"].acc
                rr = sub[sub["mode"] == "random"].acc
                r_j = (jj.iloc[0] / b) if not jj.empty else float("nan")
                r_r = (rr.iloc[0] / b) if not rr.empty else float("nan")
            ret.append(dict(dataset=ds, condition=cond,
                            clean_acc=base.iloc[0] if not base.empty else float("nan"),
                            retention_jspace=r_j, retention_random=r_r))
    ret = pd.DataFrame(ret)

    prot = []
    for ds in ret.dataset.unique():
        c = ret[(ret.dataset == ds) & (ret.condition == "cot")]
        d = ret[(ret.dataset == ds) & (ret.condition == "direct")]
        if c.empty or d.empty:
            continue
        prot.append(dict(dataset=ds,
                         retention_cot=c.retention_jspace.iloc[0],
                         retention_direct=d.retention_jspace.iloc[0],
                         protection=c.retention_jspace.iloc[0] - d.retention_jspace.iloc[0],
                         clean_cot=c.clean_acc.iloc[0], clean_direct=d.clean_acc.iloc[0]))
    prot = pd.DataFrame(prot)
    ret.to_csv(rdir / "retention.csv", index=False)
    prot.to_csv(rdir / "protection.csv", index=False)

    print("\n=== accuracy ===\n", acc.to_string(index=False))
    print("\n=== retention (ablated / clean) ===\n", ret.to_string(index=False))
    print("\n=== CoT protection margin ===\n", prot.to_string(index=False))

    # ---- figure 1: accuracy grid ----
    present = [d for d in DATASETS if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], present):
        sub = acc[acc.dataset == ds]
        x, w = range(2), 0.26
        for i, mode in enumerate(MODES):
            vals, los, his = [], [], []
            for cond in ["cot", "direct"]:
                r = sub[(sub.condition == cond) & (sub["mode"] == mode)]
                v = r.acc.iloc[0] if not r.empty else float("nan")
                vals.append(v)
                los.append(v - (r.lo.iloc[0] if not r.empty else v))
                his.append((r.hi.iloc[0] if not r.empty else v) - v)
            ax.bar([xx + (i - 1) * w for xx in x], vals, w, label=MODE_LABEL[mode],
                   color=COLORS[mode], yerr=[los, his], capsize=3, error_kw=dict(lw=1))
        ax.set_xticks(list(x)); ax.set_xticklabels(["chain of thought", "direct"])
        ax.set_title(LABELS.get(ds, ds)); ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
        ax.grid(axis="y", alpha=0.3)
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Qwen3-4B accuracy under J-space ablation (95% Wilson intervals)", fontsize=11)
    fig.tight_layout(); fig.savefig(rdir / "fig1_accuracy.png", dpi=160)

    # ---- figure 2: protection margin vs difficulty ----
    if not prot.empty:
        fig2, ax = plt.subplots(figsize=(5.2, 3.6))
        order = [d for d in DATASETS if d in set(prot.dataset)]
        p2 = prot.set_index("dataset").loc[order]
        ax.plot(range(len(order)), p2.retention_cot, "o-", color=COLORS["jspace"], label="CoT retention")
        ax.plot(range(len(order)), p2.retention_direct, "s--", color=COLORS["none"], label="direct retention")
        ax.set_xticks(range(len(order))); ax.set_xticklabels([LABELS[d] for d in order])
        ax.set_ylabel("accuracy retained under ablation"); ax.set_ylim(0, 1.35)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_title("Does CoT keep protecting as difficulty rises?", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig2.tight_layout(); fig2.savefig(rdir / "fig2_protection.png", dpi=160)

    print(f"\nwrote figures + csvs to {rdir}")


if __name__ == "__main__":
    main()
