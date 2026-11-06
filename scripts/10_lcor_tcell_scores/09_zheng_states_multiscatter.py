"""
05_abe6474_multiscatter.py
==============================
Symmetric multiscatter (same logic as 03_multiscatter_categories.py) but for
the 6 Zheng et al. (abe6474) T-cell states instead of the CellTypist-derived
lineage categories: correlate LCOR (raw, LCOR_OE) against the proportion of
each state, side by side, none singled out.

Output: 05_unbiased_analysis/figures/{name}_abe6474_multiscatter_{LCOR_raw,LCOR_OE}.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

OUTDIR = "05_unbiased_analysis/figures"
os.makedirs(OUTDIR, exist_ok=True)
subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

STATES = ["Naive", "Effector_cytotoxic", "Effector_proliferative", "Stem_Tpex", "TRM", "Exhausted"]


def multiscatter(d, xcol, xlabel, fname, title_prefix):
    fig, axes = plt.subplots(1, len(STATES), figsize=(4.6 * len(STATES), 4.6))
    for ax, cat in zip(axes, STATES):
        dd = d.dropna(subset=[xcol, cat])
        x, y = dd[xcol].values, dd[cat].values
        if len(dd) < 4 or np.std(y) == 0:
            ax.set_title(f"{cat}\n(n={len(dd)}, insufficient variation)", fontsize=9)
            ax.axis("off")
            continue
        r, p = stats.pearsonr(x, y)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        yhat = intercept + slope * xs
        n = len(x); dof = n - 2
        x_mean, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
        resid = y - (intercept + slope * x)
        s_err = np.sqrt(np.sum(resid ** 2) / dof) if dof > 0 else np.nan
        t_val = stats.t.ppf(0.975, dof) if dof > 0 else np.nan
        ci = t_val * s_err * np.sqrt(1 / n + (xs - x_mean) ** 2 / sxx) if dof > 0 else np.zeros_like(xs)
        ax.fill_between(xs, yhat - ci, yhat + ci, color="gray", alpha=0.3, linewidth=0)
        ax.plot(xs, yhat, color="black", lw=1.5)
        for st, color in subtype_colors.items():
            m = dd["subtype"].values == st
            if m.sum() == 0:
                continue
            ax.scatter(x[m], y[m], color=color, s=50, label=st, edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(0.03, 0.97, f"R={r:.2f}\np={p:.2g}\nn={n}", transform=ax.transAxes, fontsize=9,
                va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel("Proportion of T cells", fontsize=8.5)
        ax.grid(alpha=0.15)
    axes[-1].legend(title="Subtype", loc="upper right", fontsize=7, title_fontsize=8)
    fig.suptitle(f"{title_prefix}: {xlabel} vs every abe6474 T-cell state (none singled out)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png")


for name in ["Visvader", "Klughammer"]:
    print(f"\n=== {name} ===")
    percell = pd.read_csv(f"03_tcell_scores/figures/{name}_percell_tcell_labels.csv")
    mal = pd.read_csv(f"02_malignant_scores/figures/{name}_persample_malignant_scores.csv")

    prop = pd.crosstab(percell["sample"], percell["state_abe6474"], normalize="index").reset_index()
    for cat in STATES:
        if cat not in prop.columns:
            prop[cat] = 0.0

    merged = mal.merge(prop, on="sample", how="inner")
    merged.to_csv(f"{OUTDIR}/{name}_abe6474_multiscatter_data.csv", index=False)

    multiscatter(merged, "LCOR_raw_pseudobulk", "LCOR (raw, malignant)",
                 f"{name}_abe6474_multiscatter_LCOR_raw", name)
    multiscatter(merged, "LCOR_OE_score", "LCOR_OE signature score (malignant)",
                 f"{name}_abe6474_multiscatter_LCOR_OE", name)

print("\nDONE")
