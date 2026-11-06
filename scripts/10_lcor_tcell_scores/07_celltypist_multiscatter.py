"""
03_multiscatter_categories.py
================================
Fully symmetric version of the unbiased test: instead of singling out
TRM_fraction as THE outcome to report (itself a choice that could look
directed), correlate LCOR (raw, LCOR_OE) against the proportion of EVERY
lineage/state category from the composition (02_composition_plot.py) --
CD8-TRM, CD8-non-TRM, CD4, Treg, Other (non-T/unconventional T) -- side by
side in one multi-panel figure per dataset x predictor. No category is
treated differently from the others.

4 figures (2 datasets x {LCOR_raw, LCOR_OE}), 5 panels each.

Output: 05_unbiased_analysis/figures/{name}_multiscatter_{LCOR_raw,LCOR_OE}.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

OUTDIR = "05_unbiased_analysis/figures"
os.makedirs(OUTDIR, exist_ok=True)
subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

CATEGORIES = ["CD8 - TRM", "CD8 - non-TRM", "CD4", "Treg", "Other (non-T / unconventional T)"]


def display_category(row):
    if row["lineage_celltypist"] == "CD8":
        return "CD8 - TRM" if row["is_trm_celltypist"] else "CD8 - non-TRM"
    if row["lineage_celltypist"] == "CD4":
        return "CD4"
    if row["lineage_celltypist"] == "Treg":
        return "Treg"
    return "Other (non-T / unconventional T)"


def multiscatter(d, xcol, xlabel, fname, title_prefix):
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(5 * len(CATEGORIES), 4.6))
    for ax, cat in zip(axes, CATEGORIES):
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
    fig.suptitle(f"{title_prefix}: {xlabel} vs every T-cell category (none singled out)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png")


for name in ["Visvader", "Klughammer"]:
    print(f"\n=== {name} ===")
    percell = pd.read_csv(f"03_tcell_scores/figures/{name}_percell_tcell_labels.csv")
    mal = pd.read_csv(f"02_malignant_scores/figures/{name}_persample_malignant_scores.csv")

    percell["category"] = percell.apply(display_category, axis=1)
    prop = pd.crosstab(percell["sample"], percell["category"], normalize="index").reset_index()
    for cat in CATEGORIES:
        if cat not in prop.columns:
            prop[cat] = 0.0

    merged = mal.merge(prop, on="sample", how="inner")
    merged.to_csv(f"{OUTDIR}/{name}_multiscatter_data.csv", index=False)

    multiscatter(merged, "LCOR_raw_pseudobulk", "LCOR (raw, malignant)",
                 f"{name}_multiscatter_LCOR_raw", name)
    multiscatter(merged, "LCOR_OE_score", "LCOR_OE signature score (malignant)",
                 f"{name}_multiscatter_LCOR_OE", name)

print("\nDONE")
