"""
01_biased_scatterplots.py
============================
The hypothesis-driven test: LCOR (raw + LCOR_OE signature), in malignant
cells, vs CXCL16 (malignant, same patient), CXCR6 (CD8 T cells), and the
lab's own LCOR-induced T-cell signatures (UP24H, UP96H -- UP96H includes
CXCR6/ITGAE). CD8 lineage from CellTypist (03_tcell_scores).

8 outcome-predictor pairs x 2 datasets = 16 figures, Bassez-style
(per-patient scatter, subtype-colored, linear fit + 95% CI, R/p-value).

Output: 04_biased_analysis/figures/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

OUTDIR = "04_biased_analysis/figures"
os.makedirs(OUTDIR, exist_ok=True)
MIN_CD8 = 20

subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}


def bassez_plot(d, xcol, ycol, xlabel, ylabel, title, fname, ncells_col=None):
    dd = d.dropna(subset=[xcol, ycol])
    if len(dd) < 4 or np.std(dd[ycol].values) == 0:
        print(f"  SKIP {fname}: n={len(dd)}")
        return
    x, y = dd[xcol].values, dd[ycol].values
    r, p = stats.pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(7, 5.3))
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
        ax.scatter(x[m], y[m], color=color, s=70, label=f"{st} (n={m.sum()})", edgecolor="white", linewidth=0.5, zorder=3)
    stat_text = f"R = {r:.2f}\np-value = {p:.2g}"
    if ncells_col is not None and ncells_col in dd.columns:
        mean_ncells = dd[ncells_col].mean()
        stat_text += f"\nn patients = {n}\nmean cells/patient = {mean_ncells:.0f}"
    ax.text(0.02, 0.97, stat_text, transform=ax.transAxes, fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="Subtype", loc="best", fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png  (R={r:.2f}, p={p:.3g}, n={n})")


predictors = [
    ("LCOR_raw_pseudobulk", "LCOR in malignant cells (log1p CP10K)"),
    ("LCOR_OE_score", "LCOR_OE signature score (malignant cells)"),
]
outcomes = [
    ("CXCL16_raw_pseudobulk", "CXCL16 in malignant cells (log1p CP10K)", "n_malignant"),
    ("CXCR6_pseudobulk_AllT", "CXCR6 in T cells (log1p CP10K)", "n_AllT"),
    ("ITGAE_pseudobulk_AllT", "CD103 (ITGAE) in T cells (log1p CP10K)", "n_AllT"),
    ("UP24H_score_AllT", "T-cell UP24H signature score", "n_AllT"),
    ("UP96H_score_AllT", "T-cell UP96H signature score", "n_AllT"),
]

for name in ["Visvader", "Klughammer"]:
    print(f"\n=== {name} ===")
    mal = pd.read_csv(f"02_malignant_scores/figures/{name}_persample_malignant_scores.csv")
    tcl = pd.read_csv(f"03_tcell_scores/figures/{name}_persample_tcell_scores.csv")
    merged = mal.merge(tcl, on="sample", how="inner")
    merged.to_csv(f"{OUTDIR}/{name}_merged_data.csv", index=False)

    for xcol, xlabel in predictors:
        for ycol, ylabel, ncells_col in outcomes:
            d = merged[merged[ncells_col] >= MIN_CD8]
            bassez_plot(d, xcol, ycol, xlabel, ylabel, f"{name}: {xlabel} vs {ylabel}",
                        f"{name}_{xcol}_vs_{ycol}", ncells_col=ncells_col)

print("\nDONE")
