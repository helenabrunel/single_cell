"""
01_unbiased_scatterplots.py
==============================
Exploration beyond the specific CXCL16/CXCR6 hypothesis, focused on the
two phenotypic axes the working hypothesis actually describes: tissue
residency (TRM, within CD8) and (lower) exhaustion (CD8 and CD4, to check
specificity). Primary lineage/TRM calls from CellTypist; marker-based
(ITGAE/CD103+) TRM included as a secondary, independent check given the
CellTypist-vs-marker lineage concordance was only ~43-52% (expected,
dropout-driven -- see 03_tcell_scores README).

12 core figures (CellTypist-based) + 4 secondary (marker-based TRM) x 2
datasets.

Output: 05_unbiased_analysis/figures/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

OUTDIR = "05_unbiased_analysis/figures"
os.makedirs(OUTDIR, exist_ok=True)
MIN_CELLS = 20

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
# core (CellTypist-based)
outcomes_core = [
    ("TRM_fraction_CD8_celltypist", "Fraction of CD8 T cells = TRM (CellTypist)", "n_CD8_celltypist"),
]
# Exhaustion_score (PDCD1/LAG3/HAVCR2/TIGIT/CTLA4/TOX/ENTPD1) intentionally
# dropped from this unbiased block: picking that specific gene list to define
# "exhaustion" is itself a biased, hand-picked choice -- the same circularity
# concern as using CXCR6 to define TRM. TRM_fraction here, by contrast, comes
# entirely from CellTypist's own external reference classification.
# secondary (marker-based TRM, independent check)
outcomes_secondary = [
    ("TRM_fraction_CD8_marker", "Fraction of CD8 (marker-defined) = ITGAE/CD103+", "n_CD8_marker"),
]

for name in ["Visvader", "Klughammer"]:
    print(f"\n=== {name} ===")
    mal = pd.read_csv(f"02_malignant_scores/figures/{name}_persample_malignant_scores.csv")
    tcl = pd.read_csv(f"03_tcell_scores/figures/{name}_persample_tcell_scores.csv")
    merged = mal.merge(tcl, on="sample", how="inner")
    merged.to_csv(f"{OUTDIR}/{name}_merged_data.csv", index=False)

    for xcol, xlabel in predictors:
        for ycol, ylabel, mincol in outcomes_core:
            d = merged[merged[mincol] >= MIN_CELLS]
            bassez_plot(d, xcol, ycol, xlabel, ylabel, f"{name}: {xlabel} vs {ylabel}",
                        f"{name}_{xcol}_vs_{ycol}", ncells_col=mincol)
        for ycol, ylabel, mincol in outcomes_secondary:
            d = merged[merged[mincol] >= MIN_CELLS]
            bassez_plot(d, xcol, ycol, xlabel, ylabel, f"{name}: {xlabel} vs {ylabel} [secondary: marker-based]",
                        f"{name}_{xcol}_vs_{ycol}", ncells_col=mincol)

print("\nDONE")
