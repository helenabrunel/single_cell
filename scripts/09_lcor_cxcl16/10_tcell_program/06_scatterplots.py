"""
06_bassez_style_scatterplots.py
==================================
Regenerates all LCOR/LCOR_OE vs CXCL16/CXCR6/Tcell_UP24H/Tcell_UP96H
correlations in the visual style of the Bassez_LCOR_vs_ITGAE.png
reference figure: per-patient scatter (one color per patient), linear
regression line with 95% CI band, R (Pearson) and p-value annotated
top-left, patient legend on the right.

16 figures: {Visvader, Klughammer} x {LCOR_raw, LCOR_OE_score} x
{CXCL16 (malignant), CXCR6 (CD8), Tcell_UP24H (CD8), Tcell_UP96H (CD8)}

Reuses per-patient data already computed in scripts 02-05 of this folder,
plus adds LCOR raw pseudobulk per patient (not previously saved for
Klughammer; already available for Visvader in 03_pseudobulk).

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/06_scatterplots/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/06_scatterplots"
os.makedirs(OUTDIR, exist_ok=True)

def get_symbol_matrix(adata, genes):
    present_direct = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    resolved = {g: g for g in present_direct}
    if missing and adata.raw is not None and "feature_name" in adata.raw.var.columns:
        sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
        for g in missing:
            if g in sym2ens.index:
                ens = sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]
                if ens in adata.var_names:
                    resolved[g] = ens
    return resolved

def get_raw_counts(adata, symbol):
    """Always fetches true raw counts for a gene symbol via raw.var['feature_name'] -> Ensembl ID -> adata.raw.X,
    regardless of whether the symbol also happens to appear directly in adata.var_names (used for lognorm .X)."""
    sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
    match = sym2ens[symbol]
    ens = match if np.isscalar(match) else match.iloc[0]
    col = adata.raw[:, ens].X
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

# ============================================================================
# Gather LCOR_raw pseudobulk per patient -- Visvader (malignant)
# ============================================================================
print("Computing LCOR_raw pseudobulk (Visvader, malignant)...")
adata = sc.read_h5ad("compartment_epithelial.h5ad")
idx = adata.var_names.get_loc("LCOR")
raw = adata.layers["counts"][:, idx]
raw = np.asarray(raw.todense()).ravel() if issparse(raw) else np.asarray(raw).ravel()
obs = pd.DataFrame({"sample": adata.obs["sample"].astype(str).values,
                     "total_counts": adata.obs["total_counts"].values.astype(float),
                     "LCOR_raw": raw.astype(float)})
del adata
lcor_visvader = obs.groupby("sample").apply(
    lambda g: np.log1p(1e4 * g["LCOR_raw"].sum() / g["total_counts"].sum())
).rename("LCOR_raw_pseudobulk").reset_index()

# ============================================================================
# Gather LCOR_raw pseudobulk per patient -- Klughammer (malignant)
# ============================================================================
print("Computing LCOR_raw pseudobulk (Klughammer, malignant)...")
adata = sc.read_h5ad("/home/hbrunel/single_cell/Klughammer/klughammer_Malignant.h5ad")
resolved = get_symbol_matrix(adata, ["LCOR"])
raw = get_raw_counts(adata, "LCOR")
obs = pd.DataFrame({"sample": adata.obs["donor_id"].astype(str).values,
                     "total_counts": adata.obs["total_counts"].values.astype(float),
                     "LCOR_raw": raw.astype(float)})
del adata
lcor_klughammer = obs.groupby("sample").apply(
    lambda g: np.log1p(1e4 * g["LCOR_raw"].sum() / g["total_counts"].sum())
).rename("LCOR_raw_pseudobulk").reset_index()

# ============================================================================
# Assemble master per-dataset tables
# ============================================================================
vis_lcoroe = pd.read_csv("LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/02_visvader_analysis/persample_scores.csv")
vis_cxcl16cxcr6 = pd.read_csv("LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/04_visvader_lcoroe_vs_cxcl16_cxcr6/persample_data.csv")
visvader_master = lcor_visvader.merge(vis_lcoroe.drop(columns=["n_cd8"]), on="sample", how="inner").merge(
    vis_cxcl16cxcr6[["sample", "CXCL16_pseudobulk", "CXCR6_pseudobulk", "n_cd8"]], on="sample", how="left")

klu_lcoroe = pd.read_csv("LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/03_klughammer_analysis/persample_scores.csv")
klu_cxcl16cxcr6 = pd.read_csv("LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/05_klughammer_lcoroe_vs_cxcl16_cxcr6/persample_data.csv")
klughammer_master = lcor_klughammer.merge(klu_lcoroe.drop(columns=["n_cd8"]), on="sample", how="inner").merge(
    klu_cxcl16cxcr6[["sample", "CXCL16_pseudobulk", "CXCR6_pseudobulk", "n_cd8"]], on="sample", how="left")

visvader_master.to_csv(f"{OUTDIR}/visvader_master_data.csv", index=False)
klughammer_master.to_csv(f"{OUTDIR}/klughammer_master_data.csv", index=False)
print(f"\nVisvader master: {len(visvader_master)} patients")
print(f"Klughammer master: {len(klughammer_master)} patients")

# ============================================================================
# Bassez-style plotting function
# ============================================================================
def bassez_plot(df, xcol, ycol, xlabel, ylabel, title, fname, min_cd8=None):
    d = df.dropna(subset=[xcol, ycol]).copy()
    if min_cd8 is not None and "n_cd8" in d.columns:
        d = d[d["n_cd8"] >= min_cd8]
    if len(d) < 4:
        print(f"  SKIP {fname}: n={len(d)} too small")
        return
    x, y = d[xcol].values, d[ycol].values
    r, p = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    slope, intercept, _, _, se = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    yhat = intercept + slope * xs
    n = len(x)
    dof = n - 2
    x_mean, sxx = x.mean(), np.sum((x - x.mean())**2)
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid**2) / dof) if dof > 0 else np.nan
    t_val = stats.t.ppf(0.975, dof) if dof > 0 else np.nan
    ci = t_val * s_err * np.sqrt(1/n + (xs - x_mean)**2 / sxx) if dof > 0 else np.zeros_like(xs)
    ax.fill_between(xs, yhat - ci, yhat + ci, color="gray", alpha=0.3, linewidth=0)
    ax.plot(xs, yhat, color="black", lw=1.5)

    subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
    subtypes = d["subtype"].values
    for st, color in subtype_colors.items():
        m = subtypes == st
        if m.sum() == 0:
            continue
        ax.scatter(x[m], y[m], color=color, s=70, label=f"{st} (n={m.sum()})", edgecolor="white", linewidth=0.5, zorder=3)

    ax.text(0.02, 0.97, f"R = {r:.2f}\np-value = {p:.2g}", transform=ax.transAxes,
            fontsize=12, va="top", ha="left")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="Subtype", loc="best", fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png  (R={r:.2f}, p={p:.3g}, n={n})")

# ============================================================================
# Generate all 16 combinations
# ============================================================================
outcomes = [
    ("CXCL16_pseudobulk", "CXCL16 in malignant cells (log1p CP10K)", None),
    ("CXCR6_pseudobulk", "CXCR6 in CD8 T cells (log1p CP10K)", 20),
    ("mean_UP24H", "T-cell UP24H signature score (CD8)", 20),
    ("mean_UP96H", "T-cell UP96H signature score (CD8)", 20),
]
predictors = [
    ("LCOR_raw_pseudobulk", "LCOR in malignant cells (log1p CP10K)"),
    ("mean_LCOR_OE_score", "LCOR_OE signature score (malignant cells)"),
]

for dset_name, df in [("Visvader", visvader_master), ("Klughammer", klughammer_master)]:
    print(f"\n=== {dset_name} ===")
    for xcol, xlabel in predictors:
        for ycol, ylabel, min_cd8 in outcomes:
            fname = f"{dset_name}_{xcol}_vs_{ycol}"
            bassez_plot(df, xcol, ycol, xlabel, ylabel, f"{dset_name}: {xlabel} vs {ylabel}", fname, min_cd8)

print("\nDONE")
