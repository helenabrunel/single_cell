"""
05_klughammer_lcoroe_vs_cxcl16_cxcr6.py
==========================================
Replication of 04_visvader_lcoroe_vs_cxcl16_cxcr6.py in Klughammer
(independent atlas). LCOR_OE score (malignant) vs CXCL16 (malignant) and
vs CXCR6 (CD8 T cells), same patient.

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/05_klughammer_lcoroe_vs_cxcl16_cxcr6/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

PRIOR_SCORES = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/03_klughammer_analysis/persample_scores.csv"
OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/05_klughammer_lcoroe_vs_cxcl16_cxcr6"
MAL_FILE = "/home/hbrunel/single_cell/Klughammer/klughammer_Malignant.h5ad"
LYM_FILE = "/home/hbrunel/single_cell/Klughammer/klughammer_Lymphoid.h5ad"
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

lcor_scores = pd.read_csv(PRIOR_SCORES)[["sample", "subtype", "mean_LCOR_OE_score"]].drop_duplicates()
print(f"LCOR_OE scores loaded: {len(lcor_scores)} patients")

# ============================================================================
# CXCL16 pseudobulk, malignant cells
# ============================================================================
print(f"\nLoading {MAL_FILE} for CXCL16...")
adata_mal = sc.read_h5ad(MAL_FILE)
resolved = get_symbol_matrix(adata_mal, ["CXCL16"])
raw = get_raw_counts(adata_mal, "CXCL16")
obs = pd.DataFrame({
    "sample": adata_mal.obs["donor_id"].astype(str).values,
    "total_counts": adata_mal.obs["total_counts"].values.astype(float),
    "CXCL16_raw": raw.astype(float),
})
del adata_mal
cxcl16 = obs.groupby("sample").apply(
    lambda g: np.log1p(1e4 * g["CXCL16_raw"].sum() / g["total_counts"].sum())
).rename("CXCL16_pseudobulk").reset_index()
print(f"  Patients: {len(cxcl16)}")

# ============================================================================
# CXCR6 pseudobulk, CD8 T cells
# ============================================================================
print(f"\nLoading {LYM_FILE} for CXCR6...")
adata_lym = sc.read_h5ad(LYM_FILE)
is_t = adata_lym.obs["author_cell_type"].astype(str) == "T"
adata_t = adata_lym[is_t].copy()
del adata_lym

cd8_genes = get_symbol_matrix(adata_t, ["CD8A", "CD8B", "CD4", "CXCR6"])
def get_log(g):
    col = adata_t[:, cd8_genes[g]].X
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
cd8_score = np.mean(np.column_stack([get_log("CD8A"), get_log("CD8B")]), axis=1)
cd4_score = get_log("CD4")
is_cd8 = cd8_score > cd4_score
print(f"  CD8 T cells: {is_cd8.sum():,} / {adata_t.n_obs:,}")

raw_cxcr6 = get_raw_counts(adata_t, "CXCR6")
obs_t = pd.DataFrame({
    "sample": adata_t.obs["donor_id"].astype(str).values,
    "total_counts": adata_t.obs["total_counts"].values.astype(float),
    "CXCR6_raw": raw_cxcr6.astype(float),
})[is_cd8]
del adata_t
cxcr6 = obs_t.groupby("sample").apply(
    lambda g: pd.Series({
        "CXCR6_pseudobulk": np.log1p(1e4 * g["CXCR6_raw"].sum() / g["total_counts"].sum()),
        "n_cd8": len(g),
    })
).reset_index()
print(f"  Patients: {len(cxcr6)}")

# ============================================================================
# Merge + correlate
# ============================================================================
MIN_CD8 = 20
merged = lcor_scores.merge(cxcl16, on="sample", how="inner").merge(cxcr6, on="sample", how="left")
merged.to_csv(f"{OUTDIR}/persample_data.csv", index=False)

readme = [
    "05 -- LCOR_OE score vs CXCL16 (malignant) and vs CXCR6 (CD8 T cells), Klughammer",
    "="*70, "",
]

print("\n" + "="*65)
print(" LCOR_OE (malignant) vs CXCL16 (malignant), same patient")
print("="*65)
n1 = len(merged)
rho1, p1 = stats.spearmanr(merged["mean_LCOR_OE_score"], merged["CXCL16_pseudobulk"])
print(f"  n={n1}: Spearman rho={rho1:.3f}, p={p1:.3g}")
readme += [f"LCOR_OE vs CXCL16 (malignant, n={n1}): Spearman rho={rho1:.3f}, p={p1:.3g}"]

merged_cd8 = merged[merged["n_cd8"] >= MIN_CD8]
print("\n" + "="*65)
print(f" LCOR_OE (malignant) vs CXCR6 (CD8 T cells, n_cd8>={MIN_CD8}), same patient")
print("="*65)
n2 = len(merged_cd8)
rho2, p2 = stats.spearmanr(merged_cd8["mean_LCOR_OE_score"], merged_cd8["CXCR6_pseudobulk"])
print(f"  n={n2}: Spearman rho={rho2:.3f}, p={p2:.3g}")
readme += [f"LCOR_OE (malignant) vs CXCR6 (CD8 T cells, n={n2}, n_cd8>={MIN_CD8}): Spearman rho={rho2:.3f}, p={p2:.3g}"]

def resid(y, cat):
    return y - y.groupby(cat).transform("mean")
merged["cat"] = pd.Categorical(merged["subtype"])
r1 = resid(merged["mean_LCOR_OE_score"], merged["cat"])
r2 = resid(merged["CXCL16_pseudobulk"], merged["cat"])
rho1adj, p1adj = stats.spearmanr(r1, r2)
print(f"\n  [subtype-adjusted] LCOR_OE vs CXCL16: rho={rho1adj:.3f}, p={p1adj:.3g}")
readme += [f"[subtype-adjusted] LCOR_OE vs CXCL16: rho={rho1adj:.3f}, p={p1adj:.3g}"]

merged_cd8 = merged_cd8.copy()
merged_cd8["cat"] = pd.Categorical(merged_cd8["subtype"])
r3 = resid(merged_cd8["mean_LCOR_OE_score"], merged_cd8["cat"])
r4 = resid(merged_cd8["CXCR6_pseudobulk"], merged_cd8["cat"])
rho2adj, p2adj = stats.spearmanr(r3, r4)
print(f"  [subtype-adjusted] LCOR_OE vs CXCR6: rho={rho2adj:.3f}, p={p2adj:.3g}")
readme += [f"[subtype-adjusted] LCOR_OE vs CXCR6: rho={rho2adj:.3f}, p={p2adj:.3g}"]

subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for st, color in subtype_colors.items():
    d = merged[merged.subtype == st]
    axes[0].scatter(d["mean_LCOR_OE_score"], d["CXCL16_pseudobulk"], color=color, s=50, alpha=0.85, edgecolor="white", label=st)
    d2 = merged_cd8[merged_cd8.subtype == st]
    axes[1].scatter(d2["mean_LCOR_OE_score"], d2["CXCR6_pseudobulk"], color=color, s=50, alpha=0.85, edgecolor="white", label=st)
axes[0].set_xlabel("LCOR_OE score (malignant, mean/patient)", fontsize=9)
axes[0].set_ylabel("CXCL16 pseudobulk (malignant, log1p CP10K)", fontsize=9)
axes[0].set_title(f"rho={rho1:.2f}, p={p1:.3g}, n={n1}", fontsize=10, fontweight="bold")
axes[1].set_xlabel("LCOR_OE score (malignant, mean/patient)", fontsize=9)
axes[1].set_ylabel("CXCR6 pseudobulk (CD8 T cells, log1p CP10K)", fontsize=9)
axes[1].set_title(f"rho={rho2:.2f}, p={p2:.3g}, n={n2}", fontsize=10, fontweight="bold")
for ax in axes:
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.suptitle("Klughammer -- LCOR_OE signature (malignant) vs CXCL16 (malignant) / CXCR6 (CD8 T cells)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUTDIR}/scatter.png")

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"Saved: {OUTDIR}/README.txt")
print("\nDONE")
