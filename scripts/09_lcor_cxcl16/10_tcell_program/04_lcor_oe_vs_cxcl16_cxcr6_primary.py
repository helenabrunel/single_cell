"""
04_visvader_lcoroe_vs_cxcl16_cxcr6.py
========================================
The ORIGINAL correlation test, redone with the LCOR_OE
signature instead of raw LCOR (since raw LCOR is too lowly expressed):
  - LCOR_OE score (malignant cells) vs CXCL16 (malignant cells), same patient
  - LCOR_OE score (malignant cells) vs CXCR6 (CD8 T cells), same patient
Direct analogue of 03_pseudobulk/01 (LCOR vs CXCL16) and the raw-LCOR vs
CXCR6 tests earlier in this project, substituting the signature for raw
LCOR. Reuses the LCOR_OE_score already computed per patient in
02_visvader_analysis.py.

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/04_visvader_lcoroe_vs_cxcl16_cxcr6/
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

PRIOR_SCORES = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/02_visvader_analysis/persample_scores.csv"
OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/04_visvader_lcoroe_vs_cxcl16_cxcr6"
os.makedirs(OUTDIR, exist_ok=True)

IMMUNE_CELL_TYPES = {
    "CD8_T": "score_CD8_T", "CD4_T": "score_CD4_T", "Treg": "score_Treg", "NK": "score_NK",
    "Macro_M1": "score_Macro_M1", "Macro_M2": "score_Macro_M2", "B_cell": "score_B_cell",
    "Plasmablast": "score_Plasmablast", "Dendritic": "score_Dendritic", "Mast": "score_Mast",
}
def assign_immune_type(obs):
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items() if col in obs.columns}
    score_matrix = obs[[col for col in cols_present.values()]].values
    cell_types = list(cols_present.keys())
    best_idx = score_matrix.argmax(axis=1)
    best_score = score_matrix.max(axis=1)
    return np.array([cell_types[i] if s > 0 else "Unknown" for i, s in zip(best_idx, best_score)])

lcor_scores = pd.read_csv(PRIOR_SCORES)[["sample", "subtype", "mean_LCOR_OE_score"]].drop_duplicates()
print(f"LCOR_OE scores loaded: {len(lcor_scores)} patients")

# ============================================================================
# CXCL16 pseudobulk, malignant cells
# ============================================================================
print("\nLoading compartment_epithelial.h5ad for CXCL16...")
adata_mal = sc.read_h5ad("compartment_epithelial.h5ad")
idx = adata_mal.var_names.get_loc("CXCL16")
raw = adata_mal.layers["counts"][:, idx]
raw = np.asarray(raw.todense()).ravel() if issparse(raw) else np.asarray(raw).ravel()
obs = pd.DataFrame({
    "sample": adata_mal.obs["sample"].astype(str).values,
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
print("\nLoading compartment_immune.h5ad for CXCR6...")
adata_imm = sc.read_h5ad("compartment_immune.h5ad")
celltype = assign_immune_type(adata_imm.obs)
is_cd8 = celltype == "CD8_T"
idx = adata_imm.var_names.get_loc("CXCR6")
raw = adata_imm.layers["counts"][:, idx]
raw = np.asarray(raw.todense()).ravel() if issparse(raw) else np.asarray(raw).ravel()
obs_imm = pd.DataFrame({
    "sample": adata_imm.obs["sample"].astype(str).values,
    "total_counts": adata_imm.obs["total_counts"].values.astype(float),
    "CXCR6_raw": raw.astype(float),
})[is_cd8]
del adata_imm
cxcr6 = obs_imm.groupby("sample").apply(
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
    "04 -- LCOR_OE score vs CXCL16 (malignant) and vs CXCR6 (CD8 T cells), Visvader",
    "="*70,
    "",
    "This is the ORIGINAL correlation test, redone with the",
    "LCOR_OE signature instead of raw LCOR expression.",
    "",
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

# subtype-adjusted versions
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

# ============================================================================
# Figures
# ============================================================================
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
fig.suptitle("Visvader -- LCOR_OE signature (malignant) vs CXCL16 (malignant) / CXCR6 (CD8 T cells)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUTDIR}/scatter.png")

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"Saved: {OUTDIR}/README.txt")
print("\nDONE")
