"""
02_umap_trio_lcor_apm_esr1_myeloid.py
=======================================
Myeloid-compartment analogue of 01_umap_trio_lcor_apm_esr1.py -- UMAP
feature plots for LCOR / APM-score / ESR1 in the MYELOID cells
(Macro_M1+Macro_M2+Dendritic) of Visvader's compartment_immune.h5ad, where
this whole investigation started (the original myeloid question, and
05_myeloid/04_depth_artifact_apm_myeloid_cell_level.py's positive-control
result). Klughammer has no myeloid compartment (malignant cells only), so
this is Visvader-only. Not split by subtype (myeloid cell counts per
patient/subtype are already low, see 05_myeloid caveats) -- one figure
across all myeloid cells, colored by subtype for reference.

Output: LCOR-CXCL16/08_umap_gene_trios/figures/02_umap_trio_lcor_apm_esr1_myeloid/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "compartment_immune.h5ad"
OUTDIR = "LCOR-CXCL16/08_umap_gene_trios/figures/02_umap_trio_lcor_apm_esr1_myeloid"
APM_GENES = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
GENES = ["LCOR", "APM", "ESR1"]
MYELOID_GROUPS = ["Macro_M1", "Macro_M2", "Dendritic"]
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
    return np.array([cell_types[i] if s > 0 else "Unknown"
                      for i, s in zip(best_idx, best_score)])

print(f"  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
celltype = assign_immune_type(adata.obs)
is_myeloid = np.isin(celltype, MYELOID_GROUPS)
print(f"  Myeloid cells: {is_myeloid.sum():,} / {adata.n_obs:,}")

needed = ["LCOR", "ESR1"] + APM_GENES
idx_of = {g: adata.var_names.get_loc(g) for g in needed}
def get_raw(g):
    col = adata.layers["counts"][:, idx_of[g]]
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

total_counts = adata.obs["total_counts"].values.astype(float)
raw_lcor = get_raw("LCOR")
raw_esr1 = get_raw("ESR1")
raw_apm = np.column_stack([get_raw(g) for g in APM_GENES]).sum(axis=1)
umap = adata.obsm["X_umap"]

df = pd.DataFrame({
    "UMAP1": umap[:, 0], "UMAP2": umap[:, 1],
    "LCOR": np.log1p(1e4 * raw_lcor / total_counts),
    "APM": np.log1p(1e4 * raw_apm / total_counts),
    "ESR1": np.log1p(1e4 * raw_esr1 / total_counts),
    "celltype": celltype,
    "subtype": adata.obs["subtype"].astype(str).values,
})
del adata
df_myeloid = df[is_myeloid].copy()

# Full-immune-compartment background context figure (which cells are myeloid)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(df["UMAP1"], df["UMAP2"], c="#dddddd", s=3, linewidths=0, label="other immune")
ax.scatter(df_myeloid["UMAP1"], df_myeloid["UMAP2"], c="#E41A1C", s=4, linewidths=0, label="myeloid (Macro_M1/M2/Dendritic)")
ax.legend(fontsize=8, markerscale=3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Immune compartment UMAP -- myeloid highlighted (n={is_myeloid.sum():,})", fontsize=10, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/umap_myeloid_context.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/umap_myeloid_context.png")

# Trio feature plots, myeloid cells only
xlim = (df_myeloid["UMAP1"].min() - 0.5, df_myeloid["UMAP1"].max() + 0.5)
ylim = (df_myeloid["UMAP2"].min() - 0.5, df_myeloid["UMAP2"].max() + 0.5)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, gene in zip(axes, GENES):
    order = np.argsort(df_myeloid[gene].values)
    sca = ax.scatter(df_myeloid["UMAP1"].values[order], df_myeloid["UMAP2"].values[order],
                      c=df_myeloid[gene].values[order], cmap="viridis", s=6, linewidths=0)
    cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log1p(CP10K)", fontsize=7.5)
    ax.set_title(gene, fontsize=12, fontweight="bold")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=8); ax.set_ylabel("UMAP2", fontsize=8)
fig.suptitle(f"Visvader -- myeloid cells (Macro_M1+Macro_M2+Dendritic, n={len(df_myeloid):,}) -- LCOR / APM / ESR1",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/umap_trio_Visvader_myeloid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/umap_trio_Visvader_myeloid.png")

print("\nDONE")
