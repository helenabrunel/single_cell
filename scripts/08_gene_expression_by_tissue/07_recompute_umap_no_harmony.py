"""
20e_recompute_umap_no_harmony.py
========================================
20c_lcor_malignant_only.py reused the Harmony-integrated embedding already
stored in compartment_epithelial.h5ad (X_pca_harmony -> X_umap, computed by
the main pipeline in 03_split_integrate.py). But the LCOR figure series
(20/20b) deliberately avoided Harmony -- see 20_lcor_epithelial_lineage_normal.py's
"EMBEDDING CAVEAT" docstring: "WITHOUT Harmony batch integration ...
supplementary visualization only". 20c was inconsistent with that choice.

This script recomputes a plain PCA -> neighbors -> UMAP embedding (NO Harmony)
for the malignant epithelial compartment only (compartment_epithelial.h5ad,
122,101 cells), using the same recipe/parameters as 20b's joint embedding
(HVG n_top_genes=2000 flavor=seurat, PCA n_comps=30, neighbors n_pcs=30
n_neighbors=15), so the two figure series are now methodologically
consistent with each other (note: coordinates still won't be pixel-identical
to 20b's malignant-cell positions, since 20b's embedding is fit jointly with
the recovered Normal cells on a smaller shared-gene set).

Does NOT modify compartment_epithelial.h5ad. Saves the new embedding
separately, then re-renders 20c's two UMAP panels (subtype, LCOR) with it.
The dotplot panel is untouched (doesn't depend on the embedding).

OUTPUTS
-------
figures/20c_LCOR_malignant_only/
    umap_no_harmony.npy            -- (n_cells, 2) array, row order = adata.obs_names
    fig_umap_subtype.png           -- overwritten, no-Harmony embedding
    fig_umap_LCOR.png              -- overwritten, no-Harmony embedding

Run from: ~/single_cell/Visvader/
  python3 20e_recompute_umap_no_harmony.py
"""

import scanpy as sc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 13})

OUT = "figures/20c_LCOR_malignant_only"
GENE = "LCOR"
SUBTYPES = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

print("Loading compartment_epithelial.h5ad ...")
adata = sc.read_h5ad("compartment_epithelial.h5ad")
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

print("\nRecomputing embedding WITHOUT Harmony "
      "(HVG -> PCA -> neighbors -> UMAP, same recipe as 20b)...")
adata.X = adata.layers["logcounts"].copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=15)
sc.tl.umap(adata)
print("  UMAP done.")

umap = adata.obsm["X_umap"]
np.save(f"{OUT}/umap_no_harmony.npy", umap)
print(f"  Saved: {OUT}/umap_no_harmony.npy")

gidx = adata.var_names.get_loc(GENE)
lcor_log = adata.layers["logcounts"][:, gidx]
lcor_log = np.asarray(lcor_log.todense()).ravel() if issparse(lcor_log) else np.asarray(lcor_log).ravel()
subtype = adata.obs["subtype"].values


# ══════════════════════════════════════════════════════════════════════════════
# Panel: UMAP by subtype
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.0, 7.5))
plot_order = np.random.RandomState(0).permutation(adata.n_obs)
subtype_arr = subtype[plot_order]
for st in SUBTYPES:
    m = subtype_arr == st
    if not m.any():
        continue
    ax.scatter(umap[plot_order][m, 0], umap[plot_order][m, 1], s=1.2,
               color=SUBTYPE_COLORS[st], alpha=0.6,
               label=f"{st} (n={m.sum():,})", rasterized=True)
ax.set_title(f"BC subtype -- malignant epithelial compartment\n"
             f"({adata.n_obs:,} cells, no Harmony)", fontsize=17, fontweight="bold", pad=12)
leg = ax.legend(fontsize=13, markerscale=8, loc="center left",
                 bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=True)
leg.get_frame().set_edgecolor("#CCCCCC")
ax.set_xlabel("UMAP1", fontsize=14); ax.set_ylabel("UMAP2", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(f"{OUT}/fig_umap_subtype.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT}/fig_umap_subtype.png")


# ══════════════════════════════════════════════════════════════════════════════
# Panel: UMAP by LCOR expression
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 7.5))
order = np.argsort(lcor_log[plot_order])
idx_ord = plot_order[order]
sca = ax.scatter(umap[idx_ord, 0], umap[idx_ord, 1], s=1.2, c=lcor_log[idx_ord],
                  cmap="Purples", alpha=0.7, rasterized=True)
cb = plt.colorbar(sca, ax=ax, shrink=0.8)
cb.set_label("logcounts", fontsize=14)
cb.ax.tick_params(labelsize=12)
ax.set_title(f"{GENE} expression -- malignant epithelial compartment\n"
             f"(detected in {100*(lcor_log>0).mean():.1f}% of {adata.n_obs:,} cells, no Harmony)",
             fontsize=17, fontweight="bold", pad=12)
ax.set_xlabel("UMAP1", fontsize=14); ax.set_ylabel("UMAP2", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(f"{OUT}/fig_umap_LCOR.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT}/fig_umap_LCOR.png")

print("\nDONE.")
