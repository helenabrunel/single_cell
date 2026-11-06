"""
20c_umap_by_subtype.py
========================================
Companion to 20c_lcor_malignant_only.py (left untouched): same malignant
epithelial compartment, same Harmony-integrated UMAP embedding, but colored
by BC subtype (ER, HER2, TNBC) instead of LCOR expression.

OUTPUTS
-------
figures/20c_LCOR_malignant_only/fig_umap_subtype.png

Run from: ~/single_cell/Visvader/
  python3 20c_umap_by_subtype.py
"""

import scanpy as sc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

OUTDIR = "figures/20c_LCOR_malignant_only"
SUBTYPES = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

print("Loading compartment_epithelial.h5ad (malignant epithelial compartment only)...")
adata = sc.read_h5ad("compartment_epithelial.h5ad")
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
umap = adata.obsm["X_umap"]

plot_order = np.random.RandomState(0).permutation(adata.n_obs)
subtype_arr = adata.obs["subtype"].values[plot_order]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
for st in SUBTYPES:
    m = subtype_arr == st
    if not m.any():
        continue
    ax.scatter(umap[plot_order][m, 0], umap[plot_order][m, 1], s=1.2,
               color=SUBTYPE_COLORS[st], alpha=0.6,
               label=f"{st} (n={m.sum():,})", rasterized=True)
ax.set_title(f"BC subtype -- malignant epithelial compartment\n({adata.n_obs:,} cells)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, markerscale=6, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_umap_subtype.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_umap_subtype.png")
