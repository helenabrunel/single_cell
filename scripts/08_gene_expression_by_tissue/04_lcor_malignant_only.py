"""
20c_lcor_malignant_only.py
========================================
Further simplification: only the malignant (tumor)
epithelial compartment matters -- no basal/luminal lineage stratification,
no recovered Normal-tissue reference. Two outputs only:
  1. UMAP colored by LCOR expression, using the Harmony-integrated embedding
     already stored in compartment_epithelial.h5ad (X_pca_harmony -> X_umap)
     -- no re-embedding needed, unlike 20/20b's ad-hoc non-Harmony UMAP.
  2. A single-row ("Malignant") bubble/dotplot of LCOR x subtype
     (ER, HER2, TNBC) -- no lineage rows at all.

OUTPUTS
-------
figures/20c_LCOR_malignant_only/
    fig_umap_LCOR.png
    fig_dotplot_LCOR.png
    dotplot_LCOR_summary.csv

Run from: ~/single_cell/Visvader/
  python3 20c_lcor_malignant_only.py
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

OUTDIR = "figures/20c_LCOR_malignant_only"
os.makedirs(OUTDIR, exist_ok=True)
GENE = "LCOR"
SUBTYPES = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

print("Loading compartment_epithelial.h5ad (malignant epithelial compartment only)...")
adata = sc.read_h5ad("compartment_epithelial.h5ad")
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"  compartment: {adata.obs['compartment'].unique().tolist()}")
print(f"  subtype counts:\n{adata.obs['subtype'].value_counts().to_string()}")
assert GENE in adata.var_names, f"{GENE} not found in var_names"
assert "X_umap" in adata.obsm, "X_umap missing from compartment_epithelial.h5ad"

gidx = adata.var_names.get_loc(GENE)
lcor_log = adata.layers["logcounts"][:, gidx]
lcor_log = np.asarray(lcor_log.todense()).ravel() if issparse(lcor_log) else np.asarray(lcor_log).ravel()
lcor_counts = adata.layers["counts"][:, gidx]
lcor_counts = np.asarray(lcor_counts.todense()).ravel() if issparse(lcor_counts) else np.asarray(lcor_counts).ravel()
adata.obs[f"{GENE}_logcounts"] = lcor_log
adata.obs[f"{GENE}_counts"] = lcor_counts

umap = adata.obsm["X_umap"]


# ══════════════════════════════════════════════════════════════════════════════
# UMAP -- LCOR expression (Harmony-integrated embedding, malignant compartment)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding UMAP x LCOR...")
plot_order = np.random.RandomState(0).permutation(adata.n_obs)
order = np.argsort(lcor_log[plot_order])
idx_ord = plot_order[order]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
sca = ax.scatter(umap[idx_ord, 0], umap[idx_ord, 1], s=1.2, c=lcor_log[idx_ord],
                  cmap="Purples", alpha=0.7, rasterized=True)
plt.colorbar(sca, ax=ax, shrink=0.8, label="logcounts")
ax.set_title(f"{GENE} expression -- malignant epithelial compartment\n"
             f"(detected in {100*(lcor_log>0).mean():.1f}% of {adata.n_obs:,} cells)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_umap_LCOR.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_umap_LCOR.png")


# ══════════════════════════════════════════════════════════════════════════════
# Bubble/dotplot -- single "Malignant" row x subtype (ER, HER2, TNBC)
# ══════════════════════════════════════════════════════════════════════════════
print("Building single-row bubble plot (Malignant x subtype)...")
rows = []
for st in SUBTYPES:
    d = adata.obs[adata.obs["subtype"] == st]
    n = len(d)
    pct = 100 * (d[f"{GENE}_counts"] > 0).mean() if n > 0 else np.nan
    meanlog = d[f"{GENE}_logcounts"].mean() if n > 0 else np.nan
    rows.append({"row": "Malignant", "subtype": st, "n_cells": n,
                 "pct_detected": pct, "mean_logcounts": meanlog})
dot_df = pd.DataFrame(rows)
dot_df.to_csv(f"{OUTDIR}/dotplot_LCOR_summary.csv", index=False)
print(dot_df.to_string(index=False))
print(f"  Saved: {OUTDIR}/dotplot_LCOR_summary.csv")

vmax = np.nanmax(dot_df["mean_logcounts"].values)
norm = Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.plasma


def size_for(pct):
    return 20 + 6.5 * pct

fig, ax = plt.subplots(figsize=(4.8, 2.2))
for xi, st in enumerate(SUBTYPES):
    row = dot_df[dot_df.subtype == st]
    pct = row["pct_detected"].values[0]
    meanlog = row["mean_logcounts"].values[0]
    ax.scatter(xi, 0, s=size_for(pct), c=[cmap(norm(meanlog))],
               edgecolor="black", linewidth=0.5, zorder=3)

ax.set_xticks(range(len(SUBTYPES)))
ax.set_xticklabels(SUBTYPES, fontsize=9)
for xi, st in enumerate(SUBTYPES):
    ax.get_xticklabels()[xi].set_color(SUBTYPE_COLORS[st])
    ax.get_xticklabels()[xi].set_fontweight("bold")
ax.set_yticks([0])
ax.set_yticklabels(["Malignant"], fontsize=9)
ax.set_xlim(-0.6, len(SUBTYPES) - 0.4)
ax.set_ylim(-0.8, 0.8)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#EAEAEA", linewidth=0.7, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
ax.set_title(f"{GENE} expression -- malignant compartment x subtype", fontsize=10, fontweight="bold")

cbar_ax = fig.add_axes([0.98, 0.25, 0.02, 0.5])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label(f"{GENE} mean logcounts", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)

legend_ax = fig.add_axes([1.12, 0.15, 0.16, 0.7])
legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis("off")
legend_ax.set_title("% detected", fontsize=7, pad=3)
for i, pv in enumerate([10, 50, 90]):
    y = 0.82 - i * 0.36
    legend_ax.scatter([0.25], [y], s=size_for(pv), c="#999999",
                       edgecolors="black", linewidths=0.5, clip_on=False)
    legend_ax.text(0.55, y, f"{pv}%", fontsize=6.5, va="center")

plt.subplots_adjust(right=0.85)
fig.savefig(f"{OUTDIR}/fig_dotplot_LCOR.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_dotplot_LCOR.png")

print(f"""
============================================================
 DONE
============================================================
  Output: {OUTDIR}/
============================================================
""")
