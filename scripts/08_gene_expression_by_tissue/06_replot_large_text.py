"""
20d_replot_large_text.py
========================================
Re-plot only (no recomputation): regenerates the 6 panel PNGs from
20c_lcor_malignant_only.py / 20c_umap_by_subtype.py / 20b_lcor_epithelial_lineage_3cat.py
with substantially larger text everywhere (titles, axis labels, tick labels,
legends), and a bubble-plot size/color legend that is actually legible.

Reuses the already-computed h5ad objects (UMAP embeddings, LCOR values,
lineage/subtype labels) -- no PCA/neighbors/UMAP/scoring is redone.

OUTPUTS (overwrites in place, same filenames as before)
-------
figures/20c_LCOR_malignant_only/
    fig_umap_subtype.png
    fig_umap_LCOR.png
    fig_dotplot_LCOR.png
figures/20b_LCOR_epithelial_lineage_3cat/
    fig_umap_epi_lineage.png
    fig_umap_LCOR.png
    fig_dotplot_LCOR.png

Run from: ~/single_cell/Visvader/
  python3 20d_replot_large_text.py
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 13})

GENE = "LCOR"


def get_gene_vec(adata, layer, gene):
    gidx = adata.var_names.get_loc(gene)
    v = adata.layers[layer][:, gidx]
    return np.asarray(v.todense()).ravel() if issparse(v) else np.asarray(v).ravel()


def size_for(pct):
    return 30 + 9.0 * pct


def dotplot(dot_df, row_labels, col_labels, row_key, col_key, col_colors,
            title, out_path, figsize):
    vmax = np.nanmax(dot_df["mean_logcounts"].values)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=figsize)
    for xi, cv in enumerate(col_labels):
        for yi, rv in enumerate(row_labels):
            row = dot_df[(dot_df[col_key] == cv) & (dot_df[row_key] == rv)]
            if row.empty or row["n_cells"].values[0] == 0:
                continue
            pct = row["pct_detected"].values[0]
            meanlog = row["mean_logcounts"].values[0]
            ax.scatter(xi, yi, s=size_for(pct), c=[cmap(norm(meanlog))],
                       edgecolor="black", linewidth=0.6, zorder=3)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=15)
    for xi, cv in enumerate(col_labels):
        ax.get_xticklabels()[xi].set_color(col_colors[cv])
        ax.get_xticklabels()[xi].set_fontweight("bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=14)
    ax.set_xlim(-0.6, len(col_labels) - 0.4)
    ax.set_ylim(-0.6, len(row_labels) - 0.4)
    if len(row_labels) > 1:
        ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)

    # ---- color legend (mean logcounts) ----
    cbar_ax = fig.add_axes([1.04, 0.30, 0.035, 0.45])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label(f"{GENE} mean logcounts", fontsize=13)
    cb.ax.tick_params(labelsize=12)

    # ---- size legend (% detected) ----
    legend_ax = fig.add_axes([1.20, 0.28, 0.30, 0.50])
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis("off")
    legend_ax.set_title("% detected", fontsize=14, pad=8)
    for i, pv in enumerate([10, 50, 90]):
        y = 0.86 - i * 0.40
        legend_ax.scatter([0.22], [y], s=size_for(pv), c="#999999",
                           edgecolors="black", linewidths=0.6,
                           clip_on=False)
        legend_ax.text(0.55, y, f"{pv}%", fontsize=13, va="center")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def umap_categorical(umap, labels, order, colors, title, out_path, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plot_order = np.random.RandomState(0).permutation(len(labels))
    lab_arr = np.asarray(labels)[plot_order]
    for cat in order:
        m = lab_arr == cat
        if not m.any():
            continue
        ax.scatter(umap[plot_order][m, 0], umap[plot_order][m, 1], s=1.2,
                   color=colors[cat], alpha=0.6,
                   label=f"{cat} (n={m.sum():,})", rasterized=True)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=13, markerscale=8, loc="center left",
                     bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=True)
    leg.get_frame().set_edgecolor("#CCCCCC")
    ax.set_xlabel("UMAP1", fontsize=14); ax.set_ylabel("UMAP2", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def umap_continuous(umap, values, title, out_path, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plot_order = np.random.RandomState(0).permutation(len(values))
    order = np.argsort(values[plot_order])
    idx_ord = plot_order[order]
    sca = ax.scatter(umap[idx_ord, 0], umap[idx_ord, 1], s=1.2, c=values[idx_ord],
                      cmap="Purples", alpha=0.7, rasterized=True)
    cb = plt.colorbar(sca, ax=ax, shrink=0.8)
    cb.set_label("logcounts", fontsize=14)
    cb.ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("UMAP1", fontsize=14); ax.set_ylabel("UMAP2", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 -- 20c: malignant compartment only (ER/HER2/TNBC)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print(" PART 1 -- malignant compartment (20c)")
print("="*65)
OUT1 = "figures/20c_LCOR_malignant_only"
SUBTYPES1 = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS1 = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

adata = sc.read_h5ad("compartment_epithelial.h5ad")
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
umap1 = adata.obsm["X_umap"]
lcor_log1 = get_gene_vec(adata, "logcounts", GENE)
lcor_counts1 = get_gene_vec(adata, "counts", GENE)
subtype1 = adata.obs["subtype"].values

umap_categorical(
    umap1, subtype1, SUBTYPES1, SUBTYPE_COLORS1,
    f"BC subtype -- malignant epithelial compartment\n({adata.n_obs:,} cells)",
    f"{OUT1}/fig_umap_subtype.png", figsize=(8.0, 7.5))

umap_continuous(
    umap1, lcor_log1,
    f"{GENE} expression -- malignant epithelial compartment\n"
    f"(detected in {100*(lcor_log1>0).mean():.1f}% of {adata.n_obs:,} cells)",
    f"{OUT1}/fig_umap_LCOR.png", figsize=(7.5, 7.5))

rows = []
for st in SUBTYPES1:
    m = subtype1 == st
    n = int(m.sum())
    pct = 100 * (lcor_counts1[m] > 0).mean() if n > 0 else np.nan
    meanlog = lcor_log1[m].mean() if n > 0 else np.nan
    rows.append({"row": "Malignant", "subtype": st, "n_cells": n,
                 "pct_detected": pct, "mean_logcounts": meanlog})
dot_df1 = pd.DataFrame(rows)
dot_df1.to_csv(f"{OUT1}/dotplot_LCOR_summary.csv", index=False)

dotplot(dot_df1, row_labels=["Malignant"], col_labels=SUBTYPES1,
        row_key="row", col_key="subtype", col_colors=SUBTYPE_COLORS1,
        title=f"{GENE} expression -- malignant compartment x subtype",
        out_path=f"{OUT1}/fig_dotplot_LCOR.png", figsize=(6.0, 2.6))

del adata


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 -- 20b: epithelial lineage x subtype (ER/HER2/TNBC/Normal)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" PART 2 -- epithelial lineage x subtype (20b)")
print("="*65)
OUT2 = "figures/20b_LCOR_epithelial_lineage_3cat"
SUBTYPES2 = ["ER", "HER2", "TNBC", "Normal"]
SUBTYPE_COLORS2 = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C", "Normal": "#377EB8"}
LINEAGE_ORDER = ["Basal", "Luminal_progenitor", "Mature_luminal", "Others"]
LINEAGE_COLORS = {
    "Basal": "#E41A1C", "Luminal_progenitor": "#377EB8",
    "Mature_luminal": "#4DAF4A", "Others": "#BDBDBD",
}

combined = sc.read_h5ad(f"{OUT2}/epithelial_lineage_3cat.h5ad")
print(f"  {combined.n_obs:,} cells x {combined.n_vars:,} genes")
umap2 = combined.obsm["X_umap"]
lineage2 = combined.obs["epi_lineage"].values
subtype2 = combined.obs["subtype"].values
lcor_log2 = combined.obs[f"{GENE}_logcounts"].values
lcor_counts2 = combined.obs[f"{GENE}_counts"].values

umap_categorical(
    umap2, lineage2, LINEAGE_ORDER, LINEAGE_COLORS,
    "Epithelial cell population (3-category lineage)\n(Tumor + recovered Normal, joint UMAP)",
    f"{OUT2}/fig_umap_epi_lineage.png", figsize=(8.5, 7.5))

umap_continuous(
    umap2, lcor_log2,
    f"{GENE} expression\n(detected in {100*(lcor_log2>0).mean():.1f}% of cells)",
    f"{OUT2}/fig_umap_LCOR.png", figsize=(7.5, 7.5))

rows = []
for st in SUBTYPES2:
    ms = subtype2 == st
    for lin in LINEAGE_ORDER:
        m = ms & (lineage2 == lin)
        n = int(m.sum())
        pct = 100 * (lcor_counts2[m] > 0).mean() if n > 0 else np.nan
        meanlog = lcor_log2[m].mean() if n > 0 else np.nan
        rows.append({"subtype": st, "epi_lineage": lin, "n_cells": n,
                     "pct_detected": pct, "mean_logcounts": meanlog})
dot_df2 = pd.DataFrame(rows)
dot_df2.to_csv(f"{OUT2}/dotplot_LCOR_summary.csv", index=False)

dotplot(dot_df2, row_labels=LINEAGE_ORDER, col_labels=SUBTYPES2,
        row_key="epi_lineage", col_key="subtype", col_colors=SUBTYPE_COLORS2,
        title=f"{GENE} expression by epithelial lineage x subtype",
        out_path=f"{OUT2}/fig_dotplot_LCOR.png", figsize=(7.5, 4.6))

print("\nDONE.")
