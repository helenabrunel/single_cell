"""
07_hypoxia_TIM3.py
==================
Analysis of GO:0001666 (response to hypoxia) and HAVCR2 (TIM3) expression
across all 16 GSE161529 objects (4 subtypes × 4 compartments).

Hypothesis: TIM3 overexpression in response to hypoxia in the TME.

For each object:
  1. Score GO:0001666 (hypoxia response, ~200 genes)
  2. Extract HAVCR2 expression
  3. Feature plots (UMAP): hypoxia score + HAVCR2 side by side
  4. Hexbin: hypoxia score × HAVCR2 expression
  5. Correlation Pearson r per compartment × subtype → CSV

For immune objects only:
  6. Assign immune cell type (winner-takes-all from existing scores)
  7. Violin: HAVCR2 and hypoxia score by immune cell type × subtype

Output (flat):
  figures/07_hypoxia_TIM3/
    umap_[object]_hypoxia_TIM3.png
    hexbin_[object]_hypoxia_vs_TIM3.png
    violin_immune_[object]_HAVCR2.png
    violin_immune_[object]_hypoxia.png
    correlation_table.csv

Run from: ~/single_cell/Visvader/
  python3 07_hypoxia_TIM3.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings, gc
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
OUTDIR      = "figures/07_hypoxia_TIM3"
GRIDSIZE    = 90
SUBTYPE_COL = "subtype"
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}

# Immune cell type score columns (already in obs from script 03)
IMMUNE_CELL_TYPES = {
    "CD8_T"      : "score_CD8_T",
    "CD4_T"      : "score_CD4_T",
    "Treg"       : "score_Treg",
    "NK"         : "score_NK",
    "Macro_M1"   : "score_Macro_M1",
    "Macro_M2"   : "score_Macro_M2",
    "B_cell"     : "score_B_cell",
    "Plasmablast": "score_Plasmablast",
    "Dendritic"  : "score_Dendritic",
    "Mast"       : "score_Mast",
}

CELLTYPE_COLORS = {
    "CD8_T"      : "#E41A1C",
    "CD4_T"      : "#FF7F00",
    "Treg"       : "#FFFF33",
    "NK"         : "#A65628",
    "Macro_M1"   : "#377EB8",
    "Macro_M2"   : "#4DAF4A",
    "B_cell"     : "#984EA3",
    "Plasmablast": "#F781BF",
    "Dendritic"  : "#999999",
    "Mast"       : "#A6CEE3",
}

# 16 objects
OBJECTS = {
    "All_Global"       : "all_global.h5ad",
    "All_ER"           : "all_ER.h5ad",
    "All_HER2"         : "all_HER2.h5ad",
    "All_TNBC"         : "all_TNBC.h5ad",
    "Epithelial_Global": "compartment_epithelial.h5ad",
    "Epithelial_ER"    : "epithelial_ER.h5ad",
    "Epithelial_HER2"  : "epithelial_HER2.h5ad",
    "Epithelial_TNBC"  : "epithelial_TNBC.h5ad",
    "Immune_Global"    : "compartment_immune.h5ad",
    "Immune_ER"        : "immune_ER.h5ad",
    "Immune_HER2"      : "immune_HER2.h5ad",
    "Immune_TNBC"      : "immune_TNBC.h5ad",
    "Stromal_Global"   : "compartment_stromal.h5ad",
    "Stromal_ER"       : "stromal_ER.h5ad",
    "Stromal_HER2"     : "stromal_HER2.h5ad",
    "Stromal_TNBC"     : "stromal_TNBC.h5ad",
}

# ── GO:0001666 — response to hypoxia ─────────────────────────────────────────
GO_0001666 = [
    "HIF1A","EPAS1","HIF3A","ARNT","ARNT2",
    "VEGFA","VEGFB","VEGFC","VEGFD","PGF",
    "LDHA","LDHB","LDHC","PFKL","PFKM","PFKFB3","PFKFB4",
    "SLC2A1","SLC2A3","HK1","HK2","HK3","ENO1","ENO2",
    "PDK1","PDK3","PDHK1",
    "BNIP3","BNIP3L","BNIP1",
    "CA9","CA12",
    "ANGPT1","ANGPT2","ANGPTL4",
    "EPO","EPOR",
    "NOS2","NOS3",
    "HMOX1","HMOX2",
    "ADM","ADM2",
    "SERPINE1","SERPINE2",
    "FN1","COL1A1","COL4A1","COL5A1",
    "MMP2","MMP9","MMP14",
    "CXCR4","CXCL12","CXCL8",
    "TNF","IL1A","IL1B","IL6","IL8",
    "PTGS2","PTGIS",
    "TF","TFRC","SLC11A2","FTH1","FTL",
    "EGLN1","EGLN2","EGLN3",
    "VHL","PHD1","PHD2","PHD3",
    "P4HA1","P4HA2","PLOD1","PLOD2",
    "DDIT4","DDIT3",
    "ATF4","ATF3","CREB3",
    "SLC1A5","SLC7A5","SLC7A11",
    "GPI","ALDOA","ALDOC","TPI1","GAPDH","PGK1","PGAM1","ENO2","PKM",
    "MCM3","MCM4","MCM5","MCM6","MCM7",
    "CDKN1A","CDKN1B","TP53",
    "BCL2","BCL2L1","MCL1","BAX","BAK1",
    "NOTCH1","NOTCH2","DLL1","DLL4","JAG1","JAG2",
    "KDR","FLT1","FLT4","TEK","TIE1",
    "TWIST1","TWIST2","SNAI1","SNAI2","ZEB1","ZEB2","VIM","CDH2",
    "MET","HGF",
    "IGF1","IGF1R","IGF2","IGF2R","IRS1","IRS2",
    "PDGFA","PDGFB","PDGFC","PDGFD","PDGFRA","PDGFRB",
    "TGFB1","TGFB2","TGFB3","TGFBR1","TGFBR2",
    "WNT2","WNT5A","FZD1","CTNNB1",
    "MYC","MYCN",
    "NF1","NF2",
    "PTEN","PIK3CA","AKT1","AKT2","MTOR","RPTOR",
    "MAPK1","MAPK3","MAPK8","MAP2K1","MAP2K2",
    "STAT3","STAT5A","STAT5B","JAK1","JAK2",
    "NFE2L2","KEAP1","SQSTM1",
    "SOD1","SOD2","CAT","GPX1","GPX4","PRDX1","PRDX2","PRDX5",
    "HAVCR2",  # TIM3 itself — included as part of hypoxia response
]
# Remove duplicates while preserving order
GO_0001666 = list(dict.fromkeys(GO_0001666))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def score_and_znorm(adata, gene_list, col, min_pct=0.01):
    present = [g for g in gene_list if g in adata.var_names]
    if len(present) < 3:
        return False, present
    x     = adata[:, present].X
    n_exp = np.asarray((x > 0).sum(axis=1)).flatten()
    if (n_exp > 0).mean() < min_pct:
        return False, present
    sc.tl.score_genes(adata, present, score_name=col, use_raw=False)
    raw = adata.obs[col].values.astype(float)
    adata.obs[f"{col}_z"] = (raw - raw.mean()) / (raw.std() + 1e-8)
    return True, present


def get_gene_expr(adata, gene):
    """Extract expression of a single gene as array."""
    if gene not in adata.var_names:
        return None
    x = adata[:, gene].X
    if issparse(x):
        return np.asarray(x.todense()).flatten()
    return np.asarray(x).flatten()


def assign_immune_type(obs):
    """Winner-takes-all from existing immune cell type scores."""
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items()
                    if col in obs.columns}
    if not cols_present:
        return None

    score_matrix = obs[[col for col in cols_present.values()]].values
    cell_types   = list(cols_present.keys())
    best_idx     = score_matrix.argmax(axis=1)
    best_score   = score_matrix.max(axis=1)
    assigned     = [cell_types[i] if s > 0 else "Unknown"
                    for i, s in zip(best_idx, best_score)]
    return np.array(assigned)


def hexbin_panel(ax, x, y, title, xlabel, ylabel, gridsize=90,
                  filter_y_positive=False):
    mask = ~(np.isnan(x) | np.isnan(y))
    if filter_y_positive:
        mask = mask & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        ax.set_visible(False); return None

    xmin, xmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ymin, ymax = np.percentile(y, 0.5), np.percentile(y, 99.5)
    xpad = 0.20 * (xmax - xmin); ypad = 0.20 * (ymax - ymin)
    xl = (xmin - xpad, xmax + xpad)
    yl = (ymin - ypad, ymax + ypad)

    ax.set_facecolor("white")
    ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="RdYlBu_r",
                   mincnt=1, linewidths=0.02,
                   extent=[xl[0], xl[1], yl[0], yl[1]], zorder=2)
    ax.axhline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")

    r, p = stats.pearsonr(x, y)
    pstr = f"p={p:.1e}" if p >= 1e-300 else "p<1e-300"
    ax.text(0.97, 0.03, f"r={r:.3f}\n{pstr}\nn={len(x):,}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    return hb


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
all_corr_rows = []

for obj_name, h5ad_file in OBJECTS.items():

    if not os.path.exists(h5ad_file):
        print(f"\n  ⚠️  {h5ad_file} not found — skipping {obj_name}")
        continue

    print(f"\n{'='*65}")
    print(f"  {obj_name}")
    print(f"{'='*65}")

    adata = sc.read_h5ad(h5ad_file)
    print(f"  Cells: {adata.n_obs:,}")

    if "logcounts" in adata.layers:
        adata.X = adata.layers["logcounts"].copy()

    # ── Score GO:0001666 ──────────────────────────────────────────────────
    # Remove HAVCR2 from scoring list to keep it independent
    hypoxia_genes = [g for g in GO_0001666 if g != "HAVCR2"]
    ok, found = score_and_znorm(adata, hypoxia_genes, "score_hypoxia")
    print(f"  GO:0001666: {len(found)}/{len(hypoxia_genes)} genes "
          f"({'✓' if ok else '✗'})")

    # ── Extract HAVCR2 expression ─────────────────────────────────────────
    havcr2_expr = get_gene_expr(adata, "HAVCR2")
    if havcr2_expr is not None:
        adata.obs["HAVCR2_expr"] = havcr2_expr
        pct_expr = (havcr2_expr > 0).mean() * 100
        mean_expr = havcr2_expr[havcr2_expr > 0].mean() \
                    if (havcr2_expr > 0).any() else 0
        print(f"  HAVCR2: {pct_expr:.1f}% cells express "
              f"(mean={mean_expr:.3f} in expressors)")
    else:
        print(f"  ⚠️  HAVCR2 not found in var_names")

    obs = adata.obs
    is_immune = "Immune" in obj_name

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE PLOTS (UMAP)
    # ══════════════════════════════════════════════════════════════════════
    if "X_umap" in adata.obsm:
        coords = adata.obsm["X_umap"]

        plot_cols = []
        plot_labels = []
        plot_cmaps  = []

        if ok and "score_hypoxia_z" in obs.columns:
            plot_cols.append("score_hypoxia_z")
            plot_labels.append("Hypoxia score (z)")
            plot_cmaps.append("RdYlBu_r")

        if "HAVCR2_expr" in obs.columns:
            plot_cols.append("HAVCR2_expr")
            plot_labels.append("HAVCR2 / TIM3")
            plot_cmaps.append("YlOrRd")

        if SUBTYPE_COL in obs.columns:
            plot_cols.append(SUBTYPE_COL)
            plot_labels.append("Subtype")
            plot_cmaps.append(None)

        if is_immune:
            immune_types = assign_immune_type(obs)
            if immune_types is not None:
                adata.obs["immune_cell_type"] = immune_types
                obs = adata.obs
                plot_cols.append("immune_cell_type")
                plot_labels.append("Cell type")
                plot_cmaps.append(None)

        n_panels = len(plot_cols)
        if n_panels > 0:
            nc  = min(4, n_panels)
            nr  = (n_panels + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc,
                                     figsize=(5.5*nc, 5*nr))
            axes = np.array(axes).flatten()
            fig.suptitle(f"GO:0001666 × HAVCR2 — {obj_name}",
                         fontsize=11, fontweight="bold")

            for i, (col, label, cmap) in enumerate(
                    zip(plot_cols, plot_labels, plot_cmaps)):
                ax = axes[i]

                if cmap is None:
                    # Categorical
                    cats = obs[col].astype("category")
                    color_dict = SUBTYPE_COLORS if col == SUBTYPE_COL \
                                 else CELLTYPE_COLORS
                    for cat in cats.cat.categories:
                        m = cats == cat
                        color = color_dict.get(str(cat), "#AAAAAA")
                        ax.scatter(coords[m,0], coords[m,1],
                                   c=color, s=0.3, alpha=0.4,
                                   label=str(cat), rasterized=True)
                    ax.legend(fontsize=6, markerscale=3,
                              loc="upper right", ncol=2)
                else:
                    vals = obs[col].values.astype(float)
                    mask = ~np.isnan(vals)
                    vmax = np.percentile(vals[mask], 98)
                    vmin = np.percentile(vals[mask], 2) \
                           if "z" in col else 0
                    sc_p = ax.scatter(
                        coords[mask,0], coords[mask,1],
                        c=vals[mask], s=0.3, alpha=0.3,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        rasterized=True
                    )
                    plt.colorbar(sc_p, ax=ax, shrink=0.7)

                ax.set_title(label, fontsize=9, fontweight="bold")
                ax.set_xticks([]); ax.set_yticks([])

            for ax in axes[n_panels:]:
                ax.set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{OUTDIR}/umap_{obj_name}_hypoxia_TIM3.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ UMAP saved")

    # ══════════════════════════════════════════════════════════════════════
    # HEXBIN: hypoxia × HAVCR2
    # ══════════════════════════════════════════════════════════════════════
    if ok and "HAVCR2_expr" in obs.columns:
        subtypes_here = [st for st in ["ER","HER2","TNBC"]
                         if SUBTYPE_COL in obs.columns
                         and st in obs[SUBTYPE_COL].values]

        n_cols = 1 + len(subtypes_here)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.5*n_cols, 5.5))
        axes = np.array(axes).flatten()
        fig.suptitle(f"{obj_name} — Hypoxia (GO:0001666) vs HAVCR2/TIM3",
                     fontsize=10, fontweight="bold")

        x_all = obs["score_hypoxia_z"].values.astype(float)
        y_all = obs["HAVCR2_expr"].values.astype(float)

        n_pos = (y_all > 0).sum()
        hb = hexbin_panel(axes[0], x_all, y_all,
                          f"All cells (n={len(x_all):,})",
                          "Hypoxia score (z)", "HAVCR2 expression")
        if hb:
            plt.colorbar(hb, ax=axes[0], shrink=0.8, label="Cell count")

        for i, subtype in enumerate(subtypes_here):
            ax      = axes[i+1]
            st_mask = (obs[SUBTYPE_COL] == subtype).values
            hb = hexbin_panel(ax,
                              x_all[st_mask], y_all[st_mask],
                              f"{subtype} (n={st_mask.sum():,})",
                              "Hypoxia score (z)", "")
            if hb:
                plt.colorbar(hb, ax=ax, shrink=0.8, label="Cell count")
            ax.set_title(f"{subtype} (n={st_mask.sum():,})",
                         fontsize=9, fontweight="bold",
                         color=SUBTYPE_COLORS.get(subtype,"black"))

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/hexbin_{obj_name}_hypoxia_vs_TIM3.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Hexbin filtered to HAVCR2+ cells only
        n_pos = (y_all > 0).sum()
        if n_pos >= 20:
            fig2, axes2 = plt.subplots(1, n_cols, figsize=(5.5*n_cols, 5.5))
            axes2 = np.array(axes2).flatten()
            fig2.suptitle(
                f"{obj_name} — Hypoxia vs HAVCR2/TIM3 (HAVCR2+ cells only, n={n_pos:,})",
                fontsize=10, fontweight="bold"
            )
            hb2 = hexbin_panel(axes2[0], x_all, y_all,
                               f"HAVCR2+ only (n={n_pos:,})",
                               "Hypoxia score (z)", "HAVCR2 expression",
                               filter_y_positive=True)
            if hb2:
                plt.colorbar(hb2, ax=axes2[0], shrink=0.8, label="Cell count")
            for i, subtype in enumerate(subtypes_here):
                ax      = axes2[i+1]
                st_mask = (obs[SUBTYPE_COL] == subtype).values
                n_pos_st = ((y_all > 0) & st_mask).sum()
                hb2 = hexbin_panel(ax,
                                   x_all[st_mask], y_all[st_mask],
                                   f"{subtype} HAVCR2+ (n={n_pos_st:,})",
                                   "Hypoxia score (z)", "",
                                   filter_y_positive=True)
                if hb2:
                    plt.colorbar(hb2, ax=ax, shrink=0.8, label="Cell count")
                ax.set_title(f"{subtype} HAVCR2+ (n={n_pos_st:,})",
                             fontsize=9, fontweight="bold",
                             color=SUBTYPE_COLORS.get(subtype,"black"))
            for ax in axes2[n_cols:]:
                ax.set_visible(False)
            plt.tight_layout()
            plt.savefig(
                f"{OUTDIR}/hexbin_{obj_name}_hypoxia_vs_TIM3_pos_only.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close()
        print(f"  ✓ Hexbin saved")

    # ══════════════════════════════════════════════════════════════════════
    # VIOLIN BY IMMUNE CELL TYPE (immune objects only)
    # ══════════════════════════════════════════════════════════════════════
    if is_immune and "immune_cell_type" in obs.columns:

        subtypes_here = [st for st in ["ER","HER2","TNBC"]
                         if st in obs[SUBTYPE_COL].values]
        cell_type_order = [ct for ct in IMMUNE_CELL_TYPES.keys()
                           if ct in obs["immune_cell_type"].values]

        for score_col, score_label in [
            ("HAVCR2_expr",    "HAVCR2 / TIM3 expression"),
            ("score_hypoxia_z","Hypoxia score (z)"),
        ]:
            if score_col not in obs.columns:
                continue

            fname_suffix = "HAVCR2" if "HAVCR2" in score_col else "hypoxia"

            # Global violin (all subtypes together)
            fig, ax = plt.subplots(figsize=(12, 5))
            data_by_type = [
                obs.loc[obs["immune_cell_type"]==ct, score_col]
                   .dropna().values
                for ct in cell_type_order
            ]
            n_by_type = [len(d) for d in data_by_type]
            data_ok   = [(ct, d) for ct, d, n in
                         zip(cell_type_order, data_by_type, n_by_type)
                         if n >= 10]

            if data_ok:
                cts_ok, data_ok_vals = zip(*data_ok)
                parts = ax.violinplot(list(data_ok_vals),
                                      showmedians=True,
                                      showextrema=False,
                                      positions=range(len(cts_ok)))
                for pc, ct in zip(parts["bodies"], cts_ok):
                    pc.set_facecolor(CELLTYPE_COLORS.get(ct,"#AAAAAA"))
                    pc.set_alpha(0.75)
                    pc.set_edgecolor("white")
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(2)
                ax.set_xticks(range(len(cts_ok)))
                ax.set_xticklabels(
                    [f"{ct}\n(n={obs[obs['immune_cell_type']==ct].shape[0]:,})"
                     for ct in cts_ok],
                    rotation=30, ha="right", fontsize=9
                )
                ax.set_ylabel(score_label)
                ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
                ax.set_title(
                    f"{obj_name} — {score_label} by immune cell type",
                    fontsize=11, fontweight="bold"
                )
                plt.tight_layout()
                plt.savefig(
                    f"{OUTDIR}/violin_{obj_name}_{fname_suffix}_by_celltype.png",
                    dpi=150, bbox_inches="tight"
                )
                plt.close()

            # Per-subtype violin
            n_st = len(subtypes_here)
            if n_st > 0:
                fig, axes = plt.subplots(1, n_st,
                                         figsize=(6*n_st, 5),
                                         sharey=True)
                axes = np.array(axes).flatten()
                fig.suptitle(
                    f"{obj_name} — {score_label} by cell type × subtype",
                    fontsize=11, fontweight="bold"
                )
                for ax, subtype in zip(axes, subtypes_here):
                    sub = obs[obs[SUBTYPE_COL] == subtype]
                    data_st = [
                        sub.loc[sub["immune_cell_type"]==ct, score_col]
                           .dropna().values
                        for ct in cell_type_order
                    ]
                    data_st_ok = [(ct, d) for ct, d in
                                  zip(cell_type_order, data_st)
                                  if len(d) >= 10]
                    if not data_st_ok:
                        ax.set_visible(False); continue

                    cts_st, vals_st = zip(*data_st_ok)
                    parts = ax.violinplot(list(vals_st),
                                          showmedians=True,
                                          showextrema=False,
                                          positions=range(len(cts_st)))
                    for pc, ct in zip(parts["bodies"], cts_st):
                        pc.set_facecolor(CELLTYPE_COLORS.get(ct,"#AAAAAA"))
                        pc.set_alpha(0.75)
                        pc.set_edgecolor("white")
                    parts["cmedians"].set_color("black")
                    parts["cmedians"].set_linewidth(2)
                    ax.set_xticks(range(len(cts_st)))
                    ax.set_xticklabels(list(cts_st), rotation=35,
                                       ha="right", fontsize=8)
                    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
                    ax.set_title(subtype, fontsize=11, fontweight="bold",
                                 color=SUBTYPE_COLORS.get(subtype,"black"))
                    if ax == axes[0]:
                        ax.set_ylabel(score_label)

                plt.tight_layout()
                plt.savefig(
                    f"{OUTDIR}/violin_{obj_name}_{fname_suffix}_by_celltype_subtype.png",
                    dpi=150, bbox_inches="tight"
                )
                plt.close()

            print(f"  ✓ Violin {fname_suffix} saved")

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATIONS
    # ══════════════════════════════════════════════════════════════════════
    if ok and "HAVCR2_expr" in obs.columns:
        hypoxia = obs["score_hypoxia_z"].values.astype(float)
        havcr2  = obs["HAVCR2_expr"].values.astype(float)
        valid   = ~np.isnan(hypoxia)

        def mw_test(hyp, hav, label_dict):
            """Mann-Whitney: hypoxia score in HAVCR2+ vs HAVCR2- cells."""
            valid_local = ~np.isnan(hyp)
            pos = hyp[(hav > 0) & valid_local]
            neg = hyp[(hav == 0) & valid_local]
            if len(pos) < 10 or len(neg) < 10:
                return None
            stat, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            # Effect size: rank-biserial correlation
            n1, n2 = len(pos), len(neg)
            rbc = 1 - (2*stat)/(n1*n2)
            pct_pos = (hav > 0).mean() * 100
            row = {**label_dict,
                   "n_HAVCR2pos" : int(n1),
                   "n_HAVCR2neg" : int(n2),
                   "pct_HAVCR2pos": round(pct_pos, 2),
                   "median_hyp_pos": round(float(np.median(pos)), 4),
                   "median_hyp_neg": round(float(np.median(neg)), 4),
                   "rank_biserial_r": round(rbc, 4),
                   "MW_p"          : p}
            return row

        # Global
        row = mw_test(hypoxia, havcr2,
                      {"Object": obj_name, "Subtype": "All",
                       "Cell_type": "All"})
        if row: all_corr_rows.append(row)

        # Per subtype
        for subtype in ["ER","HER2","TNBC"]:
            if SUBTYPE_COL not in obs.columns: continue
            st_mask = (obs[SUBTYPE_COL] == subtype).values
            row = mw_test(hypoxia[st_mask], havcr2[st_mask],
                          {"Object": obj_name, "Subtype": subtype,
                           "Cell_type": "All"})
            if row: all_corr_rows.append(row)

        # Per immune cell type
        if is_immune and "immune_cell_type" in obs.columns:
            for ct in obs["immune_cell_type"].unique():
                ct_mask = (obs["immune_cell_type"] == ct).values
                row = mw_test(hypoxia[ct_mask], havcr2[ct_mask],
                              {"Object": obj_name, "Subtype": "All",
                               "Cell_type": ct})
                if row: all_corr_rows.append(row)

                # Per subtype within cell type
                for subtype in ["ER","HER2","TNBC"]:
                    if SUBTYPE_COL not in obs.columns: continue
                    st_mask = (obs[SUBTYPE_COL] == subtype).values
                    combo   = ct_mask & st_mask
                    row = mw_test(hypoxia[combo], havcr2[combo],
                                  {"Object": obj_name, "Subtype": subtype,
                                   "Cell_type": ct})
                    if row: all_corr_rows.append(row)


    # ══════════════════════════════════════════════════════════════════════
    # VIOLIN: HAVCR2+ vs HAVCR2- hypoxia score
    # ══════════════════════════════════════════════════════════════════════
    if ok and "HAVCR2_expr" in obs.columns and "score_hypoxia_z" in obs.columns:
        hypoxia_all = obs["score_hypoxia_z"].values.astype(float)
        havcr2_all  = obs["HAVCR2_expr"].values.astype(float)
        pos_mask    = havcr2_all > 0
        neg_mask    = havcr2_all == 0

        subtypes_here = [st for st in ["ER","HER2","TNBC"]
                         if SUBTYPE_COL in obs.columns
                         and st in obs[SUBTYPE_COL].values]
        groups = ["All"] + subtypes_here

        # ── Figure 1: all cells, HAVCR2+ vs HAVCR2- per subtype ─────────
        fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 5),
                                 sharey=True)
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"{obj_name} — Hypoxia score: HAVCR2+ vs HAVCR2-\n"
            f"(HAVCR2+ = expressing cells; HAVCR2- = non-expressing)",
            fontsize=10, fontweight="bold"
        )
        for ax, grp in zip(axes, groups):
            if grp == "All":
                hyp_pos = hypoxia_all[pos_mask & ~np.isnan(hypoxia_all)]
                hyp_neg = hypoxia_all[neg_mask & ~np.isnan(hypoxia_all)]
            else:
                st_m = (obs[SUBTYPE_COL] == grp).values
                hyp_pos = hypoxia_all[pos_mask & st_m & ~np.isnan(hypoxia_all)]
                hyp_neg = hypoxia_all[neg_mask & st_m & ~np.isnan(hypoxia_all)]

            if len(hyp_pos) < 10 or len(hyp_neg) < 10:
                ax.set_visible(False); continue

            data = [hyp_neg, hyp_pos]
            parts = ax.violinplot(data, showmedians=True, showextrema=False,
                                  positions=[0, 1])
            parts["bodies"][0].set_facecolor("#4393C3")
            parts["bodies"][0].set_alpha(0.75)
            parts["bodies"][1].set_facecolor("#D6604D")
            parts["bodies"][1].set_alpha(0.75)
            for pc in parts["bodies"]:
                pc.set_edgecolor("white")
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)

            # Mann-Whitney p-value
            if len(hyp_pos) >= 10 and len(hyp_neg) >= 10:
                _, p = stats.mannwhitneyu(hyp_pos, hyp_neg,
                                          alternative="two-sided")
                stars = "***" if p<0.001 else "**" if p<0.01                         else "*" if p<0.05 else "ns"
                ymax = max(np.percentile(hyp_pos,99),
                           np.percentile(hyp_neg,99))
                yrange = ymax - min(np.percentile(hyp_pos,1),
                                    np.percentile(hyp_neg,1))
                ax.plot([0,1], [ymax+yrange*0.05]*2, color="black", lw=1)
                ax.text(0.5, ymax+yrange*0.08, stars,
                        ha="center", fontsize=11, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"HAVCR2-\n(n={len(hyp_neg):,})",
                 f"HAVCR2+\n(n={len(hyp_pos):,})"],
                fontsize=8
            )
            ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
            title = grp if grp == "All" else grp
            ax.set_title(title, fontsize=10, fontweight="bold",
                         color=SUBTYPE_COLORS.get(grp, "black")
                         if grp != "All" else "black")
            if ax == axes[0]:
                ax.set_ylabel("Hypoxia score (z)")

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/violin_{obj_name}_hypoxia_HAVCR2pos_neg.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Violin HAVCR2+/- saved")

        # ── Figure 2: immune cell types (immune objects only) ─────────────
        if is_immune and "immune_cell_type" in obs.columns:
            cell_types_present = [ct for ct in IMMUNE_CELL_TYPES.keys()
                                  if ct in obs["immune_cell_type"].values]

            n_st = len(subtypes_here) if subtypes_here else 1
            fig, axes = plt.subplots(
                len(cell_types_present), n_st,
                figsize=(4*n_st, 3.5*len(cell_types_present)),
                sharey=False, squeeze=False
            )
            fig.suptitle(
                f"{obj_name} — Hypoxia score: HAVCR2+ vs HAVCR2-\n"
                f"by immune cell type × subtype",
                fontsize=11, fontweight="bold"
            )

            st_list = subtypes_here if subtypes_here else ["All"]

            for row_i, ct in enumerate(cell_types_present):
                ct_m = (obs["immune_cell_type"] == ct).values
                for col_i, subtype in enumerate(st_list):
                    ax = axes[row_i][col_i]
                    if subtype == "All":
                        sub_m = ct_m
                    else:
                        sub_m = ct_m & (obs[SUBTYPE_COL] == subtype).values

                    hyp_pos = hypoxia_all[sub_m & pos_mask &
                                          ~np.isnan(hypoxia_all)]
                    hyp_neg = hypoxia_all[sub_m & neg_mask &
                                          ~np.isnan(hypoxia_all)]

                    if len(hyp_pos) < 5 or len(hyp_neg) < 5:
                        ax.text(0.5, 0.5, "n<5", transform=ax.transAxes,
                                ha="center", va="center", color="gray")
                        ax.set_xticks([]); ax.set_yticks([])
                        if col_i == 0:
                            ax.set_ylabel(ct, fontsize=8, fontweight="bold",
                                          color=CELLTYPE_COLORS.get(ct,"black"))
                        continue

                    parts = ax.violinplot([hyp_neg, hyp_pos],
                                           showmedians=True,
                                           showextrema=False,
                                           positions=[0,1])
                    parts["bodies"][0].set_facecolor("#4393C3")
                    parts["bodies"][0].set_alpha(0.75)
                    parts["bodies"][1].set_facecolor("#D6604D")
                    parts["bodies"][1].set_alpha(0.75)
                    for pc in parts["bodies"]:
                        pc.set_edgecolor("white")
                    parts["cmedians"].set_color("black")
                    parts["cmedians"].set_linewidth(2)

                    if len(hyp_pos) >= 5 and len(hyp_neg) >= 5:
                        _, p = stats.mannwhitneyu(hyp_pos, hyp_neg,
                                                   alternative="two-sided")
                        stars = "***" if p<0.001 else "**" if p<0.01                                 else "*" if p<0.05 else "ns"
                        ymax = max(np.percentile(hyp_pos, 99),
                                   np.percentile(hyp_neg, 99))
                        yrange = ymax - min(np.percentile(hyp_pos, 1),
                                            np.percentile(hyp_neg, 1))
                        ax.plot([0,1], [ymax+yrange*0.05]*2,
                                color="black", lw=1)
                        ax.text(0.5, ymax+yrange*0.08, stars,
                                ha="center", fontsize=9, fontweight="bold")

                    ax.set_xticks([0,1])
                    ax.set_xticklabels(
                        [f"-(n={len(hyp_neg):,})",
                         f"+(n={len(hyp_pos):,})"],
                        fontsize=7
                    )
                    ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.5)

                    if col_i == 0:
                        ax.set_ylabel(ct, fontsize=8, fontweight="bold",
                                      color=CELLTYPE_COLORS.get(ct,"black"))
                    if row_i == 0:
                        ax.set_title(subtype, fontsize=9, fontweight="bold",
                                     color=SUBTYPE_COLORS.get(subtype,"black")
                                     if subtype != "All" else "black")

            plt.tight_layout()
            plt.savefig(
                f"{OUTDIR}/violin_{obj_name}_hypoxia_HAVCR2pos_neg_celltypes.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"  ✓ Violin HAVCR2+/- by cell type saved")

    del adata; gc.collect()
    print(f"  ✓ {obj_name} done")


# ── Save correlations ─────────────────────────────────────────────────────────
if all_corr_rows:
    df_corr = pd.DataFrame(all_corr_rows).fillna("All")
    df_corr.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)
    print(f"\n  ✓ {OUTDIR}/correlation_table.csv ({len(df_corr):,} rows)")

    # Summary: Mann-Whitney rank-biserial r per object
    df_summary = df_corr[
        (df_corr["Subtype"] == "All") &
        (df_corr["Cell_type"] == "All")
    ].copy()

    if len(df_summary) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Hypoxia (GO:0001666) — HAVCR2+ vs HAVCR2- cells\n"
            "Mann-Whitney rank-biserial r "
            "(positive = HAVCR2+ cells have higher hypoxia score)",
            fontsize=11, fontweight="bold"
        )
        ax = axes[0]
        rs = df_summary["rank_biserial_r"].values
        colors = ["#C41E3A" if r > 0 else "#1565C0" for r in rs]
        ax.barh(range(len(df_summary)), rs, color=colors, alpha=0.8)
        ax.set_yticks(range(len(df_summary)))
        ax.set_yticklabels(df_summary["Object"].tolist(), fontsize=8)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Rank-biserial r")
        ax.set_title("Effect size by object", fontsize=10, fontweight="bold")
        ax.set_xlim(-1, 1); ax.grid(axis="x", alpha=0.3)
        ax2 = axes[1]
        pcts = df_summary["pct_HAVCR2pos"].values
        ax2.barh(range(len(df_summary)), pcts, color="#888888", alpha=0.7)
        ax2.set_yticks(range(len(df_summary)))
        ax2.set_yticklabels(df_summary["Object"].tolist(), fontsize=8)
        ax2.set_xlabel("% HAVCR2+ cells")
        ax2.set_title("HAVCR2 expression prevalence", fontsize=10, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/summary_MW_barplot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {OUTDIR}/summary_MW_barplot.png")

    # Immune cell type heatmap
    df_ct = df_corr[
        (df_corr["Object"].str.contains("Immune")) &
        (df_corr["Cell_type"] != "All") &
        (df_corr["Subtype"] == "All")
    ].copy()

    if len(df_ct) > 0:
        pivot = df_ct.pivot_table(index="Cell_type", columns="Object",
                                   values="rank_biserial_r")
        fig, ax = plt.subplots(
            figsize=(max(8, len(pivot.columns)*2.5), max(5, len(pivot.index)*0.8))
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
                    cbar_kws={"label": "Rank-biserial r"})
        ax.set_title(
            "Hypoxia: HAVCR2+ vs HAVCR2- by immune cell type × object\n"
            "(positive = HAVCR2+ cells more hypoxic)",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/summary_MW_immune_celltypes.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {OUTDIR}/summary_MW_immune_celltypes.png")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  figures/07_hypoxia_TIM3/ (flat)
    umap_[object]_hypoxia_TIM3.png        feature plots
    hexbin_[object]_hypoxia_vs_TIM3.png   hexbins
    violin_[object]_HAVCR2_by_celltype*.png
    violin_[object]_hypoxia_by_celltype*.png
    correlation_table.csv
    summary_r_barplot.png
══════════════════════════════════════════════════════════════
""")
