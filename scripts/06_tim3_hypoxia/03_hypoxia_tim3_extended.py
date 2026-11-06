"""
07b_hypoxia_TIM3_extended.py
=============================
Extended analysis of GO:0001666 (hypoxia) × HAVCR2 (TIM3).
Builds on script 07 adding two types of figures:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — Immune objects only: colour by immune cell type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each immune object (Global + ER + HER2 + TNBC):

  A) Scatter hipoxia × HAVCR2 — coloured by immune cell type
       - Background: HAVCR2− cells in light grey
       - Foreground: HAVCR2+ cells coloured by cell type
       - Repeated per subtype (Global / ER / HER2 / TNBC panels)

  B) Hexbin hipoxia × HAVCR2 — one panel per immune cell type
       - All cells of that type shown
       - Subtype columns (ER / HER2 / TNBC)

Output:
  figures/07b/
    p1_scatter_[object]_celltype.png
    p1_hexbin_[object]_[celltype].png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — All compartments: colour by continuous signature
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signatures: HALLMARK_EMT + 9 stemness signatures from signatures_confirmed.json

For each object (Epithelial / Immune / Stromal × Global + subtipos):
For each signature:
  Two side-by-side panels, same axes (hypoxia × HAVCR2):
    Left  — HAVCR2− cells, colour = signature score (grey-blue gradient)
    Right — HAVCR2+ cells, colour = signature score (YlOrRd gradient)
  Subtype panels as rows (Global / ER / HER2 / TNBC)

Output:
  figures/07b/
    p2_[object]_[signature]_hypoxia_TIM3.png

Run from: ~/single_cell/Visvader/
  python3 07b_hypoxia_TIM3_extended.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import json, os, warnings, gc
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
SIGFILE  = "signatures_confirmed.json"
OUTDIR   = "figures/07b"
GRIDSIZE = 90
SUBTYPE_COL = "subtype"
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}

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
    "Unknown"    : "#DDDDDD",
}

STEMNESS_KEYS = [
    "ESC_ASSOU", "ESC_WONG", "BENPORATH_ES1", "stem_consensus",
    "fMaSC_over_aMaSC", "LIM_MAMMARY_STEM_CELL_UP",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP", "LIM_MAMMARY_LUMINAL_MATURE_UP",
    "EMP"
]

HALLMARK_EMT = [
    "ABI3BP","ACTA2","ADAM12","ANPEP","APLP1","AREG","BASP1","BDNF","BGN",
    "BMP1","CADM1","CALD1","CALU","CAP2","CAPG","CCN1","CCN2","CD44","CD59",
    "CDH11","CDH2","CDH6","COL11A1","COL12A1","COL16A1","COL1A1","COL1A2",
    "COL3A1","COL4A1","COL4A2","COL5A1","COL5A2","COL5A3","COL6A2","COL6A3",
    "COL7A1","COL8A2","COLGALT1","COMP","COPA","CRLF1","CTHRC1","CXCL1",
    "CXCL12","CXCL6","CXCL8","DAB2","DCN","DKK1","DPYSL3","DST","ECM1",
    "ECM2","EDIL3","EFEMP2","ELN","EMP3","ENO2","FAP","FAS","FBLN1","FBLN2",
    "FBLN5","FBN1","FBN2","FERMT2","FGF2","FLNA","FMOD","FN1","FOXC2",
    "FSTL1","FSTL3","FUCA1","FZD8","GADD45A","GADD45B","GAS1","GEM","GJA1",
    "GLIPR1","GPC1","GPX7","GREM1","HTRA1","ID2","IGFBP2","IGFBP3","IGFBP4",
    "IL15","IL32","IL6","INHBA","ITGA2","ITGA5","ITGAV","ITGB1","ITGB3",
    "ITGB5","JUN","LAMA1","LAMA2","LAMA3","LAMC1","LAMC2","LGALS1","LOX",
    "LOXL1","LOXL2","LRP1","LRRC15","LUM","MAGEE1","MATN2","MATN3","MCM7",
    "MEST","MFAP5","MGP","MMP1","MMP14","MMP2","MMP3","MSX1","MXRA5","MYL9",
    "MYLK","NID2","NNMT","NOTCH2","NT5E","NTM","OXTR","P3H1","PCOLCE",
    "PCOLCE2","PDGFRB","PDLIM4","PFN2","PLAUR","PLOD1","PLOD2","PLOD3",
    "PMEPA1","PMP22","POSTN","PPIB","PRRX1","PRSS2","PTHLH","PTX3","PVR",
    "QSOX1","RGS4","RHOB","SAT1","SCG2","SDC1","SDC4","SERPINE1","SERPINE2",
    "SERPINH1","SFRP1","SFRP4","SGCB","SGCD","SGCG","SLC6A8","SLIT2","SLIT3",
    "SNAI2","SNTB1","SPARC","SPOCK1","SPP1","TAGLN","TFPI2","TGFB1","TGFBI",
    "TGFBR3","TGM2","THBS1","THBS2","THY1","TIMP1","TIMP3","TNC","TNFAIP3",
    "TNFRSF11B","TNFRSF12A","TPM1","TPM2","TPM4","VCAM1","VCAN","VEGFA",
    "VEGFC","VIM","WIPF1","WNT5A",
]

# 16 objects split by compartment
IMMUNE_OBJECTS = {
    "Immune_Global": "compartment_immune.h5ad",
    "Immune_ER"    : "immune_ER.h5ad",
    "Immune_HER2"  : "immune_HER2.h5ad",
    "Immune_TNBC"  : "immune_TNBC.h5ad",
}

# Part 2: all compartment objects
PART2_OBJECTS = {
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

# GO:0001666 — same as script 07
GO_0001666 = list(dict.fromkeys([
    "HIF1A","EPAS1","HIF3A","ARNT","ARNT2",
    "VEGFA","VEGFB","VEGFC","VEGFD","PGF",
    "LDHA","LDHB","LDHC","PFKL","PFKM","PFKFB3","PFKFB4",
    "SLC2A1","SLC2A3","HK1","HK2","HK3","ENO1","ENO2",
    "PDK1","PDK3","PDHK1",
    "BNIP3","BNIP3L","BNIP1","CA9","CA12",
    "ANGPT1","ANGPT2","ANGPTL4","EPO","EPOR",
    "NOS2","NOS3","HMOX1","HMOX2","ADM","ADM2",
    "SERPINE1","SERPINE2","FN1","COL1A1","COL4A1","COL5A1",
    "MMP2","MMP9","MMP14","CXCR4","CXCL12","CXCL8",
    "TNF","IL1A","IL1B","IL6","IL8","PTGS2","PTGIS",
    "TF","TFRC","SLC11A2","FTH1","FTL",
    "EGLN1","EGLN2","EGLN3","VHL","PHD1","PHD2","PHD3",
    "P4HA1","P4HA2","PLOD1","PLOD2","DDIT4","DDIT3",
    "ATF4","ATF3","CREB3","SLC1A5","SLC7A5","SLC7A11",
    "GPI","ALDOA","ALDOC","TPI1","GAPDH","PGK1","PGAM1","PKM",
    "MCM3","MCM4","MCM5","MCM6","MCM7",
    "CDKN1A","CDKN1B","TP53",
    "BCL2","BCL2L1","MCL1","BAX","BAK1",
    "NOTCH1","NOTCH2","DLL1","DLL4","JAG1","JAG2",
    "KDR","FLT1","FLT4","TEK","TIE1",
    "TWIST1","TWIST2","SNAI1","SNAI2","ZEB1","ZEB2","VIM","CDH2",
    "MET","HGF","IGF1","IGF1R","IGF2","IGF2R","IRS1","IRS2",
    "PDGFA","PDGFB","PDGFC","PDGFD","PDGFRA","PDGFRB",
    "TGFB1","TGFB2","TGFB3","TGFBR1","TGFBR2",
    "WNT2","WNT5A","FZD1","CTNNB1","MYC","MYCN","NF1","NF2",
    "PTEN","PIK3CA","AKT1","AKT2","MTOR","RPTOR",
    "MAPK1","MAPK3","MAPK8","MAP2K1","MAP2K2",
    "STAT3","STAT5A","STAT5B","JAK1","JAK2",
    "NFE2L2","KEAP1","SQSTM1",
    "SOD1","SOD2","CAT","GPX1","GPX4","PRDX1","PRDX2","PRDX5",
]))


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
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
    if gene not in adata.var_names:
        return None
    x = adata[:, gene].X
    return np.asarray(x.todense()).flatten() if issparse(x) \
           else np.asarray(x).flatten()


def assign_immune_type(obs):
    """Winner-takes-all — same logic as script 07."""
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items()
                    if col in obs.columns}
    if not cols_present:
        return None
    mat        = obs[[c for c in cols_present.values()]].values
    cell_types = list(cols_present.keys())
    best_idx   = mat.argmax(axis=1)
    best_score = mat.max(axis=1)
    return np.array([cell_types[i] if s > 0 else "Unknown"
                     for i, s in zip(best_idx, best_score)])


def axis_limits(hyp, havcr):
    """Consistent axis limits for all scatter/hexbin panels."""
    xlim = (
        np.percentile(hyp, 0.5)  - 0.15 * np.ptp(np.percentile(hyp, [0.5, 99.5])),
        np.percentile(hyp, 99.5) + 0.15 * np.ptp(np.percentile(hyp, [0.5, 99.5]))
    )
    pos = havcr[havcr > 0]
    ymax = np.percentile(pos, 99.5) if len(pos) > 0 else 1.0
    ylim = (-0.05 * ymax, ymax * 1.20)
    return xlim, ylim, ymax


def format_scatter_ax(ax, xlim, ylim, ymax, xlabel=True, ylabel=True):
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.4, zorder=3)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.axhspan(ylim[0], 0, color="#F5F5F5", alpha=0.6, zorder=0)
    ax.text(xlim[1] * 0.98, ymax * 0.02, "HAVCR2=0",
            ha="right", va="bottom", fontsize=6, color="#AAAAAA",
            style="italic")
    if xlabel:
        ax.set_xlabel("Hypoxia score (GO:0001666, z)", fontsize=8)
    if ylabel:
        ax.set_ylabel("HAVCR2 / TIM3 (log-norm)", fontsize=8)


def hexbin_panel(ax, x, y, title, xlabel="", ylabel="",
                 gridsize=GRIDSIZE, filter_pos=False):
    """Hexbin matching script-07 style."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if filter_pos:
        mask &= (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        ax.text(0.5, 0.5, "n<20", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        ax.set_visible(True); return

    xmin, xmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ymin, ymax = np.percentile(y, 0.5), np.percentile(y, 99.5)
    xpad = 0.20*(xmax-xmin); ypad = 0.20*(ymax-ymin)
    xl = (xmin-xpad, xmax+xpad); yl = (ymin-ypad, ymax+ypad)

    ax.set_facecolor("white")
    ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="RdYlBu_r",
                   mincnt=1, linewidths=0.02,
                   extent=[xl[0],xl[1],yl[0],yl[1]], zorder=2)
    ax.axhline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")
    r, p = stats.pearsonr(x, y)
    pstr = f"p={p:.1e}" if p >= 1e-300 else "p<1e-300"
    ax.text(0.97, 0.03, f"r={r:.3f}\n{pstr}\nn={len(x):,}",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    plt.colorbar(hb, ax=ax, shrink=0.7, label="Cell count")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SCATTER COLOURED BY IMMUNE CELL TYPE
# ══════════════════════════════════════════════════════════════════════════════

def p1_scatter_celltype(obs, obj_name, xlim, ylim, ymax, fname):
    """
    Scatter hypoxia × HAVCR2 coloured by immune cell type.
    Columns = All + ER + HER2 + TNBC.
    Background: HAVCR2− grey. Foreground: HAVCR2+ by cell type.
    """
    hyp   = obs["score_hypoxia_z"].values.astype(float)
    havcr = obs["HAVCR2_expr"].values.astype(float)
    ctype = obs["immune_cell_type"].values

    subtypes    = [st for st in ["ER","HER2","TNBC"]
                   if SUBTYPE_COL in obs.columns
                   and st in obs[SUBTYPE_COL].values]
    groups      = [("All", np.ones(len(obs), bool))] + \
                  [(st, (obs[SUBTYPE_COL]==st).values) for st in subtypes]
    n_cols      = len(groups)

    draw_order  = [ct for ct in IMMUNE_CELL_TYPES if ct in ctype] + \
                  (["Unknown"] if "Unknown" in ctype else [])

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5*n_cols, 6),
                             squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f"{obj_name} — Hypoxia × TIM3 · colour = immune cell type",
                 fontsize=11, fontweight="bold")

    # Build legend handles once
    handles = []
    for ct in draw_order:
        if ct == "Unknown":
            continue
        m_pos = (ctype == ct) & (havcr > 0)
        if m_pos.sum() == 0:
            continue
        handles.append(Line2D([0],[0], marker="o", color="w",
                               markerfacecolor=CELLTYPE_COLORS.get(ct,"#888"),
                               markersize=7, label=f"{ct} ({m_pos.sum():,}+)"))

    for col_i, (label, mask) in enumerate(groups):
        ax  = axes[col_i]
        hyp_m   = hyp[mask];   havcr_m = havcr[mask]
        ctype_m = ctype[mask]
        n_m     = mask.sum()
        n_pos_m = (havcr_m > 0).sum()

        # HAVCR2+ only — coloured by cell type
        if n_pos_m < 10:
            ax.text(0.5, 0.5, f"HAVCR2+\nn<10",
                    transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
        else:
            for ct in draw_order:
                m = (ctype_m == ct) & (havcr_m > 0)
                if m.sum() == 0:
                    continue
                ax.scatter(hyp_m[m], havcr_m[m],
                           c=CELLTYPE_COLORS.get(ct, "#888"),
                           s=20, alpha=0.80,
                           linewidths=0.2, edgecolors="white",
                           rasterized=True, zorder=2)

        # Use HAVCR2+ ylim only (y starts at first positive value)
        pos_all = havcr[havcr > 0]
        ymin_pos = np.percentile(pos_all, 0.5) * 0.8 if len(pos_all) > 0 else 0
        ax.set_ylim(ymin_pos, ylim[1])

        format_scatter_ax(ax, xlim, (ymin_pos, ylim[1]), ymax,
                          xlabel=True, ylabel=(col_i==0))
        color = SUBTYPE_COLORS.get(label, "black") if label != "All" else "black"
        ax.set_title(f"{label}  (n={n_pos_m:,} HAVCR2+)",
                     fontsize=9, fontweight="bold", color=color)

        if col_i == 0 and handles:
            ax.legend(handles=handles, fontsize=6.5, loc="upper left",
                      framealpha=0.88,
                      title="Cell type (HAVCR2+)",
                      title_fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/{fname}.png")


def p1_hexbin_per_celltype(obs, obj_name, xlim, fname_prefix):
    """
    One figure per immune cell type:
    Hexbin hypoxia × HAVCR2 for cells of that type only.
    Columns = All + ER + HER2 + TNBC.
    """
    hyp   = obs["score_hypoxia_z"].values.astype(float)
    havcr = obs["HAVCR2_expr"].values.astype(float)
    ctype = obs["immune_cell_type"].values

    subtypes = [st for st in ["ER","HER2","TNBC"]
                if SUBTYPE_COL in obs.columns
                and st in obs[SUBTYPE_COL].values]
    groups   = [("All", np.ones(len(obs), bool))] + \
               [(st, (obs[SUBTYPE_COL]==st).values) for st in subtypes]
    n_cols   = len(groups)

    cell_types_present = [ct for ct in IMMUNE_CELL_TYPES
                          if ct in ctype]

    for ct in cell_types_present:
        ct_mask = ctype == ct
        if ct_mask.sum() < 20:
            continue

        fig, axes = plt.subplots(1, n_cols, figsize=(5.5*n_cols, 5.5),
                                 squeeze=False)
        axes = axes.flatten()
        color = CELLTYPE_COLORS.get(ct, "#888888")
        fig.suptitle(
            f"{obj_name} — {ct} · Hypoxia × TIM3 (hexbin)",
            fontsize=11, fontweight="bold", color=color
        )

        for col_i, (label, mask) in enumerate(groups):
            ax      = axes[col_i]
            combo   = ct_mask & mask
            n_combo = combo.sum()
            st_color = SUBTYPE_COLORS.get(label,"black") \
                       if label != "All" else color
            pos_combo   = combo & (havcr > 0)
            n_pos_combo = pos_combo.sum()
            hexbin_panel(
                ax,
                hyp[pos_combo], havcr[pos_combo],
                title=f"{label} HAVCR2+ (n={n_pos_combo:,})",
                xlabel="Hypoxia (z)",
                ylabel="HAVCR2 (log-norm)" if col_i == 0 else ""
            )
            ax.set_title(f"{label} HAVCR2+ (n={n_pos_combo:,})",
                         fontsize=9, fontweight="bold", color=st_color)

        plt.tight_layout()
        safe_ct = ct.replace("/","_")
        fname   = f"{fname_prefix}_{safe_ct}.png"
        plt.savefig(f"{OUTDIR}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {OUTDIR}/{fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SCATTER COLOURED BY CONTINUOUS SIGNATURE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def p2_scatter_signature(obs, obj_name, sig_col_z, sig_label,
                          xlim, ylim, ymax, fname):
    """
    HAVCR2+ cells only, coloured by signature score (YlOrRd).
    One panel per subtype row (All / ER / HER2 / TNBC).
    Pearson r(hypoxia, sig) annotated per panel.
    """
    hyp   = obs["score_hypoxia_z"].values.astype(float)
    havcr = obs["HAVCR2_expr"].values.astype(float)
    sig   = obs[sig_col_z].values.astype(float) \
            if sig_col_z in obs.columns else np.full(len(obs), np.nan)

    pos_mask = havcr > 0

    subtypes = [st for st in ["ER","HER2","TNBC"]
                if SUBTYPE_COL in obs.columns
                and st in obs[SUBTYPE_COL].values]
    groups   = [("All", np.ones(len(obs), bool))] + \
               [(st, (obs[SUBTYPE_COL]==st).values) for st in subtypes]
    n_rows   = len(groups)

    # Shared colour scale: p2–p98 of all HAVCR2+ cells with sig score
    sig_pos  = sig[pos_mask & ~np.isnan(sig)]
    vmin_pos = np.nanpercentile(sig_pos, 2)  if len(sig_pos) > 0 else -2
    vmax_pos = np.nanpercentile(sig_pos, 98) if len(sig_pos) > 0 else  2

    # y-axis: HAVCR2+ range only
    pos_vals = havcr[pos_mask]
    ymin_pos = np.percentile(pos_vals, 0.5) * 0.8 if len(pos_vals) > 0 else 0
    ylim_pos = (ymin_pos, ylim[1])

    fig, axes = plt.subplots(1, n_rows,
                             figsize=(5.5 * n_rows, 5.5),
                             squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        f"{obj_name} — Hypoxia × TIM3 (HAVCR2+ only)\n"
        f"Colour = {sig_label}",
        fontsize=11, fontweight="bold"
    )

    for col_i, (label, mask) in enumerate(groups):
        ax       = axes[col_i]
        combo    = mask & pos_mask
        hyp_m    = hyp[combo]
        havcr_m  = havcr[combo]
        sig_m    = sig[combo]
        n_pos    = combo.sum()
        st_color = SUBTYPE_COLORS.get(label,"black") if label != "All" else "black"

        if n_pos < 10:
            ax.text(0.5, 0.5, f"HAVCR2+\nn<10",
                    transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
            format_scatter_ax(ax, xlim, ylim_pos, ymax,
                              xlabel=True, ylabel=(col_i==0))
            ax.set_title(f"{label}  (n<10 HAVCR2+)",
                         fontsize=9, fontweight="bold", color=st_color)
            continue

        valid = ~np.isnan(sig_m)
        sc = ax.scatter(
            hyp_m[valid], havcr_m[valid],
            c=sig_m[valid], cmap="YlOrRd",
            vmin=vmin_pos, vmax=vmax_pos,
            s=20, alpha=0.82,
            linewidths=0.2, edgecolors="white",
            rasterized=True, zorder=2
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
        cbar.set_label(f"{sig_label} (z)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        format_scatter_ax(ax, xlim, ylim_pos, ymax,
                          xlabel=True, ylabel=(col_i==0))
        ax.set_title(f"{label}  (n={n_pos:,} HAVCR2+)",
                     fontsize=9, fontweight="bold", color=st_color)

        # Pearson r(hypoxia, sig) in HAVCR2+ cells
        if valid.sum() > 30:
            r, p = stats.pearsonr(hyp_m[valid], sig_m[valid])
            pstr = f"p={p:.1e}" if p >= 1e-300 else "p<1e-300"
            ax.text(0.97, 0.03,
                    f"r(hyp,sig)={r:.3f}\n{pstr}",
                    transform=ax.transAxes, fontsize=6.5,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/{fname}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
with open(SIGFILE) as f:
    SIGNATURES = json.load(f)

print("\n" + "="*65)
print(" 07b — HYPOXIA × TIM3 EXTENDED")
print("="*65)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: immune objects — colour by cell type
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print(" PART 1 — Immune cell type colouring")
print("─"*65)

for obj_name, h5ad_file in IMMUNE_OBJECTS.items():
    if not os.path.exists(h5ad_file):
        print(f"\n  ⚠️  {h5ad_file} not found — skipping {obj_name}")
        continue

    print(f"\n{'='*65}\n  {obj_name}\n{'='*65}")
    adata = sc.read_h5ad(h5ad_file)
    print(f"  Cells: {adata.n_obs:,}")

    if "logcounts" in adata.layers:
        adata.X = adata.layers["logcounts"].copy()

    # Score hypoxia
    hypoxia_genes = [g for g in GO_0001666 if g != "HAVCR2"]
    ok, found = score_and_znorm(adata, hypoxia_genes, "score_hypoxia")
    print(f"  GO:0001666: {len(found)}/{len(hypoxia_genes)} genes "
          f"({'✓' if ok else '✗'})")
    if not ok:
        del adata; gc.collect(); continue

    # HAVCR2
    havcr2 = get_gene_expr(adata, "HAVCR2")
    if havcr2 is None:
        print("  ⚠️  HAVCR2 not found"); del adata; gc.collect(); continue
    adata.obs["HAVCR2_expr"] = havcr2
    pct = (havcr2 > 0).mean() * 100
    print(f"  HAVCR2: {pct:.1f}% expressing")

    # Immune cell types
    immune_types = assign_immune_type(adata.obs)
    if immune_types is None:
        print("  ⚠️  No immune scores"); del adata; gc.collect(); continue
    adata.obs["immune_cell_type"] = immune_types
    for ct, n in pd.Series(immune_types).value_counts().items():
        print(f"      {ct:20s}: {n:6,}")

    obs  = adata.obs
    safe = obj_name.replace("/","_")
    hyp_all   = obs["score_hypoxia_z"].values.astype(float)
    havcr_all = obs["HAVCR2_expr"].values.astype(float)
    xlim, ylim, ymax = axis_limits(hyp_all, havcr_all)

    # 1A — scatter coloured by cell type
    p1_scatter_celltype(obs, obj_name, xlim, ylim, ymax,
                        fname=f"p1_scatter_{safe}_celltype")

    # 1B — hexbin per cell type
    p1_hexbin_per_celltype(obs, obj_name, xlim,
                           fname_prefix=f"p1_hexbin_{safe}")

    del adata; gc.collect()
    print(f"  ✓ Part 1 done: {obj_name}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: all compartments — colour by continuous signature
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print(" PART 2 — Continuous signature colouring (EMT + stemness)")
print("─"*65)

# Build signature dict: name → (gene_list, short_label)
SIG_DICT = {"HALLMARK_EMT": (HALLMARK_EMT, "EMT")}
for key in STEMNESS_KEYS:
    if key in SIGNATURES:
        SIG_DICT[key] = (SIGNATURES[key],
                         key.replace("LIM_MAMMARY_","LIM_").replace("_"," "))

print(f"  Signatures to score: {list(SIG_DICT.keys())}")

for obj_name, h5ad_file in PART2_OBJECTS.items():
    if not os.path.exists(h5ad_file):
        print(f"\n  ⚠️  {h5ad_file} not found — skipping {obj_name}")
        continue

    print(f"\n{'='*65}\n  {obj_name}\n{'='*65}")
    adata = sc.read_h5ad(h5ad_file)
    print(f"  Cells: {adata.n_obs:,}")

    if "logcounts" in adata.layers:
        adata.X = adata.layers["logcounts"].copy()

    # Score hypoxia
    hypoxia_genes = [g for g in GO_0001666 if g != "HAVCR2"]
    ok, found = score_and_znorm(adata, hypoxia_genes, "score_hypoxia")
    print(f"  GO:0001666: {len(found)}/{len(hypoxia_genes)} genes "
          f"({'✓' if ok else '✗'})")
    if not ok:
        del adata; gc.collect(); continue

    # HAVCR2
    havcr2 = get_gene_expr(adata, "HAVCR2")
    if havcr2 is None:
        print("  ⚠️  HAVCR2 not found"); del adata; gc.collect(); continue
    adata.obs["HAVCR2_expr"] = havcr2
    pct = (havcr2 > 0).mean() * 100
    print(f"  HAVCR2: {pct:.1f}% expressing")

    obs  = adata.obs
    safe = obj_name.replace("/","_")
    hyp_all   = obs["score_hypoxia_z"].values.astype(float)
    havcr_all = obs["HAVCR2_expr"].values.astype(float)
    xlim, ylim, ymax = axis_limits(hyp_all, havcr_all)

    # Score each signature and generate figure
    for sig_name, (gene_list, sig_label) in SIG_DICT.items():
        col     = f"score_{sig_name}"
        col_z   = f"{col}_z"
        ok_sig, found_sig = score_and_znorm(adata, gene_list, col)
        print(f"  {sig_name}: {len(found_sig)}/{len(gene_list)} genes "
              f"({'✓' if ok_sig else '✗'})")
        if not ok_sig:
            continue

        safe_sig = sig_name.replace(" ","_").replace("/","_")
        p2_scatter_signature(
            adata.obs, obj_name,
            sig_col_z=col_z,
            sig_label=sig_label,
            xlim=xlim, ylim=ylim, ymax=ymax,
            fname=f"p2_{safe}_{safe_sig}"
        )

    del adata; gc.collect()
    print(f"  ✓ Part 2 done: {obj_name}")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Output: {OUTDIR}/
  PART 1 (immune cell type):
    p1_scatter_[object]_celltype.png
    p1_hexbin_[object]_[celltype].png
  PART 2 (signature colour):
    p2_[object]_HALLMARK_EMT.png
    p2_[object]_[stemness_sig].png
══════════════════════════════════════════════════════════════
""")
