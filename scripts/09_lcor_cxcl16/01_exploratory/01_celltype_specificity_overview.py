"""
01_celltype_specificity_overview.py
===================================
Exploratory expression overview of LCOR, CXCL16, CXCR6 across tissue
compartments, immune cell subtypes, and breast cancer subtypes: where are
these 3 genes actually expressed (bubble/dot plots only -- pure "where do
these genes sit" exploration).

WHY THIS SCRIPT EXISTS
-----------------------
Block 2 (correlation) and block 4 (LIANA interaction) both assume LCOR/
CXCL16 sit in tumor (Epithelial) cells and CXCR6 sits in immune cells.
This script is the basic sanity/exploration layer checking that
assumption directly, before any correlation or interaction test is run.

CELL GROUPS (13, fixed x-axis order)
-------------------------------------
  Epithelial | CD8_T CD4_T Treg NK Macro_M1 Macro_M2 B_cell Plasmablast
               Dendritic Mast Unknown | Stromal

Epithelial/Stromal come from compartment_epithelial.h5ad / compartment_stromal.h5ad
directly (single group per file). The 11 immune subtypes come from
compartment_immune.h5ad, which already carries precomputed per-cell marker
scores (score_CD8_T, score_CD4_T, ... -- from sc.tl.score_genes, see
07_hipoxia_tim3.py) -- IMMUNE_CELL_TYPES and assign_immune_type() below are
copied verbatim from that script (winner-takes-all argmax over the existing
score columns; "Unknown" = a cell where every score <= 0).

SPLIT HISTORY
--------------
This script used to also produce a hexbin correlation grid (pairwise gene
co-expression per group x subtype). That content is a correlation
analysis, not exploration, so it was split out into
01_compartment_hexbin_scan.py (Block 2 -- Correlation), which reuses the
same cell-grouping code below.

OUTPUTS
-------
LCOR-CXCL16/01_exploratorio/figures/01_celltype_specificity_overview/
    fig_bubble_All.png / _ER.png / _HER2.png / _TNBC.png
        Dot plot: rows=genes, columns=the 13 groups (Epithelial/Immune/Stromal
        background bands), dot size=%detected, dot color=mean logcounts
        (shared color scale across the 4 figures).
    bubble_summary_table.csv
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/01_exploratorio/scripts/01_celltype_specificity_overview.py
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
sc.settings.verbosity = 0
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Config ───────────────────────────────────────────────────────────────────
OUTDIR = "LCOR-CXCL16/01_exploratorio/figures/01_celltype_specificity_overview"
GENES  = ["LCOR", "CXCL16", "CXCR6"]
SUBTYPES = ["All", "ER", "HER2", "TNBC"]
os.makedirs(OUTDIR, exist_ok=True)

# Immune subtype scoring -- copied verbatim from 07_hipoxia_tim3.py
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

GROUP_ORDER = ["Epithelial"] + list(IMMUNE_CELL_TYPES.keys()) + ["Unknown", "Stromal"]


def assign_immune_type(obs):
    """Winner-takes-all from existing immune cell type scores. (07_hipoxia_tim3.py)"""
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items() if col in obs.columns}
    score_matrix = obs[[col for col in cols_present.values()]].values
    cell_types   = list(cols_present.keys())
    best_idx     = score_matrix.argmax(axis=1)
    best_score   = score_matrix.max(axis=1)
    return np.array([cell_types[i] if s > 0 else "Unknown"
                      for i, s in zip(best_idx, best_score)])


def extract(h5ad_file, genes):
    print(f"  Loading {h5ad_file}...")
    adata = sc.read_h5ad(h5ad_file)
    print(f"    Cells: {adata.n_obs:,}")
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise SystemExit(f"  Gene(s) not found in {h5ad_file}: {missing}")

    idx = [adata.var_names.get_loc(g) for g in genes]
    counts = adata.layers["counts"][:, idx]
    counts = np.asarray(counts.todense()) if issparse(counts) else np.asarray(counts)
    logc = adata.layers["logcounts"][:, idx]
    logc = np.asarray(logc.todense()) if issparse(logc) else np.asarray(logc)

    df = pd.DataFrame({"subtype": adata.obs["subtype"].astype(str).values})
    for i, g in enumerate(genes):
        df[f"{g}_counts"] = counts[:, i]
        df[f"{g}_logcounts"] = logc[:, i]
    return df, adata.obs


# ══════════════════════════════════════════════════════════════════════════════
# LOAD + BUILD THE 13-GROUP TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print(" 01 -- LCOR / CXCL16 / CXCR6 -- celltype specificity overview (exploratory)")
print("="*65)

epi_df, _ = extract("compartment_epithelial.h5ad", GENES)
epi_df["group"] = "Epithelial"

imm_df, imm_obs = extract("compartment_immune.h5ad", GENES)
imm_df["group"] = assign_immune_type(imm_obs)

str_df, _ = extract("compartment_stromal.h5ad", GENES)
str_df["group"] = "Stromal"

combined = pd.concat([epi_df, imm_df, str_df], ignore_index=True)
print(f"\n  Combined: {len(combined):,} cells across {combined['group'].nunique()} groups")

print("\n  Group sizes (All subtypes):")
print(combined["group"].value_counts().reindex(GROUP_ORDER).to_string())
print("\n  Group sizes by subtype:")
print(pd.crosstab(combined["group"], combined["subtype"]).reindex(GROUP_ORDER).to_string())


def subset(df, subtype):
    return df if subtype == "All" else df[df["subtype"] == subtype]


# ══════════════════════════════════════════════════════════════════════════════
# BUBBLE / DOT PLOT -- gene x group, per subtype
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Bubble plot: detection %% (size) x mean logcounts (color)")
print("="*65)

rows = []
for subtype in SUBTYPES:
    d = subset(combined, subtype)
    for grp in GROUP_ORDER:
        dg = d[d["group"] == grp]
        n = len(dg)
        for g in GENES:
            if n == 0:
                pct, meanlog = np.nan, np.nan
            else:
                pct = 100 * (dg[f"{g}_counts"] > 0).mean()
                meanlog = dg[f"{g}_logcounts"].mean()
            rows.append({"subtype": subtype, "group": grp, "gene": g,
                         "n_cells": n, "pct_detected": pct, "mean_logcounts": meanlog})
bubble_df = pd.DataFrame(rows)
bubble_df.to_csv(f"{OUTDIR}/bubble_summary_table.csv", index=False)
print(f"  Saved: {OUTDIR}/bubble_summary_table.csv")

print("\n  Detection %% by gene x group (All subtypes):")
print(bubble_df[bubble_df.subtype == "All"].pivot(index="group", columns="gene",
      values="pct_detected").reindex(GROUP_ORDER).round(1).to_string())

vmax = np.nanmax(bubble_df["mean_logcounts"].values)
norm = Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.viridis


def size_for(pct):
    return 20 + 6.5 * pct  # pct in [0,100] -> size in [20, 670]


BAND_COLORS = {"Epithelial": "#E41A1C", "Immune": "#377EB8", "Stromal": "#4DAF4A"}


def plot_bubble(subtype, outpath):
    d = bubble_df[bubble_df["subtype"] == subtype]
    n_g = len(GROUP_ORDER)
    fig, ax = plt.subplots(figsize=(0.68 * n_g + 1.6, 3.6))

    ax.axvspan(-0.5, 0.5, color=BAND_COLORS["Epithelial"], alpha=0.08, zorder=0)
    ax.axvspan(0.5, n_g - 1.5, color=BAND_COLORS["Immune"], alpha=0.08, zorder=0)
    ax.axvspan(n_g - 1.5, n_g - 0.5, color=BAND_COLORS["Stromal"], alpha=0.08, zorder=0)
    for x0, x1, label in [(-0.5, 0.5, "Epithelial"), (0.5, n_g - 1.5, "Immune"),
                           (n_g - 1.5, n_g - 0.5, "Stromal")]:
        ax.text((x0 + x1) / 2, -0.85, label, ha="center", fontsize=8.5,
                fontweight="bold", color=BAND_COLORS[label])

    for gi, gene in enumerate(GENES):
        for xi, grp in enumerate(GROUP_ORDER):
            row = d[(d.group == grp) & (d.gene == gene)]
            if row.empty or row["n_cells"].values[0] == 0:
                continue
            pct = row["pct_detected"].values[0]
            meanlog = row["mean_logcounts"].values[0]
            ax.scatter(xi, gi, s=size_for(pct), c=[cmap(norm(meanlog))],
                       edgecolor="black", linewidth=0.4, zorder=3)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(GROUP_ORDER, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(GENES)))
    ax.set_yticklabels(GENES, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.6, n_g - 0.4)
    ax.set_ylim(-1.05, len(GENES) - 0.4)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#EAEAEA", linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Subtype: {subtype}", fontsize=10, fontweight="bold", pad=8)

    cbar_ax = fig.add_axes([0.93, 0.30, 0.015, 0.42])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("mean logcounts", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)

    legend_ax = fig.add_axes([0.90, 0.76, 0.10, 0.16])
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis("off")
    legend_ax.set_title("% detected", fontsize=7, pad=3)
    for i, pv in enumerate([10, 50, 90]):
        y = 0.78 - i * 0.42
        legend_ax.scatter([0.25], [y], s=size_for(pv), c="#999999",
                           edgecolors="black", linewidths=0.4, clip_on=False)
        legend_ax.text(0.55, y, f"{pv}%", fontsize=6.5, va="center")

    plt.subplots_adjust(right=0.89, bottom=0.30, top=0.86)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


for subtype in SUBTYPES:
    out = f"{OUTDIR}/fig_bubble_{subtype}.png"
    plot_bubble(subtype, out)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
det_all = bubble_df[bubble_df.subtype == "All"]
headline = []
for g in GENES:
    dg = det_all[det_all.gene == g].set_index("group").reindex(GROUP_ORDER)
    top = dg["pct_detected"].idxmax()
    headline.append(f"  {g}: highest detection in '{top}' ({dg['pct_detected'].max():.1f}%%)")

readme = [
    "01 -- LCOR / CXCL16 / CXCR6 -- celltype specificity overview (Block 1 -- Exploratory)",
    "="*70,
    "",
    "SPLIT HISTORY",
    "-"*70,
    "This script used to also produce a hexbin correlation grid. That content",
    "is a correlation analysis, not exploration, so it was split out into",
    "01_compartment_hexbin_scan.py (Block 2 -- Correlation).",
    "",
    "PURPOSE",
    "-"*70,
    "Exploratory check of where these 3 genes are expressed, across cell",
    "groups and breast cancer subtypes -- a sanity check on the assumption",
    "behind 01_liana_interaction_vs_lcor.py's LIANA analysis (LCOR/CXCL16 in",
    "tumor cells, CXCR6 in immune cells).",
    "",
    "CELL GROUPS (13, fixed order)",
    "-"*70,
    "  " + " | ".join(GROUP_ORDER),
    "",
    "  Epithelial / Stromal: all cells in compartment_epithelial.h5ad /",
    "  compartment_stromal.h5ad respectively.",
    "  The 11 immune groups: winner-takes-all argmax over the precomputed",
    "  score_CD8_T / score_CD4_T / score_Treg / score_NK / score_Macro_M1 /",
    "  score_Macro_M2 / score_B_cell / score_Plasmablast / score_Dendritic /",
    "  score_Mast columns already present in compartment_immune.h5ad (from",
    "  sc.tl.score_genes -- see 07_hipoxia_tim3.py, IMMUNE_CELL_TYPES /",
    "  assign_immune_type(), reused verbatim here). 'Unknown' = every score <= 0.",
    "",
    "OUTPUTS",
    "-"*70,
    "  fig_bubble_{All,ER,HER2,TNBC}.png",
    "    Dot plot: rows=genes, columns=the 13 groups (compartment background",
    "    bands via axvspan). Dot size = %% cells with counts>0 (detection),",
    "    dot color = mean logcounts across ALL cells in the group (zero-",
    "    inflated mean, standard scanpy-dotplot convention). Color scale is",
    "    shared across all 4 figures for comparability.",
    "  bubble_summary_table.csv",
    "    subtype, group, gene, n_cells, pct_detected, mean_logcounts",
    "",
    "HEADLINE NUMBERS (detection %%, all subtypes combined)",
    "-"*70,
] + headline
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Output: {OUTDIR}/
============================================================
""")
