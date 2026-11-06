"""
01_compartment_hexbin_scan.py
=============================
Broad, uncorrected per-cell hexbin correlation scan of LCOR/CXCL16/CXCR6,
across ALL 13 cell groups (compartments/immune subtypes) and all 4 subtype
splits (All/ER/HER2/TNBC). No depth correction, no restriction to the
malignant compartment -- this is the first, broadest correlation look,
predating the refined tests in 05-06 (malignant-only, depth-corrected).

WHY THIS SCRIPT EXISTS / SPLIT HISTORY
-----------------------------------------
Originally part of 01_celltype_specificity_overview.py (that script's
hexbin section) -- split out because its content is a correlation
analysis (Pearson r / Spearman rho per panel), not exploratory cell-type
specificity, even though it reuses the same 13-group cell classification
and shares the same purpose statement below. 01 keeps only the bubble/dot
plots (pure "where are these genes expressed" exploration); this script
keeps the hexbin grids (pairwise gene correlation, per group x subtype).

CELL GROUPS (13, fixed x-axis order) -- identical scheme to
01_celltype_specificity_overview.py
-------------------------------------------------------------
  Epithelial | CD8_T CD4_T Treg NK Macro_M1 Macro_M2 B_cell Plasmablast
               Dendritic Mast Unknown | Stromal

Epithelial/Stromal come from compartment_epithelial.h5ad / compartment_stromal.h5ad
directly (single group per file). The 11 immune subtypes come from
compartment_immune.h5ad, which already carries precomputed per-cell marker
scores (score_CD8_T, score_CD4_T, ... -- from sc.tl.score_genes, see
07_hipoxia_tim3.py) -- IMMUNE_CELL_TYPES and assign_immune_type() below are
copied verbatim from that script (winner-takes-all argmax over the existing
score columns; "Unknown" = a cell where every score <= 0).

IMPORTANT CAVEAT
-----------------
LCOR-CXCL16 panels test genuine per-cell co-expression (both plausibly
present in the same, Epithelial, cells) -- this is a real (if
depth-uncorrected, all-compartment) correlation test, superseded in rigor
by 02_depth_artifact_cell_level.py's depth-offset model restricted to
malignant cells. CXCL16-CXCR6 and LCOR-CXCR6 panels, computed per-cell
WITHIN a single group, mostly show these gene pairs occupy DIFFERENT cell
types -- i.e. those two pairs are a compartmental-specificity check, NOT
a ligand-receptor interaction test (that needs cross-cell-type analysis;
see 01_liana_interaction_vs_lcor.py's per-patient LIANA run).

OUTPUTS
-------
LCOR-CXCL16/02_correlacion/figures/01_compartment_hexbin_scan/
    hexbin_LCOR_vs_CXCL16.png / hexbin_LCOR_vs_CXCR6.png / hexbin_CXCL16_vs_CXCR6.png
        4 rows (All/ER/HER2/TNBC) x 13 columns (groups) hexbin grids, per gene pair.
    hexbin_correlation_table.csv
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/02_correlacion/scripts/01_compartment_hexbin_scan.py
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Config ───────────────────────────────────────────────────────────────────
OUTDIR = "LCOR-CXCL16/02_correlacion/figures/01_compartment_hexbin_scan"
GENES  = ["LCOR", "CXCL16", "CXCR6"]
GENE_PAIRS = [("LCOR", "CXCL16"), ("LCOR", "CXCR6"), ("CXCL16", "CXCR6")]
SUBTYPES = ["All", "ER", "HER2", "TNBC"]
GRIDSIZE = 90
os.makedirs(OUTDIR, exist_ok=True)

# Immune subtype scoring -- copied verbatim from 07_hipoxia_tim3.py (and 01_celltype_specificity_overview.py)
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


def hexbin_panel(ax, x, y, title, xlabel, ylabel, gridsize=90):
    """Same footprint as 14_lcor_cxcl16_hexbin.py's hexbin_panel (log color scale)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 20:
        ax.set_visible(False)
        return None

    xmin, xmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ymin, ymax = np.percentile(y, 0.5), np.percentile(y, 99.5)
    xpad = 0.20 * (xmax - xmin); ypad = 0.20 * (ymax - ymin)
    xl = (xmin - xpad, xmax + xpad)
    yl = (ymin - ypad, ymax + ypad)
    if xl[0] == xl[1]:
        xl = (xl[0] - 0.5, xl[1] + 0.5)
    if yl[0] == yl[1]:
        yl = (yl[0] - 0.5, yl[1] + 0.5)

    ax.set_facecolor("white")
    ax.grid(True, color="#DDDDDD", lw=0.4, zorder=0)
    ax.hexbin(x, y, gridsize=gridsize, cmap="RdYlBu_r", mincnt=1, linewidths=0.02,
              bins="log", extent=[xl[0], xl[1], yl[0], yl[1]], zorder=2)
    ax.axhline(0, color="black", ls="--", lw=0.7, alpha=0.5, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=0.7, alpha=0.5, zorder=3)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    ax.text(0.96, 0.04, f"r={r:.2f}\nρ={rho:.2f}", transform=ax.transAxes,
            fontsize=5.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))
    return r, p, rho, p_s, len(x)


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
print(" 01 -- LCOR / CXCL16 / CXCR6 -- compartment hexbin scan (correlation)")
print("="*65)

epi_df, _ = extract("compartment_epithelial.h5ad", GENES)
epi_df["group"] = "Epithelial"

imm_df, imm_obs = extract("compartment_immune.h5ad", GENES)
imm_df["group"] = assign_immune_type(imm_obs)

str_df, _ = extract("compartment_stromal.h5ad", GENES)
str_df["group"] = "Stromal"

combined = pd.concat([epi_df, imm_df, str_df], ignore_index=True)
print(f"\n  Combined: {len(combined):,} cells across {combined['group'].nunique()} groups")


def subset(df, subtype):
    return df if subtype == "All" else df[df["subtype"] == subtype]


# ══════════════════════════════════════════════════════════════════════════════
# HEXBIN GRIDS -- one figure per gene pair, rows=subtype, cols=group
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Hexbin grids (per gene pair x group x subtype)")
print("="*65)
print("""
  NOTE: LCOR-CXCL16 panels test real per-cell co-expression (both plausibly
  present in Epithelial cells) -- a genuine, if depth-uncorrected and
  all-compartment, correlation test. CXCL16-CXCR6 and LCOR-CXCR6 panels,
  computed per-cell WITHIN a single group, mostly show these genes occupy
  DIFFERENT cell types -- a compartmental-specificity check, not a
  ligand-receptor interaction test (that needs cross-cell-type analysis;
  see 01_liana_interaction_vs_lcor.py's per-patient LIANA run).
""")

corr_rows = []
n_g = len(GROUP_ORDER)

for gene_x, gene_y in GENE_PAIRS:
    print(f"  {gene_x} vs {gene_y} ...")
    fig, axes = plt.subplots(len(SUBTYPES), n_g,
                              figsize=(2.05 * n_g, 2.15 * len(SUBTYPES)))
    for ri, subtype in enumerate(SUBTYPES):
        d = subset(combined, subtype)
        for ci, grp in enumerate(GROUP_ORDER):
            ax = axes[ri, ci]
            dg = d[d["group"] == grp]
            x = dg[f"{gene_x}_logcounts"].values.astype(float)
            y = dg[f"{gene_y}_logcounts"].values.astype(float)
            title = grp if ri == 0 else ""
            xlabel = gene_x if ri == len(SUBTYPES) - 1 else ""
            ylabel = gene_y if ci == 0 else ""
            out = hexbin_panel(ax, x, y, title, xlabel, ylabel, gridsize=GRIDSIZE)
            if out:
                r, p, rho, p_s, n = out
                corr_rows.append({"gene_pair": f"{gene_x}_vs_{gene_y}", "subtype": subtype,
                                   "group": grp, "n": n, "pearson_r": r, "pearson_p": p,
                                   "spearman_rho": rho, "spearman_p": p_s})
            if ci == 0:
                ax.annotate(subtype, xy=(-0.55, 0.5), xycoords="axes fraction",
                            fontsize=9, fontweight="bold", rotation=90, ha="center", va="center")

    fig.suptitle(f"{gene_x} vs {gene_y} -- logcounts, per group x subtype "
                 f"(log color scale; dashed lines = gene not detected)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0.01, 0, 1, 0.97])
    outpath = f"{OUTDIR}/hexbin_{gene_x}_vs_{gene_y}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(f"{OUTDIR}/hexbin_correlation_table.csv", index=False)
print(f"  Saved: {OUTDIR}/hexbin_correlation_table.csv  ({len(df_corr)} non-empty panels)")

# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
readme = [
    "01 -- LCOR / CXCL16 / CXCR6 -- compartment hexbin scan (correlation, Block 2)",
    "="*70,
    "",
    "SPLIT HISTORY",
    "-"*70,
    "Originally the hexbin section of 01_celltype_specificity_overview.py.",
    "Split out into its own script because this content is a correlation",
    "analysis (Pearson r / Spearman rho per panel), not exploratory cell-type",
    "specificity -- 01 keeps only the bubble/dot plots.",
    "",
    "PURPOSE",
    "-"*70,
    "Broadest, uncorrected per-cell hexbin correlation of LCOR/CXCL16/CXCR6,",
    "across all 13 cell groups and 4 subtype splits, no depth correction, no",
    "restriction to the malignant compartment -- the first correlation look,",
    "predating the refined tests in 02_depth_artifact_cell_level.py",
    "(malignant-only, depth-offset GLM) and 03_depth_artifact_diagnostic_hexbin.py.",
    "",
    "CELL GROUPS (13, fixed order) -- identical scheme to",
    "01_celltype_specificity_overview.py",
    "-"*70,
    "  " + " | ".join(GROUP_ORDER),
    "",
    "OUTPUTS",
    "-"*70,
    "  hexbin_{LCOR_vs_CXCL16,LCOR_vs_CXCR6,CXCL16_vs_CXCR6}.png",
    "    4 rows (All/ER/HER2/TNBC) x 13 columns (groups), each panel a hexbin",
    "    of the two genes' logcounts among cells in that group x subtype",
    "    (gridsize=90, log color scale, Pearson r + Spearman rho annotated;",
    "    panels with <20 cells are blank).",
    "  hexbin_correlation_table.csv",
    "    gene_pair, subtype, group, n, pearson_r, pearson_p, spearman_rho, spearman_p",
    "",
    "IMPORTANT CAVEAT",
    "-"*70,
    "  LCOR-CXCL16 panels test genuine per-cell co-expression (both plausibly",
    "  present in the same, Epithelial, cells) -- a real but depth-uncorrected,",
    "  all-compartment correlation test, superseded in rigor by",
    "  02_depth_artifact_cell_level.py's depth-offset model restricted to",
    "  malignant cells. CXCL16-CXCR6 and LCOR-CXCR6, computed per-cell WITHIN",
    "  a single group, mostly show these gene pairs occupy DIFFERENT cell",
    "  types (near-zero for one gene in groups where it isn't expected) --",
    "  i.e. those two pairs are a compartmental-specificity check, NOT a",
    "  ligand-receptor interaction test. Testing the CXCL16(tumor)->CXCR6",
    "  (T cell) interaction requires cross-cell-type analysis and is",
    "  covered by 01_liana_interaction_vs_lcor.py's per-patient LIANA run.",
]
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
