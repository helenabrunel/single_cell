"""
05_hexbin_correlation.py
========================
Cell state scoring, hexbin density analysis and cell cycle quadrant
analysis for all 16 GSE161529 objects (4 subtypes × 4 compartments).

For each object:
  1. Score stemness signatures (9 from signatures_confirmed.json)
  2. Score proliferation (Tirosh S/G2M gene lists)
  3. Assign cell cycle phase (G1/S/G2M)
  4. Score immunogenicity (APM consensus)
  5. Load CytoTRACE2 scores from CSV

Hexbins (style matching reference figure):
  - gridsize=90, p0.5-p99.5 + 20% padding, linear scale, RdYlBu_r
  - White background with grid, no LogNorm
  - All cells + per subtype as columns (for global objects)
  6 combinations:
  1. Stemness × Immunogenicity
  2. Proliferation × Immunogenicity
  3. Stemness × Proliferation
  4. Stemness × CytoTRACE2
  5. CytoTRACE2 × Immunogenicity
  6. CytoTRACE2 × Proliferation

Cell cycle quadrant analysis (per stemness signature + CytoTRACE2):
  - Stem-high:      top 33% of stemness score (within object)
  - Immune-evasive: bottom 33% of immunogenicity score (within object)
  - 4 quadrants: stem-high/low × evasive/non-evasive
  - Q2 (stem-high + evasive) = population of interest
  - G1/S/G2M distribution per quadrant × subtype

Output (flat, no subdirectories):
  figures/05_analysis/
    hexbin_[object]_[x]_vs_[y].png
    umap_[object].png
    cellcycle_[object]_[signature].png
    correlation_table.csv
    summary_r_barplot.png

Run from: ~/single_cell/
  python3 05_hexbin_correlation.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings, gc
from scipy import stats
from scipy.sparse import issparse
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
SIGFILE     = "signatures_confirmed.json"
OUTDIR      = "figures/05_analysis"
GRIDSIZE    = 90
TERTILE     = 33
SUBTYPE_COL = "subtype"
SAMPLE_COL  = "sample"
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}
PHASE_COLORS = {
    "G1" : "#4393C3",
    "S"  : "#F4A582",
    "G2M": "#D6604D",
}

# 16 objects
OBJECTS = {
    "All_Global"       : ("all_global.h5ad",
                          "cytotrace2_scores_All_Global.csv"),
    "All_ER"           : ("all_ER.h5ad",
                          "cytotrace2_scores_All_ER.csv"),
    "All_HER2"         : ("all_HER2.h5ad",
                          "cytotrace2_scores_All_HER2.csv"),
    "All_TNBC"         : ("all_TNBC.h5ad",
                          "cytotrace2_scores_All_TNBC.csv"),
    "Epithelial_Global": ("compartment_epithelial.h5ad",
                          "cytotrace2_scores_Epithelial_Global.csv"),
    "Epithelial_ER"    : ("epithelial_ER.h5ad",
                          "cytotrace2_scores_Epithelial_ER.csv"),
    "Epithelial_HER2"  : ("epithelial_HER2.h5ad",
                          "cytotrace2_scores_Epithelial_HER2.csv"),
    "Epithelial_TNBC"  : ("epithelial_TNBC.h5ad",
                          "cytotrace2_scores_Epithelial_TNBC.csv"),
    "Immune_Global"    : ("compartment_immune.h5ad",
                          "cytotrace2_scores_Immune_Global.csv"),
    "Immune_ER"        : ("immune_ER.h5ad",
                          "cytotrace2_scores_Immune_ER.csv"),
    "Immune_HER2"      : ("immune_HER2.h5ad",
                          "cytotrace2_scores_Immune_HER2.csv"),
    "Immune_TNBC"      : ("immune_TNBC.h5ad",
                          "cytotrace2_scores_Immune_TNBC.csv"),
    "Stromal_Global"   : ("compartment_stromal.h5ad",
                          "cytotrace2_scores_Stromal_Global.csv"),
    "Stromal_ER"       : ("stromal_ER.h5ad",
                          "cytotrace2_scores_Stromal_ER.csv"),
    "Stromal_HER2"     : ("stromal_HER2.h5ad",
                          "cytotrace2_scores_Stromal_HER2.csv"),
    "Stromal_TNBC"     : ("stromal_TNBC.h5ad",
                          "cytotrace2_scores_Stromal_TNBC.csv"),
}

STEMNESS_KEYS = [
    "ESC_ASSOU", "ESC_WONG", "BENPORATH_ES1", "stem_consensus",
    "fMaSC_over_aMaSC", "LIM_MAMMARY_STEM_CELL_UP",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP", "LIM_MAMMARY_LUMINAL_MATURE_UP",
    "EMP"
]

IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B"
]

S_GENES = [
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP",
    "RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2",
    "RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
    "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"
]
G2M_GENES = [
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
    "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
    "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
    "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1",
    "ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def score_and_znorm(adata, gene_list, col, min_pct=0.01):
    present = [g for g in gene_list if g in adata.var_names]
    if len(present) < 3:
        return False
    x     = adata[:, present].X
    n_exp = np.asarray((x > 0).sum(axis=1)).flatten()
    if (n_exp > 0).mean() < min_pct:
        return False
    sc.tl.score_genes(adata, present, score_name=col, use_raw=False)
    raw = adata.obs[col].values.astype(float)
    adata.obs[f"{col}_z"] = (raw - raw.mean()) / (raw.std() + 1e-8)
    return True


def assign_phase(s_score, g2m_score, threshold=0.0):
    """Winner-takes-all phase assignment from Tirosh scores."""
    phases = []
    for s, g in zip(s_score, g2m_score):
        if s > threshold and s >= g:
            phases.append("S")
        elif g > threshold and g > s:
            phases.append("G2M")
        else:
            phases.append("G1")
    return np.array(phases)


def hexbin_panel(ax, x, y, title, xlabel, ylabel, gridsize=90):
    """Hexbin matching reference figure style."""
    mask = ~(np.isnan(x) | np.isnan(y))
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


def make_hexbin_figure(obs, obj_name, x_col, y_col,
                       x_label, y_label, fname):
    """All cells + per subtype columns. Flat output."""
    if x_col not in obs.columns or y_col not in obs.columns:
        return
    x_all = obs[x_col].values.astype(float)
    y_all = obs[y_col].values.astype(float)
    mask  = ~(np.isnan(x_all) | np.isnan(y_all))
    if mask.sum() < 20:
        return

    subtypes_here = [st for st in ["ER","HER2","TNBC"]
                     if SUBTYPE_COL in obs.columns
                     and st in obs[SUBTYPE_COL].values]

    n_cols = 1 + len(subtypes_here)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5*n_cols, 5.5))
    axes = np.array(axes).flatten()
    fig.suptitle(f"{obj_name} — {x_label} vs {y_label}",
                 fontsize=10, fontweight="bold")

    hb = hexbin_panel(axes[0], x_all, y_all,
                      f"All (n={mask.sum():,})",
                      x_label, y_label)
    if hb:
        plt.colorbar(hb, ax=axes[0], shrink=0.8, label="Cell count")

    for i, subtype in enumerate(subtypes_here):
        ax      = axes[i+1]
        st_mask = (obs[SUBTYPE_COL] == subtype).values
        hb = hexbin_panel(ax,
                          x_all[st_mask], y_all[st_mask],
                          f"{subtype} (n={st_mask.sum():,})",
                          x_label, "")
        if hb:
            plt.colorbar(hb, ax=ax, shrink=0.8, label="Cell count")
        ax.set_title(f"{subtype} (n={st_mask.sum():,})",
                     fontsize=9, fontweight="bold",
                     color=SUBTYPE_COLORS.get(subtype, "black"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()


def quadrant_cellcycle_figure(obs, obj_name, stem_col, stem_label,
                               immuno_col, phase_col, tertile=33, fname=None):
    """
    Barplot of G1/S/G2M distribution across 4 quadrants
    defined by stem × immunogenicity tertiles.
    Q2 (stem-high + immune-evasive) is the population of interest.
    """
    required = [stem_col, immuno_col, phase_col]
    if any(c not in obs.columns for c in required):
        return

    stem_hi_thr   = np.percentile(obs[stem_col].dropna(),  100 - tertile)
    immuno_lo_thr = np.percentile(obs[immuno_col].dropna(), tertile)

    stem_high = obs[stem_col]   >= stem_hi_thr
    evasive   = obs[immuno_col] <= immuno_lo_thr

    obs = obs.copy()
    obs["quadrant"] = "Q4: stem-low / non-evasive"
    obs.loc[ stem_high & ~evasive, "quadrant"] = "Q1: stem-high / non-evasive"
    obs.loc[ stem_high &  evasive, "quadrant"] = "Q2: stem-high / evasive ★"
    obs.loc[~stem_high &  evasive, "quadrant"] = "Q3: stem-low / evasive"

    QUADRANT_ORDER = [
        "Q1: stem-high / non-evasive",
        "Q2: stem-high / evasive ★",
        "Q3: stem-low / evasive",
        "Q4: stem-low / non-evasive",
    ]

    subtypes_here = [st for st in ["ER","HER2","TNBC"]
                     if SUBTYPE_COL in obs.columns
                     and st in obs[SUBTYPE_COL].values]
    if not subtypes_here:
        subtypes_here = ["All"]
        obs["_st"] = "All"
        st_col = "_st"
    else:
        st_col = SUBTYPE_COL

    n_cols = len(subtypes_here)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5.5), sharey=True)
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"{obj_name} — Cell cycle by quadrant\n"
        f"Stem: {stem_label} (top {tertile}%) × "
        f"Immune-evasive (bottom {tertile}% APM)",
        fontsize=10, fontweight="bold"
    )

    for ax, subtype in zip(axes, subtypes_here):
        sub = obs if subtype == "All" else obs[obs[st_col] == subtype]
        if len(sub) < 30:
            ax.set_visible(False); continue

        rows = []
        n_labels = {}
        for q in QUADRANT_ORDER:
            q_cells = sub[sub["quadrant"] == q]
            n_q = len(q_cells)
            n_labels[q] = n_q
            if n_q == 0:
                continue
            for phase in ["G1","S","G2M"]:
                frac = (q_cells[phase_col] == phase).sum() / n_q * 100
                rows.append({"quadrant": q, "phase": phase,
                             "pct": frac, "n": n_q})

        if not rows:
            ax.set_visible(False); continue

        df_q  = pd.DataFrame(rows)
        pivot = df_q.pivot_table(
            index="quadrant", columns="phase",
            values="pct", fill_value=0
        ).reindex([q for q in QUADRANT_ORDER
                   if q in df_q["quadrant"].values])

        bottom = np.zeros(len(pivot))
        for phase in ["G1","S","G2M"]:
            if phase not in pivot.columns:
                continue
            vals = pivot[phase].values
            ax.barh(range(len(pivot)), vals, left=bottom,
                    color=PHASE_COLORS[phase], label=phase,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
            bottom += vals

        # Y tick labels with n + Q2 highlighted
        ylabels_data = [(f"{q}\n(n={n_labels.get(q,0):,})",
                         "#D6604D" if "★" in q else "black",
                         "bold"    if "★" in q else "normal")
                        for q in pivot.index]
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([l for l,c,w in ylabels_data], fontsize=8)
        for tick, (l,c,w) in zip(ax.get_yticklabels(), ylabels_data):
            tick.set_color(c); tick.set_fontweight(w)

        # Highlight Q2 row background
        for i, q in enumerate(pivot.index):
            if "★" in q:
                ax.axhspan(i-0.4, i+0.4, color="#FFF0EE",
                           alpha=0.5, zorder=0)

        ax.set_xlim(0, 100)
        ax.set_xlabel("% cells", fontsize=9)
        ax.set_title(subtype, fontsize=10, fontweight="bold",
                     color=SUBTYPE_COLORS.get(subtype, "black"))
        ax.axvline(50, color="gray", ls="--", lw=0.8, alpha=0.5)

        if ax == axes[0]:
            ax.legend(handles=[
                mpatches.Patch(color=PHASE_COLORS[p], label=p)
                for p in ["G1","S","G2M"]
            ], fontsize=8, loc="lower right")

    plt.tight_layout()
    if fname:
        plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
with open(SIGFILE) as f:
    SIGNATURES = json.load(f)

all_corr_rows = []

for obj_name, (h5ad_file, ct2_csv) in OBJECTS.items():

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

    # ── Score stemness ────────────────────────────────────────────────────
    stem_z_cols = []
    for key in STEMNESS_KEYS:
        if key not in SIGNATURES:
            continue
        col = f"score_{key}"
        if score_and_znorm(adata, SIGNATURES[key], col):
            stem_z_cols.append((key, f"{col}_z"))

    if stem_z_cols:
        z_vals = adata.obs[[c for _,c in stem_z_cols]].values
        adata.obs["score_stemness_composite_z"] = np.nanmean(z_vals, axis=1)
    print(f"  ✓ {len(stem_z_cols)} stemness signatures")

    # ── Score proliferation + assign phase ────────────────────────────────
    score_and_znorm(adata, S_GENES,   "score_S_phase")
    score_and_znorm(adata, G2M_GENES, "score_G2M_phase")

    if "score_S_phase" in adata.obs and "score_G2M_phase" in adata.obs:
        s   = adata.obs["score_S_phase"].values.astype(float)
        g2m = adata.obs["score_G2M_phase"].values.astype(float)
        adata.obs["phase_assigned"] = assign_phase(s, g2m)
        prolif = (s + g2m) / 2
        adata.obs["score_proliferation_z"] = \
            (prolif - prolif.mean()) / (prolif.std() + 1e-8)
        phase_counts = dict(
            zip(*np.unique(adata.obs["phase_assigned"], return_counts=True))
        )
        print(f"  ✓ Phase: {phase_counts}")
    else:
        print(f"  ⚠️  Proliferation scoring failed")

    phase_col = "phase_assigned" \
        if "phase_assigned" in adata.obs.columns else None

    # ── Score immunogenicity ──────────────────────────────────────────────
    score_and_znorm(adata, IMMUNOGENIC, "score_immunogenic")
    print(f"  ✓ Immunogenicity scored")

    # ── Load CytoTRACE2 ───────────────────────────────────────────────────
    has_ct2 = False
    if os.path.exists(ct2_csv):
        ct2 = pd.read_csv(ct2_csv, index_col=0)
        ct2.index = ct2.index.astype(str)
        common = adata.obs_names.intersection(ct2.index)
        if common.shape[0] > 0 and \
           not ct2.loc[common, "CytoTRACE2_Score"].isna().all():
            adata.obs["CytoTRACE2_Score"] = np.nan
            adata.obs.loc[common, "CytoTRACE2_Score"] = \
                ct2.loc[common, "CytoTRACE2_Score"].values.astype(float)
            valid = ~adata.obs["CytoTRACE2_Score"].isna()
            raw   = adata.obs.loc[valid, "CytoTRACE2_Score"].values
            adata.obs.loc[valid, "CytoTRACE2_Score_z"] = \
                (raw - raw.mean()) / (raw.std() + 1e-8)
            has_ct2 = True
            print(f"  ✓ CytoTRACE2: {valid.sum():,} cells")
        else:
            print(f"  ⚠️  CytoTRACE2 CSV empty or all NaN")
    else:
        print(f"  ⚠️  {ct2_csv} not found")

    obs = adata.obs

    # ══════════════════════════════════════════════════════════════════════
    # HEXBINS
    # ══════════════════════════════════════════════════════════════════════
    safe = obj_name.replace("/", "_")

    for key, stem_z in stem_z_cols:
        label = key.replace("_", " ").replace("LIM MAMMARY", "LIM")

        if "score_immunogenic_z" in obs.columns:
            make_hexbin_figure(obs, obj_name, stem_z,
                               "score_immunogenic_z",
                               f"{label} (z)", "Immunogenic (z)",
                               f"hexbin_{safe}_{key}_vs_immuno")

        if "score_proliferation_z" in obs.columns:
            make_hexbin_figure(obs, obj_name, stem_z,
                               "score_proliferation_z",
                               f"{label} (z)", "Proliferation (z)",
                               f"hexbin_{safe}_{key}_vs_prolif")

        if has_ct2 and "CytoTRACE2_Score_z" in obs.columns:
            make_hexbin_figure(obs, obj_name, stem_z,
                               "CytoTRACE2_Score_z",
                               f"{label} (z)", "CytoTRACE2 (z)",
                               f"hexbin_{safe}_{key}_vs_ct2")

    if "score_proliferation_z" in obs.columns and \
       "score_immunogenic_z" in obs.columns:
        make_hexbin_figure(obs, obj_name,
                           "score_proliferation_z",
                           "score_immunogenic_z",
                           "Proliferation (z)", "Immunogenic (z)",
                           f"hexbin_{safe}_prolif_vs_immuno")

    if has_ct2 and "score_immunogenic_z" in obs.columns:
        make_hexbin_figure(obs, obj_name,
                           "CytoTRACE2_Score_z",
                           "score_immunogenic_z",
                           "CytoTRACE2 (z)", "Immunogenic (z)",
                           f"hexbin_{safe}_ct2_vs_immuno")

    if has_ct2 and "score_proliferation_z" in obs.columns:
        make_hexbin_figure(obs, obj_name,
                           "CytoTRACE2_Score_z",
                           "score_proliferation_z",
                           "CytoTRACE2 (z)", "Proliferation (z)",
                           f"hexbin_{safe}_ct2_vs_prolif")

    print(f"  ✓ Hexbins done")

    # ══════════════════════════════════════════════════════════════════════
    # UMAP
    # ══════════════════════════════════════════════════════════════════════
    if "X_umap" in adata.obsm:
        coords    = adata.obsm["X_umap"]
        umap_cols = (
            [c for _, c in stem_z_cols] +
            [c for c in ["score_proliferation_z",
                         "score_immunogenic_z",
                         "score_stemness_composite_z",
                         "CytoTRACE2_Score_z"]
             if c in obs.columns]
        )
        nc = 4; nr = (len(umap_cols) + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
        axes = np.array(axes).flatten()
        fig.suptitle(f"UMAP scores — {obj_name}",
                     fontsize=11, fontweight="bold")

        for i, col in enumerate(umap_cols):
            ax   = axes[i]
            vals = obs[col].values.astype(float)
            mask = ~np.isnan(vals)
            vabs = max(abs(np.nanpercentile(vals, 2)),
                       abs(np.nanpercentile(vals, 98)))
            sc_p = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=vals[mask], s=0.4, alpha=0.3,
                cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                rasterized=True
            )
            plt.colorbar(sc_p, ax=ax, shrink=0.7)
            ax.set_title(col.replace("score_", "").replace("_z", ""),
                         fontsize=8, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

        for ax in axes[len(umap_cols):]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/umap_{safe}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ UMAP saved")

    # ══════════════════════════════════════════════════════════════════════
    # CELL CYCLE QUADRANT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    if phase_col and "score_immunogenic_z" in obs.columns:

        for key, stem_z in stem_z_cols:
            label = key.replace("_", " ").replace("LIM MAMMARY", "LIM")
            quadrant_cellcycle_figure(
                obs, obj_name,
                stem_col=stem_z,
                stem_label=label,
                immuno_col="score_immunogenic_z",
                phase_col=phase_col,
                tertile=TERTILE,
                fname=f"cellcycle_{safe}_{key}"
            )

        if has_ct2 and "CytoTRACE2_Score_z" in obs.columns:
            quadrant_cellcycle_figure(
                obs, obj_name,
                stem_col="CytoTRACE2_Score_z",
                stem_label="CytoTRACE2",
                immuno_col="score_immunogenic_z",
                phase_col=phase_col,
                tertile=TERTILE,
                fname=f"cellcycle_{safe}_CytoTRACE2"
            )

        print(f"  ✓ Cell cycle quadrant figures done")

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATIONS
    # ══════════════════════════════════════════════════════════════════════
    x_cols = [c for _, c in stem_z_cols] + \
             [c for c in ["score_proliferation_z", "CytoTRACE2_Score_z"]
              if c in obs.columns]
    y_cols = [c for c in ["score_immunogenic_z", "score_proliferation_z",
                           "CytoTRACE2_Score_z", "score_stemness_composite_z"]
              if c in obs.columns]

    for xc in x_cols:
        for yc in y_cols:
            if xc == yc:
                continue
            x = obs[xc].values.astype(float)
            y = obs[yc].values.astype(float)
            mv = ~(np.isnan(x) | np.isnan(y))
            if mv.sum() < 30:
                continue
            r, p = stats.pearsonr(x[mv], y[mv])
            all_corr_rows.append({
                "Object": obj_name,
                "X": xc.replace("score_", "").replace("_z", ""),
                "Y": yc.replace("score_", "").replace("_z", ""),
                "r": round(r, 4), "p": p, "n": int(mv.sum())
            })

    del adata; gc.collect()
    print(f"  ✓ {obj_name} done")


# ── Save correlations ─────────────────────────────────────────────────────────
if all_corr_rows:
    df_corr = pd.DataFrame(all_corr_rows)
    df_corr.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)
    print(f"\n  ✓ {OUTDIR}/correlation_table.csv ({len(df_corr):,} rows)")

    # Summary barplot
    pivot_pairs = [
        ("stemness composite", "immunogenic",   "Stemness vs Immunogenicity"),
        ("stemness composite", "proliferation", "Stemness vs Proliferation"),
        ("CytoTRACE2 Score",   "proliferation", "CytoTRACE2 vs Proliferation"),
        ("CytoTRACE2 Score",   "immunogenic",   "CytoTRACE2 vs Immunogenicity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle("Pearson r — key pairs (GSE161529)",
                 fontsize=13, fontweight="bold")

    for ax, (xl, yl, title) in zip(axes, pivot_pairs):
        sub = df_corr[
            df_corr["X"].str.contains(xl.split()[0], case=False) &
            df_corr["Y"].str.contains(yl.split()[0], case=False)
        ][["Object", "r"]].drop_duplicates("Object")
        if sub.empty:
            ax.set_visible(False); continue
        sub   = sub.set_index("Object")
        order = [o for o in OBJECTS.keys() if o in sub.index]
        sub   = sub.loc[order]
        colors = ["#C41E3A" if v > 0 else "#1565C0" for v in sub["r"]]
        ax.barh(range(len(sub)), sub["r"].values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=8)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Pearson r")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(-1, 1)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/summary_r_barplot.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/summary_r_barplot.png")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  figures/05_analysis/ (flat structure)
    hexbin_[object]_[x]_vs_[y].png
    umap_[object].png
    cellcycle_[object]_[signature].png
    correlation_table.csv
    summary_r_barplot.png
══════════════════════════════════════════════════════════════
""")
