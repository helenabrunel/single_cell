"""
08_stem_cc_corrected.py
=======================
Stemness × immunogenicity analysis with cell-cycle correction.

Rationale
---------
In scRNA-seq, stemness signatures (especially ESC-like) are substantially
correlated with cell cycle scores (r=0.3–0.8 depending on compartment/subtype)
because sc.tl.score_genes() operates on raw log-normalized expression (layers
["logcounts"]) with no prior regression. This is not a pipeline artifact —
sc.pp.scale(vars.to.regress=...) only affects X_scaled used for PCA/UMAP,
not the expression used for gene scoring.

Correction approach
-------------------
For each stemness signature and each object independently:
    stem_corrected = stem_raw - LinearRegression(S_score, G2M_score).predict()

The residual captures stemness variance NOT explained by cell cycle phase.
This is equivalent to partial correlation (stem ~ immune | S_score, G2M_score)
but produces a per-cell score usable in hexbins, UMAPs and quadrant analysis.

Regression is done BEFORE z-normalisation, per object, using raw (non-z)
scores. The regressors are score_S_phase and score_G2M_phase (raw, not z),
which are already computed by sc.tl.score_genes() in the same way.

Outputs (flat structure in figures/08_cc_corrected/)
-----------------------------------------------------
For each object:
  hexbin_[object]_[sig]_corrected_vs_immuno.png   — corrected stem × immune
  hexbin_[object]_[sig]_raw_vs_immuno.png         — original stem × immune
  (both panels side-by-side for direct comparison)

Summary:
  correlation_table_corrected.csv   — r values before/after for all objects
  summary_r_comparison.png          — before vs after barplot

Run from: ~/single_cell/
  python3 08_stem_cc_corrected.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings, gc
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
SIGFILE  = "signatures_confirmed.json"
OUTDIR   = "figures/08_cc_corrected"
GRIDSIZE = 90
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}

# Same 16 objects as script 05
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

def score_genes_raw(adata, gene_list, col, min_pct=0.01):
    """Score genes → raw (non-z) score in adata.obs[col]. Returns bool."""
    present = [g for g in gene_list if g in adata.var_names]
    if len(present) < 3:
        return False
    from scipy.sparse import issparse
    x     = adata[:, present].X
    n_exp = np.asarray((x > 0).sum(axis=1)).flatten()
    if (n_exp > 0).mean() < min_pct:
        return False
    sc.tl.score_genes(adata, present, score_name=col, use_raw=False)
    return True


def cc_regress(stem_raw, s_raw, g2m_raw):
    """
    Regress stem_raw ~ S_raw + G2M_raw per-object.
    Returns residuals (same scale as stem_raw).
    Both regressors and target are raw (non-z) scores.
    """
    X_cc  = np.column_stack([s_raw, g2m_raw])
    valid = ~(np.isnan(stem_raw) | np.isnan(s_raw) | np.isnan(g2m_raw))
    residuals = stem_raw.copy()
    if valid.sum() > 10:
        reg = LinearRegression().fit(X_cc[valid], stem_raw[valid])
        residuals[valid] = stem_raw[valid] - reg.predict(X_cc[valid])
        r2 = reg.score(X_cc[valid], stem_raw[valid])
    else:
        r2 = np.nan
    return residuals, r2


def znorm(arr):
    """Z-normalise a 1-D array, ignoring NaNs."""
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    return (arr - mu) / (sd + 1e-8)


def hexbin_panel(ax, x, y, title, xlabel, ylabel, gridsize=90):
    """Hexbin matching script-05 style."""
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
    return hb, r


def assign_phase(s_raw, g2m_raw, threshold=0.0):
    """Winner-takes-all phase assignment (same as script 05)."""
    phases = []
    for s, g in zip(s_raw, g2m_raw):
        if s > threshold and s >= g:
            phases.append("S")
        elif g > threshold and g > s:
            phases.append("G2M")
        else:
            phases.append("G1")
    return np.array(phases)


PHASE_COLORS = {"G1": "#4393C3", "S": "#F4A582", "G2M": "#D6604D"}


def validation_figure(adata_obsm, obs, obj_name, key,
                      raw_z_col, cor_z_col, prolif_z_col, fname):
    """
    Validation grid for one signature × object.

    Layout (3 rows × 3 cols):

    Row 0 — UMAP coloured by: raw stem | corrected stem | phase
    Row 1 — Hexbin: raw × prolif | corrected × prolif | raw × corrected
    Row 2 — Hexbin: raw × immuno | corrected × immuno | phase violin

    Expectations if correction worked:
      - Row 0 col 1 (corrected UMAP): phase clusters should disappear
      - Row 1 col 0: r(raw × prolif) moderate/high
      - Row 1 col 1: r(corrected × prolif) ≈ 0  ← key check
      - Row 1 col 2: r(raw × corrected) high     ← stem signal preserved
      - Row 2: corrected × immuno vs raw × immuno (did r change?)
    """
    has_umap = "X_umap" in adata_obsm
    phase    = obs["phase_assigned"].values \
               if "phase_assigned" in obs.columns else None
    raw_z    = obs[raw_z_col].values.astype(float)
    cor_z    = obs[cor_z_col].values.astype(float)
    pro_z    = obs[prolif_z_col].values.astype(float)
    imm_z    = obs["score_immunogenic_z"].values.astype(float)

    sig_label = key.replace("_", " ").replace("LIM MAMMARY", "LIM")
    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle(
        f"Validation: cell-cycle correction — {obj_name} — {sig_label}",
        fontsize=11, fontweight="bold"
    )

    # ── Row 0: UMAPs ──────────────────────────────────────────────────────
    if has_umap:
        coords = adata_obsm["X_umap"]
        for col_idx, (vals, title, cmap, vtype) in enumerate([
            (raw_z,  f"{sig_label} RAW",       "RdBu_r", "cont"),
            (cor_z,  f"{sig_label} CORRECTED",  "RdBu_r", "cont"),
            (phase,  "Cell cycle phase",        None,     "cat"),
        ]):
            ax = axes[0, col_idx]
            mask = ~np.isnan(vals.astype(float)) \
                   if vtype == "cont" else np.ones(len(coords), dtype=bool)
            if vtype == "cont":
                vabs = max(abs(np.nanpercentile(vals, 2)),
                           abs(np.nanpercentile(vals, 98)))
                sc_p = ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=vals[mask], s=0.4, alpha=0.3,
                    cmap=cmap, vmin=-vabs, vmax=vabs, rasterized=True
                )
                plt.colorbar(sc_p, ax=ax, shrink=0.7)
            else:
                if phase is not None:
                    for ph, color in PHASE_COLORS.items():
                        m = phase == ph
                        ax.scatter(
                            coords[m, 0], coords[m, 1],
                            c=color, s=0.4, alpha=0.3,
                            label=ph, rasterized=True
                        )
                    ax.legend(markerscale=5, fontsize=8, loc="upper right")
                else:
                    ax.text(0.5, 0.5, "phase N/A",
                            transform=ax.transAxes, ha="center")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    else:
        for col_idx in range(3):
            axes[0, col_idx].text(0.5, 0.5, "No UMAP embedding",
                                  transform=axes[0, col_idx].transAxes,
                                  ha="center", va="center", fontsize=9)
            axes[0, col_idx].set_visible(True)

    # ── Row 1: stem × prolif validation ───────────────────────────────────
    pairs_r1 = [
        (raw_z, pro_z, f"{sig_label} raw", "Proliferation (z)",
         "r(raw × prolif)"),
        (cor_z, pro_z, f"{sig_label} corrected", "Proliferation (z)",
         "r(corrected × prolif)  ← should be ≈0"),
        (raw_z, cor_z, f"{sig_label} raw", f"{sig_label} corrected",
         "r(raw × corrected)  ← should be high"),
    ]
    for col_idx, (x, y, xl, yl, title) in enumerate(pairs_r1):
        res = hexbin_panel(axes[1, col_idx], x, y,
                           title=title, xlabel=xl, ylabel=yl)
        if res:
            hb, _ = res
            plt.colorbar(hb, ax=axes[1, col_idx], shrink=0.7,
                         label="Cell count")

    # ── Row 2: stem × immuno + phase violin ───────────────────────────────
    pairs_r2 = [
        (raw_z, imm_z, f"{sig_label} raw",       "Immunogenic (z)",
         "raw × immuno"),
        (cor_z, imm_z, f"{sig_label} corrected",  "Immunogenic (z)",
         "corrected × immuno  ← main result"),
    ]
    for col_idx, (x, y, xl, yl, title) in enumerate(pairs_r2):
        res = hexbin_panel(axes[2, col_idx], x, y,
                           title=title, xlabel=xl, ylabel=yl)
        if res:
            hb, _ = res
            plt.colorbar(hb, ax=axes[2, col_idx], shrink=0.7,
                         label="Cell count")

    # Violin: corrected stem score per phase ← phase should not predict it
    ax = axes[2, 2]
    if phase is not None:
        data_by_phase = [cor_z[phase == ph] for ph in ["G1", "S", "G2M"]]
        data_by_phase = [d[~np.isnan(d)] for d in data_by_phase]
        parts = ax.violinplot(data_by_phase, positions=[0, 1, 2],
                              showmedians=True, showextrema=False)
        for pc, ph in zip(parts["bodies"], ["G1", "S", "G2M"]):
            pc.set_facecolor(PHASE_COLORS[ph])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["G1", "S", "G2M"])
        ax.set_ylabel(f"{sig_label} corrected (z)", fontsize=8)
        ax.set_title(
            "Corrected score by phase\n(flat = CC removed successfully)",
            fontsize=9, fontweight="bold"
        )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        # Annotate with Kruskal-Wallis p
        from scipy.stats import kruskal
        valid_groups = [d for d in data_by_phase if len(d) > 5]
        if len(valid_groups) >= 2:
            stat, p_kw = kruskal(*valid_groups)
            pstr = f"KW p={p_kw:.2e}" if p_kw >= 1e-300 else "KW p<1e-300"
            ax.text(0.97, 0.97, pstr, transform=ax.transAxes,
                    fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "phase N/A",
                transform=ax.transAxes, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()


def comparison_hexbin(obs, obj_name, raw_col, cor_col, immuno_col,
                      sig_label, fname):
    """
    Side-by-side comparison: raw stem × immune  vs  corrected stem × immune.
    One row = All cells. If global object, additional rows per subtype.
    """
    subtypes = [st for st in ["ER", "HER2", "TNBC"]
                if "subtype" in obs.columns and st in obs["subtype"].values]
    groups   = [("All", np.ones(len(obs), dtype=bool))] + \
               [(st, (obs["subtype"] == st).values) for st in subtypes]

    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(11, 5.2 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"{obj_name}  —  {sig_label}\n"
        f"Left: raw score × immunogenicity   |   "
        f"Right: CC-corrected score × immunogenicity",
        fontsize=10, fontweight="bold"
    )

    raw_vals = obs[raw_col].values.astype(float)
    cor_vals = obs[cor_col].values.astype(float)
    imm_vals = obs[immuno_col].values.astype(float)

    for row_idx, (label, mask) in enumerate(groups):
        n = mask.sum()
        color = SUBTYPE_COLORS.get(label, "black")

        res_raw = hexbin_panel(
            axes[row_idx, 0],
            raw_vals[mask], imm_vals[mask],
            title=f"{label} (n={n:,})  RAW",
            xlabel=f"{sig_label} raw (z)",
            ylabel="Immunogenic (z)"
        )
        res_cor = hexbin_panel(
            axes[row_idx, 1],
            cor_vals[mask], imm_vals[mask],
            title=f"{label} (n={n:,})  CC-CORRECTED",
            xlabel=f"{sig_label} corrected (z)",
            ylabel=""
        )

        for ax_obj, result in [(axes[row_idx, 0], res_raw),
                               (axes[row_idx, 1], res_cor)]:
            if result is not None:
                hb, _ = result
                plt.colorbar(hb, ax=ax_obj, shrink=0.8, label="Cell count")

        # Subtype colour stripe on y-axis
        if label != "All":
            for col_idx in range(2):
                axes[row_idx, col_idx].set_ylabel(
                    label, color=color, fontsize=10, fontweight="bold"
                )

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
with open(SIGFILE) as f:
    SIGNATURES = json.load(f)

all_corr_rows = []

print("\n" + "="*65)
print(" 08 — STEMNESS × IMMUNE  with  CELL-CYCLE CORRECTION")
print("="*65)
print("  Method : LinearRegression(S_score, G2M_score) residuals")
print("  Scope  : per-object regression (not global)")
print(f"  Objects: {len(OBJECTS)}")

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

    # ── 1. Score cell cycle (raw, no z-norm) ─────────────────────────────
    ok_s   = score_genes_raw(adata, S_GENES,   "score_S_phase_raw")
    ok_g2m = score_genes_raw(adata, G2M_GENES, "score_G2M_phase_raw")

    if not (ok_s and ok_g2m):
        print("  ⚠️  Cell cycle scoring failed — skipping")
        del adata; gc.collect()
        continue

    s_raw   = adata.obs["score_S_phase_raw"].values.astype(float)
    g2m_raw = adata.obs["score_G2M_phase_raw"].values.astype(float)
    print(f"  ✓ S_score   range: [{s_raw.min():.3f}, {s_raw.max():.3f}]")
    print(f"  ✓ G2M_score range: [{g2m_raw.min():.3f}, {g2m_raw.max():.3f}]")

    # ── 2. Score immunogenicity ───────────────────────────────────────────
    ok_imm = score_genes_raw(adata, IMMUNOGENIC, "score_immunogenic_raw")
    if not ok_imm:
        print("  ⚠️  Immunogenicity scoring failed — skipping")
        del adata; gc.collect()
        continue
    imm_raw = adata.obs["score_immunogenic_raw"].values.astype(float)
    adata.obs["score_immunogenic_z"] = znorm(imm_raw)
    print("  ✓ Immunogenicity scored")

    # ── 3. Score stemness + regress + z-norm ──────────────────────────────
    stem_results = []   # (key, raw_z_col, cor_z_col, r2)

    for key in STEMNESS_KEYS:
        if key not in SIGNATURES:
            continue
        col_raw = f"score_{key}_raw"
        col_cor = f"score_{key}_corrected_raw"
        col_raw_z = f"score_{key}_raw_z"
        col_cor_z = f"score_{key}_corrected_z"

        if not score_genes_raw(adata, SIGNATURES[key], col_raw):
            continue

        stem_raw_vals = adata.obs[col_raw].values.astype(float)

        # Regress out S + G2M
        residuals, r2 = cc_regress(stem_raw_vals, s_raw, g2m_raw)
        adata.obs[col_cor] = residuals

        # Z-norm both
        adata.obs[col_raw_z] = znorm(stem_raw_vals)
        adata.obs[col_cor_z] = znorm(residuals)

        stem_results.append((key, col_raw_z, col_cor_z, r2))
        print(f"  ✓ {key:40s}  R²(cc)={r2:.3f}")

    if not stem_results:
        print("  ⚠️  No stemness signatures scored — skipping")
        del adata; gc.collect()
        continue

    # ── 4. Assign phase (needed for validation violin) ────────────────────
    adata.obs["phase_assigned"] = assign_phase(s_raw, g2m_raw)
    prolif_raw = (s_raw + g2m_raw) / 2
    adata.obs["score_proliferation_z"] = znorm(prolif_raw)
    phase_counts = dict(
        zip(*np.unique(adata.obs["phase_assigned"], return_counts=True))
    )
    print(f"  ✓ Phase: {phase_counts}")

    obs  = adata.obs
    safe = obj_name.replace("/", "_")

    # ── 5. Validation figures (Global + TNBC objects only) ────────────────
    # One 3×3 grid per signature showing: UMAP, stem×prolif, stem×immuno,
    # and phase violin. Key check: r(corrected × prolif) ≈ 0.
    if obj_name.endswith("_Global") or obj_name.endswith("_TNBC"):
        for key, raw_z, cor_z, r2 in stem_results:
            validation_figure(
                adata_obsm=adata.obsm,
                obs=obs,
                obj_name=obj_name,
                key=key,
                raw_z_col=raw_z,
                cor_z_col=cor_z,
                prolif_z_col="score_proliferation_z",
                fname=f"validation_{safe}_{key}"
            )
        print(f"  ✓ Validation figures saved")

    # ── 6. Comparison hexbin figures (all objects) ────────────────────────
    for key, raw_z, cor_z, r2 in stem_results:
        label = key.replace("_", " ").replace("LIM MAMMARY", "LIM")
        comparison_hexbin(
            obs, obj_name,
            raw_col=raw_z, cor_col=cor_z,
            immuno_col="score_immunogenic_z",
            sig_label=label,
            fname=f"hexbin_{safe}_{key}_raw_vs_corrected"
        )

    print(f"  ✓ Comparison hexbins saved")

    # ── 7. Correlations: raw vs corrected × immune ────────────────────────
    for key, raw_z, cor_z, r2 in stem_results:
        imm_z = obs["score_immunogenic_z"].values.astype(float)

        for version, col in [("raw", raw_z), ("corrected", cor_z)]:
            x = obs[col].values.astype(float)
            mv = ~(np.isnan(x) | np.isnan(imm_z))
            if mv.sum() < 30:
                continue
            r, p = stats.pearsonr(x[mv], imm_z[mv])

            # Also compute r(stem × prolif) for verification
            prolif = (s_raw + g2m_raw) / 2
            prolif_z = znorm(prolif)
            r_prolif, _ = stats.pearsonr(x[mv], prolif_z[mv])

            all_corr_rows.append({
                "Object"       : obj_name,
                "Signature"    : key,
                "Version"      : version,
                "r_stem_immune": round(r, 4),
                "r_stem_prolif": round(r_prolif, 4),
                "R2_cc"        : round(r2, 4) if not np.isnan(r2) else np.nan,
                "p"            : p,
                "n"            : int(mv.sum())
            })

    # Subtype-level correlations
    if "subtype" in obs.columns:
        for subtype in ["ER", "HER2", "TNBC"]:
            st_mask = (obs["subtype"] == subtype).values
            if st_mask.sum() < 30:
                continue
            imm_z = obs["score_immunogenic_z"].values.astype(float)
            prolif_z = znorm((s_raw + g2m_raw) / 2)

            for key, raw_z, cor_z, r2 in stem_results:
                for version, col in [("raw", raw_z), ("corrected", cor_z)]:
                    x = obs[col].values.astype(float)
                    mv = st_mask & ~(np.isnan(x) | np.isnan(imm_z))
                    if mv.sum() < 30:
                        continue
                    r, p = stats.pearsonr(x[mv], imm_z[mv])
                    r_prolif, _ = stats.pearsonr(x[mv], prolif_z[mv])
                    all_corr_rows.append({
                        "Object"       : f"{obj_name}_{subtype}",
                        "Signature"    : key,
                        "Version"      : version,
                        "r_stem_immune": round(r, 4),
                        "r_stem_prolif": round(r_prolif, 4),
                        "R2_cc"        : round(r2, 4) if not np.isnan(r2) else np.nan,
                        "p"            : p,
                        "n"            : int(mv.sum())
                    })

    del adata; gc.collect()
    print(f"  ✓ {obj_name} done")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES + FIGURES
# ══════════════════════════════════════════════════════════════════════════════
if not all_corr_rows:
    print("\n  ⚠️  No correlation rows — check that .h5ad files are present")
else:
    df = pd.DataFrame(all_corr_rows)
    out_csv = f"{OUTDIR}/correlation_table_corrected.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  ✓ {out_csv}  ({len(df):,} rows)")

    # ── Summary figure: raw vs corrected r(stem×immune) per signature ─────
    # Focus on Epithelial objects (most interpretable for stemness)
    epi_objs = [o for o in OBJECTS if o.startswith("Epithelial")]

    df_epi = df[df["Object"].isin(epi_objs)].copy()
    if df_epi.empty:
        print("  ⚠️  No Epithelial objects in results — skipping summary figure")
    else:
        pivot = df_epi.groupby(["Signature", "Version"])[
            ["r_stem_immune", "r_stem_prolif"]
        ].mean().reset_index()

        sigs_ordered = (
            pivot[pivot["Version"] == "corrected"]
            .sort_values("r_stem_immune")["Signature"].tolist()
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "Effect of cell-cycle correction on stemness × immunogenicity\n"
            "(mean across Epithelial objects: Global + ER + HER2 + TNBC)",
            fontsize=11, fontweight="bold"
        )

        x     = np.arange(len(sigs_ordered))
        width = 0.38

        for ax, metric, title in [
            (axes[0], "r_stem_immune",
             "r(stem × immunogenicity)\nHigher = more immune signal"),
            (axes[1], "r_stem_prolif",
             "r(stem × proliferation)\nLower |r| = less CC contamination"),
        ]:
            raw_vals = [
                pivot.loc[(pivot["Signature"] == s) &
                          (pivot["Version"] == "raw"), metric].values
                for s in sigs_ordered
            ]
            cor_vals = [
                pivot.loc[(pivot["Signature"] == s) &
                          (pivot["Version"] == "corrected"), metric].values
                for s in sigs_ordered
            ]
            raw_vals = [v[0] if len(v) else np.nan for v in raw_vals]
            cor_vals = [v[0] if len(v) else np.nan for v in cor_vals]

            ax.barh(x - width/2, raw_vals, width,
                    label="raw", color="#4C72B0", alpha=0.8)
            ax.barh(x + width/2, cor_vals, width,
                    label="CC-corrected", color="#DD8452", alpha=0.8)
            ax.set_yticks(x)
            ax.set_yticklabels(
                [s.replace("LIM_MAMMARY_", "LIM_").replace("_", " ")
                 for s in sigs_ordered],
                fontsize=9
            )
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("Pearson r", fontsize=10)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        out_fig = f"{OUTDIR}/summary_r_comparison.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out_fig}")

    # ── R² table: how much variance does CC explain per signature? ─────────
    print("\n  R² of cell-cycle regression per signature "
          "(mean across all objects):")
    r2_summary = (
        df[df["Version"] == "corrected"]
        .groupby("Signature")["R2_cc"]
        .mean()
        .sort_values(ascending=False)
        .round(3)
    )
    for sig, r2 in r2_summary.items():
        bar = "█" * int(r2 * 30)
        print(f"    {sig:45s}  R²={r2:.3f}  {bar}")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Output: {OUTDIR}/
    hexbin_[object]_[sig]_raw_vs_corrected.png   (side-by-side)
    correlation_table_corrected.csv
    summary_r_comparison.png

  Key question answered:
    Does correcting for cell cycle change the stem × immune
    relationship, and for which signatures / subtypes?

  Next: compare r_stem_immune raw vs corrected per subtype in
  correlation_table_corrected.csv to decide which signature
  to use for the main stem × immune analysis.
══════════════════════════════════════════════════════════════
""")
