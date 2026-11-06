"""
09_stem_immune_metrics.py
==========================
Comparative analysis of metrics to capture the stem × immune relationship.

Motivation
----------
Pearson r underestimates the stem × immune association because the pattern
is not linear — it's a triangular distribution with a depleted upper-right
quadrant (stem-high / immune-high). This script compares four metrics:

  1. Pearson r         — linear correlation (baseline)
  2. Mutual Information (MI) — any dependence, no assumption of shape
  3. Distance Correlation (dCor) — detects non-linear dependence, has p-value
  4. % quadrant depletion — % cells in stem-high/immune-high vs expected
                             under independence (fold-depletion)

Outputs
-------
  figures/09_metrics/
    metric_comparison_table.csv       — all objects × signatures × 4 metrics
    summary_heatmap_[object].png      — heatmap of metrics for key objects
    metric_scatter_[object]_[sig].png — annotated hexbin with all 4 metrics
    summary_comparison_barplot.png    — Pearson vs dCor vs MI vs %quad

Run from: ~/single_cell/Visvader/
  python3 09_stem_immune_metrics.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os, warnings, gc
from scipy import stats
from scipy.sparse import issparse
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import dcor

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
SIGFILE = "signatures_confirmed.json"
OUTDIR  = "figures/09_metrics"
os.makedirs(OUTDIR, exist_ok=True)

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

# Key objects for detailed figures (hexbin + summary)
KEY_OBJECTS = ["Epithelial_Global", "Epithelial_TNBC",
               "Immune_Global", "All_Global"]

QUADRANT_THRESHOLD = 67   # top 33% = above 67th percentile

SUBTYPE_COLORS = {
    "ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"
}


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return round(r, 4), p


def mutual_information(x, y, n_bins=20):
    """
    Discretised MI using equal-frequency binning.
    Normalised by H(X,Y) to give range [0,1].
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return np.nan
    x_m, y_m = x[mask].reshape(-1,1), y[mask].reshape(-1,1)

    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                           strategy='quantile')
    x_d = kbd.fit_transform(x_m).flatten().astype(int)
    kbd2 = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                            strategy='quantile')
    y_d = kbd2.fit_transform(y_m).flatten().astype(int)

    mi = mutual_info_score(x_d, y_d)
    # Normalise: MI / sqrt(H(X)*H(Y))
    def entropy(z):
        counts = np.bincount(z)
        p = counts[counts > 0] / counts.sum()
        return -np.sum(p * np.log(p + 1e-12))
    norm = np.sqrt(entropy(x_d) * entropy(y_d))
    return round(mi / norm if norm > 0 else 0, 4)


def distance_correlation(x, y, n_subsample=5000):
    """
    dCor via the dcor library.
    Subsample for large datasets to keep runtime manageable.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return np.nan, np.nan
    x_m, y_m = x[mask], y[mask]
    if len(x_m) > n_subsample:
        idx = np.random.choice(len(x_m), n_subsample, replace=False)
        x_m, y_m = x_m[idx], y_m[idx]
    try:
        dc = dcor.distance_correlation(x_m, y_m)
        # p-value via permutation (fast homogeneous test)
        res = dcor.independence.distance_covariance_test(
            x_m, y_m, num_resamples=199, random_state=42
        )
        return round(float(dc), 4), res.pvalue
    except Exception:
        return np.nan, np.nan


def quadrant_depletion(x, y, threshold_pct=QUADRANT_THRESHOLD):
    """
    Computes quadrant metrics using p33/p67 thresholds:
      Q1   : stem-high / immune-high (depleted)
      SCIe : Stem Cell-like Immune Evasion (stem-high / immune-low)
      NSI  : Non-Stem Immunogenic (stem-low / immune-high)

    Returns:
      obs_Q1%, exp%, fold_depletion,
      pct_SCIe, pct_NSI, SCIe_NSI_ratio
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    x_m, y_m = x[mask], y[mask]

    q_x_hi = np.percentile(x_m, threshold_pct)       # p67 stem
    q_x_lo = np.percentile(x_m, 100 - threshold_pct) # p33 stem
    q_y_hi = np.percentile(y_m, threshold_pct)        # p67 immune
    q_y_lo = np.percentile(y_m, 100 - threshold_pct)  # p33 immune

    # Q1: stem-high / immune-high (depleted quadrant)
    obs_Q1 = ((x_m > q_x_hi) & (y_m > q_y_hi)).mean() * 100
    exp     = ((100 - threshold_pct) / 100) ** 2 * 100
    fold    = round(obs_Q1 / exp, 3) if exp > 0 else np.nan

    # SCIe: stem-high / immune-low
    pct_SCIe = ((x_m > q_x_hi) & (y_m < q_y_lo)).mean() * 100

    # NSI: stem-low / immune-high
    pct_NSI  = ((x_m < q_x_lo) & (y_m > q_y_hi)).mean() * 100

    # SCIe/NSI ratio (>1 = immune evasion predominates)
    ratio = round(pct_SCIe / pct_NSI, 3) if pct_NSI > 0 else np.nan

    return (round(obs_Q1, 3), round(exp, 3), fold,
            round(pct_SCIe, 3), round(pct_NSI, 3), ratio)


def znorm(arr):
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    return (arr - mu) / (sd + 1e-8)


def score_and_znorm(adata, gene_list, col, min_pct=0.01):
    present = [g for g in gene_list if g in adata.var_names]
    if len(present) < 3:
        return False
    x = adata[:, present].X
    n_exp = np.asarray((x > 0).sum(axis=1)).flatten()
    if (n_exp > 0).mean() < min_pct:
        return False
    sc.tl.score_genes(adata, present, score_name=col, use_raw=False)
    raw = adata.obs[col].values.astype(float)
    adata.obs[f"{col}_z"] = znorm(raw)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: annotated hexbin with all 4 metrics
# ══════════════════════════════════════════════════════════════════════════════

def hexbin_annotated(obs, obj_name, stem_col, sig_label, fname):
    """
    Hexbin stem × immune annotated with Pearson, MI, dCor, %quad.
    Subtype panels as columns.
    """
    x_all = obs[stem_col].values.astype(float)
    y_all = obs["score_immunogenic_z"].values.astype(float)

    subtypes = [st for st in ["ER","HER2","TNBC"]
                if "subtype" in obs.columns and
                st in obs["subtype"].values]
    groups = [("Global", np.ones(len(obs), bool))] + \
             [(st, (obs["subtype"]==st).values) for st in subtypes]

    fig, axes = plt.subplots(1, len(groups),
                             figsize=(5.5*len(groups), 5.5),
                             squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        f"{obj_name} — {sig_label} × Immunogenicity\n"
        f"Annotated with 4 association metrics",
        fontsize=11, fontweight="bold"
    )

    for col_i, (label, mask) in enumerate(groups):
        ax   = axes[col_i]
        x, y = x_all[mask], y_all[mask]
        mv   = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mv], y[mv]

        if len(x) < 20:
            ax.text(0.5, 0.5, "n<20", transform=ax.transAxes,
                    ha="center", va="center"); continue

        xmin,xmax = np.percentile(x,0.5), np.percentile(x,99.5)
        ymin,ymax = np.percentile(y,0.5), np.percentile(y,99.5)
        xpad = 0.2*(xmax-xmin); ypad = 0.2*(ymax-ymin)
        xl = (xmin-xpad, xmax+xpad); yl = (ymin-ypad, ymax+ypad)

        ax.set_facecolor("white")
        ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
        hb = ax.hexbin(x, y, gridsize=90, cmap="RdYlBu_r",
                       mincnt=1, linewidths=0.02,
                       extent=[xl[0],xl[1],yl[0],yl[1]], zorder=2)
        ax.set_xlim(xl); ax.set_ylim(yl)
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5, zorder=3)
        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5, zorder=3)

        # Draw quadrant threshold lines
        qx = np.percentile(x, QUADRANT_THRESHOLD)
        qy = np.percentile(y, QUADRANT_THRESHOLD)
        ax.axhline(qy, color="#FF6600", ls=":", lw=1.0, alpha=0.7, zorder=3)
        ax.axvline(qx, color="#FF6600", ls=":", lw=1.0, alpha=0.7, zorder=3)

        # Shade depleted quadrant (stem-high / immune-high)
        ax.axhspan(qy, yl[1], xmin=(qx-xl[0])/(xl[1]-xl[0]), xmax=1.0,
                   color="#FF6600", alpha=0.06, zorder=1)

        plt.colorbar(hb, ax=ax, shrink=0.7, label="Cell count")
        color = SUBTYPE_COLORS.get(label,"black") if label!="Global" else "black"
        ax.set_title(f"{label} (n={len(x):,})",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel(f"{sig_label} (z)", fontsize=8)
        if col_i == 0:
            ax.set_ylabel("Immunogenicity (z)", fontsize=8)

        # Compute all metrics
        r_p, p_p   = pearson(x, y)
        mi_val     = mutual_information(x, y)
        dc, p_dc   = distance_correlation(x, y)
        obs_q, exp_q, fold, pct_scie, pct_nsi, scie_nsi = quadrant_depletion(x, y)

        def fmt_p(p):
            if p is None or np.isnan(p): return "n.s."
            return f"p={p:.1e}" if p >= 1e-300 else "p<1e-300"

        annotation = (
            f"Pearson r = {r_p:.3f}  ({fmt_p(p_p)})\n"
            f"dCor      = {dc:.3f}  ({fmt_p(p_dc)})\n"
            f"norm-MI   = {mi_val:.3f}\n"
            f"Q1 (hi/hi): {obs_q:.1f}% obs / {exp_q:.1f}% exp  fold={fold:.2f}×\n"
            f"SCIe = {pct_scie:.1f}%  NSI = {pct_nsi:.1f}%  "
            f"SCIe/NSI = {scie_nsi:.2f}"
        )
        ax.text(0.02, 0.98, annotation,
                transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", alpha=0.90))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/{fname}.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: summary comparison barplot (key objects)
# ══════════════════════════════════════════════════════════════════════════════

def summary_comparison_figure(df_key):
    """
    For key objects: 4-panel figure comparing metrics across signatures.
    One row per metric, columns = key objects.
    """
    metrics   = ["pearson_r", "dcor", "norm_mi", "fold_depletion"]
    ylabels   = ["Pearson r", "Distance Correlation", "Norm. Mutual Info",
                 "Fold depletion\n(stem-high/immune-high)"]
    ref_lines = [0, 0, 0, 1]   # reference line value per metric

    key_objs = df_key["Object"].unique()
    sigs     = STEMNESS_KEYS

    n_rows = len(metrics)
    n_cols = len(key_objs)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5*n_cols, 4*n_rows),
                             squeeze=False)
    fig.suptitle(
        "Stem × Immune: comparison of association metrics\n"
        "(Pearson, Distance Correlation, Mutual Information, Quadrant depletion)",
        fontsize=12, fontweight="bold"
    )

    x_pos = np.arange(len(sigs))
    sig_labels = [s.replace("LIM_MAMMARY_","LIM_").replace("_"," ")
                  for s in sigs]

    for col_i, obj in enumerate(key_objs):
        sub = df_key[df_key["Object"]==obj].set_index("Signature")

        for row_i, (metric, ylabel, ref) in enumerate(
            zip(metrics, ylabels, ref_lines)
        ):
            ax = axes[row_i, col_i]
            vals = [sub.loc[s, metric] if s in sub.index else np.nan
                    for s in sigs]

            colors = ["#E41A1C" if (v < ref and metric in ["pearson_r","fold_depletion"])
                      else "#377EB8" if not np.isnan(v) else "#CCCCCC"
                      for v in vals]

            bars = ax.bar(x_pos, vals, color=colors, alpha=0.80,
                          edgecolor="white", width=0.7)
            ax.axhline(ref, color="black", lw=1.0, ls="--", alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sig_labels, rotation=40, ha="right",
                               fontsize=7.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis="y", alpha=0.25)

            if row_i == 0:
                ax.set_title(obj, fontsize=10, fontweight="bold")

            # Annotate bars with values
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005 * ax.get_ylim()[1],
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6, rotation=0)

    plt.tight_layout()
    out = f"{OUTDIR}/summary_comparison_barplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: heatmap — all objects × signatures for each metric
# ══════════════════════════════════════════════════════════════════════════════

def metrics_heatmap(df_all, metric, title, fname, cmap="RdBu_r",
                    center=0, vmin=None, vmax=None):
    pivot = df_all.pivot_table(index="Signature", columns="Object",
                               values=metric)
    # Reorder signatures
    pivot = pivot.reindex([s for s in STEMNESS_KEYS if s in pivot.index])
    sig_labels = [s.replace("LIM_MAMMARY_","LIM_").replace("_"," ")
                  for s in pivot.index]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)*0.9),
                                    len(pivot)*0.7 + 2))
    im = ax.imshow(pivot.values.astype(float),
                   cmap=cmap, aspect="auto",
                   vmin=vmin if vmin is not None else np.nanpercentile(pivot.values, 2),
                   vmax=vmax if vmax is not None else np.nanpercentile(pivot.values, 98))
    plt.colorbar(im, ax=ax, shrink=0.6, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(sig_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/{fname}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
with open(SIGFILE) as f:
    SIGNATURES = json.load(f)

print("\n" + "="*65)
print(" 09 — STEM × IMMUNE: COMPARATIVE METRICS")
print("="*65)
print(f"  Metrics: Pearson r, Distance Correlation, "
      f"Normalised MI, Quadrant depletion")
print(f"  Objects : {len(OBJECTS)}")
print(f"  Signatures: {len(STEMNESS_KEYS)}")

all_rows = []

for obj_name, h5ad_file in OBJECTS.items():
    if not os.path.exists(h5ad_file):
        print(f"\n  ⚠️  {h5ad_file} not found — skipping {obj_name}")
        continue

    print(f"\n{'='*65}\n  {obj_name}\n{'='*65}")
    adata = sc.read_h5ad(h5ad_file)
    if "logcounts" in adata.layers:
        adata.X = adata.layers["logcounts"].copy()

    # Score immunogenicity
    if not score_and_znorm(adata, IMMUNOGENIC, "score_immunogenic"):
        print("  ⚠️  Immunogenicity scoring failed"); del adata; gc.collect(); continue

    imm_z = adata.obs["score_immunogenic_z"].values.astype(float)

    # Score each stemness signature
    for key in STEMNESS_KEYS:
        if key not in SIGNATURES:
            continue
        col = f"score_{key}"
        if not score_and_znorm(adata, SIGNATURES[key], col):
            continue

        x = adata.obs[f"{col}_z"].values.astype(float)
        y = imm_z

        r_p, p_p     = pearson(x, y)
        mi_val       = mutual_information(x, y)
        dc, p_dc     = distance_correlation(x, y)
        obs_q, exp_q, fold, pct_scie, pct_nsi, scie_nsi = quadrant_depletion(x, y)

        row = {
            "Object"        : obj_name,
            "Signature"     : key,
            "n"             : int((~(np.isnan(x)|np.isnan(y))).sum()),
            "pearson_r"     : r_p,
            "pearson_p"     : p_p,
            "dcor"          : dc,
            "dcor_p"        : p_dc,
            "norm_mi"       : mi_val,
            "pct_quad_obs"  : obs_q,
            "pct_quad_exp"  : exp_q,
            "fold_depletion": fold,
            "pct_SCIe"      : pct_scie,
            "pct_NSI"       : pct_nsi,
            "SCIe_NSI_ratio": scie_nsi,
        }
        all_rows.append(row)
        print(f"  {key:40s}  r={r_p:+.3f}  dCor={dc:.3f}  "
              f"MI={mi_val:.3f}  SCIe={pct_scie:.1f}%  NSI={pct_nsi:.1f}%  "
              f"SCIe/NSI={scie_nsi:.2f}")

        # Annotated hexbins for key objects only
        if obj_name in KEY_OBJECTS:
            sig_label = key.replace("LIM_MAMMARY_","LIM_").replace("_"," ")
            safe = obj_name.replace("/","_")
            hexbin_annotated(
                adata.obs, obj_name,
                stem_col=f"{col}_z",
                sig_label=sig_label,
                fname=f"hexbin_{safe}_{key}_metrics"
            )

    # Subtype-level metrics
    if "subtype" in adata.obs.columns:
        for subtype in ["ER","HER2","TNBC"]:
            st_mask = (adata.obs["subtype"]==subtype).values
            if st_mask.sum() < 30:
                continue
            for key in STEMNESS_KEYS:
                col_z = f"score_{key}_z"
                if col_z not in adata.obs.columns:
                    continue
                x = adata.obs.loc[st_mask, col_z].values.astype(float)
                y = imm_z[st_mask]
                r_p, p_p     = pearson(x, y)
                mi_val       = mutual_information(x, y)
                dc, p_dc     = distance_correlation(x, y)
                obs_q, exp_q, fold, pct_scie, pct_nsi, scie_nsi = quadrant_depletion(x, y)
                all_rows.append({
                    "Object"        : f"{obj_name}_{subtype}",
                    "Signature"     : key,
                    "n"             : int((~(np.isnan(x)|np.isnan(y))).sum()),
                    "pearson_r"     : r_p,
                    "pearson_p"     : p_p,
                    "dcor"          : dc,
                    "dcor_p"        : p_dc,
                    "norm_mi"       : mi_val,
                    "pct_quad_obs"  : obs_q,
                    "pct_quad_exp"  : exp_q,
                    "fold_depletion": fold,
                    "pct_SCIe"      : pct_scie,
                    "pct_NSI"       : pct_nsi,
                    "SCIe_NSI_ratio": scie_nsi,
                })

    del adata; gc.collect()


# ── Save table ────────────────────────────────────────────────────────────────
df_all = pd.DataFrame(all_rows)
out_csv = f"{OUTDIR}/metric_comparison_table.csv"
df_all.to_csv(out_csv, index=False)
print(f"\n  ✓ {out_csv}  ({len(df_all):,} rows)")

# ── Summary figures ───────────────────────────────────────────────────────────
df_global = df_all[df_all["Object"].str.endswith("Global") &
                   ~df_all["Object"].str.contains("_ER|_HER2|_TNBC")]

# Heatmaps per metric
for metric, title, cmap in [
    ("pearson_r",      "Pearson r — stem × immune",               "RdBu_r"),
    ("dcor",           "Distance Correlation — stem × immune",     "YlOrRd"),
    ("norm_mi",        "Normalised Mutual Information",             "YlOrRd"),
    ("fold_depletion", "Fold depletion Q1 (stem-high/immune-high)","RdYlGn"),
    ("pct_SCIe",       "% SCIe cells (stem-high / immune-low)",    "YlOrRd"),
    ("pct_NSI",        "% NSI cells (stem-low / immune-high)",     "YlGnBu"),
    ("SCIe_NSI_ratio", "SCIe/NSI ratio  (>1 = immune evasion)",   "RdYlGn"),
]:
    metrics_heatmap(df_global, metric, title,
                    fname=f"heatmap_{metric}",
                    cmap=cmap,
                    vmin=None, vmax=None)

# Comparison barplot for key objects
df_key = df_all[df_all["Object"].isin(KEY_OBJECTS)]
if not df_key.empty:
    summary_comparison_figure(df_key)

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Output: {OUTDIR}/
    metric_comparison_table.csv
    hexbin_[object]_[sig]_metrics.png   (key objects only)
    heatmap_pearson_r.png
    heatmap_dcor.png
    heatmap_norm_mi.png
    heatmap_fold_depletion.png
    summary_comparison_barplot.png

  Key question: does dCor / MI / fold-depletion reveal a stronger
  stem × immune relationship than Pearson alone?
══════════════════════════════════════════════════════════════
""")
