"""
Hexbin: stemness score vs S.Score, before and after CC residualisation.
Per-patient mean-centered then pooled — removes inter-patient score shifts
so the density reflects only within-patient covariation.

Layout: 2 rows (Visvader / Klughammer) × 2 cols (before / after)
Signature: ESC_WONG (recommended) + stem_consensus (excluded) as reference

Output: figures/part1/hexbin_stem_vs_cc.png
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = "figures/part1"
plt.rcParams.update({"font.family": "DejaVu Sans"})

DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_ccfree.h5ad",   "batch": "sample"},
    "Klughammer": {"file": "klughammer_malignant_ccfree.h5ad",  "batch": "sampleid"},
}

SIGS = [
    ("ESC_WONG", "ESC_WONG", "#059669"),
]

BG     = "#FFFFFF"
DARK   = "#1F2937"
GREY   = "#6B7280"
BORDER = "#E5E7EB"

# Blue-to-red colormap matching the reference format
CMAP = matplotlib.colormaps.get_cmap("RdYlBu_r")

GRIDSIZE = 55


def mean_center_per_patient(obs, batch_col, cols):
    """Subtract per-patient mean from each column in place."""
    obs = obs.copy()
    for pat in obs[batch_col].unique():
        mask = obs[batch_col] == pat
        for c in cols:
            obs.loc[mask, c] = obs.loc[mask, c] - obs.loc[mask, c].mean()
    return obs


def spearman_annot(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[m], y[m])
    p_str = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
    return r, p, m.sum(), p_str


def hexbin_panel(ax, x, y, xlabel, ylabel, title, color_accent,
                 annot_loc="lower right"):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=8)

    # hexbin
    hb = ax.hexbin(x, y, gridsize=GRIDSIZE, cmap=CMAP,
                   linewidths=0.2, mincnt=1, bins="log")
    cb = plt.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("log₁₀(cell count)", fontsize=7.5, color=GREY)
    cb.ax.tick_params(labelsize=7, colors=GREY)

    # reference lines
    ax.axhline(0, color=GREY, lw=0.9, ls="--", alpha=0.6, zorder=3)
    ax.axvline(0, color=GREY, lw=0.9, ls="--", alpha=0.6, zorder=3)

    # annotation
    r, p, n, p_str = spearman_annot(x, y)
    props = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor=BORDER, alpha=0.9)
    xpos = 0.97 if "right" in annot_loc else 0.03
    ha   = "right" if "right" in annot_loc else "left"
    ypos = 0.04 if "lower" in annot_loc else 0.96
    va   = "bottom" if "lower" in annot_loc else "top"
    ax.text(xpos, ypos,
            f"r = {r:+.3f}\np = {p_str}\nn = {n:,}",
            transform=ax.transAxes, fontsize=8, ha=ha, va=va,
            bbox=props, color=DARK)

    ax.set_xlabel(xlabel, fontsize=9, color="#374151")
    ax.set_ylabel(ylabel, fontsize=9, color="#374151")
    ax.set_title(title, fontsize=10, fontweight="bold", color=color_accent,
                 loc="left", pad=4)


# ══════════════════════════════════════════════════════════════════════════════
# Load & mean-center
# ══════════════════════════════════════════════════════════════════════════════
data = {}
for ds_name, cfg in DATASETS.items():
    print(f"Loading {ds_name} …")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs.copy()
    del adata

    batch = cfg["batch"]
    sig_cols = []
    for sig, _, _ in SIGS:
        raw_col  = f"score_{sig}"
        free_col = f"score_{sig}_ccfree"
        if raw_col in obs.columns and free_col in obs.columns:
            sig_cols += [raw_col, free_col]

    cols_to_center = sig_cols + ["S.Score", "CytoTRACE2"]
    obs_c = mean_center_per_patient(obs, batch, cols_to_center)
    data[ds_name] = obs_c
    print(f"  {obs[batch].nunique()} patients, {len(obs):,} cells")


# ══════════════════════════════════════════════════════════════════════════════
# Figure: ESC_WONG — 2 datasets × 3 panels (before | after | potency)
# ══════════════════════════════════════════════════════════════════════════════
sig, sig_label, sig_color = SIGS[0]
raw_col  = f"score_{sig}"
free_col = f"score_{sig}_ccfree"

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.998,
         f"ESC_WONG — stemness score: CC contamination and residual potency signal",
         ha="center", va="top", fontsize=14, fontweight="bold", color=DARK)
fig.text(0.5, 0.972,
         "Per-patient mean-centered, then pooled  ·  each hex = log₁₀(cell count)",
         ha="center", va="top", fontsize=9.5, color=GREY, style="italic")

col_titles = [
    ("1 — BEFORE correction", "#DC2626",
     "Does ESC_WONG correlate with cell cycle?"),
    ("2 — AFTER correction  (sig_ccfree vs S.Score)", "#2563EB",
     "Is CC contamination removed?"),
    ("3 — POTENCY signal  (sig_ccfree vs CytoTRACE2)", "#059669",
     "Does genuine potency survive?"),
]

for col_i, (col_title, col_color, col_sub) in enumerate(col_titles):
    fig.text((col_i + 0.5) / 3, 0.952, col_title,
             ha="center", va="top", fontsize=10.5,
             fontweight="bold", color=col_color)
    fig.text((col_i + 0.5) / 3, 0.932, col_sub,
             ha="center", va="top", fontsize=8.5,
             color=GREY, style="italic")

for row_i, (ds_name, obs_c) in enumerate(data.items()):
    ss   = obs_c["S.Score"].values.astype(float)
    raw  = obs_c[raw_col].values.astype(float)
    free = obs_c[free_col].values.astype(float)
    ct2  = obs_c["CytoTRACE2"].values.astype(float)
    n    = len(obs_c)

    # row label
    fig.text(0.01, 0.75 - row_i * 0.48, ds_name,
             fontsize=11, fontweight="bold", color=DARK,
             va="center", rotation=90)

    # Panel 1 — BEFORE: raw sig vs S.Score
    hexbin_panel(
        axes[row_i, 0],
        x=ss, y=raw,
        xlabel="S.Score  (per-patient centered)",
        ylabel="ESC_WONG  (raw)",
        title=f"{ds_name}  ({n:,} cells)",
        color_accent=DARK,
    )

    # Panel 2 — AFTER: sig_ccfree vs S.Score
    hexbin_panel(
        axes[row_i, 1],
        x=ss, y=free,
        xlabel="S.Score  (per-patient centered)",
        ylabel="ESC_WONG_ccfree",
        title=f"{ds_name}  ({n:,} cells)",
        color_accent=DARK,
    )

    # Panel 3 — POTENCY: sig_ccfree vs CytoTRACE2
    hexbin_panel(
        axes[row_i, 2],
        x=ct2, y=free,
        xlabel="CytoTRACE2  (per-patient centered)",
        ylabel="ESC_WONG_ccfree",
        title=f"{ds_name}  ({n:,} cells)",
        color_accent=DARK,
        annot_loc="upper left",
    )

plt.tight_layout(rect=[0.03, 0, 1, 0.928], h_pad=3.0, w_pad=2.5)
outfile = os.path.join(OUT, "hexbin_stem_vs_cc.png")
fig.savefig(outfile, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"✓ {outfile}")
