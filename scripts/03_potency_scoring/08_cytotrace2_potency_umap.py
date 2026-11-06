"""
22f_cytotrace2_potency_umap.py
==============================
Per subtype (Harmony-integrated epithelial objects), the two CytoTRACE2 UMAPs
from cytotrace2_py.common.plot.plot_all_outputs, with the package's palette:
  - potency category UMAP (Differentiated → Totipotent)
  - potency score UMAP (fixed 0–1 scale, colorbar labelled by category)

Input: cytotrace2_scores_Epithelial_{subtype}.csv (04_cytotrace2_scoring.py)
Outputs: figures/22_axis_signatures/phenotype/ (copied to axis_based signatures/UMAPs/)
  umap_{subtype}_CytoTRACE2_potency_category.png
  umap_{subtype}_CytoTRACE2_potency_score.png
  umap_{subtype}_CytoTRACE2_2panels.png
  cytotrace2_potency_counts.csv
"""

import os, sys, shutil, warnings
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 8})

DATA_DIR = "/home/hbrunel/single_cell/Visvader"
FIG_DIR  = os.path.join(DATA_DIR, "figures", "22_axis_signatures", "phenotype")
COPY_DIR = "/home/hbrunel/single_cell/axis_based signatures/UMAPs"

SUBTYPES = ["ER", "HER2", "TNBC"]
EPI_H5AD = "epithelial_{}.h5ad"

# --er-filtered: ER only, re-integrated without 11 samples (23_reintegrate_ER_filtered.py)
ER_FILTERED = "--er-filtered" in sys.argv
if ER_FILTERED:
    SUBTYPES = ["ER"]
    EPI_H5AD = "epithelial_{}_filtered.h5ad"
    FIG_DIR  = os.path.join(DATA_DIR, "figures", "22_axis_signatures_ER_filtered", "phenotype")
    COPY_DIR = "/home/hbrunel/single_cell/axis_based signatures/UMAPs_ER_filtered"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(COPY_DIR, exist_ok=True)

# Palette and scale from cytotrace2_py.common.plot.plot_all_outputs
POTENCY_LABELS = ["Differentiated", "Unipotent", "Oligopotent",
                  "Multipotent", "Pluripotent", "Totipotent"]
POTENCY_COLORS = ["#5E4FA2", "#66C2A5", "#E6F598", "#FEE08B", "#F46D43", "#9E0142"]
POTENCY_PALETTE = dict(zip(POTENCY_LABELS, POTENCY_COLORS))
# Package plots 6*score - 5.5 with vmin=-5, vmax=0: colour k sits at score
# (2k+1)/12, i.e. the centre of each 1/6-wide potency band. Same mapping on a
# 0-1 axis, flat at both ends, so the colorbar spans the full score range.
_centres = (2 * np.arange(6) + 1) / 12
POTENCY_CMAP = LinearSegmentedColormap.from_list(
    "potency_cols",
    [(0, POTENCY_COLORS[0])] + list(zip(_centres, POTENCY_COLORS)) + [(1, POTENCY_COLORS[-1])])
SCORE_VMIN, SCORE_VMAX = 0, 1


def style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticks([]); ax.set_yticks([])


def plot_category(ax, coords, potency, title):
    pot = np.asarray(potency)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pot))
    ax.scatter(coords[order, 0], coords[order, 1],
               c=[POTENCY_PALETTE[p] for p in pot[order]],
               s=2, alpha=0.8, linewidths=0, rasterized=True)
    handles = [Line2D([], [], marker="o", ls="", color=POTENCY_PALETTE[l], markersize=6,
                      label=f"{l} ({(pot == l).sum():,}; {100 * (pot == l).mean():.1f}%)")
               for l in POTENCY_LABELS]
    ax.legend(handles=handles, title="Potency category", fontsize=8, title_fontsize=8,
              frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    style(ax, title)


def plot_score(ax, coords, score, title):
    order = np.argsort(score)
    p = ax.scatter(coords[order, 0], coords[order, 1], c=score[order], cmap=POTENCY_CMAP,
                   vmin=SCORE_VMIN, vmax=SCORE_VMAX, s=2, alpha=0.8,
                   linewidths=0, rasterized=True)
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title("Potency score", fontsize=8, loc="left", pad=8)
    cbar.set_ticks(_centres)
    cbar.set_ticklabels(POTENCY_LABELS, fontsize=7)
    for b in np.arange(1, 6) / 6:
        cbar.ax.axhline(b, color="white", lw=0.5)
    style(ax, title)


counts, outputs = [], []

for subtype in SUBTYPES:
    print(f"=== {subtype} ===")
    a = ad.read_h5ad(os.path.join(DATA_DIR, EPI_H5AD.format(subtype)), backed="r")
    cells = a.obs_names
    coords = np.asarray(a.obsm["X_umap"])
    a.file.close()

    ct2 = pd.read_csv(os.path.join(DATA_DIR, f"cytotrace2_scores_Epithelial_{subtype}.csv"),
                      index_col=0).reindex(cells)
    assert ct2["CytoTRACE2_Score"].notna().all()
    score = ct2["CytoTRACE2_Score"].values.astype(float)
    potency = ct2["CytoTRACE2_Potency"].astype(str).values

    vc = pd.Series(potency).value_counts()
    for l in POTENCY_LABELS:
        counts.append({"subtype": subtype, "potency": l, "n_cells": int(vc.get(l, 0)),
                       "pct": round(100 * vc.get(l, 0) / len(potency), 2)})
    print("  " + ", ".join(f"{l} {vc.get(l, 0):,}" for l in POTENCY_LABELS))

    panels = {
        "potency_category": lambda ax: plot_category(ax, coords, potency,
                                                     f"{subtype} — CytoTRACE2 potency"),
        "potency_score":    lambda ax: plot_score(ax, coords, score,
                                                  f"{subtype} — CytoTRACE2 score"),
    }
    for name, draw in panels.items():
        fig, ax = plt.subplots(figsize=(7, 6.5))
        draw(ax)
        out = os.path.join(FIG_DIR, f"umap_{subtype}_CytoTRACE2_{name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
        outputs.append(out)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, draw in zip(axes, panels.values()):
        draw(ax)
    fig.suptitle(f"{subtype} — epithelial cells, Harmony UMAP (n={len(cells):,})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"umap_{subtype}_CytoTRACE2_2panels.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    outputs.append(out)

out = os.path.join(FIG_DIR, "cytotrace2_potency_counts.csv")
pd.DataFrame(counts).to_csv(out, index=False)
outputs.append(out)

for f in outputs:
    shutil.copy(f, COPY_DIR)
print("Done.")
