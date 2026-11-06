"""
22c_axis_signatures_labels.py
=============================
One UMAP per subtype (Harmony-integrated epithelial cells) where every cell is
labelled SCIE / NSI / SCI / NC by comparing its SCIE and NSI scores.

Input: per-cell scores from 22_axis_signatures_umap.py
       (figures/22_axis_signatures/scores_Epithelial_{subtype}.csv)

Rule (per subtype; z = score standardised within the subtype):
  SCIE : z_SCIE - z_NSI > DIFF  and  z_SCIE > HIGH   (SCIE much higher than NSI)
  NSI  : z_NSI - z_SCIE > DIFF  and  z_NSI  > HIGH   (NSI much higher than SCIE)
  SCI  : z_SCIE > HIGH and z_NSI > HIGH, not dominated by either (both high)
  NC   : everything else (both low / no clear call)

Outputs (figures/22_axis_signatures/labels/ and copied to axis_based signatures/):
  umap_{subtype}_SCIE_NSI_SCI_NC.png  — the UMAP per subtype
  zscatter_{subtype}.png              — z_SCIE vs z_NSI coloured by label (rule check)
  labels_{subtype}.csv                — per-cell scores, z-scores and label
  label_counts.csv
"""

import os, shutil, warnings
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 8})

DATA_DIR = "/home/hbrunel/single_cell/Visvader"
SIG_DIR  = "/home/hbrunel/single_cell/axis_based signatures"
SCORE_DIR = os.path.join(DATA_DIR, "figures", "22_axis_signatures")
FIG_DIR  = os.path.join(SCORE_DIR, "labels")
os.makedirs(FIG_DIR, exist_ok=True)

SUBTYPES = ["ER", "HER2", "TNBC"]
HIGH = 0.0   # "high" = above the subtype mean (z > 0)
DIFF = 1.0   # "much higher" = at least 1 SD above the other signature

LABEL_COLORS = {
    "SCIE": "#00B0F0",
    "NSI":  "#FBA094",
    "SCI":  "#AFCB4B",
    "NC":   "#D9D9D9",
}
PLOT_ORDER = ["NC", "NSI", "SCIE", "SCI"]  # grey at the back


def label_cells(zs, zn):
    d = zs - zn
    return pd.Series(np.select(
        [(d > DIFF) & (zs > HIGH),
         (-d > DIFF) & (zn > HIGH),
         (zs > HIGH) & (zn > HIGH)],
        ["SCIE", "NSI", "SCI"], default="NC"), index=zs.index)


def legend_handles(labels):
    return [Line2D([], [], marker="o", ls="", color=LABEL_COLORS[l], markersize=6,
                   label=f"{l} ({(labels == l).sum():,}; {100 * (labels == l).mean():.1f}%)")
            for l in ["SCIE", "NSI", "SCI", "NC"]]


counts, outputs = [], []

for subtype in SUBTYPES:
    df = pd.read_csv(os.path.join(SCORE_DIR, f"scores_Epithelial_{subtype}.csv"), index_col=0)
    a = ad.read_h5ad(os.path.join(DATA_DIR, f"epithelial_{subtype}.h5ad"), backed="r")
    assert (a.obs_names == df.index).all()
    coords = np.asarray(a.obsm["X_umap"])
    a.file.close()

    for s in ["SCIE", "NSI"]:
        df[f"z_{s}"] = (df[s] - df[s].mean()) / df[s].std()
    df["label"] = label_cells(df["z_SCIE"], df["z_NSI"])
    df.to_csv(os.path.join(FIG_DIR, f"labels_{subtype}.csv"))
    labels = df["label"]

    vc = labels.value_counts()
    for l in LABEL_COLORS:
        counts.append({"subtype": subtype, "label": l, "n_cells": int(vc.get(l, 0)),
                       "pct": round(100 * vc.get(l, 0) / len(labels), 2)})
    print(f"{subtype}: " + ", ".join(f"{l} {vc.get(l, 0):,}" for l in LABEL_COLORS))

    # UMAP
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for l in PLOT_ORDER:
        m = (labels == l).values
        ax.scatter(coords[m, 0], coords[m, 1], c=LABEL_COLORS[l], s=2,
                   alpha=0.5 if l == "NC" else 0.85, linewidths=0, rasterized=True)
    ax.legend(handles=legend_handles(labels), fontsize=8, frameon=False, loc="best")
    ax.set_title(f"{subtype} — epithelial cells (n={len(labels):,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(FIG_DIR, f"umap_{subtype}_SCIE_NSI_SCI_NC.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    outputs.append(out)

    # Rule check: z_SCIE vs z_NSI
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for l in PLOT_ORDER:
        m = (labels == l).values
        ax.scatter(df["z_SCIE"].values[m], df["z_NSI"].values[m], c=LABEL_COLORS[l],
                   s=1.5, alpha=0.6, linewidths=0, rasterized=True)
    lim = np.nanpercentile(np.abs(df[["z_SCIE", "z_NSI"]].values), 99.9)
    x = np.linspace(-lim, lim, 2)
    ax.plot(x, x - DIFF, "k--", lw=0.6); ax.plot(x, x + DIFF, "k--", lw=0.6)
    ax.axvline(HIGH, color="k", lw=0.6); ax.axhline(HIGH, color="k", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("SCIE (z-score)"); ax.set_ylabel("NSI (z-score)")
    ax.set_title(f"{subtype} — labelling rule (HIGH={HIGH}, DIFF={DIFF})",
                 fontsize=10, fontweight="bold")
    ax.legend(handles=legend_handles(labels), fontsize=7, frameon=False, loc="upper left")
    out = os.path.join(FIG_DIR, f"zscatter_{subtype}.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    outputs.append(out)

out = os.path.join(FIG_DIR, "label_counts.csv")
pd.DataFrame(counts).to_csv(out, index=False)
outputs.append(out)

for f in outputs:
    shutil.copy(f, SIG_DIR)
print("Done.")
