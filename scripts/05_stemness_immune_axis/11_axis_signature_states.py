"""
22b_axis_signatures_states.py
=============================
Classifies each cell by its dominant axis-based signature, per subtype, and
plots the classes on the Harmony UMAP.

Input: per-cell SCIE / NSI scores from 22_axis_signatures_umap.py
       (figures/22_axis_signatures/scores_{dataset}_{subtype}.csv)
       + X_umap from the Harmony-integrated objects.

Rule (per subtype and dataset):
  a signature is "high" in a cell if its score > HIGH_QUANTILE of that
  signature's scores in the subtype.
    SCIE high only  -> SCIE  (#00B0F0)
    NSI high only   -> NSI   (#FBA094)
    both high       -> SCI   (#AFCB4B)
    neither         -> None  (light grey)

Outputs (figures/22_axis_signatures/ and copied to axis_based signatures/):
  umap_{dataset}_{subtype}_axis_state.png
  umap_{dataset}_axis_state_all_subtypes.png
  axis_state_counts.csv
  axis_state_{dataset}_{subtype}.csv  — per-cell scores + state
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
FIG_DIR  = os.path.join(DATA_DIR, "figures", "22_axis_signatures")

SUBTYPES = ["ER", "HER2", "TNBC"]
DATASETS = {
    "Epithelial": lambda st: f"epithelial_{st}.h5ad",
    "All":        lambda st: f"all_{st.lower()}.h5ad",
}
HIGH_QUANTILE = 0.75

STATE_COLORS = {
    "SCIE": "#00B0F0",
    "NSI":  "#FBA094",
    "SCI":  "#AFCB4B",
    "None": "#D9D9D9",
}
PLOT_ORDER = ["None", "NSI", "SCIE", "SCI"]  # grey at the back


def classify(scores):
    hi = {s: scores[s] > scores[s].quantile(HIGH_QUANTILE) for s in ["SCIE", "NSI"]}
    state = np.select(
        [hi["SCIE"] & hi["NSI"], hi["SCIE"], hi["NSI"]],
        ["SCI", "SCIE", "NSI"], default="None")
    return pd.Series(state, index=scores.index)


def plot_states(ax, coords, state, title):
    handles = []
    for st in PLOT_ORDER:
        m = (state == st).values
        ax.scatter(coords[m, 0], coords[m, 1], c=STATE_COLORS[st], s=1.5,
                   alpha=0.8 if st != "None" else 0.5, linewidths=0, rasterized=True)
    for st in ["SCIE", "NSI", "SCI", "None"]:
        n = int((state == st).sum())
        handles.append(Line2D([], [], marker="o", ls="", color=STATE_COLORS[st],
                              markersize=6,
                              label=f"{st} ({n:,}; {100 * n / len(state):.1f}%)"))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")


counts = []
outputs = []

for dataset, fname in DATASETS.items():
    fig_all, axes_all = plt.subplots(1, len(SUBTYPES), figsize=(6.5 * len(SUBTYPES), 6))
    for ax_all, subtype in zip(axes_all, SUBTYPES):
        print(f"=== {dataset} / {subtype} ===")
        scores = pd.read_csv(os.path.join(FIG_DIR, f"scores_{dataset}_{subtype}.csv"),
                             index_col=0)
        a = ad.read_h5ad(os.path.join(DATA_DIR, fname(subtype)), backed="r")
        assert (a.obs_names == scores.index).all()
        coords = np.asarray(a.obsm["X_umap"])
        a.file.close()

        state = classify(scores)
        scores["axis_state"] = state
        scores.to_csv(os.path.join(FIG_DIR, f"axis_state_{dataset}_{subtype}.csv"))

        vc = state.value_counts()
        for st in STATE_COLORS:
            counts.append({"dataset": dataset, "subtype": subtype, "state": st,
                           "n_cells": int(vc.get(st, 0)),
                           "pct": 100 * vc.get(st, 0) / len(state)})
        print("  " + ", ".join(f"{st}: {vc.get(st, 0):,}" for st in STATE_COLORS))

        title = f"{subtype} — {dataset} cells (Harmony UMAP)"
        fig, ax = plt.subplots(figsize=(6.5, 6))
        plot_states(ax, coords, state, title)
        out = os.path.join(FIG_DIR, f"umap_{dataset}_{subtype}_axis_state.png")
        fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
        outputs.append(out)

        plot_states(ax_all, coords, state, subtype)

    fig_all.suptitle(f"Dominant axis signature — {dataset} cells "
                     f"(high = score > P{int(HIGH_QUANTILE * 100)} within subtype)",
                     fontsize=13, fontweight="bold", y=1.02)
    fig_all.tight_layout()
    out = os.path.join(FIG_DIR, f"umap_{dataset}_axis_state_all_subtypes.png")
    fig_all.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig_all)
    outputs.append(out)

counts_path = os.path.join(FIG_DIR, "axis_state_counts.csv")
pd.DataFrame(counts).to_csv(counts_path, index=False)
outputs.append(counts_path)

for f in outputs:
    shutil.copy(f, SIG_DIR)
print("Done.")
