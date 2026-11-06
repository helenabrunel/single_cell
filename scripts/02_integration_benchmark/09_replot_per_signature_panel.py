"""
part0_replot_persig_panel.py
=============================
For each stemness signature: one panel figure mirroring the _umap_panel.png
layout but with that signature in the stemness column.

Rows = integration methods (Unintegrated / Harmony / scVI / DRVI)
Cols = Patient ID | Subtype | <this signature> | Immunogenic (consensus) | CytoTRACE2

Produces 9 figures per dataset → figures/part0/sigs_panel/
Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 part0_replot_persig_panel.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os

OUTDIR = "figures/part0/sigs_panel"
os.makedirs(OUTDIR, exist_ok=True)

with open("Visvader/signatures_confirmed.json") as f:
    ALL_SIGS = json.load(f)
STEM_SIGS = list(ALL_SIGS.keys())

METHODS = ["Unintegrated", "Harmony", "scVI", "DRVI"]
METHOD_COLORS = {
    "Unintegrated": "#888888",
    "Harmony"     : "#4393C3",
    "scVI"        : "#D6604D",
    "DRVI"        : "#762A83",
}
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

DATASETS = {
    "Visvader"  : {
        "file"       : "visvader_malignant_processed.h5ad",
        "batch_col"  : "sample",
        "subtype_col": "subtype",
    },
    "Klughammer": {
        "file"       : "klughammer_malignant_processed.h5ad",
        "batch_col"  : "sampleid",
        "subtype_col": "subtype",
    },
}

SIG_LABELS = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BENPORATH_ES1"                    : "BenPorath ES1",
    "EMP"                              : "EMP",
    "fMaSC_over_aMaSC"                 : "fMaSC / aMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"         : "LIM Mammary Stem",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM Luminal Progenitor",
    "LIM_MAMMARY_LUMINAL_MATURE_UP"    : "LIM Luminal Mature",
    "stem_consensus"                   : "Stem consensus",
}


def _patient_panel(ax, coords, batch_vals, n_patients):
    cats  = pd.Categorical(batch_vals)
    n     = len(cats.categories)
    cmap_ = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=cats.codes, cmap=cmap_, s=0.3, alpha=0.35, rasterized=True)
    ax.set_title(f"Patient (n={n_patients})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _subtype_panel(ax, coords, subtype_vals):
    for st, color in SUBTYPE_COLORS.items():
        m = subtype_vals == st
        if m.sum():
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=color, s=0.3, alpha=0.35, rasterized=True)
    ax.legend(
        handles=[mpatches.Patch(color=c, label=l)
                 for l, c in SUBTYPE_COLORS.items()
                 if l in subtype_vals],
        fontsize=6, framealpha=0.5)
    ax.set_title("Subtype", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _continuous_panel(ax, coords, scores, title, cmap="RdYlBu_r"):
    fv   = np.asarray(scores, dtype=float)
    vmin = np.nanpercentile(fv, 2)
    vmax = np.nanpercentile(fv, 98)
    sc_p = ax.scatter(coords[:, 0], coords[:, 1],
                      c=fv, cmap=cmap, vmin=vmin, vmax=vmax,
                      s=0.3, alpha=0.4, rasterized=True)
    plt.colorbar(sc_p, ax=ax, shrink=0.6, pad=0.01)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


for dataset_name, cfg in DATASETS.items():
    print(f"\n=== {dataset_name} ===")
    adata       = sc.read_h5ad(cfg["file"])
    batch_col   = cfg["batch_col"]
    subtype_col = cfg["subtype_col"]
    n_patients  = adata.obs[batch_col].nunique()
    methods     = [m for m in METHODS if f"X_umap_{m}" in adata.obsm]
    has_ct      = "CytoTRACE2" in adata.obs.columns
    print(f"  {adata.n_obs:,} cells | {n_patients} patients | "
          f"methods: {methods} | CytoTRACE2: {has_ct}")

    n_cols = 4 + (1 if has_ct else 0)   # patient | subtype | sig | immune | [CT2]
    col_titles = ["Patient", "Subtype", "<sig>",
                  "Immunogenic\n(consensus)"] + (["CytoTRACE2"] if has_ct else [])

    sigs = [s for s in STEM_SIGS if f"score_{s}" in adata.obs.columns]

    for sig in sigs:
        label     = SIG_LABELS.get(sig, sig)
        score_col = f"score_{sig}"
        n_rows    = len(methods)

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.8 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"{dataset_name}  —  stemness: {label}",
                     fontsize=12, fontweight="bold")

        for r, method in enumerate(methods):
            coords = adata.obsm[f"X_umap_{method}"]
            color  = METHOD_COLORS[method]

            # row label + spine colour
            axes[r, 0].set_ylabel(method, fontsize=10, fontweight="bold",
                                   rotation=90, labelpad=6, color=color)
            for c_idx in range(n_cols):
                for spine in axes[r, c_idx].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)

            # col 0: patient
            _patient_panel(axes[r, 0], coords,
                           adata.obs[batch_col].values, n_patients)

            # col 1: subtype
            _subtype_panel(axes[r, 1], coords,
                           adata.obs[subtype_col].values)

            # col 2: this stemness signature
            _continuous_panel(axes[r, 2], coords,
                               adata.obs[score_col].values,
                               label, cmap="RdYlBu_r")

            # col 3: immunogenic consensus
            if "score_immune" in adata.obs.columns:
                _continuous_panel(axes[r, 3], coords,
                                   adata.obs["score_immune"].values,
                                   "Immunogenic (consensus)", cmap="PuRd")

            # col 4: CytoTRACE2
            if has_ct:
                _continuous_panel(axes[r, 4], coords,
                                   adata.obs["CytoTRACE2"].values,
                                   "CytoTRACE2", cmap="YlOrRd_r")

        plt.tight_layout()
        fname = f"{dataset_name}_{sig}_panel.png"
        out   = os.path.join(OUTDIR, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {fname}")

print(f"\nAll figures in {OUTDIR}/")
