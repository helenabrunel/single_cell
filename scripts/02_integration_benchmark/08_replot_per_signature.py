"""
part0_replot_persig.py
======================
For each stemness signature: one panel figure with
  rows = integration methods (Unintegrated / Harmony / scVI / DRVI)
  cols = Patient ID (reference) | signature score

Produces 9 figures per dataset in figures/part0/sigs/
Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 part0_replot_persig.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os

OUTDIR  = "figures/part0/sigs"
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
DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_processed.h5ad",   "batch_col": "sample"},
    "Klughammer": {"file": "klughammer_malignant_processed.h5ad", "batch_col": "sampleid"},
}

SIG_LABELS = {
    "ESC_ASSOU"                    : "ESC_ASSOU",
    "ESC_WONG"                     : "ESC_WONG",
    "BENPORATH_ES1"                : "BenPorath_ES1",
    "EMP"                          : "EMP",
    "fMaSC_over_aMaSC"             : "fMaSC / aMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"     : "LIM Mammary Stem",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM Luminal Progenitor",
    "LIM_MAMMARY_LUMINAL_MATURE_UP": "LIM Luminal Mature",
    "stem_consensus"               : "Stem consensus",
}


def _patient_scatter(ax, coords, batch_vals, batch_col, n_patients):
    cats  = pd.Categorical(batch_vals)
    n     = len(cats.categories)
    cmap_ = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=cats.codes, cmap=cmap_, s=0.3, alpha=0.35, rasterized=True)
    ax.set_title(f"Patient ID  (n={n_patients})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _sig_scatter(ax, coords, scores, sig_label):
    fv   = scores.astype(float)
    vmin = np.nanpercentile(fv, 2)
    vmax = np.nanpercentile(fv, 98)
    sc_p = ax.scatter(coords[:, 0], coords[:, 1],
                      c=fv, cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                      s=0.3, alpha=0.4, rasterized=True)
    plt.colorbar(sc_p, ax=ax, shrink=0.6, pad=0.01)
    ax.set_title(sig_label, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


for dataset_name, cfg in DATASETS.items():
    print(f"\n=== {dataset_name} ===")
    adata      = sc.read_h5ad(cfg["file"])
    batch_col  = cfg["batch_col"]
    n_patients = adata.obs[batch_col].nunique()
    methods    = [m for m in METHODS if f"X_umap_{m}" in adata.obsm]
    print(f"  {adata.n_obs:,} cells | {n_patients} patients | methods: {methods}")

    sigs = [s for s in STEM_SIGS if f"score_{s}" in adata.obs.columns]

    for sig in sigs:
        label     = SIG_LABELS.get(sig, sig)
        score_col = f"score_{sig}"
        n_rows    = len(methods)

        fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"{dataset_name}  —  {label}",
                     fontsize=12, fontweight="bold")

        for r, method in enumerate(methods):
            coords = adata.obsm[f"X_umap_{method}"]
            color  = METHOD_COLORS[method]

            # col 0: patient reference
            _patient_scatter(axes[r, 0], coords,
                             adata.obs[batch_col].values,
                             batch_col, n_patients)
            axes[r, 0].set_ylabel(method, fontsize=10, fontweight="bold",
                                   rotation=90, labelpad=6, color=color)
            for spine in axes[r, 0].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2)

            # col 1: signature score
            _sig_scatter(axes[r, 1], coords,
                         adata.obs[score_col].values, label)
            for spine in axes[r, 1].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2)

        plt.tight_layout()
        fname = f"{dataset_name}_{sig}_panel.png"
        out   = os.path.join(OUTDIR, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {fname}")

print(f"\nAll figures in {OUTDIR}/")
