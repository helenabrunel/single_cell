"""
part0_replot_allsigs.py
=======================
Regenerate only the _allsigs_umap.png figures from the already-saved
processed h5ads.  Does NOT redo QC, scoring, or integration.

Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 part0_replot_allsigs.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os

OUTDIR  = "figures/part0"
SIGFILE = "Visvader/signatures_confirmed.json"
with open(SIGFILE) as f:
    ALL_SIGS = json.load(f)
STEM_SIGS = list(ALL_SIGS.keys())

METHOD_COLORS = {
    "Unintegrated": "#888888",
    "Harmony"     : "#4393C3",
    "scVI"        : "#D6604D",
    "DRVI"        : "#762A83",
}

DATASETS = {
    "Visvader"   : {"file": "visvader_malignant_processed.h5ad",    "batch_col": "sample"},
    "Klughammer" : {"file": "klughammer_malignant_processed.h5ad",  "batch_col": "sampleid"},
}


def _umap_scatter(ax, coords, values, title,
                  cmap="RdYlBu_r", categorical_cmap="tab20",
                  s=0.3, alpha=0.4):
    vals = np.asarray(values)
    if vals.dtype.kind in ("U", "O", "S"):
        cats  = pd.Categorical(vals)
        n     = len(cats.categories)
        cmap_ = plt.cm.get_cmap(categorical_cmap, n)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=cats.codes, cmap=cmap_, s=s, alpha=alpha, rasterized=True)
        handles = [mpatches.Patch(color=cmap_(i / max(n - 1, 1)), label=str(c))
                   for i, c in enumerate(cats.categories)]
        ax.legend(handles=handles, fontsize=4, ncol=2,
                  loc="upper right", framealpha=0.4)
    else:
        fv   = vals.astype(float)
        vmin = np.nanpercentile(fv, 2)
        vmax = np.nanpercentile(fv, 98)
        sc_p = ax.scatter(coords[:, 0], coords[:, 1],
                          c=fv, cmap=cmap, vmin=vmin, vmax=vmax,
                          s=s, alpha=alpha, rasterized=True)
        plt.colorbar(sc_p, ax=ax, shrink=0.55, pad=0.01)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def plot_all_sigs_umap(adata, batch_col, dataset_name):
    """
    Rows = integration methods (Unintegrated, Harmony, scVI, DRVI — whichever available).
    Col 0 = patient ID reference.
    Col 1…n = one stemness signature per column.
    """
    methods = [m for m in ["Unintegrated", "Harmony", "scVI", "DRVI"]
               if f"X_umap_{m}" in adata.obsm]
    if not methods:
        print(f"  No UMAP embeddings found for {dataset_name} — skipping")
        return

    sigs   = [s for s in STEM_SIGS if f"score_{s}" in adata.obs.columns]
    n_cols = 1 + len(sigs)
    n_rows = len(methods)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"All stemness signatures — {dataset_name}\n"
        f"col 0 = patient ID (batch reference) | each row = integration method",
        fontsize=11, fontweight="bold"
    )

    for r, method in enumerate(methods):
        coords = adata.obsm[f"X_umap_{method}"]
        color  = METHOD_COLORS.get(method, "#333")

        # col 0: patient reference
        _umap_scatter(axes[r, 0], coords,
                      adata.obs[batch_col].values,
                      f"Patient ID\n({method})",
                      categorical_cmap="tab20")
        axes[r, 0].set_ylabel(method, fontsize=9, fontweight="bold",
                               rotation=90, labelpad=6,
                               color=color)
        # colour the spine to match method
        for spine in axes[r, 0].spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

        # cols 1…: signatures
        for c, sig in enumerate(sigs, start=1):
            short = (sig.replace("LIM_MAMMARY_STEM_CELL_UP",      "LIM_Stem")
                        .replace("LIM_MAMMARY_LUMINAL_PROGENITOR_UP", "LIM_LumProg")
                        .replace("LIM_MAMMARY_LUMINAL_MATURE_UP",     "LIM_LumMat")
                        .replace("BENPORATH_ES1",                  "BP_ES1")
                        .replace("fMaSC_over_aMaSC",               "fMaSC/aMaSC")
                        .replace("stem_consensus",                 "stem_cons"))
            _umap_scatter(axes[r, c], coords,
                          adata.obs[f"score_{sig}"].values,
                          short, cmap="RdYlBu_r")

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{dataset_name}_allsigs_umap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}  ({n_rows} methods × {n_cols} cols)")


for dataset_name, cfg in DATASETS.items():
    print(f"\n--- {dataset_name} ---")
    adata = sc.read_h5ad(cfg["file"])
    print(f"  {adata.n_obs:,} cells | embeddings: {[k for k in adata.obsm if 'umap' in k.lower()]}")
    plot_all_sigs_umap(adata, cfg["batch_col"], dataset_name)

print("\nDone — figures in figures/part0/")
