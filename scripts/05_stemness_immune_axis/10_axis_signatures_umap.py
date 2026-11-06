"""
22_axis_signatures_umap.py
==========================
Scores the subtype-specific axis-based signatures (SCIE and NSI) on the
Harmony-integrated objects of each subtype and plots them on the Harmony UMAP.

Signatures: ~/single_cell/axis_based signatures/{SCIE,NSI}_{ER,HER2,TNBC}_consensus.csv
Each subtype is scored only with its own SCIE / NSI signature
(sc.tl.score_genes on logcounts).

Datasets (Harmony-integrated per subtype, from 03_split_integrate.py):
  Epithelial : epithelial_{ER,HER2,TNBC}.h5ad
  All cells  : all_{er,her2,tnbc}.h5ad

Outputs (figures/22_axis_signatures/):
  umap_{dataset}_{subtype}_{SCIE|NSI}.png   — one UMAP per signature
  umap_{dataset}_{subtype}_SCIE_NSI.png     — SCIE | NSI side by side
                                               (+ compartment panel for All)
  scores_{dataset}_{subtype}.csv            — per-cell SCIE / NSI scores
  genes_used.csv                            — genes found / missing per signature
"""

import os, sys, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 8})

DATA_DIR = "/home/hbrunel/single_cell/Visvader"
SIG_DIR  = "/home/hbrunel/single_cell/axis_based signatures"
FIG_DIR  = os.path.join(DATA_DIR, "figures", "22_axis_signatures")

SUBTYPES   = ["ER", "HER2", "TNBC"]
SIGNATURES = ["SCIE", "NSI"]
DATASETS = {
    "Epithelial": lambda st: f"epithelial_{st}.h5ad",
    "All":        lambda st: f"all_{st.lower()}.h5ad",
}

# --er-filtered: ER only, re-integrated without 11 samples (23_reintegrate_ER_filtered.py)
ER_FILTERED = "--er-filtered" in sys.argv
if ER_FILTERED:
    SUBTYPES = ["ER"]
    DATASETS = {"Epithelial": lambda st: f"epithelial_{st}_filtered.h5ad"}
    FIG_DIR  = os.path.join(DATA_DIR, "figures", "22_axis_signatures_ER_filtered")
os.makedirs(FIG_DIR, exist_ok=True)
SIG_COLORS = {"SCIE": "#00B0F0", "NSI": "#FBA094"}
CMAPS = {sig: LinearSegmentedColormap.from_list(f"{sig}_cmap", ["#E6E6E6", col])
         for sig, col in SIG_COLORS.items()}
COMP_COLORS = {
    "Epithelial": "#E41A1C",
    "Immune":     "#377EB8",
    "Stromal":    "#4DAF4A",
    "Unassigned": "#CCCCCC",
}


def load_signature(sig, subtype):
    path = os.path.join(SIG_DIR, f"{sig}_{subtype}_consensus.csv")
    return pd.read_csv(path)["x"].dropna().astype(str).tolist()


def plot_score(ax, coords, values, cmap, title, label):
    vmin, vmax = np.nanpercentile(values, [1, 99])
    order = np.argsort(values)  # high-scoring cells on top
    p = ax.scatter(coords[order, 0], coords[order, 1], c=values[order],
                   cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.7,
                   linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=7); cbar.ax.tick_params(labelsize=6)


def plot_compartment(ax, coords, comp):
    for c, col in COMP_COLORS.items():
        m = (comp == c)
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], c=col, s=1.5, alpha=0.6,
                       linewidths=0, rasterized=True, label=f"{c} ({m.sum():,})")
    ax.set_title("Compartment", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=6, fontsize=7, frameon=False, loc="best")


genes_log = []

for dataset, fname in DATASETS.items():
    for subtype in SUBTYPES:
        h5ad = os.path.join(DATA_DIR, fname(subtype))
        if not os.path.exists(h5ad):
            print(f"Missing {h5ad}, skipping"); continue

        print(f"=== {dataset} / {subtype} ===")
        ad = sc.read_h5ad(h5ad)
        ad.X = ad.layers["logcounts"]
        coords = ad.obsm["X_umap"]

        for sig in SIGNATURES:
            genes = load_signature(sig, subtype)
            ok = [g for g in genes if g in ad.var_names]
            missing = [g for g in genes if g not in ad.var_names]
            genes_log.append({"dataset": dataset, "subtype": subtype, "signature": sig,
                              "n_genes": len(genes), "n_used": len(ok),
                              "missing": ";".join(missing)})
            print(f"  {sig}: {len(ok)}/{len(genes)} genes found"
                  + (f" (missing: {', '.join(missing)})" if missing else ""))
            sc.tl.score_genes(ad, ok, score_name=sig, use_raw=False, random_state=0)

        ad.obs[["sample", "compartment"] + SIGNATURES].to_csv(
            os.path.join(FIG_DIR, f"scores_{dataset}_{subtype}.csv"))

        # Individual UMAPs
        for sig in SIGNATURES:
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_score(ax, coords, ad.obs[sig].values.astype(float), CMAPS[sig],
                       f"{sig} — {subtype} ({dataset}, Harmony)", f"{sig} score")
            out = os.path.join(FIG_DIR, f"umap_{dataset}_{subtype}_{sig}.png")
            fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
            print(f"  Saved: {out}")

        # Combined panel
        n = len(SIGNATURES) + (1 if dataset == "All" else 0)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        for ax, sig in zip(axes, SIGNATURES):
            plot_score(ax, coords, ad.obs[sig].values.astype(float), CMAPS[sig],
                       f"{sig} — {subtype}", f"{sig} score")
        if dataset == "All":
            plot_compartment(axes[-1], coords, ad.obs["compartment"].astype(str).values)
        r = np.corrcoef(ad.obs["SCIE"], ad.obs["NSI"])[0, 1]
        fig.suptitle(f"{subtype} — {dataset} cells (n={ad.n_obs:,}) | "
                     f"Pearson r(SCIE, NSI) = {r:.2f}",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"umap_{dataset}_{subtype}_SCIE_NSI.png")
        fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
        print(f"  Saved: {out}")

        del ad; gc.collect()

pd.DataFrame(genes_log).to_csv(os.path.join(FIG_DIR, "genes_used.csv"), index=False)
print("Done.")
