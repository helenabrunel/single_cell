"""
22e_stem_signatures_umap.py
===========================
Per subtype (Harmony-integrated epithelial objects): UMAPs coloured by
  - fMaSC_over_aMaSC
  - LIM_MAMMARY_STEM_CELL_UP
(gene lists from signatures_confirmed.json; sc.tl.score_genes on logcounts).

Outputs: figures/22_axis_signatures/phenotype/ (copied to axis_based signatures/UMAPs/)
  umap_{subtype}_{fMaSC|LIM_MaSC}.png, umap_{subtype}_stem_2panels.png,
  stem_scores_{subtype}.csv, stem_genes_used.csv
"""

import os, sys, json, shutil, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt

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
with open(os.path.join(DATA_DIR, "signatures_confirmed.json")) as f:
    SIGS = json.load(f)
STEM = {  # short name -> (json key, title)
    "fMaSC":    ("fMaSC_over_aMaSC", "fMaSC"),
    "LIM_MaSC": ("LIM_MAMMARY_STEM_CELL_UP", "LIM mammary stem cell"),
}
CMAP = "magma_r"


def plot_continuous(ax, coords, values, title, label):
    vmin, vmax = np.nanpercentile(values, [1, 99])
    order = np.argsort(values)
    p = ax.scatter(coords[order, 0], coords[order, 1], c=values[order], cmap=CMAP,
                   vmin=vmin, vmax=vmax, s=2, alpha=0.8, linewidths=0, rasterized=True)
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticks([]); ax.set_yticks([])


genes_log, outputs = [], []

for subtype in SUBTYPES:
    print(f"=== {subtype} ===")
    a = sc.read_h5ad(os.path.join(DATA_DIR, EPI_H5AD.format(subtype)))
    a.X = a.layers["logcounts"]
    coords = np.asarray(a.obsm["X_umap"])

    for name, (key, _) in STEM.items():
        genes = SIGS[key]
        ok = [g for g in genes if g in a.var_names]
        genes_log.append({"subtype": subtype, "signature": key,
                          "n_genes": len(genes), "n_used": len(ok)})
        print(f"  {key}: {len(ok)}/{len(genes)} genes found")
        sc.tl.score_genes(a, ok, score_name=name, use_raw=False, random_state=0)

    a.obs[["sample"] + list(STEM)].to_csv(os.path.join(FIG_DIR, f"stem_scores_{subtype}.csv"))

    for name, (_, title) in STEM.items():
        fig, ax = plt.subplots(figsize=(7, 6.5))
        plot_continuous(ax, coords, a.obs[name].values.astype(float),
                        f"{subtype} — {title}", f"{title} score")
        out = os.path.join(FIG_DIR, f"umap_{subtype}_{name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
        outputs.append(out)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (name, (_, title)) in zip(axes, STEM.items()):
        plot_continuous(ax, coords, a.obs[name].values.astype(float),
                        f"{subtype} — {title}", f"{title} score")
    r = np.corrcoef(a.obs["fMaSC"], a.obs["LIM_MaSC"])[0, 1]
    fig.suptitle(f"{subtype} — epithelial cells, Harmony UMAP (n={a.n_obs:,}) | "
                 f"Pearson r = {r:.2f}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"umap_{subtype}_stem_2panels.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    outputs.append(out)

    del a; gc.collect()

out = os.path.join(FIG_DIR, "stem_genes_used.csv")
pd.DataFrame(genes_log).to_csv(out, index=False)
outputs.append(out)

for f in outputs:
    shutil.copy(f, COPY_DIR)
print("Done.")
