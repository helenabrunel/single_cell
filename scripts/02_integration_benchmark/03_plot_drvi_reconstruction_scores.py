"""
part0_drvi_reconstruction_scores.py
=====================================
Plot DRVI latent dimension statistics (reconstruction scores) for every
per-subtype model in figures/part0/.

Workflow per model
------------------
1. Detect dataset / subtype from directory name
2. Load HVG gene list from {model_dir}_hvg_genes.txt
3. Load source h5ad, filter to correct compartment + subtype
4. Load DRVI model, set up AnnData with same HVG genes
5. get_latent_representation(return_anndata=True)
6. set_latent_dimension_stats → embed.var populated
7. plot_latent_dimension_stats(show=False) → save PNG

Outputs  →  figures/part0/
  {Dataset}_{Subtype}_drvi_reconstruction_scores.png
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, gc, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from drvi.model import DRVI
from drvi.utils import tl as drvi_tl
from drvi.utils import pl as drvi_pl

OUTDIR = "figures/part0"

# ── dataset source configs (mirrors part0_malignant_integration.py) ────────────
DATASET_CFG = {
    "Visvader": {
        "path"             : "Visvader/compartment_epithelial.h5ad",
        "compartment_col"  : None,
        "compartment_value": None,
        "batch_col"        : "sample",
        "subtype_col"      : "subtype",
        "count_layer"      : "counts",
    },
    "Klughammer": {
        "path"             : "Klughammer/klughammer_prepared.h5ad",
        "compartment_col"  : "compartments",
        "compartment_value": "Malignant",
        "batch_col"        : "sampleid",
        "subtype_col"      : "subtype",
        "count_layer"      : None,
    },
}
MIN_CELLS = 500

# ── discover available per-subtype models ──────────────────────────────────────
models = []
for fname in sorted(os.listdir(OUTDIR)):
    # pattern: {Dataset}_{Subtype}_drvi_model  (directory)
    if not fname.endswith("_drvi_model"):
        continue
    model_dir = os.path.join(OUTDIR, fname)
    if not os.path.isdir(model_dir):
        continue
    hvg_file = model_dir + "_hvg_genes.txt"
    if not os.path.exists(hvg_file):
        print(f"  Skipping {fname}: no HVG gene file found")
        continue

    # parse Dataset_Subtype
    stem = fname[: -len("_drvi_model")]
    parts = stem.split("_")
    if len(parts) < 2:
        continue
    dataset_name = parts[0]
    subtype      = parts[1]
    if dataset_name not in DATASET_CFG:
        continue

    models.append((dataset_name, subtype, model_dir, hvg_file))

print(f"Found {len(models)} per-subtype DRVI models:")
for ds, st, mdir, _ in models:
    print(f"  {ds} {st} → {mdir}")

# ── process each model ────────────────────────────────────────────────────────
for dataset_name, subtype, model_dir, hvg_file in models:
    print(f"\n{'─'*60}")
    print(f" {dataset_name}  ·  {subtype}")
    print(f"{'─'*60}")

    out_png = os.path.join(OUTDIR, f"{dataset_name}_{subtype}_drvi_reconstruction_scores.png")

    cfg = DATASET_CFG[dataset_name]

    # Load HVG gene list
    with open(hvg_file) as f:
        hvg_genes = [l.strip() for l in f.readlines() if l.strip()]
    print(f"  HVG genes: {len(hvg_genes)}")

    # Load and subset data
    print(f"  Loading {cfg['path']} ...")
    adata = sc.read_h5ad(cfg["path"])

    if cfg["compartment_col"]:
        adata = adata[adata.obs[cfg["compartment_col"]] == cfg["compartment_value"]].copy()

    # QC filter (same as integration script)
    counts = adata.obs[cfg["batch_col"]].value_counts()
    keep   = counts[counts >= MIN_CELLS].index
    adata  = adata[adata.obs[cfg["batch_col"]].isin(keep)].copy()

    # Subset to this subtype
    adata = adata[adata.obs[cfg["subtype_col"]] == subtype].copy()
    print(f"  {adata.n_obs:,} cells | {adata.obs[cfg['batch_col']].nunique()} patients")

    # Filter to HVG genes present in this dataset
    present = [g for g in hvg_genes if g in adata.var_names]
    if len(present) < len(hvg_genes):
        print(f"  ⚠  {len(hvg_genes) - len(present)} HVG genes missing from dataset")
    ad_hvg = adata[:, present].copy()

    # Set up AnnData for DRVI
    has_counts = cfg["count_layer"] is not None
    if has_counts:
        DRVI.setup_anndata(ad_hvg, layer=cfg["count_layer"],
                           batch_key=cfg["batch_col"], is_count_data=True)
    else:
        DRVI.setup_anndata(ad_hvg, batch_key=cfg["batch_col"], is_count_data=False)

    # Load model
    print(f"  Loading model: {model_dir}")
    try:
        model = DRVI.load(model_dir, adata=ad_hvg)
    except Exception as e:
        print(f"  ⚠  Could not load model: {e}")
        del adata, ad_hvg; gc.collect()
        continue

    # Get latent representation and wrap as AnnData manually
    print("  Computing latent representation ...")
    Z     = model.get_latent_representation(ad_hvg)   # (n_cells, n_latent)
    embed = ad.AnnData(
        X   = Z,
        obs = ad_hvg.obs.copy(),
        var = pd.DataFrame(index=[f"dim_{i}" for i in range(Z.shape[1])]),
    )

    # Compute reconstruction stats
    print("  Computing latent dimension stats ...")
    drvi_tl.set_latent_dimension_stats(model, embed, inplace=True)

    n_active   = (~embed.var["vanished"]).sum()
    n_vanished = embed.var["vanished"].sum()
    print(f"  Active dims: {n_active}  |  Vanished: {n_vanished}")

    # Plot
    fig = drvi_pl.plot_latent_dimension_stats(embed, show=False,
                                              figsize=(5, 3), ncols=4)
    fig.suptitle(
        f"{dataset_name}  ·  {subtype}  —  DRVI latent dimension statistics\n"
        f"Active: {n_active}  |  Vanished: {n_vanished}  |  "
        f"Total dims: {embed.n_vars}",
        fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_png}")

    del adata, ad_hvg, embed, model; gc.collect()

print(f"\n{'═'*60}")
print(" DONE")
print(f"{'═'*60}")
