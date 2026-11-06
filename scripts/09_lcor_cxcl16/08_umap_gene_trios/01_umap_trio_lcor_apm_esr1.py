"""
01_umap_trio_lcor_apm_esr1.py
===============================
Standard single-cell UMAP feature plots for the LCOR / APM-score / ESR1
trio, split by subtype (ER+, TNBC), in both Visvader and Klughammer
(independent atlas). One 1x3 panel figure per dataset x subtype
combination (4 figures total): each cell is a point on that dataset's
precomputed UMAP, colored by that gene's expression.

EXPRESSION METRIC
------------------
log1p(CP10K) -- 1e4 * raw_count / cell_total_counts, log1p-transformed --
computed directly from raw counts for consistency with every other script
in this project (rather than relying on each dataset's own precomputed
normalization, which differs in method between Visvader/scanpy and
Klughammer/Seurat). APM score uses the same 9-gene meta-gene sum as
07_apm_positive_control (TAP1, TAP2, TAPBP, HLA-A/B/C, PSMB8, PSMB9, B2M).

UMAP EMBEDDING
--------------
Visvader: X_umap from compartment_epithelial.h5ad (malignant cells only).
Klughammer: X_umap_Harmony from klughammer_malignant_processed.h5ad
(batch-corrected embedding, most appropriate for a multi-donor atlas).

Output: LCOR-CXCL16/08_umap_gene_trios/figures/01_umap_trio_lcor_apm_esr1/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "LCOR-CXCL16/08_umap_gene_trios/figures/01_umap_trio_lcor_apm_esr1"
APM_GENES = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
GENES = ["LCOR", "APM", "ESR1"]
os.makedirs(OUTDIR, exist_ok=True)


def load_expr(name, h5ad_path, umap_key, sample_col, ensembl_lookup):
    print(f"\n  Loading {name}: {h5ad_path} (umap key: {umap_key})...")
    adata = sc.read_h5ad(h5ad_path)
    needed = ["LCOR", "ESR1"] + APM_GENES
    if ensembl_lookup:
        sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
        ens_map = {g: (sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]) for g in needed}
        raw_var_index = pd.Index(adata.raw.var_names)
        idx_of = {g: raw_var_index.get_loc(ens_map[g]) for g in needed}
        def get_raw(g):
            col = adata.raw.X[:, idx_of[g]]
            return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
    else:
        idx_of = {g: adata.var_names.get_loc(g) for g in needed}
        def get_raw(g):
            col = adata.layers["counts"][:, idx_of[g]]
            return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

    total_counts = adata.obs["total_counts"].values.astype(float)
    raw_lcor = get_raw("LCOR")
    raw_esr1 = get_raw("ESR1")
    raw_apm = np.column_stack([get_raw(g) for g in APM_GENES]).sum(axis=1)

    umap = adata.obsm[umap_key]
    df = pd.DataFrame({
        "UMAP1": umap[:, 0], "UMAP2": umap[:, 1],
        "LCOR": np.log1p(1e4 * raw_lcor / total_counts),
        "APM": np.log1p(1e4 * raw_apm / total_counts),
        "ESR1": np.log1p(1e4 * raw_esr1 / total_counts),
        "subtype": adata.obs["subtype"].astype(str).values,
    })
    del adata
    return df


def plot_trio(df, subtype, dataset_name, xlim, ylim, fname):
    d = df[df["subtype"] == subtype]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, gene in zip(axes, GENES):
        order = np.argsort(d[gene].values)  # plot high-expressors on top
        sca = ax.scatter(d["UMAP1"].values[order], d["UMAP2"].values[order],
                          c=d[gene].values[order], cmap="viridis", s=4, linewidths=0)
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("log1p(CP10K)", fontsize=7.5)
        ax.set_title(gene, fontsize=12, fontweight="bold")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=8); ax.set_ylabel("UMAP2", fontsize=8)
    fig.suptitle(f"{dataset_name} -- {subtype} malignant cells (n={len(d):,}) -- LCOR / APM / ESR1",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTDIR}/{fname}.png  (n={len(d):,} cells)")


for dataset_name, h5ad_path, umap_key, sample_col, ens_lookup, subtypes in [
    ("Visvader", "compartment_epithelial.h5ad", "X_umap", "sample", False, ["ER", "TNBC"]),
    ("Klughammer", "/home/hbrunel/single_cell/klughammer_malignant_processed.h5ad", "X_umap_Harmony", "donor_id", True, ["ER", "TNBC"]),
]:
    df = load_expr(dataset_name, h5ad_path, umap_key, sample_col, ens_lookup)
    xlim = (df["UMAP1"].min() - 0.5, df["UMAP1"].max() + 0.5)
    ylim = (df["UMAP2"].min() - 0.5, df["UMAP2"].max() + 0.5)
    for st in subtypes:
        plot_trio(df, st, dataset_name, xlim, ylim, f"umap_trio_{dataset_name}_{st}")

print("\nDONE")
