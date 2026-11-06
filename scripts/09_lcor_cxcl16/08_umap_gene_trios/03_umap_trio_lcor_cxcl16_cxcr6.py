"""
03_umap_trio_lcor_cxcl16_cxcr6.py
====================================
UMAP feature plots for the LCOR / CXCL16 / CXCR6 trio, split by subtype
(ER+, TNBC). CXCR6 is a T-cell gene (near-absent in malignant cells, see
01_exploratorio) -- a meaningful combined view needs an embedding spanning
malignant AND immune cells together.

Visvader: all_global.h5ad (226,137 cells, Epithelial+Immune+Stromal+
Unassigned together on one UMAP) -- LCOR/CXCL16 signal shows in the
Epithelial region, CXCR6 in the Immune (T-cell) region.

Klughammer: malignant-cells-only atlas -- CXCR6 is expected to be flat/
near-zero everywhere here (no T cells present). Included for completeness
and as an explicit illustration of that limitation, not because a
meaningful CXCR6 pattern is expected.

Output: LCOR-CXCL16/08_umap_gene_trios/figures/03_umap_trio_lcor_cxcl16_cxcr6/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "LCOR-CXCL16/08_umap_gene_trios/figures/03_umap_trio_lcor_cxcl16_cxcr6"
GENES = ["LCOR", "CXCL16", "CXCR6"]
os.makedirs(OUTDIR, exist_ok=True)


def load_expr(name, h5ad_path, umap_key, ensembl_lookup):
    print(f"\n  Loading {name}: {h5ad_path} (umap key: {umap_key})...")
    adata = sc.read_h5ad(h5ad_path)
    if ensembl_lookup:
        sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
        ens_map = {g: (sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]) for g in GENES}
        raw_var_index = pd.Index(adata.raw.var_names)
        idx_of = {g: raw_var_index.get_loc(ens_map[g]) for g in GENES}
        def get_raw(g):
            col = adata.raw.X[:, idx_of[g]]
            return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
    else:
        idx_of = {g: adata.var_names.get_loc(g) for g in GENES}
        def get_raw(g):
            col = adata.layers["counts"][:, idx_of[g]]
            return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

    total_counts = adata.obs["total_counts"].values.astype(float)
    umap = adata.obsm[umap_key]
    df = pd.DataFrame({
        "UMAP1": umap[:, 0], "UMAP2": umap[:, 1],
        "subtype": adata.obs["subtype"].astype(str).values,
    })
    for g in GENES:
        df[g] = np.log1p(1e4 * get_raw(g) / total_counts)
    if "compartment" in adata.obs.columns:
        df["compartment"] = adata.obs["compartment"].astype(str).values
    del adata
    return df


def plot_trio(df, subtype, dataset_name, xlim, ylim, fname):
    d = df[df["subtype"] == subtype]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, gene in zip(axes, GENES):
        order = np.argsort(d[gene].values)
        sca = ax.scatter(d["UMAP1"].values[order], d["UMAP2"].values[order],
                          c=d[gene].values[order], cmap="viridis", s=3, linewidths=0)
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("log1p(CP10K)", fontsize=7.5)
        ax.set_title(gene, fontsize=12, fontweight="bold")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=8); ax.set_ylabel("UMAP2", fontsize=8)
    fig.suptitle(f"{dataset_name} -- {subtype} (n={len(d):,}) -- LCOR / CXCL16 / CXCR6",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTDIR}/{fname}.png  (n={len(d):,} cells)")


# --- Visvader: combined all-compartment atlas ---
df_v = load_expr("Visvader", "all_global.h5ad", "X_umap", ensembl_lookup=False)
xlim = (df_v["UMAP1"].min() - 0.5, df_v["UMAP1"].max() + 0.5)
ylim = (df_v["UMAP2"].min() - 0.5, df_v["UMAP2"].max() + 0.5)
for st in ["ER", "TNBC"]:
    plot_trio(df_v, st, "Visvader (all compartments)", xlim, ylim, f"umap_trio_Visvader_{st}")

# Compartment-context figure (where is Epithelial vs Immune on this UMAP)
if "compartment" in df_v.columns:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    comp_colors = {"Epithelial": "#377EB8", "Immune": "#E41A1C", "Stromal": "#4DAF4A", "Unassigned": "#dddddd"}
    for comp, color in comp_colors.items():
        d = df_v[df_v["compartment"] == comp]
        ax.scatter(d["UMAP1"], d["UMAP2"], c=color, s=2, linewidths=0, label=f"{comp} (n={len(d):,})")
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Visvader -- compartment context for the trio above", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/umap_compartment_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTDIR}/umap_compartment_context.png")

# --- Klughammer: malignant-only (CXCR6 not expected to show signal) ---
df_k = load_expr("Klughammer", "/home/hbrunel/single_cell/klughammer_malignant_processed.h5ad",
                  "X_umap_Harmony", ensembl_lookup=True)
xlim_k = (df_k["UMAP1"].min() - 0.5, df_k["UMAP1"].max() + 0.5)
ylim_k = (df_k["UMAP2"].min() - 0.5, df_k["UMAP2"].max() + 0.5)
for st in ["ER", "TNBC"]:
    plot_trio(df_k, st, "Klughammer (malignant cells only -- CXCR6 not expected here)",
              xlim_k, ylim_k, f"umap_trio_Klughammer_{st}")

print("\nDONE")
