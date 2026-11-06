"""
02_process_and_TNK.py
=====================
Per platform (10x / inDrop, kept separate):
  1. Normalise (1e4 + log1p), HVG (seurat_v3 on counts, batch-aware),
     PCA, Harmony (batch = sample), UMAP, Leiden
  2. Broad lineage annotation per Leiden cluster from marker scores
     (T, NK, B, Plasma, Myeloid, pDC, Mast, Epithelial/Stromal)
  3. Subset the T/NK compartment (γδ cells can sit next to NK cells, so
     NK clusters are kept) and re-integrate it

Output: data/azizi_{10x,indrop}_all.h5ad   (all cells, annotated)
        data/azizi_{10x,indrop}_TNK.h5ad   (T/NK compartment)
        figures/02_process/

Run from: ~/single_cell/Azizi/
  python3 02_process_and_TNK.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, gc, warnings
import harmonypy as hm

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1
sc.settings.n_jobs = 8

OUTDIR = "figures/02_process"
os.makedirs(OUTDIR, exist_ok=True)
sc.settings.figdir = OUTDIR

PLATFORMS = ["10x", "indrop"]
PARAMS = {
    "all": {"n_hvg": 3000, "n_pcs": 40, "res": 1.0},
    "TNK": {"n_hvg": 2000, "n_pcs": 30, "res": 1.0},
}

LINEAGE = {
    "T"         : ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRDC"],
    "NK"        : ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1", "FCGR3A"],
    "B"         : ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1"],
    "Plasma"    : ["JCHAIN", "MZB1", "IGHG1", "IGKC", "XBP1"],
    "Myeloid"   : ["LYZ", "CD14", "CD68", "FCER1G", "TYROBP", "CST3", "AIF1"],
    "pDC"       : ["LILRA4", "IL3RA", "CLEC4C", "TCF4"],
    "Mast"      : ["TPSAB1", "TPSB2", "CPA3", "KIT"],
    "Epi_Stroma": ["EPCAM", "KRT8", "KRT18", "COL1A1", "DCN", "PECAM1"],
}
GD_GENES = ["TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "TRDV3", "TRGV9",
            "TRAC", "TRBC1", "TRBC2", "CD3E", "CD8A", "CD4", "NKG7"]


def integrate(adata, p, batch="sample"):
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=p["n_hvg"], flavor="seurat_v3",
                                    layer="counts", batch_key=batch, span=0.5)
    except ValueError:   # loess fails on very small inDrop libraries
        print("  seurat_v3 HVG failed → falling back to flavor='seurat'")
        sc.pp.highly_variable_genes(adata, n_top_genes=p["n_hvg"], flavor="seurat",
                                    batch_key=batch)
    # keep TCR V/C genes out of HVGs so clonality does not drive clustering
    tcr = adata.var_names.str.match(r"^TR[ABDG][VJC]")
    adata.var.loc[tcr, "highly_variable"] = False
    sc.pp.pca(adata, n_comps=p["n_pcs"], mask_var="highly_variable")
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch, max_iter_harmony=20,
                        verbose=False)
    Z = np.asarray(ho.Z_corr)
    adata.obsm["X_pca_harmony"] = Z if Z.shape[0] == adata.n_obs else Z.T
    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=p["res"], flavor="igraph", n_iterations=2)
    return adata


def annotate_lineage(adata):
    for lin, genes in LINEAGE.items():
        g = [x for x in genes if x in adata.var_names]
        sc.tl.score_genes(adata, g, score_name=f"score_{lin}")
    cols = [f"score_{l}" for l in LINEAGE]
    cl = adata.obs.groupby("leiden", observed=True)[cols].mean()
    # absolute (control-subtracted) scores: z-scoring across clusters breaks
    # when one lineage dominates (e.g. the 10x set is ~100% T cells)
    lab = cl.idxmax(axis=1).str.replace("score_", "")
    adata.obs["lineage"] = adata.obs["leiden"].map(lab).astype("category")
    cl.assign(lineage=lab).to_csv(f"{OUTDIR}/cluster_lineage_scores_{adata.uns['name']}.csv")
    return adata


def plots(adata, tag):
    name = adata.uns["name"]
    color = ["sample", "patient", "tissue", "leiden", "lineage"]
    color = [c for c in color if c in adata.obs]
    sc.pl.umap(adata, color=color, ncols=3, wspace=0.4, show=False, frameon=False)
    plt.savefig(f"{OUTDIR}/umap_{name}_{tag}_meta.png", dpi=150, bbox_inches="tight")
    plt.close()
    g = [x for x in GD_GENES if x in adata.var_names]
    sc.pl.umap(adata, color=g, ncols=5, cmap="Reds", show=False, frameon=False,
               vmax="p99.5")
    plt.savefig(f"{OUTDIR}/umap_{name}_{tag}_gdgenes.png", dpi=150, bbox_inches="tight")
    plt.close()
    allg = [x for gs in LINEAGE.values() for x in gs if x in adata.var_names]
    sc.pl.dotplot(adata, allg, groupby="leiden", standard_scale="var", show=False)
    plt.savefig(f"{OUTDIR}/dotplot_{name}_{tag}_lineage.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for plat in PLATFORMS:
        print(f"══ {plat} ══")
        adata = sc.read_h5ad(f"data/azizi_{plat}_raw.h5ad")
        adata.uns["name"] = plat
        adata = integrate(adata, PARAMS["all"])
        adata = annotate_lineage(adata)
        plots(adata, "all")
        print(adata.obs["lineage"].value_counts().to_string())
        adata.write_h5ad(f"data/azizi_{plat}_all.h5ad", compression="gzip")

        tnk = adata[adata.obs["lineage"].isin(["T", "NK"])].copy()
        del adata; gc.collect()
        tnk.uns["name"] = f"{plat}_TNK"
        for k in ["X_pca", "X_pca_harmony", "X_umap"]:
            tnk.obsm.pop(k, None)
        tnk = integrate(tnk, PARAMS["TNK"])
        tnk.obs["lineage_all"] = tnk.obs["lineage"]
        tnk = annotate_lineage(tnk)
        plots(tnk, "TNK")
        print(f"  T/NK compartment: {tnk.n_obs} cells")
        tnk.write_h5ad(f"data/azizi_{plat}_TNK.h5ad", compression="gzip")
        del tnk; gc.collect()
    print("Done.")
