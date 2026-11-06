"""
02_process_TNK.py
=================
Normalise and embed the Klughammer T/NK compartment (author labels "T" and
"NK"): 1e4 + log1p, HVG (seurat_v3 on counts, batch-aware), PCA, Harmony
(batch = patient), UMAP, Leiden. TCR V/J/C genes are excluded from HVGs.

Patients with < 20 T/NK cells are kept in the object but excluded from the
HVG batch computation (loess fails on tiny batches).

Input:  data/klughammer_TNK_raw.h5ad
Output: data/klughammer_TNK.h5ad, figures/02_process/

Run from: ~/single_cell/Klughammer/gdT/
  python3 02_process_TNK.py
"""

import scanpy as sc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
import harmonypy as hm

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1
sc.settings.n_jobs = 8

OUTDIR = "figures/02_process"
os.makedirs(OUTDIR, exist_ok=True)
N_HVG, N_PCS, RES = 2000, 30, 1.0
MARKERS = ["CD3E", "CD3D", "CD4", "CD8A", "CD8B", "FOXP3", "TRDC", "TRGC1", "TRGC2",
           "KLRF1", "FCER1G", "NCAM1", "FCGR3A", "NKG7", "GZMB", "MKI67"]


if __name__ == "__main__":
    a = sc.read_h5ad("data/klughammer_TNK_raw.h5ad")
    a.X = a.layers["counts"].copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    big = a.obs["patient"].value_counts()
    big = big[big >= 20].index
    h = a[a.obs["patient"].isin(big)].copy()
    try:
        sc.pp.highly_variable_genes(h, n_top_genes=N_HVG, flavor="seurat_v3", layer="counts",
                                    batch_key="patient", span=0.5)
    except ValueError:   # loess fails on some small/sparse patients
        print("  seurat_v3 HVG failed → falling back to flavor='seurat'")
        sc.pp.highly_variable_genes(h, n_top_genes=N_HVG, flavor="seurat", batch_key="patient")
    a.var["highly_variable"] = h.var["highly_variable"].reindex(a.var_names).fillna(False)
    a.var.loc[a.var_names.str.match(r"^TR[ABDG][VJC]"), "highly_variable"] = False
    del h

    sc.pp.pca(a, n_comps=N_PCS, mask_var="highly_variable")
    ho = hm.run_harmony(a.obsm["X_pca"], a.obs, "patient", max_iter_harmony=20, verbose=False)
    Z = np.asarray(ho.Z_corr)
    a.obsm["X_pca_harmony"] = Z if Z.shape[0] == a.n_obs else Z.T
    sc.pp.neighbors(a, use_rep="X_pca_harmony")
    sc.tl.umap(a)
    sc.tl.leiden(a, resolution=RES, flavor="igraph", n_iterations=2)

    sc.pl.umap(a, color=["author_cell_type", "patient", "assay", "site_biopsy", "subtype", "leiden"],
               ncols=3, wspace=0.4, show=False, frameon=False, legend_fontsize=6)
    plt.savefig(f"{OUTDIR}/umap_TNK_meta.png", dpi=150, bbox_inches="tight"); plt.close()
    sc.pl.umap(a, color=MARKERS, ncols=4, cmap="Reds", vmax="p99.5", show=False, frameon=False)
    plt.savefig(f"{OUTDIR}/umap_TNK_markers.png", dpi=150, bbox_inches="tight"); plt.close()
    sc.pl.dotplot(a, MARKERS, groupby="leiden", standard_scale="var", show=False)
    plt.savefig(f"{OUTDIR}/dotplot_TNK_leiden.png", dpi=150, bbox_inches="tight"); plt.close()
    print(a)
    a.write_h5ad("data/klughammer_TNK.h5ad", compression="gzip")
