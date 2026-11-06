"""
05_export_for_report.py
=======================
Export lightweight tables (UMAP coordinates, metadata, expression of a few
genes) from the h5ad objects so the R Markdown report
(report/azizi_gdT_report.Rmd) can be rendered with plain R/ggplot2,
without reading h5ad from R.

Input:  data/azizi_{10x,indrop}_{all,TNK,gdT}.h5ad
Output: report/data/*.csv.gz

Run from: ~/single_cell/Azizi/
  python3 05_export_for_report.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from scipy.sparse import issparse

OUT = "report/data"
os.makedirs(OUT, exist_ok=True)

GENES_TNK = ["TRDC", "TRDV1", "TRDV2", "TRDV3", "TRGV9", "TRGC1", "TRGC2", "TRAC", "CD3E", "CD8A", "CD4",
             "NKG7", "KLRF1", "FCGR3A"]
GENES_GD = ["TRDC", "TRDV1", "TRDV2", "TRDV3", "TRGV9", "TRGC1", "TRGC2",
            "CD3E", "CD8A", "CD8B", "CD4",
            # cytotoxicity / NK receptors
            "GZMA", "GZMB", "GZMK", "PRF1", "GNLY", "NKG7", "FCGR3A", "KLRD1",
            "KLRC1", "KLRK1", "NCR1", "KIR2DL4",
            # cytokines / TFs
            "IFNG", "TNF", "IL17A", "RORC", "TBX21", "EOMES", "ZBTB16",
            # residency / exhaustion / memory
            "ITGAE", "CD69", "CXCR6", "PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX",
            "CTLA4", "ENTPD1", "CCR7", "SELL", "IL7R", "TCF7", "KLF2", "S1PR1",
            "MKI67"]
OBS_COMMON = ["patient", "tissue", "sample", "leiden", "lineage", "gd_call",
              "gd_subset", "Vd_family", "is_gdT", "gd_cluster_frac",
              "stcat_label", "stcat_l0_cell", "stcat_P_Tgd", "consensus_gd"]
OBS_10X = ["has_TRA_productive", "has_TRB_productive", "has_clonotype",
           "has_tcr_contig", "TRD_v"]


def expr(adata, genes):
    g = [x for x in genes if x in adata.var_names]
    X = adata[:, g].X
    X = X.toarray() if issparse(X) else np.asarray(X)
    return pd.DataFrame(X, index=adata.obs_names, columns=g).round(3)


def raw_counts(adata, genes):
    g = [x for x in genes if x in adata.var_names]
    X = adata[:, g].layers["counts"]
    X = X.toarray() if issparse(X) else np.asarray(X)
    return pd.DataFrame(X, index=adata.obs_names, columns=[f"{x}_counts" for x in g])


def export(adata, name, genes, extra_obs=()):
    cols = [c for c in list(OBS_COMMON) + list(extra_obs) if c in adata.obs]
    df = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names,
                      columns=["UMAP1", "UMAP2"])
    df = df.join(adata.obs[cols])
    if genes:
        df = df.join(expr(adata, genes))
    df.index.name = "cell"
    df.to_csv(f"{OUT}/{name}.csv.gz")
    print(f"  {name}: {df.shape}")


if __name__ == "__main__":
    for plat in ["10x", "indrop"]:
        extra = OBS_10X if plat == "10x" else ()
        a = sc.read_h5ad(f"data/azizi_{plat}_all.h5ad")
        export(a, f"{plat}_all", ["CD3E", "PTPRC", "LYZ", "CD79A", "NKG7", "TRDC"])
        a = sc.read_h5ad(f"data/azizi_{plat}_TNK.h5ad")
        export(a, f"{plat}_TNK", GENES_TNK, extra)
        rc = raw_counts(a, ["TRDC", "TRAC", "TRBC1", "TRBC2", "CD3D", "CD3E", "CD3G"])
        rc.index.name = "cell"
        rc.to_csv(f"{OUT}/{plat}_TNK_counts.csv.gz")
        g = sc.read_h5ad(f"data/azizi_{plat}_gdT.h5ad")
        g.obs["gd_leiden"] = g.obs["gd_leiden"]
        export(g, f"{plat}_gdT", GENES_GD, list(extra) + ["gd_leiden"])
    print("Done.")
