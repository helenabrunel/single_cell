"""
04a_export_for_decontx.py
=========================
Export raw counts of the FULL Klughammer atlas, one HDF5 file per sample,
for ambient-RNA correction with decontX (04b_decontx.R). All cells of a
sample are needed so the ambient profile reflects every cell type (e.g.
NK cells, the main TRDC source).

Each file holds the cells × genes CSR arrays (= genes × cells CSC in R),
author cell types (decontX clusters, z) and a T/NK flag.

Input:  ../data/dataset.h5ad
Output: data/decontx/in/<sampleid>.h5, data/decontx/genes.csv

Run from: ~/single_cell/Klughammer/gdT/
  python3 04a_export_for_decontx.py
"""

import anndata as ad
import numpy as np
import pandas as pd
import h5py, os
from scipy import sparse

OUT = "data/decontx/in"
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    a = ad.read_h5ad("../data/dataset.h5ad", backed="r")
    genes = pd.DataFrame({"gene_id": a.raw.var.index,
                          "symbol": a.var.loc[a.raw.var.index, "feature_name"].astype(str).values})
    genes.to_csv("data/decontx/genes.csv", index=False)
    for s in a.obs["sampleid"].cat.categories:
        idx = np.where((a.obs["sampleid"] == s).values)[0]
        X = sparse.csr_matrix(a.raw.X[idx]).astype(np.float64)
        X.sort_indices()
        o = a.obs.iloc[idx]
        with h5py.File(f"{OUT}/{s}.h5", "w") as f:
            f["data"] = X.data
            f["indices"] = X.indices.astype(np.int32)
            f["indptr"] = X.indptr.astype(np.int64)
            f["shape"] = np.array(X.shape)                       # cells, genes
            f["cells"] = o.index.astype(str).values.astype("S")
            f["z"] = o["author_cell_type"].astype(str).values.astype("S")
            f["tnk"] = o["author_cell_type"].isin(["T", "NK"]).values.astype(np.int8)
        print(f"{s}: {X.shape[0]} cells, {int(o['author_cell_type'].isin(['T','NK']).sum())} T/NK")
