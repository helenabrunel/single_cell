"""
10a_export_for_landscent.py
===========================
Exports logcounts matrices and metadata from the epithelial h5ad files
into Matrix Market format for reading by the R LandSCENT scoring script.

Run before: Rscript 10_landscent_scoring.R
"""

import os
import anndata as ad
import scipy.io
import pandas as pd

DATA_DIR   = "/home/hbrunel/single_cell/Visvader"
EXPORT_DIR = "/home/hbrunel/single_cell/landscent_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

SUBTYPES = ["ER", "HER2", "TNBC"]

for subtype in SUBTYPES:
    mtx_path   = os.path.join(EXPORT_DIR, f"logcounts_{subtype}.mtx")
    genes_path = os.path.join(EXPORT_DIR, f"genes_{subtype}.txt")
    cells_path = os.path.join(EXPORT_DIR, f"cells_{subtype}.csv")

    if all(os.path.exists(p) for p in [mtx_path, genes_path, cells_path]):
        print(f"{subtype}: already exported — skipping")
        continue

    print(f"\n{subtype}: reading h5ad ...")
    adata = ad.read_h5ad(os.path.join(DATA_DIR, f"epithelial_{subtype}.h5ad"))
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    X = adata.layers["counts"]   # raw counts, cells × genes sparse CSR

    # Write as genes × cells (transpose) in Matrix Market format
    print(f"  Writing sparse matrix ...")
    scipy.io.mmwrite(mtx_path, X.T)

    # Gene symbols (one per line, order matches matrix rows)
    symbols = adata.var["symbol"].values
    pd.Series(symbols).to_csv(genes_path, index=False, header=False)

    # Cell metadata
    adata.obs[["sample", "subtype"]].to_csv(cells_path)

    print(f"  Done: {os.path.getsize(mtx_path)/1e6:.0f} MB matrix")

print("\nExport complete. Run: Rscript Visvader/10_landscent_scoring.R")
