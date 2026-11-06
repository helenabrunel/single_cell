"""
01_export_reference.py
======================
Prepares the Python-side reference for transferring Klughammer and Regner to
R/Seurat WITHOUT converting any object (no SeuratDisk / sceasy / zellkonverter).

Strategy
--------
The count matrices are NOT rewritten here. R reads the raw counts directly
from the original source (02_import_counts.R):
  - Klughammer : raw/X CSR arrays inside the original CELLxGENE h5ad
                 (Klughammer/data/dataset.h5ad; .X is log-normalised, .raw.X
                 holds the integer UMI counts)
  - Regner     : the original GEO 10x mtx files (GSE243526_RNA/)

This script only exports, per dataset:
  obs.csv.gz        cell metadata (index = cell barcode, in matrix order)
  var.csv.gz        gene table (ensembl_id, symbol, ...) in matrix order
  cell_checks.csv.gz per-cell total UMIs and n detected genes
  gene_checks.csv.gz per-gene total UMIs and n cells detected
  entry_checks.csv.gz 5000 random non-zero entries (cell, gene, value)
  summary.json      dims, nnz, grand total

R then rebuilds the matrix independently and must reproduce every check
exactly (integer equality, no tolerance).

Run from: ~/single_cell/
  python seurat_transfer/01_export_reference.py [klughammer|regner|both]
"""

import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT = "seurat_transfer"
RNG = np.random.default_rng(20261013)
N_ENTRIES = 5000


def export(name, X, obs, var):
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    X = sp.csr_matrix(X)                       # cells x genes
    assert X.shape == (obs.shape[0], var.shape[0])

    data = X.data
    assert np.all(data == np.round(data)), f"{name}: non-integer counts"
    assert data.min() >= 0, f"{name}: negative counts"
    # Exact integer arithmetic: float32 sums lose precision above 2^24
    # (e.g. MALAT1 total in Klughammer was off by 4 UMIs)
    X = sp.csr_matrix((data.astype(np.int64), X.indices, X.indptr), shape=X.shape)
    data = X.data
    assert obs.index.is_unique, f"{name}: duplicated cell barcodes"

    obs = obs.copy()
    for c in obs.columns:                      # categoricals -> plain strings
        if isinstance(obs[c].dtype, pd.CategoricalDtype):
            obs[c] = obs[c].astype(str).replace("nan", np.nan)
    obs.index.name = "cell"
    obs.to_csv(os.path.join(out, "obs.csv.gz"))
    var.index.name = "row_id"
    var.to_csv(os.path.join(out, "var.csv.gz"))

    pd.DataFrame({
        "cell": obs.index,
        "total": np.asarray(X.sum(1)).ravel().astype(np.int64),
        "n_genes": np.diff(X.indptr),
    }).to_csv(os.path.join(out, "cell_checks.csv.gz"), index=False)

    Xc = X.tocsc()
    pd.DataFrame({
        "row_id": var.index,
        "total": np.asarray(Xc.sum(0)).ravel().astype(np.int64),
        "n_cells": np.diff(Xc.indptr),
    }).to_csv(os.path.join(out, "gene_checks.csv.gz"), index=False)
    del Xc

    k = RNG.choice(X.nnz, N_ENTRIES, replace=False)
    rows = np.searchsorted(X.indptr, k, side="right") - 1
    pd.DataFrame({
        "cell_idx": rows + 1,                  # 1-based for R
        "gene_idx": X.indices[k] + 1,
        "value": X.data[k].astype(np.int64),
    }).to_csv(os.path.join(out, "entry_checks.csv.gz"), index=False)

    summary = {"n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]),
               "nnz": int(X.nnz), "grand_total": int(data.sum(dtype=np.int64))}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {name}: {summary}")


which = sys.argv[1] if len(sys.argv) > 1 else "both"

# ── Klughammer ────────────────────────────────────────────────────────────────
if which in ("klughammer", "both"):
    print("Klughammer …")
    a = ad.read_h5ad("Klughammer/data/dataset.h5ad")
    assert (a.raw.var_names == a.var_names).all()
    var = a.var[["feature_name", "feature_biotype", "feature_reference"]].copy()
    var.insert(0, "ensembl_id", a.var_names)
    var = var.rename(columns={"feature_name": "symbol"})
    export("klughammer", a.raw.X, a.obs, var)
    del a

# ── Regner ────────────────────────────────────────────────────────────────────
# Reference = gse243526_raw.h5ad, built by scipy.io.mmread from the same GEO
# files R will read with Matrix::readMM -> two independent readers.
if which in ("regner", "both"):
    print("Regner …")
    a = ad.read_h5ad("Regner/gse243526_raw.h5ad")
    kept = ad.read_h5ad("Regner/gse243526_processed.h5ad", backed="r").obs_names
    obs = a.obs.copy()
    obs["passed_python_qc"] = obs.index.isin(kept)
    var = a.var[["ensembl_id", "symbol", "feature_type"]].copy()
    export("regner", a.X, obs, var)

print("Done.")
