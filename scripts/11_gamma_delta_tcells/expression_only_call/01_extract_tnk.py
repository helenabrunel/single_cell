"""
01_extract_TNK.py
=================
Extract the T/NK compartment of the Klughammer et al. (Nat Med 2024)
metastatic breast cancer atlas (HTAPP, CELLxGENE release) with RAW counts,
for γδ T-cell analysis (same approach as ~/single_cell/Azizi/).

Dataset notes (data/dataset.h5ad, 157,531 cells):
  - 10x 3' v2/v3, no TCR sequencing → γδ calls rely on expression only.
  - X is log-normalised; integer counts are in .raw.X (same genes).
  - Genes are Ensembl IDs; symbols in var["feature_name"].
  - 30 patients (donor_id), one metastatic biopsy each (sampleid).
  - Author labels: author_cell_type ("T", "NK", …). NK cells are kept
    because γδ cells can be annotated as NK (shared cytotoxic programme).

Patient-level clinical table (data/clinical_patients.csv):
  subtype from receptors_biopsy: HER2+ if HER2+; HR+ if ER+ or PR+
  (incl. PR-low); TNBC if ER-/PR-/HER2-.

Input:  ../data/dataset.h5ad
Output: data/klughammer_TNK_raw.h5ad, data/clinical_patients.csv

Run from: ~/single_cell/Klughammer/gdT/
  python3 01_extract_TNK.py
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import re
from scipy import sparse

SRC = "../data/dataset.h5ad"
KEEP_TYPES = ["T", "NK"]
OBS_KEEP = ["donor_id", "sampleid", "author_cell_type", "assay", "site_biopsy",
            "receptors_biopsy", "receptors_primary", "ER_biopsy", "PR_biopsy",
            "HER2_biopsy", "stage_at_diagnosis", "metastatic_presentation",
            "biopsy_days_after_metastasis", "histology_breast", "development_stage",
            "nCount_RNA", "nFeature_RNA", "percent.mito"]


def subtype(r):
    r = str(r)
    if r == "nan":
        return np.nan
    if "HER2+" in r:
        return "HER2+"
    if "ER+" in r or "PR+" in r or "PR-low" in r:
        return "HR+"
    return "TNBC"


if __name__ == "__main__":
    a = ad.read_h5ad(SRC, backed="r")
    idx = np.where(a.obs["author_cell_type"].isin(KEEP_TYPES).values)[0]
    print(f"T/NK cells: {len(idx)}")
    X = a.raw.X[idx]
    X = sparse.csr_matrix(X).astype(np.float32)
    obs = a.obs.iloc[idx][OBS_KEEP].copy()
    var = a.raw.var[[]].copy()
    var["gene_ids"] = var.index
    var.index = a.var.loc[a.raw.var.index, "feature_name"].astype(str).values

    t = ad.AnnData(X=X, obs=obs, var=var)
    t.var_names_make_unique()
    t.obs["patient"] = t.obs["donor_id"].astype(str)
    t.obs["subtype"] = t.obs["receptors_biopsy"].map(subtype)
    t.obs["age"] = t.obs["development_stage"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for c in t.obs.columns:
        if t.obs[c].dtype == object:
            t.obs[c] = t.obs[c].astype("category")
    t.layers["counts"] = t.X.copy()
    t.var["mt"] = t.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(t, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    print(t)
    t.write_h5ad("data/klughammer_TNK_raw.h5ad", compression="gzip")

    cl = (t.obs.groupby("patient", observed=True)
            .agg(sampleid=("sampleid", "first"), assay=("assay", "first"),
                 site_biopsy=("site_biopsy", "first"), receptors_biopsy=("receptors_biopsy", "first"),
                 subtype=("subtype", "first"), stage_at_diagnosis=("stage_at_diagnosis", "first"),
                 metastatic_presentation=("metastatic_presentation", "first"),
                 histology_breast=("histology_breast", "first"), age=("age", "first"),
                 n_T=("author_cell_type", lambda x: (x == "T").sum()),
                 n_NK=("author_cell_type", lambda x: (x == "NK").sum())))
    # patients with no T/NK cells are still listed from the full object
    allp = a.obs.groupby("donor_id", observed=True).agg(
        site_biopsy=("site_biopsy", "first"), receptors_biopsy=("receptors_biopsy", "first"))
    missing = sorted(set(allp.index.astype(str)) - set(cl.index))
    print("patients without T/NK cells:", missing)
    cl.to_csv("data/clinical_patients.csv")
    print(cl.to_string())
