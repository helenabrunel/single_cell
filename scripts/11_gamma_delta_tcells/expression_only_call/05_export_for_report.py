"""
05_export_for_report.py
=======================
Tables for report/klughammer_gdT_feasibility.Rmd.

Output (report/data/):
  tnk_umap.csv.gz          UMAP + metadata + selected genes (log-normalised)
  detection_by_type.csv    % detection of TCR/lineage genes in author T and NK cells
  rule_calibration.csv     expression rules: precision/recall in Azizi 10x (TCR truth)
                           and cells passing in Klughammer
  depth_stratified.csv     TRDC % in sure-αβ vs DN candidates by UMI-depth bin
                           (Klughammer and Azizi 10x)
  background_per_patient.csv  TRDC % in sure-αβ / DN per patient, % NK, assay, site
(background_test.csv and validation_genes_not_in_rule.csv are read from figures/)

Run from: ~/single_cell/Klughammer/gdT/
  python3 05_export_for_report.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from scipy.sparse import issparse

OUT = "report/data"
os.makedirs(OUT, exist_ok=True)
AZIZI = "../../Azizi/data/azizi_10x_TNK.h5ad"
UMAP_GENES = ["TRDC", "TRGC1", "TRGC2", "CD3E", "CD8B", "CD4", "KLRF1", "FCER1G"]
DET_GENES = ["TRDC", "TRDV1", "TRDV2", "TRDV3", "TRGV9", "TRGC1", "TRGC2", "TRAC",
             "TRBC1", "TRBC2", "CD3E", "CD3D", "KLRF1", "IL17A"]


def counts_getter(a):
    C = a.layers["counts"]
    return lambda x: (np.asarray(C[:, a.var_names.get_loc(x)].todense()).ravel()
                      if x in a.var_names else np.zeros(a.n_obs))


def groups(g, t_mask):
    cd3 = (g("CD3D") + g("CD3E") + g("CD3G")) > 0
    nk = (g("KLRF1") + g("FCER1G")) > 0
    cd = (g("CD4") + g("CD8B")) > 0
    return t_mask & cd3 & ~nk & cd, t_mask & cd3 & ~nk & ~cd


RULES = {
    "TRDC>0": lambda g: g("TRDC") > 0,
    "+ CD3>0": lambda g: (g("TRDC") > 0) & ((g("CD3D") + g("CD3E") + g("CD3G")) > 0),
    "+ CD4=0 & CD8B=0": lambda g: (g("TRDC") > 0) & ((g("CD3D") + g("CD3E") + g("CD3G")) > 0)
                                  & ((g("CD4") + g("CD8B")) == 0),
    "+ KLRF1=0 & FCER1G=0 (final rule)": lambda g: (g("TRDC") > 0)
        & ((g("CD3D") + g("CD3E") + g("CD3G")) > 0) & ((g("CD4") + g("CD8B")) == 0)
        & ((g("KLRF1") + g("FCER1G")) == 0),
}


if __name__ == "__main__":
    a = sc.read_h5ad("data/klughammer_TNK.h5ad")
    g = counts_getter(a)
    T = (a.obs["author_cell_type"] == "T").values

    X = a[:, [x for x in UMAP_GENES if x in a.var_names]].X
    X = X.toarray() if issparse(X) else X
    df = pd.DataFrame(a.obsm["X_umap"], index=a.obs_names, columns=["UMAP1", "UMAP2"])
    df = df.join(a.obs[["patient", "author_cell_type", "site_biopsy", "subtype", "assay",
                        "leiden", "gd_call", "decontX_contamination"]])
    df = df.join(pd.DataFrame(np.round(X, 3), index=a.obs_names,
                              columns=[x for x in UMAP_GENES if x in a.var_names]))
    df.index.name = "cell"
    df.to_csv(f"{OUT}/tnk_umap.csv.gz")

    det = pd.DataFrame({t: [100 * (g(x)[(a.obs["author_cell_type"] == t).values] > 0).mean()
                            for x in DET_GENES] for t in ["T", "NK"]}, index=DET_GENES).round(1)
    det.index.name = "gene"
    det.to_csv(f"{OUT}/detection_by_type.csv")

    az = sc.read_h5ad(AZIZI)
    ga = counts_getter(az)
    truth = az.obs["gd_call"].isin(["gdT_high", "gdT_medium"]).values
    rows = []
    for k, f in RULES.items():
        r = f(ga); rk = f(g)
        rows.append({"rule": k, "Azizi n": int(r.sum()),
                     "Azizi precision": round((r & truth).sum() / r.sum(), 2),
                     "Azizi recall": round((r & truth).sum() / truth.sum(), 2),
                     "Klughammer T cells passing": int((rk & T).sum()),
                     "Klughammer NK cells passing": int((rk & ~T).sum()),
                     "Klughammer % of T": round(100 * (rk & T).sum() / T.sum(), 1)})
    pd.DataFrame(rows).to_csv(f"{OUT}/rule_calibration.csv", index=False)

    rows = []
    for name, obj, gg, tm in [("Klughammer", a, g, T), ("Azizi 10x", az, ga, np.ones(az.n_obs, bool))]:
        C = obj.layers["counts"]; tot = np.asarray(C.sum(1)).ravel()
        ab, dn = groups(gg, tm)
        trdc = gg("TRDC") > 0
        q = np.quantile(tot[ab | dn], np.linspace(0, 1, 6))
        for i, (lo, hi) in enumerate(zip(q[:-1], q[1:]), 1):
            b = (tot >= lo) & (tot < hi)
            rows.append({"dataset": name, "bin": i, "UMI range": f"{lo:.0f}–{hi:.0f}",
                         "sure-αβ": 100 * trdc[ab & b].mean(), "DN candidates": 100 * trdc[dn & b].mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/depth_stratified.csv", index=False)

    ab, dn = groups(g, T)
    trdc = g("TRDC") > 0
    rows = []
    for p in a.obs["patient"].cat.categories:
        pm = (a.obs["patient"] == p).values
        if (ab & pm).sum() >= 30 and (dn & pm).sum() >= 30:
            rows.append({"patient": p, "sure-αβ": 100 * trdc[ab & pm].mean(),
                         "DN candidates": 100 * trdc[dn & pm].mean(),
                         "pct_NK": 100 * (a.obs.loc[pm, "author_cell_type"] == "NK").mean(),
                         "assay": a.obs.loc[pm, "assay"].iloc[0], "site": a.obs.loc[pm, "site_biopsy"].iloc[0]})
    pd.DataFrame(rows).to_csv(f"{OUT}/background_per_patient.csv", index=False)
    print("Done.")
