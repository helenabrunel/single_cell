"""
08_nk_subclusters.py
====================
Are there T-like (possibly γδ) cells among author-NK cells?

The γδ rule of 03_gd_calling.py excluded KLRF1+/FCER1G+ cells (TRDC is
detected in 83 % of NK cells), and the background test used author-T cells
only, so cells labelled NK were never evaluated as γδ. Here author-NK cells
are re-embedded (HVG without TCR genes, PCA, Harmony by patient, UMAP, Leiden
1.0) and each subcluster is profiled for T, γδ-associated and NK genes.
Subclusters with CD3D detected in > 50 % of cells are flagged "T-like".

Also: CD3E+ vs CD3E- author-NK cells, and the reference profiles of author-T
CD8B+ cells and TCR-validated γδ cells of Azizi 10x.

Input:  data/klughammer_TNK.h5ad, ../../Azizi/data/azizi_10x_TNK.h5ad
Output: data/klughammer_NK_sub.h5ad,
        report/data/nk_subclusters.csv, report/data/nk_cd3e_coexpression.csv,
        report/data/nk_umap.csv.gz

Run from: ~/single_cell/Klughammer/gdT/
  python3 08_nk_subclusters.py > logs/08_nk_subclusters.log
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os, warnings
import harmonypy as hm

warnings.filterwarnings("ignore")
OUT = "report/data"
os.makedirs(OUT, exist_ok=True)
AZIZI = "../../Azizi/data/azizi_10x_TNK.h5ad"
GENES = ["CD3E", "CD3D", "CD3G", "TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2",
         "CD8A", "CD8B", "CD5", "IL7R", "KLRF1", "FCER1G", "NCAM1", "FCGR3A", "KLRC1",
         "TYROBP", "GZMK"]
UMAP_GENES = ["CD3D", "TRAC", "TRGC1", "CD8B", "KLRF1", "FCER1G"]


def getter(a):
    C = a.layers["counts"]
    return lambda x: (np.asarray(C[:, a.var_names.get_loc(x)].todense()).ravel()
                      if x in a.var_names else np.zeros(a.n_obs))


def pct(g, m):
    return {x: round(100 * (g(x)[m] > 0).mean(), 1) for x in GENES}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    a = sc.read_h5ad("data/klughammer_TNK.h5ad")
    g = getter(a)
    T = (a.obs["author_cell_type"] == "T").values

    nk = a[~T].copy()
    for k in list(nk.obsm):
        nk.obsm.pop(k)
    sc.pp.highly_variable_genes(nk, n_top_genes=1500, flavor="seurat")
    nk.var.loc[nk.var_names.str.match(r"^TR[ABDG][VJC]"), "highly_variable"] = False
    sc.pp.pca(nk, n_comps=20, mask_var="highly_variable")
    Z = np.asarray(hm.run_harmony(nk.obsm["X_pca"], nk.obs, "patient", verbose=False).Z_corr)
    nk.obsm["X_pca_harmony"] = Z if Z.shape[0] == nk.n_obs else Z.T
    sc.pp.neighbors(nk, use_rep="X_pca_harmony")
    sc.tl.umap(nk)
    sc.tl.leiden(nk, resolution=1.0, flavor="igraph", n_iterations=2, key_added="nk_sub")

    gn = getter(nk)
    tot = np.asarray(nk.layers["counts"].sum(1)).ravel()
    rows = []
    for c in sorted(nk.obs["nk_sub"].unique(), key=int):
        m = (nk.obs["nk_sub"] == c).values
        r = {"subcluster": c, "n": int(m.sum()), "patients": nk.obs["patient"][m].nunique(),
             "median UMI": int(np.median(tot[m]))}
        r.update(pct(gn, m))
        rows.append(r)
    sub = pd.DataFrame(rows)
    sub["T-like"] = sub["CD3D"] > 50
    nk.obs["T_like"] = nk.obs["nk_sub"].isin(sub.loc[sub["T-like"], "subcluster"])
    print("── author-NK subclusters (% detected) ──\n" + sub.to_string(index=False))

    # pooled T-like vs other NK, plus references
    tl = nk.obs["T_like"].values
    ref = pd.DataFrame({
        "NK T-like subclusters": pct(gn, tl),
        "NK other subclusters": pct(gn, ~tl),
        "author T CD8B+": pct(g, T & (g("CD8B") > 0))})
    az = sc.read_h5ad(AZIZI)
    ga = getter(az)
    ref["Azizi γδ (TCR)"] = pd.Series(pct(ga, az.obs["gd_call"].isin(["gdT_high", "gdT_medium"]).values))
    pooled = pd.DataFrame({"n": [int(tl.sum()), int((~tl).sum()), int((T & (g("CD8B") > 0)).sum()),
                                 int(az.obs["gd_call"].isin(["gdT_high", "gdT_medium"]).sum())],
                           "patients": [nk.obs["patient"][tl].nunique(), nk.obs["patient"][~tl].nunique(),
                                        np.nan, 3],
                           "median UMI": [int(np.median(tot[tl])), int(np.median(tot[~tl])),
                                          int(np.median(np.asarray(a.layers["counts"][T & (g("CD8B") > 0)].sum(1)))),
                                          np.nan]},
                          index=ref.columns).T
    print("\n── pooled ──\n" + pd.concat([pooled, ref]).to_string())
    sub.to_csv(f"{OUT}/nk_subclusters.csv", index=False)
    pd.concat([pooled, ref]).rename_axis("gene").to_csv(f"{OUT}/nk_tlike_vs_references.csv")

    # CD3E+ vs CD3E- author-NK cells
    c3 = gn("CD3E") > 0
    co = pd.DataFrame({"NK CD3E+": pct(gn, c3), "NK CD3E-": pct(gn, ~c3), "author T": pct(g, T)})
    co.loc["n"] = [int(c3.sum()), int((~c3).sum()), int(T.sum())]
    co.loc["% in T-like subclusters"] = [round(100 * tl[c3].mean(), 1), round(100 * tl[~c3].mean(), 1), np.nan]
    co.loc["% with CD3E = 1 UMI"] = [round(100 * (gn("CD3E")[c3] == 1).mean(), 1), np.nan,
                                     round(100 * (g("CD3E")[T & (g("CD3E") > 0)] == 1).mean(), 1)]
    print("\n── CD3E+ vs CD3E- author-NK ──\n" + co.to_string())
    co.rename_axis("gene").to_csv(f"{OUT}/nk_cd3e_coexpression.csv")

    df = pd.DataFrame(nk.obsm["X_umap"], index=nk.obs_names, columns=["UMAP1", "UMAP2"])
    df["nk_sub"] = nk.obs["nk_sub"].astype(str).values
    df["T_like"] = tl
    df["patient"] = nk.obs["patient"].astype(str).values
    for x in UMAP_GENES:
        df[x] = np.log1p(gn(x) / tot * 1e4).round(3)
    df.rename_axis("cell").to_csv(f"{OUT}/nk_umap.csv.gz")
    nk.write_h5ad("data/klughammer_NK_sub.h5ad", compression="gzip")
