"""
03_gd_calling.py
================
Expression-only γδ T-cell call for Klughammer (10x 3', no TCR data).

Rule (calibrated on Azizi 10x 5' data, where the TCR gives the truth;
see logs/03_gd_calling.log and ~/single_cell/Azizi/03_gd_calling.py):
    gdT = TRDC>0 & CD3(D/E/G)>0 & CD4==0 & CD8B==0 & KLRF1==0 & FCER1G==0
  - TRDC alone: precision 0.69 in Azizi (doublets/ambient) and, here, TRDC is
    detected in 83 % of NK cells (germline transcripts).
  - TRAC/TRBC cannot be used: true γδ express germline TRAC/TRBC at levels
    ≥ αβ cells.
  - CD4 / CD8B (CD8αβ) are almost absent from γδ (0 % / 4 %) but common in
    αβ T cells → used as exclusion; KLRF1/FCER1G exclude NK cells.
  - Azizi 10x: precision 0.88, recall 0.63 vs the TCR-based call.
TRDV genes are essentially undetected in this 3' data (≤ 0.4 % of TRDC+
cells) → no Vδ subsets.

Other categories (excluded, kept for transparency):
  TRDC+_NKlike   TRDC>0 & (KLRF1>0 | FCER1G>0)     (germline TRDC in NK)
  TRDC+_CD4CD8B  TRDC>0 & CD3>0 & (CD4>0 | CD8B>0)  (likely αβ)
  TRDC+_CD3neg   TRDC>0 & CD3==0 & no NK markers

Validation with genes NOT used in the rule (TRGC1, TRGC2, TYROBP, KLRC1,
CD5, CD28): detection in γδ-called vs TRDC- T cells, compared with the
TCR-validated γδ and αβ cells of Azizi 10x.

Input:  data/klughammer_TNK.h5ad
Output: data/klughammer_TNK.h5ad (gd_call, is_gdT), data/klughammer_gdT.h5ad,
        figures/03_gd_calling/

Run from: ~/single_cell/Klughammer/gdT/
  python3 03_gd_calling.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
import harmonypy as hm

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

OUTDIR = "figures/03_gd_calling"
os.makedirs(OUTDIR, exist_ok=True)
VALID_GENES = ["TRGC1", "TRGC2", "TYROBP", "KLRC1", "CD5", "CD28"]
# % detected in Azizi 10x (TCR-validated γδ vs αβ-pair cells), from Azizi analysis
AZIZI_REF = pd.DataFrame({"Azizi γδ (TCR)": [24, 28, 37, 45, 22, 19],
                          "Azizi αβ (TCR)": [2, 7, 1, 5, 49, 32]}, index=VALID_GENES)


def counts(a, *genes):
    C = a.layers["counts"]
    out = np.zeros(a.n_obs)
    for g in genes:
        if g in a.var_names:
            out += np.asarray(C[:, a.var_names.get_loc(g)].todense()).ravel()
    return out


def call(a):
    trdc = counts(a, "TRDC"); cd3 = counts(a, "CD3D", "CD3E", "CD3G")
    cd4cd8b = counts(a, "CD4", "CD8B"); nk = counts(a, "KLRF1", "FCER1G")
    c = np.full(a.n_obs, "non_gd", dtype=object)
    c[(trdc > 0) & (cd3 == 0)] = "TRDC+_CD3neg"
    c[(trdc > 0) & (cd3 > 0) & (cd4cd8b > 0)] = "TRDC+_CD4CD8B"
    c[(trdc > 0) & (nk > 0)] = "TRDC+_NKlike"
    c[(trdc > 0) & (cd3 > 0) & (cd4cd8b == 0) & (nk == 0)] = "gdT"
    a.obs["gd_call"] = pd.Categorical(c)
    a.obs["is_gdT"] = a.obs["gd_call"] == "gdT"
    frac = a.obs.groupby("leiden", observed=True)["is_gdT"].mean()
    a.obs["gd_cluster_frac"] = a.obs["leiden"].map(frac).astype(float)
    return a


def validate(a):
    cd3 = counts(a, "CD3D", "CD3E", "CD3G"); trdc = counts(a, "TRDC")
    ref = (a.obs["author_cell_type"] == "T").values & (trdc == 0) & (cd3 > 0)
    gd = a.obs["is_gdT"].values
    tab = pd.DataFrame({
        "Klughammer γδ call": [100 * (counts(a, g)[gd] > 0).mean() for g in VALID_GENES],
        "Klughammer TRDC- T": [100 * (counts(a, g)[ref] > 0).mean() for g in VALID_GENES]},
        index=VALID_GENES).round(1)
    return tab.join(AZIZI_REF)


def reembed(a):
    g = a[a.obs["is_gdT"]].copy()
    for k in list(g.obsm):
        g.obsm.pop(k)
    sc.pp.highly_variable_genes(g, n_top_genes=1500, flavor="seurat")
    g.var.loc[g.var_names.str.match(r"^TR[ABDG][VJC]"), "highly_variable"] = False
    sc.pp.pca(g, n_comps=20, mask_var="highly_variable")
    ho = hm.run_harmony(g.obsm["X_pca"], g.obs, "patient", verbose=False)
    Z = np.asarray(ho.Z_corr)
    g.obsm["X_pca_harmony"] = Z if Z.shape[0] == g.n_obs else Z.T
    sc.pp.neighbors(g, use_rep="X_pca_harmony", n_neighbors=15)
    sc.tl.umap(g)
    sc.tl.leiden(g, resolution=0.8, flavor="igraph", n_iterations=2, key_added="gd_leiden")
    return g


if __name__ == "__main__":
    a = call(sc.read_h5ad("data/klughammer_TNK.h5ad"))
    print("── gd_call × author label ──")
    print(pd.crosstab(a.obs["gd_call"], a.obs["author_cell_type"]).to_string())

    v = validate(a)
    v.to_csv(f"{OUTDIR}/validation_genes_not_in_rule.csv")
    print("\n── Validation (% detected; genes not used in the rule) ──\n" + v.to_string())

    top = (a.obs.groupby("leiden", observed=True)
             .agg(n=("is_gdT", "size"), gd_frac=("is_gdT", "mean"))
             .sort_values("gd_frac", ascending=False).head(6).round(3))
    print("\n── Leiden clusters most enriched in γδ calls ──\n" + top.to_string())

    T = a.obs[a.obs["author_cell_type"] == "T"]
    per = (T.groupby("patient", observed=True)
             .agg(n_T=("is_gdT", "size"), n_gd=("is_gdT", "sum"))
             .assign(pct_gd_of_T=lambda d: (100 * d.n_gd / d.n_T).round(2)))
    per.to_csv(f"{OUTDIR}/gd_per_patient.csv")
    print("\n── γδ per patient (% of author T cells) ──\n" + per.describe().round(2).to_string())

    sc.pl.umap(a, color=["gd_call", "gd_cluster_frac", "author_cell_type", "leiden"], ncols=2,
               wspace=0.4, show=False, frameon=False, legend_fontsize=7)
    plt.savefig(f"{OUTDIR}/umap_TNK_gdcall.png", dpi=150, bbox_inches="tight"); plt.close()

    g = reembed(a)
    sc.pl.umap(g, color=["patient", "site_biopsy", "subtype", "gd_leiden", "TRDC", "TRGC1",
                         "TRGC2", "TYROBP", "KLRC1", "CD8A", "GZMB", "GZMK"],
               ncols=4, cmap="Reds", vmax="p99", show=False, frameon=False, legend_fontsize=6)
    plt.savefig(f"{OUTDIR}/umap_gdT.png", dpi=150, bbox_inches="tight"); plt.close()
    a.write_h5ad("data/klughammer_TNK.h5ad", compression="gzip")
    g.write_h5ad("data/klughammer_gdT.h5ad", compression="gzip")
    print(f"\nsaved {g.n_obs} γδ cells")
