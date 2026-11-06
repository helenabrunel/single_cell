"""
03_gd_calling.py
================
Marker/TCR-based identification of γδ T cells in the T/NK compartment.

Notes from calibration on the 10x data (see logs/03_gd_calling.log):
  - The 2018 10x V(D)J kit only amplifies TRA/TRB. TRD/TRG contigs in the
    contig files are V-only fragments (no J/C, not productive) → they are
    NOT evidence of γδ identity and are ignored.
  - TCR is therefore used as NEGATIVE evidence: a γδ cell should not carry
    a productive α+β pair. ~90% of TRDC+ / no-αβ-pair cells have no TCR
    contig at all, as expected for γδ.
  - γδ cells express germline TRAC/TRBC transcripts (median ~6 UMIs), so
    TRDC-vs-TRAC/TRBC ratios are NOT used.
  - NK cells can express germline TRDC → require CD3 or a TRDV transcript.

Per-cell call (obs["gd_call"]):
  10x   gdT_high   : TRDC>0, no productive TRA/TRB chain (non-productive
                     TRB contigs are common in γδ cells and allowed)
        gdT_medium : TRDC>0, single productive α or β chain (no pair)
                     or TRDV1/2>0 & TRDC==0 & no productive TRA/TRB
        TRDC+_abPair : TRDC>0 but productive αβ pair (doublet/ambient; excluded)
  inDrop (no TCR; 3' with poor V-gene capture)
        gdT_medium : TRDC>0 & CD3(D/E/G)>0
        gdT_low    : TRDC>0 & CD3==0 & TRDV>0
        TRDC+_CD3neg : TRDC>0, no CD3/TRDV (possible NK germline; excluded)
  other cells → "non_gd"

δ-chain subset (obs["gd_subset"], gd cells only): Vd1 / Vd2 / Vd3 by
dominant TRDV UMI count; "Vg9Vd2" if TRDV2 & TRGV9; "Vd_undetermined" if
no TRDV detected.

A cluster-level view is added too: fraction of gd-called cells per Leiden
cluster of the T/NK object (obs["gd_cluster_frac"]).

Input:  data/azizi_{10x,indrop}_TNK.h5ad
Output: same files updated with gd columns,
        data/azizi_{10x,indrop}_gdT.h5ad  (γδ cells, re-embedded)
        figures/03_gd_calling/
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

OUTDIR = "figures/03_gd_calling"
os.makedirs(OUTDIR, exist_ok=True)

GD_CALLS = ["gdT_high", "gdT_medium", "gdT_low"]
GD_GENES = ["TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "TRDV3", "TRGV9",
            "CD3E", "CD8A", "CD4", "TRAC", "NKG7", "KLRF1"]


def counts(adata, *genes):
    C = adata.layers["counts"]
    out = np.zeros(adata.n_obs)
    for g in genes:
        if g in adata.var_names:
            out += np.asarray(C[:, adata.var_names.get_loc(g)].todense()).ravel()
    return out


def call_gd(adata, plat):
    o = adata.obs
    trdc = counts(adata, "TRDC")
    cd3  = counts(adata, "CD3D", "CD3E", "CD3G")
    v1, v2, v3 = (counts(adata, g) for g in ["TRDV1", "TRDV2", "TRDV3"])
    g9 = counts(adata, "TRGV9")
    vd = v1 + v2 + v3
    call = np.full(adata.n_obs, "non_gd", dtype=object)

    if plat == "10x":
        has_ab = (o["has_TRA_productive"] | o["has_TRB_productive"]).values
        pair = (o["has_TRA_productive"] & o["has_TRB_productive"]).values
        call[(trdc > 0) & pair] = "TRDC+_abPair"
        call[(trdc > 0) & ~pair & has_ab] = "gdT_medium"
        call[(trdc == 0) & ((v1 + v2) > 0) & ~has_ab] = "gdT_medium"
        call[(trdc > 0) & ~has_ab] = "gdT_high"
    else:
        call[(trdc > 0)] = "TRDC+_CD3neg"
        call[(trdc > 0) & (cd3 == 0) & (vd > 0)] = "gdT_low"
        call[(trdc > 0) & (cd3 > 0)] = "gdT_medium"

    o["gd_call"] = pd.Categorical(call)
    o["is_gdT"] = o["gd_call"].isin(GD_CALLS)

    V = np.vstack([v1, v2, v3]).T
    sub = np.array(["Vd1", "Vd2", "Vd3"])[V.argmax(1)].astype(object)
    sub[vd == 0] = "Vd_undetermined"
    sub[(sub == "Vd2") & (g9 > 0)] = "Vg9Vd2"
    sub[~o["is_gdT"].values] = np.nan
    o["gd_subset"] = pd.Categorical(sub)
    o["Vd_family"] = o["gd_subset"].astype(str).replace(
        {"Vg9Vd2": "Vd2", "nan": np.nan}).astype("category")

    frac = o.groupby("leiden", observed=True)["is_gdT"].mean()
    o["gd_cluster_frac"] = o["leiden"].map(frac).astype(float)
    return adata


def reembed_gd(adata):
    gd = adata[adata.obs["is_gdT"]].copy()
    for k in list(gd.obsm):
        gd.obsm.pop(k)
    gd.X = gd.layers["counts"].copy()
    sc.pp.normalize_total(gd, target_sum=1e4)
    sc.pp.log1p(gd)
    sc.pp.highly_variable_genes(gd, n_top_genes=1500, flavor="seurat")
    gd.var.loc[gd.var_names.str.match(r"^TR[ABDG][VJC]"), "highly_variable"] = False
    sc.pp.pca(gd, n_comps=20, mask_var="highly_variable")
    batch = "sample"
    if gd.obs[batch].nunique() > 1:
        import harmonypy as hm
        ho = hm.run_harmony(gd.obsm["X_pca"], gd.obs, batch, verbose=False)
        Z = np.asarray(ho.Z_corr)
        gd.obsm["X_pca_harmony"] = Z if Z.shape[0] == gd.n_obs else Z.T
        rep = "X_pca_harmony"
    else:
        rep = "X_pca"
    sc.pp.neighbors(gd, use_rep=rep, n_neighbors=15)
    sc.tl.umap(gd)
    sc.tl.leiden(gd, resolution=0.8, flavor="igraph", n_iterations=2,
                 key_added="gd_leiden")
    return gd


def summarise(adata, plat):
    o = adata.obs
    print(f"\n── {plat}: gd_call ──")
    print(o["gd_call"].value_counts().to_string())
    gd = o[o["is_gdT"]]
    print(f"\n── {plat}: δ-chain subset (gd cells) ──")
    print(gd["gd_subset"].value_counts().to_string())
    tab = pd.crosstab([o["patient"], o["tissue"]], o["is_gdT"])
    tab.columns = ["non_gd", "gdT"]
    tab["pct_gdT_of_TNK"] = (100 * tab["gdT"] / tab.sum(1)).round(2)
    print(f"\n── {plat}: % γδ of T/NK per patient × tissue ──")
    print(tab.to_string())
    tab.to_csv(f"{OUTDIR}/gd_fraction_{plat}.csv")
    pd.crosstab([gd["patient"], gd["tissue"]], gd["gd_subset"]).to_csv(
        f"{OUTDIR}/gd_subset_counts_{plat}.csv")
    top = (o.groupby("leiden", observed=True)
             .agg(n=("is_gdT", "size"), gd_frac=("is_gdT", "mean"))
             .sort_values("gd_frac", ascending=False).head(5).round(3))
    print(f"\n── {plat}: top γδ-enriched T/NK clusters ──\n{top.to_string()}")


def plots(adata, gd, plat):
    sc.pl.umap(adata, color=["gd_call", "gd_subset", "gd_cluster_frac", "leiden"],
               ncols=2, wspace=0.4, show=False, frameon=False, na_in_legend=False)
    plt.savefig(f"{OUTDIR}/umap_TNK_{plat}_gdcall.png", dpi=150, bbox_inches="tight")
    plt.close()
    g = [x for x in GD_GENES if x in gd.var_names]
    sc.pl.umap(gd, color=["gd_subset", "patient", "tissue", "gd_leiden"] + g,
               ncols=4, cmap="Reds", show=False, frameon=False, vmax="p99")
    plt.savefig(f"{OUTDIR}/umap_gdT_{plat}.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for plat in ["10x", "indrop"]:
        adata = sc.read_h5ad(f"data/azizi_{plat}_TNK.h5ad")
        adata = call_gd(adata, plat)
        summarise(adata, plat)
        gd = reembed_gd(adata)
        plots(adata, gd, plat)
        adata.write_h5ad(f"data/azizi_{plat}_TNK.h5ad", compression="gzip")
        gd.write_h5ad(f"data/azizi_{plat}_gdT.h5ad", compression="gzip")
        print(f"  saved {gd.n_obs} γδ cells → data/azizi_{plat}_gdT.h5ad")
