"""
04_stcat.py
===========
STCAT annotation (Guo lab; Science Bulletin 2025; celltypist-based,
hierarchical) of the T/NK compartment, and comparison with the
marker/TCR-based γδ call from 03.

STCAT needs Python 3.9 + anndata 0.10, so it lives in the conda env
"stcat" (conda create -n stcat python=3.9.16 && pip install STCAT) and
cannot read h5ad files written by anndata 0.12. This script therefore
has three steps:

  export  (base env)  data/stcat/{plat}_input.h5ad  — raw counts, no metadata
  run     (stcat env) data/stcat/{plat}_stcat.csv
            - STCAT.STCAT(): final hierarchical label ("Prediction"),
              cluster-level majority voting + marker correction
            - level-0 model per cell (no voting): label + P(Tgd), so that
              small γδ groups absorbed by CD8 clusters are not lost
  import  (base env)  adds stcat_* columns to data/azizi_{plat}_TNK.h5ad
                      and data/azizi_{plat}_gdT.h5ad, plus comparison
                      tables/plots in figures/04_stcat/

STCAT level 0 has a single "Tgd" class (no Vδ1/Vδ2 split); the δ-chain
subset comes from 03.

RESULT (2026-10-16): STCAT does not identify γδ cells in this dataset.
  - final labels: 0 cells "Tgd" (10x and inDrop)
  - per-cell P(Tgd): AUC 0.59 (10x, vs TRDC+/no-αβ γδ) and 0.55 (inDrop);
    0/473 high-confidence γδ with P>0.5
  - γδ are mostly CD4-/CD8- (DN); STCAT splits CD4 vs CD8 with
    `CD8 > CD4`, so ties (0/0) fall into the CD4 branch → Vγ9Vδ2 cells end
    up as "CD4 Tem cytotoxicity". STCAT state labels are therefore not
    meaningful for γδ cells; keep them only as a reference for αβ cells.

Run from: ~/single_cell/Azizi/
  python3 04_stcat.py export
  conda run -n stcat python 04_stcat.py run
  python3 04_stcat.py import
"""

import sys, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PLATFORMS = ["10x", "indrop"]
SDIR = "data/stcat"
OUTDIR = "figures/04_stcat"
os.makedirs(SDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)


def export():
    import scanpy as sc
    import anndata as ad
    for plat in PLATFORMS:
        a = sc.read_h5ad(f"data/azizi_{plat}_TNK.h5ad")
        m = ad.AnnData(X=a.layers["counts"].copy())
        m.obs_names = a.obs_names.astype(str)
        m.var_names = a.var_names.astype(str)
        m.write_h5ad(f"{SDIR}/{plat}_input.h5ad")
        print(f"exported {plat}: {m.shape}")


def run():
    import scanpy as sc
    import STCAT
    from STCAT import annotate
    model0 = os.path.join(os.path.dirname(STCAT.__file__), "data", "T_cell.pkl")
    for plat in PLATFORMS:
        a = sc.read_h5ad(f"{SDIR}/{plat}_input.h5ad")
        print(f"── STCAT {plat}: {a.shape}")
        res = STCAT.STCAT(a.copy())
        out = res.obs[["Prediction", "Cluster", "Uncertainty score"]].copy()
        out.columns = ["stcat_label", "stcat_cluster", "stcat_uncertainty"]

        # level-0 per-cell prediction (no majority voting)
        b = a.copy()
        sc.pp.normalize_total(b, target_sum=1e4)
        sc.pp.log1p(b)
        pr = annotate.annotate(b, model=model0, majority_voting=False)
        lab = pr.predicted_labels["predicted_labels"].rename("stcat_l0_cell")
        prob = pr.probability_matrix
        p_gd = prob["Tgd"].rename("stcat_P_Tgd")
        out = out.join(lab, how="outer").join(p_gd, how="outer")
        out.index.name = "cell"
        out.to_csv(f"{SDIR}/{plat}_stcat.csv")
        print(out["stcat_label"].value_counts().head(15).to_string())


def import_():
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for plat in PLATFORMS:
        res = pd.read_csv(f"{SDIR}/{plat}_stcat.csv", index_col=0)
        for f in [f"data/azizi_{plat}_TNK.h5ad", f"data/azizi_{plat}_gdT.h5ad"]:
            a = sc.read_h5ad(f)
            a.obs = a.obs.drop(columns=[c for c in a.obs if c.startswith("stcat_")])
            a.obs = a.obs.join(res, how="left")
            for c in ["stcat_label", "stcat_l0_cell"]:
                a.obs[c] = a.obs[c].fillna("NA").astype("category")
            a.obs["stcat_is_Tgd"] = a.obs["stcat_label"] == "Tgd"
            a.obs["stcat_l0_is_Tgd"] = a.obs["stcat_l0_cell"] == "Tgd"
            a.write_h5ad(f, compression="gzip")
            if f.endswith("_TNK.h5ad"):
                tnk = a

        o = tnk.obs
        print(f"\n══ {plat} ══")
        print("STCAT final labels (top 20):")
        print(o["stcat_label"].value_counts().head(20).to_string())

        # agreement: marker/TCR call vs STCAT (final & per-cell level 0)
        ct = pd.crosstab(o["gd_call"], [o["stcat_is_Tgd"], o["stcat_l0_is_Tgd"]],
                         margins=True)
        ct.columns = [f"final={a}|l0={b}" if a != "All" else "All" for a, b in ct.columns]
        print("\nmarker/TCR gd_call × STCAT Tgd (final | per-cell level 0):")
        print(ct.to_string())
        ct.to_csv(f"{OUTDIR}/gdcall_vs_stcat_{plat}.csv")

        gd = o[o["is_gdT"]]
        print("\nWhat STCAT calls the marker-defined γδ cells (final label):")
        print(pd.crosstab(gd["stcat_label"], gd["gd_subset"]).loc[
            lambda d: d.sum(1) > 0].sort_values("Vd1" if "Vd1" in gd["gd_subset"].values
                                                else d.columns[0], ascending=False)
            .head(12).to_string())
        st = o[o["stcat_label"] == "Tgd"]
        print(f"\nSTCAT Tgd cells: {len(st)}; of these marker-gd: {st['is_gdT'].mean():.2f}; "
              f"TRDC+ αβ-pair: {(st['gd_call']=='TRDC+_abPair').mean():.2f}")

        o["consensus_gd"] = np.select(
            [o["is_gdT"] & (o["stcat_is_Tgd"] | o["stcat_l0_is_Tgd"]),
             o["is_gdT"],
             o["stcat_is_Tgd"] | o["stcat_l0_is_Tgd"]],
            ["both", "marker_only", "stcat_only"], "neither")
        print("\nconsensus:\n" + o["consensus_gd"].value_counts().to_string())
        tnk.obs["consensus_gd"] = o["consensus_gd"].astype("category")
        tnk.write_h5ad(f"data/azizi_{plat}_TNK.h5ad", compression="gzip")

        sc.pl.umap(tnk, color=["gd_call", "stcat_label", "stcat_P_Tgd", "consensus_gd"],
                   ncols=2, wspace=0.6, show=False, frameon=False, legend_fontsize=6)
        plt.savefig(f"{OUTDIR}/umap_TNK_{plat}_stcat.png", dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"export": export, "run": run, "import": import_}.get(
        step, lambda: sys.exit("usage: 04_stcat.py export|run|import"))()
