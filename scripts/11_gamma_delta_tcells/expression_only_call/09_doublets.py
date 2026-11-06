"""
09_doublets.py
==============
Are the T-like subclusters of NK-annotated cells (08_nk_subclusters.py) NK+T
doublets?

Scrublet (scanpy.pp.scrublet) is run per 10x channel (channel_id; doublets only
form within a channel) on the WHOLE atlas, all cell types, raw counts.
Expected doublet rate = 0.8 % per 1,000 cells in the channel (10x guidance),
clipped to 1–10 %. Fixed score threshold 0.25 (automatic thresholding needs
scikit-image, not installed); continuous scores are compared as well.

Then doublet scores / calls are compared between: T-like NK subclusters,
other NK-annotated cells, author T cells (CD8B+ and all) and, for reference,
the γδ calls of 03_gd_calling.py.

Input:  ../data/dataset.h5ad, data/klughammer_NK_sub.h5ad, data/klughammer_TNK.h5ad
Output: data/scrublet_scores.csv, report/data/doublets_by_group.csv,
        report/data/nk_subclusters_doublets.csv

Run from: ~/single_cell/Klughammer/gdT/
  python3 09_doublets.py > logs/09_doublets.log
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os, warnings
from scipy import sparse

warnings.filterwarnings("ignore")
SRC = "../data/dataset.h5ad"
OUT = "report/data"
THRESH = 0.25


def run_scrublet(a):
    rows = []
    for ch in a.obs["channel_id"].astype(str).unique():
        idx = np.where(a.obs["channel_id"].astype(str).values == ch)[0]
        X = sparse.csr_matrix(a.raw.X[idx]).astype(np.float32)
        b = ad.AnnData(X=X, obs=pd.DataFrame(index=a.obs_names[idx]))
        rate = float(np.clip(0.008 * len(idx) / 1000, 0.01, 0.10))
        try:
            sc.pp.scrublet(b, expected_doublet_rate=rate, threshold=THRESH, random_state=0, verbose=False)
            sc_ = b.obs["doublet_score"].values
            pr = (sc_ > THRESH).astype(float)
        except Exception as e:     # very small channels → NaN, not "singlet"
            print(f"  {ch}: scrublet failed ({e})")
            sc_, pr = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
        print(f"  {ch}: {len(idx)} cells, rate {rate:.3f}, predicted doublets {np.nansum(pr):.0f} "
              f"({100 * np.nanmean(pr):.1f} %)")
        rows.append(pd.DataFrame({"doublet_score": sc_, "predicted_doublet": pr,
                                  "channel_id": ch}, index=a.obs_names[idx]))
    return pd.concat(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    if os.path.exists("data/scrublet_scores.csv"):
        d = pd.read_csv("data/scrublet_scores.csv", index_col=0)
    else:
        a = ad.read_h5ad(SRC, backed="r")
        d = run_scrublet(a)
        d["author_cell_type"] = a.obs.loc[d.index, "author_cell_type"].astype(str).values
        d.to_csv("data/scrublet_scores.csv")

    print("\n── Predicted doublets by author cell type ──")
    print(d.groupby("author_cell_type").agg(n=("doublet_score", "size"),
                                            median_score=("doublet_score", "median"),
                                            pct_doublet=("predicted_doublet", lambda x: round(100 * x.mean(), 1)))
          .sort_values("n", ascending=False).to_string())

    nk = sc.read_h5ad("data/klughammer_NK_sub.h5ad")
    t = sc.read_h5ad("data/klughammer_TNK.h5ad", backed="r")
    C = t.layers["counts"] if not t.isbacked else None
    T = t.obs["author_cell_type"].astype(str).values == "T"
    cd8b = np.asarray(sc.read_h5ad("data/klughammer_TNK.h5ad")[:, "CD8B"].layers["counts"].todense()).ravel() > 0

    groups = {
        "NK T-like subclusters": nk.obs_names[nk.obs["T_like"].values.astype(bool)],
        "NK other subclusters": nk.obs_names[~nk.obs["T_like"].values.astype(bool)],
        "author T, CD8B+": t.obs_names[T & cd8b],
        "author T, all": t.obs_names[T],
        "γδ call (03, author T + NK)": t.obs_names[t.obs["gd_call"].astype(str).values == "gdT"],
    }
    rows = []
    for k, ids in groups.items():
        x = d.loc[ids]
        rows.append({"group": k, "n": len(x), "median doublet score": round(x["doublet_score"].median(), 3),
                     "% predicted doublet": round(100 * x["predicted_doublet"].mean(), 1),
                     "% score > 0.25": round(100 * (x["doublet_score"] > 0.25).mean(), 1)})
    gr = pd.DataFrame(rows)
    print("\n── Doublet scores by group ──\n" + gr.to_string(index=False))
    gr.to_csv(f"{OUT}/doublets_by_group.csv", index=False)

    nk.obs = nk.obs.join(d[["doublet_score", "predicted_doublet"]])
    sub = (nk.obs.groupby("nk_sub", observed=True)
             .agg(n=("doublet_score", "size"), T_like=("T_like", "first"),
                  median_score=("doublet_score", "median"),
                  pct_doublet=("predicted_doublet", lambda x: round(100 * x.mean(), 1))))
    print("\n── NK subclusters ──\n" + sub.round(3).to_string())
    sub.round(3).to_csv(f"{OUT}/nk_subclusters_doublets.csv")

    # T-like NK cells NOT called doublets: do they keep the T-like profile?
    tl = nk.obs["T_like"].values.astype(bool)
    print(f"\nNK cells without a score (scrublet failed): {int(nk.obs['doublet_score'].isna().sum())}")
    dbl = (nk.obs["predicted_doublet"] == 1).values
    sing = (nk.obs["predicted_doublet"] == 0).values
    G = ["CD3D", "CD3G", "TRAC", "TRGC1", "TRGC2", "CD8A", "CD8B", "KLRF1", "FCER1G", "FCGR3A", "IL7R", "CD5"]
    Cn = nk.layers["counts"]
    pc = lambda m: {x: round(100 * (np.asarray(Cn[m][:, nk.var_names.get_loc(x)].todense()).ravel() > 0).mean(), 1) for x in G}
    prof = pd.DataFrame({"T-like, all": pc(tl), "T-like, singlets": pc(tl & sing),
                         "T-like, doublets": pc(tl & dbl), "other NK, singlets": pc(~tl & sing)})
    prof.loc["n"] = [tl.sum(), (tl & sing).sum(), (tl & dbl).sum(), (~tl & sing).sum()]
    print("\n── T-like NK cells: singlets vs doublets (% detected) ──\n" + prof.to_string())
    prof.rename_axis("gene").to_csv(f"{OUT}/nk_tlike_singlet_profile.csv")

    # ── Doublet expectation from UMI depth, per patient (same-patient comparison) ──
    tot = np.asarray(Cn.sum(1)).ravel()
    tt = sc.read_h5ad("data/klughammer_TNK.h5ad")
    tot_t = np.asarray(tt.layers["counts"].sum(1)).ravel()
    pat, pat_t = nk.obs["patient"].astype(str).values, tt.obs["patient"].astype(str).values
    rows = []
    for p in np.unique(pat[tl]):
        a_, b_, c_ = tl & (pat == p), ~tl & (pat == p), T & (pat_t == p)
        if min(a_.sum(), b_.sum(), c_.sum()) >= 10:
            rows.append({"patient": p, "n T-like": int(a_.sum()), "T-like": np.median(tot[a_]),
                         "NK other": np.median(tot[b_]), "author T": np.median(tot_t[c_])})
    um = pd.DataFrame(rows)
    um["NK + T (doublet expectation)"] = um["NK other"] + um["author T"]
    print("\n── Median UMI per cell, per patient (≥ 10 cells per group) ──\n" + um.round(0).to_string(index=False))
    print(f"median ratio T-like / NK other = {(um['T-like'] / um['NK other']).median():.2f}; "
          f"T-like / doublet expectation = {(um['T-like'] / um['NK + T (doublet expectation)']).median():.2f}")
    um.round(0).to_csv(f"{OUT}/nk_tlike_umi_doublet_expectation.csv", index=False)

    # ── Background test inside the T-like cells: CD8B+ (αβ) vs CD8B- ──
    gk = lambda x: np.asarray(Cn[:, nk.var_names.get_loc(x)].todense()).ravel()
    c8 = gk("CD8B") > 0
    t_core = (gk("CD3D") > 0) & (gk("TRAC") > 0)
    rows = []
    for k, m in [("T-like, CD8B+ (αβ)", tl & c8), ("T-like, CD8B−", tl & ~c8),
                 ("T-like CD3D+TRAC+, CD8B+ (αβ)", tl & c8 & t_core),
                 ("T-like CD3D+TRAC+, CD8B−", tl & ~c8 & t_core), ("NK other", ~tl)]:
        r = {"group": k, "n": int(m.sum())}
        r.update({f"% {x}+": round(100 * (gk(x)[m] > 0).mean(), 1) for x in ["TRDC", "TRGC1", "TRGC2", "CD3D"]})
        r["% TRDC ≥ 2"] = round(100 * (gk("TRDC")[m] >= 2).mean(), 1)
        rows.append(r)
    bt = pd.DataFrame(rows)
    print("\n── Background test inside T-like NK cells ──\n" + bt.to_string(index=False))
    bt.to_csv(f"{OUT}/nk_tlike_background_test.csv", index=False)
