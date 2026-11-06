"""
04c_background_test.py
======================
Does ambient-RNA correction (decontX) make TRDC / TRGC informative for γδ
T cells in Klughammer?

1. Adds decontX counts (rounded) as layer "decontX" and per-cell
   contamination to data/klughammer_TNK.h5ad.
2. Background test, for raw and decontX counts and several markers:
     sure-αβ   = author T, CD3+, (CD4 or CD8B)+, KLRF1-/FCER1G-
     candidates (DN) = author T, CD3+, CD4-/CD8B-, KLRF1-/FCER1G-
   A γδ marker should be clearly MORE frequent in DN candidates than in
   sure-αβ cells. est. precision = (rate_DN - rate_αβ) / rate_DN.
   Same test on Azizi 10x (TCR-validated) as reference.
3. Per-patient background of TRDC in sure-αβ cells, raw vs decontX.

Input:  data/klughammer_TNK.h5ad, data/decontx/out/*.h5,
        ~/single_cell/Azizi/data/azizi_10x_TNK.h5ad
Output: data/klughammer_TNK.h5ad (layer decontX, obs decontX_contamination),
        figures/04_decontx/

Run from: ~/single_cell/Klughammer/gdT/
  python3 04c_background_test.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import h5py, glob, os
from scipy import sparse

OUTDIR = "figures/04_decontx"
os.makedirs(OUTDIR, exist_ok=True)
AZIZI = "../../Azizi/data/azizi_10x_TNK.h5ad"
MARKERS = {
    "TRDC":            lambda g: g("TRDC") > 0,
    "TRGC1":           lambda g: g("TRGC1") > 0,
    "TRGC2":           lambda g: g("TRGC2") > 0,
    "TRGC1|TRGC2":     lambda g: (g("TRGC1") + g("TRGC2")) > 0,
    "TRGV9":           lambda g: g("TRGV9") > 0,
    "TRDC & TRGC":     lambda g: (g("TRDC") > 0) & ((g("TRGC1") + g("TRGC2")) > 0),
    "TRDC>=2":         lambda g: g("TRDC") >= 2,
}


def add_decontx(a):
    genes = pd.read_csv("data/decontx/genes.csv")
    mats, cells, cont = [], [], []
    for f in sorted(glob.glob("data/decontx/out/*.h5")):
        with h5py.File(f, "r") as h:
            n, m = h["shape"][:]
            mats.append(sparse.csr_matrix((h["data"][:], h["indices"][:], h["indptr"][:]),
                                          shape=(int(n), int(m))))
            cells += [c.decode() for c in h["cells"][:]]
            cont.append(h["contamination"][:])
    D = sparse.vstack(mats).tocsr()
    pos = pd.Index(cells).get_indexer(a.obs_names)
    assert (pos >= 0).all(), "cells missing from decontX output"
    gid = genes.set_index("gene_id").index.get_indexer(a.var["gene_ids"])
    D = D[pos][:, gid]
    D.data = np.round(D.data)
    D.eliminate_zeros()
    a.layers["decontX"] = D.astype(np.float32)
    a.obs["decontX_contamination"] = np.concatenate(cont)[pos]
    return a


def getter(a, layer):
    C = a.layers[layer]
    return lambda x: (np.asarray(C[:, a.var_names.get_loc(x)].todense()).ravel()
                      if x in a.var_names else np.zeros(a.n_obs))


def groups(a, g, t_mask):
    cd3 = (g("CD3D") + g("CD3E") + g("CD3G")) > 0
    nk = (g("KLRF1") + g("FCER1G")) > 0
    cd = (g("CD4") + g("CD8B")) > 0
    return t_mask & cd3 & ~nk & cd, t_mask & cd3 & ~nk & ~cd


def test(a, layer, t_mask, label):
    g = getter(a, layer)
    ab, dn = groups(a, g, t_mask)
    rows = []
    for k, f in MARKERS.items():
        m = f(g)
        r_ab, r_dn = m[ab].mean(), m[dn].mean()
        rows.append({"data": label, "marker": k, "% in sure-αβ": round(100 * r_ab, 2),
                     "% in DN candidates": round(100 * r_dn, 2),
                     "ratio DN/αβ": round(r_dn / r_ab, 2) if r_ab > 0 else np.inf,
                     "est. precision": round((r_dn - r_ab) / r_dn, 2) if r_dn > 0 else np.nan,
                     "n DN+": int((m & dn).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = add_decontx(sc.read_h5ad("data/klughammer_TNK.h5ad"))
    T = (a.obs["author_cell_type"] == "T").values
    print(f"decontX contamination (median): T {np.median(a.obs.loc[T,'decontX_contamination']):.3f}, "
          f"NK {np.median(a.obs.loc[~T,'decontX_contamination']):.3f}")

    az = sc.read_h5ad(AZIZI)
    res = pd.concat([test(az, "counts", np.ones(az.n_obs, bool), "Azizi 10x 5' (reference)"),
                     test(a, "counts", T, "Klughammer raw"),
                     test(a, "decontX", T, "Klughammer decontX")])
    res.to_csv(f"{OUTDIR}/background_test.csv", index=False)
    print(res.to_string(index=False))

    rows = []
    for layer in ["counts", "decontX"]:
        g = getter(a, layer)
        ab, dn = groups(a, g, T)
        trdc = g("TRDC") > 0
        for p in a.obs["patient"].cat.categories:
            pm = (a.obs["patient"] == p).values
            if (ab & pm).sum() >= 30 and (dn & pm).sum() >= 30:
                rows.append({"patient": p, "layer": layer,
                             "TRDC % sure-αβ": 100 * trdc[ab & pm].mean(),
                             "TRDC % DN": 100 * trdc[dn & pm].mean(),
                             "pct_NK": 100 * (a.obs.loc[pm, "author_cell_type"] == "NK").mean()})
    pp = pd.DataFrame(rows)
    pp.to_csv(f"{OUTDIR}/background_per_patient.csv", index=False)
    w = pp.pivot(index="patient", columns="layer", values=["TRDC % sure-αβ", "TRDC % DN"]).round(1)
    print("\nPer patient (≥30 cells per group):\n" + w.to_string())
    print("\npatients with DN > αβ:", {l: int((d["TRDC % DN"] > d["TRDC % sure-αβ"]).sum())
                                      for l, d in pp.groupby("layer")}, "of", pp.patient.nunique())
    a.write_h5ad("data/klughammer_TNK.h5ad", compression="gzip")
