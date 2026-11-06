"""
10_combined_score_call.py
=========================
γδ call with the combined score proposed in review, applied directly:
    score = mean z(log TRDC, TRGC1, TRGC2) − mean z(log TRAC, TRBC1, TRBC2)
    among CD3E+ CD4− cells, call the top X % by score (X = 1, 3, 5, 10 %).
z-scores are computed within each dataset.

Same procedure in Azizi 10x (TCR truth → precision/recall) and Klughammer
(all T/NK cells, author labels not used). For the called cells:
  - Azizi: precision / recall vs the TCR-based γδ call.
  - both: background test — % called among CD8B+ cells (αβ) vs CD8B− cells,
    restricted to KLRF1−FCER1G− cells so NK cells do not drive the ratio;
    % of called cells annotated as NK / KLRF1+;
    detection of genes NOT in the score that separate γδ from αβ in Azizi
    (KLRC1, TYROBP, CD5, CD28, CD8B).

Input:  data/klughammer_TNK.h5ad, ../../Azizi/data/azizi_10x_TNK.h5ad
Output: report/data/combined_score_call.csv, report/data/combined_score_validation.csv

Run from: ~/single_cell/Klughammer/gdT/
  python3 10_combined_score_call.py > logs/10_combined_score_call.log
"""

import scanpy as sc
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
OUT = "report/data"
AZIZI = "../../Azizi/data/azizi_10x_TNK.h5ad"
TOP = [1, 3, 5, 10]
VALID = ["KLRC1", "TYROBP", "CD5", "CD28", "CD8B", "KLRF1"]


def getter(a):
    C = a.layers["counts"]
    return lambda x: np.asarray(C[:, a.var_names.get_loc(x)].todense()).ravel()


def score(a, g):
    tot = np.asarray(a.layers["counts"].sum(1)).ravel()
    L = lambda x: np.log1p(g(x) / tot * 1e4)
    z = lambda x: (L(x) - L(x).mean()) / (L(x).std() + 1e-9)
    return (z("TRDC") + z("TRGC1") + z("TRGC2")) / 3 - (z("TRAC") + z("TRBC1") + z("TRBC2")) / 3


def evaluate(name, a, truth=None, nk_label=None):
    g = getter(a)
    s = score(a, g)
    elig = (g("CD3E") > 0) & (g("CD4") == 0)
    nkm = (g("KLRF1") + g("FCER1G")) > 0
    cd8b = g("CD8B") > 0
    rows, calls = [], {}
    for x in TOP:
        thr = np.quantile(s[elig], 1 - x / 100)
        call = elig & (s >= thr)
        calls[x] = call
        ab, dn = ~nkm & cd8b, ~nkm & ~cd8b
        r = {"dataset": name, "top %": x, "cells called": int(call.sum()),
             "% KLRF1/FCER1G+": round(100 * nkm[call].mean(), 1),
             "% of CD8B+ (αβ) called": round(100 * call[ab].mean(), 2),
             "% of CD8B− called": round(100 * call[dn].mean(), 2)}
        r["ratio CD8B−/CD8B+"] = round(r["% of CD8B− called"] / max(r["% of CD8B+ (αβ) called"], 1e-9), 2)
        if nk_label is not None:
            r["% annotated NK"] = round(100 * nk_label[call].mean(), 1)
        if truth is not None:
            r["precision (TCR)"] = round((call & truth).sum() / call.sum(), 2)
            r["recall (TCR)"] = round((call & truth).sum() / truth.sum(), 2)
        rows.append(r)
    val = {f"{name} top 3 %, not NK-like": {v: round(100 * (g(v)[calls[3] & ~nkm] > 0).mean(), 1) for v in VALID},
           f"{name} other CD3E+CD4− not NK-like": {v: round(100 * (g(v)[elig & ~calls[3] & ~nkm] > 0).mean(), 1)
                                                 for v in VALID}}
    if truth is not None:
        val["Azizi γδ (TCR)"] = {v: round(100 * (g(v)[truth] > 0).mean(), 1) for v in VALID}
    return pd.DataFrame(rows), pd.DataFrame(val)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    az = sc.read_h5ad(AZIZI)
    truth = az.obs["gd_call"].isin(["gdT_high", "gdT_medium"]).values
    ra, va = evaluate("Azizi", az, truth=truth)
    k = sc.read_h5ad("data/klughammer_TNK.h5ad")
    rk, vk = evaluate("Klughammer", k, nk_label=(k.obs["author_cell_type"] == "NK").values)
    res = pd.concat([ra, rk])
    print("── Combined-score call ──\n" + res.to_string(index=False))
    res.to_csv(f"{OUT}/combined_score_call.csv", index=False)
    val = va.join(vk)
    val.index.name = "gene"
    print("\n── Genes not in the score (% detected) ──\n" + val.to_string())
    val.to_csv(f"{OUT}/combined_score_validation.csv")
