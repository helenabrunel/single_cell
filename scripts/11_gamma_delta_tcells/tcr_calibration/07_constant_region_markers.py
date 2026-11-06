"""
07_constant_region_markers.py
=============================
Which TCR genes can mark γδ cells when the TCR is not sequenced? (Question
raised for Klughammer, a 3' dataset without TCR: why use TRDC only, and not a
combined score TRDC/TRGC1/TRGC2 positive, TRAC/TRBC1/TRBC2 negative, CD3E
high, CD4 negative?)

Tested on the 10x data, where the TCR (α/β kit) gives an independent truth:
  1. TCR-only specificity: for each marker (> 0 UMI), % of positive cells with
     a productive αβ pair (→ αβ) or with no productive chain. Does not use the
     γδ call, so it does not favour TRDC.
  2. Detection of TCR constant genes, CD3E, CD4, CD8B, KLRF1, FCER1G in γδ
     (gd_call gdT_high/medium) vs αβ-pair cells.
  3. Combined rules (precision/recall vs the γδ call) and AUC of a continuous
     score: mean z(log) of TRDC/TRGC1/TRGC2 minus mean z of TRAC/TRBC1/TRBC2.

Note: the 10x cells are sorted CD3+ T cells, so NK-like γδ cells with low CD3
are probably under-represented; NK-marker exclusion cannot be calibrated here.

Input:  data/azizi_10x_TNK.h5ad
Output: report/data/marker_specificity_tcr.csv, report/data/constant_genes_gd_vs_ab.csv,
        report/data/combined_rules.csv

Run from: ~/single_cell/Azizi/
  python3 07_constant_region_markers.py > logs/07_constant_region_markers.log
"""

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = "report/data"
DET_GENES = ["TRDC", "TRGC1", "TRGC2", "TRAC", "TRBC1", "TRBC2", "CD3E", "CD4", "CD8B",
             "KLRF1", "FCER1G"]


def getter(a):
    C = a.layers["counts"]
    return lambda x: np.asarray(C[:, a.var_names.get_loc(x)].todense()).ravel()


def lognorm(a):
    tot = np.asarray(a.layers["counts"].sum(1)).ravel()
    g = getter(a)
    return lambda x: np.log1p(g(x) / tot * 1e4)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    a = sc.read_h5ad("data/azizi_10x_TNK.h5ad")
    g = getter(a)
    tra = a.obs["has_TRA_productive"].astype(str).eq("True").values
    trb = a.obs["has_TRB_productive"].astype(str).eq("True").values
    pair, none = tra & trb, ~tra & ~trb
    gd = a.obs["gd_call"].isin(["gdT_high", "gdT_medium"]).values
    ab = pair & ~gd

    trgc = (g("TRGC1") + g("TRGC2")) > 0
    abc0 = (g("TRAC") + g("TRBC1") + g("TRBC2")) == 0
    markers = {
        "TRDC": g("TRDC") > 0, "TRDV1": g("TRDV1") > 0, "TRDV2": g("TRDV2") > 0,
        "TRGC1": g("TRGC1") > 0, "TRGC2": g("TRGC2") > 0, "TRGC1 or TRGC2": trgc,
        "TRDC and (TRGC1 or TRGC2)": (g("TRDC") > 0) & trgc,
        "TRAC = TRBC1 = TRBC2 = 0": abc0,
        "all cells (baseline)": np.ones(a.n_obs, bool)}
    spec = pd.DataFrame([{"marker": k, "positive cells": int(m.sum()),
                          "% with productive αβ pair": round(100 * pair[m].mean(), 1),
                          "% with no productive chain": round(100 * none[m].mean(), 1)}
                         for k, m in markers.items()])
    print("── 1. TCR-only specificity ──\n" + spec.to_string(index=False))
    spec.to_csv(f"{OUT}/marker_specificity_tcr.csv", index=False)

    det = pd.DataFrame({
        "γδ: % detected": [100 * (g(x)[gd] > 0).mean() for x in DET_GENES],
        "γδ: median UMI": [np.median(g(x)[gd]) for x in DET_GENES],
        "αβ: % detected": [100 * (g(x)[ab] > 0).mean() for x in DET_GENES],
        "αβ: median UMI": [np.median(g(x)[ab]) for x in DET_GENES]}, index=DET_GENES).round(1)
    det.index.name = "gene"
    print(f"\n── 2. γδ (n={gd.sum()}) vs αβ pair (n={ab.sum()}) ──\n" + det.to_string())
    print(f"TRAC = TRBC1 = TRBC2 = 0: γδ {100 * abc0[gd].mean():.1f} %, αβ {100 * abc0[ab].mean():.1f} %")
    det.to_csv(f"{OUT}/constant_genes_gd_vs_ab.csv")

    gdc = g("TRDC") + g("TRGC1") + g("TRGC2")
    cd3e, cd4 = g("CD3E"), g("CD4")
    rules = {
        "TRDC > 0": g("TRDC") > 0,
        "(TRDC or TRGC)+, TRAC/TRBC = 0, CD3E+, CD4−": (gdc > 0) & abc0 & (cd3e > 0) & (cd4 == 0),
        "TRDC+ and TRGC+, TRAC/TRBC = 0, CD3E+, CD4−": (g("TRDC") > 0) & trgc & abc0 & (cd3e > 0) & (cd4 == 0),
        "TRDC+TRGC UMI > TRAC+TRBC UMI, CD3E+, CD4−":
            (gdc > g("TRAC") + g("TRBC1") + g("TRBC2")) & (cd3e > 0) & (cd4 == 0),
        "Klughammer rule: TRDC+, CD3+, CD4−CD8B−, KLRF1−FCER1G−":
            (g("TRDC") > 0) & ((g("CD3D") + cd3e + g("CD3G")) > 0) & ((cd4 + g("CD8B")) == 0)
            & ((g("KLRF1") + g("FCER1G")) == 0)}
    L = lognorm(a)
    z = lambda x: (L(x) - L(x).mean()) / (L(x).std() + 1e-9)
    score = (z("TRDC") + z("TRGC1") + z("TRGC2")) / 3 - (z("TRAC") + z("TRBC1") + z("TRBC2")) / 3
    keep = gd | ab
    rr = pd.DataFrame([{"rule": k, "cells passing": int(m.sum()),
                        "precision": round((m & gd).sum() / max(m.sum(), 1), 2),
                        "recall": round((m & gd).sum() / gd.sum(), 2)} for k, m in rules.items()])
    rr = pd.concat([rr, pd.DataFrame([
        {"rule": "continuous score (TRDC/TRGC z − TRAC/TRBC z): AUC γδ vs αβ",
         "precision": round(roc_auc_score(gd[keep], score[keep]), 3)},
        {"rule": "TRDC alone (log-normalised): AUC γδ vs αβ",
         "precision": round(roc_auc_score(gd[keep], L("TRDC")[keep]), 3)}])])
    print("\n── 3. Combined rules (precision/recall vs γδ call; AUC rows in 'precision') ──\n"
          + rr.to_string(index=False))
    rr.to_csv(f"{OUT}/combined_rules.csv", index=False)
