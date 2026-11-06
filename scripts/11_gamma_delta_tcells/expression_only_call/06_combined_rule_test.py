"""
06_combined_rule_test.py
========================
Follow-up questions (Oct 2026):
  (1) 34.7 % of author-NK cells express CD3E → are the author NK labels wrong?
  (2) test a combined rule: TRDC/TRGC1/TRGC2 positive, TRAC/TRBC1/TRBC2
      negative, CD3E high, CD4 negative (instead of TRDC alone).

Part A — Azizi 10x 5' (TCR truth = gd_call gdT_high/medium; αβ = productive
         TRA+TRB pair): detection of TRAC/TRBC in true γδ, precision/recall of
         the combined rules, and AUC of a continuous combined score.
Part B — Klughammer, author labels IGNORED: rules applied to all T/NK cells;
         background test (rate in CD8B+ cells, which are αβ, vs CD4-CD8B- cells).
Part C — author-NK cells that express CD3E: co-expression of other T/NK genes
         and Leiden-cluster membership.

Input:  data/klughammer_TNK.h5ad, ../../Azizi/data/azizi_10x_TNK.h5ad
Output: figures/06_combined_rule/, logs via stdout

Run from: ~/single_cell/Klughammer/gdT/
  python3 06_combined_rule_test.py > logs/06_combined_rule_test.log
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

OUTDIR = "figures/06_combined_rule"
os.makedirs(OUTDIR, exist_ok=True)
AZIZI = "../../Azizi/data/azizi_10x_TNK.h5ad"


def getter(a):
    C = a.layers["counts"]
    return lambda x: (np.asarray(C[:, a.var_names.get_loc(x)].todense()).ravel()
                      if x in a.var_names else np.zeros(a.n_obs))


def lognorm(a):
    tot = np.asarray(a.layers["counts"].sum(1)).ravel()
    g = getter(a)
    return lambda x: np.log1p(g(x) / tot * 1e4)


def rules(g):
    gd = g("TRDC") + g("TRGC1") + g("TRGC2")
    ab = g("TRAC") + g("TRBC1") + g("TRBC2")
    cd3e = g("CD3E")
    cd4 = g("CD4")
    return {
        "TRDC>0 (reference)": g("TRDC") > 0,
        "S1: (TRDC|TRGC)+ & TRAC/TRBC=0 & CD3E+ & CD4-":
            (gd > 0) & (ab == 0) & (cd3e > 0) & (cd4 == 0),
        "S2: TRDC+ & TRGC+ & TRAC/TRBC=0 & CD3E+ & CD4-":
            (g("TRDC") > 0) & ((g("TRGC1") + g("TRGC2")) > 0) & (ab == 0) & (cd3e > 0) & (cd4 == 0),
        "S3: (TRDC|TRGC) > TRAC+TRBC & CD3E+ & CD4-":
            (gd > ab) & (cd3e > 0) & (cd4 == 0),
        "S4: S1 with CD3E>=2":
            (gd > 0) & (ab == 0) & (cd3e >= 2) & (cd4 == 0),
        "previous rule (TRDC+ CD3+ CD4-CD8B- KLRF1-FCER1G-)":
            (g("TRDC") > 0) & ((g("CD3D") + g("CD3E") + g("CD3G")) > 0)
            & ((g("CD4") + g("CD8B")) == 0) & ((g("KLRF1") + g("FCER1G")) == 0),
    }


POSITIVE = {
    "TRDC": lambda g: g("TRDC") > 0,
    "TRGC1": lambda g: g("TRGC1") > 0,
    "TRGC2": lambda g: g("TRGC2") > 0,
    "TRGC1 or TRGC2": lambda g: (g("TRGC1") + g("TRGC2")) > 0,
    "TRDC or TRGC1 or TRGC2": lambda g: (g("TRDC") + g("TRGC1") + g("TRGC2")) > 0,
    "TRDC and (TRGC1 or TRGC2)": lambda g: (g("TRDC") > 0) & ((g("TRGC1") + g("TRGC2")) > 0),
}


def exclusion(g):
    """Exclusion part of the previous rule: CD3+ CD4- CD8B- KLRF1- FCER1G-."""
    return (((g("CD3D") + g("CD3E") + g("CD3G")) > 0) & ((g("CD4") + g("CD8B")) == 0)
            & ((g("KLRF1") + g("FCER1G")) == 0))


def positive_markers(ga, truth, g, T):
    """Each γδ constant-region marker alone and with the exclusion filters:
    precision/recall in Azizi (TCR truth); background ratio in Klughammer author-T cells
    (DN candidates vs sure-αβ, as in 04c)."""
    cd3 = (g("CD3D") + g("CD3E") + g("CD3G")) > 0
    nkm = (g("KLRF1") + g("FCER1G")) > 0
    cd = (g("CD4") + g("CD8B")) > 0
    ab, dn = T & cd3 & ~nkm & cd, T & cd3 & ~nkm & ~cd
    rows = []
    for k, f in POSITIVE.items():
        m_a = f(ga); m_ae = m_a & exclusion(ga); m_k = f(g)
        rows.append({"positive marker": k,
                     "Azizi precision (alone)": round((m_a & truth).sum() / m_a.sum(), 2),
                     "Azizi recall (alone)": round((m_a & truth).sum() / truth.sum(), 2),
                     "Azizi precision (+ exclusion)": round((m_ae & truth).sum() / m_ae.sum(), 2),
                     "Azizi recall (+ exclusion)": round((m_ae & truth).sum() / truth.sum(), 2),
                     "Klughammer ratio DN/αβ": round(m_k[dn].mean() / m_k[ab].mean(), 2)})
    return pd.DataFrame(rows)


def score(a):
    """Continuous score: mean z(log) of γδ constant genes minus mean z of αβ constant genes."""
    L = lognorm(a)
    z = lambda x: (L(x) - L(x).mean()) / (L(x).std() + 1e-9)
    return (z("TRDC") + z("TRGC1") + z("TRGC2")) / 3 - (z("TRAC") + z("TRBC1") + z("TRBC2")) / 3


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    # ── Part A: Azizi (TCR truth) ──
    az = sc.read_h5ad(AZIZI)
    ga = getter(az)
    truth = az.obs["gd_call"].isin(["gdT_high", "gdT_medium"]).values
    abpair = (az.obs["has_TRA_productive"].astype(bool) & az.obs["has_TRB_productive"].astype(bool)).values & ~truth
    print(f"Azizi: TCR-γδ {truth.sum()}, TCR-αβ pair {abpair.sum()}, total {az.n_obs}")

    genes = ["TRDC", "TRGC1", "TRGC2", "TRAC", "TRBC1", "TRBC2", "CD3E", "CD4", "CD8B"]
    det = pd.DataFrame({
        "TCR-γδ % detected": [100 * (ga(x)[truth] > 0).mean() for x in genes],
        "TCR-γδ median UMI": [np.median(ga(x)[truth]) for x in genes],
        "TCR-αβ % detected": [100 * (ga(x)[abpair] > 0).mean() for x in genes],
        "TCR-αβ median UMI": [np.median(ga(x)[abpair]) for x in genes]}, index=genes).round(1)
    abz = (ga("TRAC") + ga("TRBC1") + ga("TRBC2")) == 0
    print("\n── A1. Azizi: detection in TCR-defined γδ vs αβ ──\n" + det.to_string())
    print(f"TRAC=TRBC1=TRBC2=0 in TCR-γδ: {100*abz[truth].mean():.1f} %, in TCR-αβ: {100*abz[abpair].mean():.1f} %")
    det.to_csv(f"{OUTDIR}/azizi_detection_gd_vs_ab.csv")

    rows = []
    for k, m in rules(ga).items():
        rows.append({"rule": k, "n passing": int(m.sum()),
                     "precision": round((m & truth).sum() / max(m.sum(), 1), 2),
                     "recall": round((m & truth).sum() / truth.sum(), 2)})
    ra = pd.DataFrame(rows)
    print("\n── A2. Azizi: rules vs TCR truth ──\n" + ra.to_string(index=False))
    ra.to_csv(f"{OUTDIR}/azizi_rules.csv", index=False)

    s = score(az)
    keep = truth | abpair
    print(f"\nA3. Azizi combined score (γδ C genes − αβ C genes): AUC γδ vs αβ = "
          f"{roc_auc_score(truth[keep], s[keep]):.3f}")
    print(f"    AUC of TRDC alone = {roc_auc_score(truth[keep], np.log1p(ga('TRDC'))[keep]):.3f}")

    # ── Part B: Klughammer, author labels ignored ──
    a = sc.read_h5ad("data/klughammer_TNK.h5ad")
    g = getter(a)
    T = (a.obs["author_cell_type"] == "T").values

    pm = positive_markers(ga, truth, g, T)
    print("\n── A4/B0. Positive marker choice (exclusion = CD3+ CD4- CD8B- KLRF1- FCER1G-) ──\n"
          + pm.to_string(index=False))
    pm.to_csv(f"{OUTDIR}/positive_markers.csv", index=False)
    cd8b = g("CD8B") > 0
    dn = (g("CD4") == 0) & (g("CD8B") == 0)
    rows = []
    for k, m in rules(g).items():
        rows.append({"rule": k, "n passing (all T/NK)": int(m.sum()),
                     "of which author NK": int((m & ~T).sum()),
                     "% in CD8B+ (αβ)": round(100 * m[cd8b].mean(), 2),
                     "% in CD4-CD8B-": round(100 * m[dn].mean(), 2),
                     "ratio DN/CD8B+": round(m[dn].mean() / max(m[cd8b].mean(), 1e-9), 2)})
    # Azizi with the same test, as reference
    cd8b_a = ga("CD8B") > 0; dn_a = (ga("CD4") == 0) & (ga("CD8B") == 0)
    for r, (k, m) in zip(rows, rules(ga).items()):
        r["Azizi ratio DN/CD8B+"] = round(m[dn_a].mean() / max(m[cd8b_a].mean(), 1e-9), 2)
    rk = pd.DataFrame(rows)
    print("\n── B1. Klughammer (all T/NK, no author labels): background test ──\n" + rk.to_string(index=False))
    rk.to_csv(f"{OUTDIR}/klughammer_rules_background.csv", index=False)

    s_k = score(a)
    print("\nB2. Combined score: mean in CD8B+ vs CD4-CD8B- (Klughammer / Azizi)")
    print(f"    Klughammer: CD8B+ {s_k[cd8b].mean():.3f}, DN {s_k[dn].mean():.3f}")
    print(f"    Azizi:      CD8B+ {s[cd8b_a].mean():.3f}, DN {s[dn_a].mean():.3f}, TCR-γδ {s[truth].mean():.3f}")
    top = s_k >= np.quantile(s_k, 0.95)
    print(f"    Klughammer top-5 % score cells: {100*(~T[top]).mean():.1f} % author NK, "
          f"KLRF1+ {100*(g('KLRF1')[top]>0).mean():.1f} %, CD8B+ {100*cd8b[top].mean():.1f} %")

    # ── Part C: CD3E in author-NK cells ──
    nk = ~T
    c3 = g("CD3E") > 0
    cg = ["CD3D", "CD3G", "TRAC", "TRBC1", "TRBC2", "CD8B", "CD4", "CD5", "CD28", "IL7R",
          "KLRF1", "FCER1G", "NCAM1", "FCGR3A", "TYROBP", "TRDC"]
    tab = pd.DataFrame({
        "NK CD3E+": [100 * (g(x)[nk & c3] > 0).mean() for x in cg],
        "NK CD3E-": [100 * (g(x)[nk & ~c3] > 0).mean() for x in cg],
        "T (author)": [100 * (g(x)[T] > 0).mean() for x in cg]}, index=cg).round(1)
    print(f"\n── C1. Author-NK cells: CD3E+ {int((nk&c3).sum())} / {int(nk.sum())} ──")
    print("CD3E UMI distribution in NK CD3E+:", pd.Series(g("CD3E")[nk & c3]).value_counts().sort_index().head(6).to_dict())
    print("CD3E UMI distribution in T CD3E+:  median", np.median(g("CD3E")[T & (g("CD3E") > 0)]))
    print(tab.to_string())
    tab.to_csv(f"{OUTDIR}/nk_cd3e_coexpression.csv")

    tot = np.asarray(a.layers["counts"].sum(1)).ravel()
    print(f"median UMI/cell: NK CD3E+ {np.median(tot[nk&c3]):.0f}, NK CD3E- {np.median(tot[nk&~c3]):.0f}, T {np.median(tot[T]):.0f}")

    # Leiden clusters: are CD3E+ NK cells in NK clusters or T clusters?
    cl = a.obs["leiden"].astype(str).values
    comp = pd.crosstab(cl, a.obs["author_cell_type"].values)
    nk_clusters = comp.index[(comp["NK"] / comp.sum(1)) > 0.5]
    print(f"\nNK-dominated Leiden clusters (>50 % author NK): {list(nk_clusters)}")
    in_nkcl = np.isin(cl, nk_clusters)
    print(f"NK CD3E+ cells in NK-dominated clusters: {100*in_nkcl[nk&c3].mean():.1f} %; "
          f"NK CD3E- cells: {100*in_nkcl[nk&~c3].mean():.1f} %")
    # strict 'T-like' NK: CD3E+ & (CD3D|CD3G)+ & TRAC+ & KLRF1- & FCER1G-
    tl = nk & c3 & ((g("CD3D") + g("CD3G")) > 0) & (g("TRAC") > 0) & ((g("KLRF1") + g("FCER1G")) == 0)
    print(f"author-NK with full T-like profile (CD3E+ CD3D/G+ TRAC+ KLRF1- FCER1G-): {int(tl.sum())} "
          f"({100*tl.sum()/nk.sum():.1f} % of NK)")
