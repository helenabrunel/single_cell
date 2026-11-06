"""
06_gd_signatures.py
===================
Functional-state signatures for γδ T cells.

Signatures are scored (sc.tl.score_genes) on the WHOLE T/NK object, so the
background/control genes come from all T cells and αβ CD4/CD8 cells can be
used as a reference. Detection was checked first (see logs):
  - IL-17 / γδT17 (IL17A, IL17F, RORC, IL23R, CCR6): IL17A/F detected in 0
    γδ cells on both platforms → not scored, reported as detection only.
  - Proliferation (MKI67, TOP2A…): ~1 % of γδ cells → not scored.
  - Exhaustion and tissue residency are sparse in inDrop (2–10 % detection)
    → computed but flagged as low-information on that platform.

Analyses
  10x  (tumour, 3 patients, Vδ resolved)
    - mean score per group: Vδ1 / Vδ2 (incl. Vγ9Vδ2) / Vδ3 / αβ CD8 / αβ CD4 / Treg
      (αβ groups from STCAT labels, which are reasonable for αβ cells)
    - Vδ1 vs Vδ2: cell-level Wilcoxon (BH) + per-patient mean difference,
      to check the direction is consistent in each patient
    - DE genes Vδ1 vs Vδ2 (Wilcoxon), γδ subclusters × signatures
  inDrop (8 patients, 4 tissues, no Vδ)
    - γδ score per patient × tissue (pseudobulk means, ≥ 10 cells)

Main findings (2026-10-16, see logs/06_gd_signatures.log):
  - IL-17 programme not detected in γδ cells on either platform.
  - Vδ1 vs Vδ2 (10x): Exhaustion ↑ in Vδ1, Naive/memory and NK receptors ↑ in
    Vδ2 in 3/3 patients, but effect sizes come mostly from BC09 (largest
    sample); cytotoxicity similar. Cell-level p-values are pseudo-replicated —
    use the per-patient differences.
  - γδ subclusters are partly patient-driven (cl.5 = BC11, cl.1 = BC09 Vδ2);
    only cl.0 (Vδ1, exhausted/resident/IFN-γ) is shared by all 3 patients.

Input:  data/azizi_{10x,indrop}_TNK.h5ad, data/azizi_{10x,indrop}_gdT.h5ad
Output: obs columns sig_* in both objects, figures/06_gd_signatures/,
        report/data/{plat}_signatures.csv.gz

Run from: ~/single_cell/Azizi/
  python3 06_gd_signatures.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "figures/06_gd_signatures"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs("report/data", exist_ok=True)

SIGNATURES = {
    "Cytotoxicity"    : ["GZMA", "GZMB", "GZMH", "PRF1", "GNLY", "NKG7", "CST7", "CTSW"],
    "Exhaustion"      : ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4", "ENTPD1", "LAYN"],
    "Tissue_residency": ["ITGAE", "CD69", "CXCR6", "ZNF683", "RGS1", "ITGA1"],
    "Naive_memory"    : ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "KLF2", "S1PR1"],
    "IFNg_type1"      : ["IFNG", "TBX21", "TNF", "CCL3", "CCL4", "CCL5"],
    "NK_receptors"    : ["KLRC1", "KLRC2", "KLRK1", "NCR1", "NCR3", "KIR2DL4", "FCGR3A", "TYROBP"],
}
DETECTION_ONLY = {
    "IL17_type17"  : ["IL17A", "IL17F", "RORC", "IL23R", "CCR6"],
    "Proliferation": ["MKI67", "TOP2A", "CENPF", "TYMS"],
}
LOW_INFO_INDROP = ["Exhaustion", "Tissue_residency"]
SIG_COLS = [f"sig_{k}" for k in SIGNATURES]
MIN_CELLS = 10


def score(adata):
    for k, genes in SIGNATURES.items():
        g = [x for x in genes if x in adata.var_names]
        sc.tl.score_genes(adata, g, score_name=f"sig_{k}", random_state=0)
    return adata


def group_10x(o):
    lab = o["stcat_label"].astype(str)
    grp = np.where(lab.str.contains("Treg"), "αβ Treg",
          np.where(lab.str.startswith("CD8"), "αβ CD8",
          np.where(lab.str.startswith("CD4"), "αβ CD4", "αβ other")))
    grp = pd.Series(grp, index=o.index)
    grp[o["gd_call"] == "TRDC+_abPair"] = "excluded"
    fam = o["Vd_family"].astype(str)
    grp[o["is_gdT"]] = ("γδ " + fam[o["is_gdT"]]).str.replace("γδ Vd", "γδ Vδ")
    return grp


def detection_table(gd, plat):
    rows = []
    for k, genes in {**SIGNATURES, **DETECTION_ONLY}.items():
        for g in genes:
            if g in gd.var_names:
                det = float((gd[:, g].layers["counts"].toarray() > 0).mean() * 100)
            else:
                det = np.nan
            rows.append({"platform": plat, "signature": k, "gene": g,
                         "pct_detected_in_gd": round(det, 1),
                         "scored": k in SIGNATURES})
    return pd.DataFrame(rows)


def heatmap(means, title, fname, annot_n=None):
    z = (means - means.mean()) / means.std()
    fig, ax = plt.subplots(figsize=(1.1 * z.shape[1] + 2.5, 0.45 * z.shape[0] + 1.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels([c.replace("sig_", "") for c in z.columns], rotation=45, ha="right")
    ylab = [f"{i} (n={annot_n[i]})" if annot_n is not None else i for i in z.index]
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels(ylab)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{means.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(z.values[i, j]) > 1.2 else "black")
    plt.colorbar(im, ax=ax, label="z (across groups)", shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}", dpi=150)
    plt.close()


def vd1_vs_vd2(o):
    d = o[o["group"].isin(["γδ Vδ1", "γδ Vδ2"])]
    rows = []
    for s in SIG_COLS:
        a = d.loc[d.group == "γδ Vδ1", s]; b = d.loc[d.group == "γδ Vδ2", s]
        p = mannwhitneyu(a, b).pvalue
        per_pat = {}
        for pat, dd in d.groupby("patient", observed=True):
            n1 = (dd.group == "γδ Vδ1").sum(); n2 = (dd.group == "γδ Vδ2").sum()
            if n1 >= MIN_CELLS and n2 >= MIN_CELLS:
                per_pat[pat] = (dd.loc[dd.group == "γδ Vδ1", s].mean()
                                - dd.loc[dd.group == "γδ Vδ2", s].mean())
        diff = a.mean() - b.mean()
        same = sum(np.sign(v) == np.sign(diff) for v in per_pat.values())
        rows.append({"signature": s.replace("sig_", ""), "mean_Vd1": a.mean(),
                     "mean_Vd2": b.mean(), "diff_Vd1_minus_Vd2": diff, "p": p,
                     "patients_same_direction": f"{same}/{len(per_pat)}",
                     **{f"diff_{k}": v for k, v in per_pat.items()}})
    r = pd.DataFrame(rows)
    r["p_adj"] = multipletests(r["p"], method="fdr_bh")[1]
    return r, d


def violins_by_patient(d, fname):
    fig, axs = plt.subplots(1, len(SIG_COLS), figsize=(3 * len(SIG_COLS), 3.6), sharey=False)
    pats = sorted(d.patient.unique())
    for ax, s in zip(axs, SIG_COLS):
        data, pos, cols = [], [], []
        for i, p in enumerate(pats):
            for j, (gname, c) in enumerate([("γδ Vδ1", "#1b9e77"), ("γδ Vδ2", "#7570b3")]):
                v = d.loc[(d.patient == p) & (d.group == gname), s].values
                if len(v) >= 3:
                    data.append(v); pos.append(i * 3 + j); cols.append(c)
        parts = ax.violinplot(data, positions=pos, showmedians=True, widths=0.9)
        for b, c in zip(parts["bodies"], cols):
            b.set_facecolor(c); b.set_alpha(0.7)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(pats))])
        ax.set_xticklabels(pats, fontsize=8)
        ax.set_title(s.replace("sig_", ""), fontsize=10)
    axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#1b9e77"),
                           plt.Rectangle((0, 0), 1, 1, color="#7570b3")],
                  labels=["Vδ1", "Vδ2"], fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}", dpi=150)
    plt.close()


def de_vd1_vd2(gd):
    g = gd[gd.obs["Vd_family"].isin(["Vd1", "Vd2"])].copy()
    # exclude TCR genes: they define the groups
    keep = ~g.var_names.str.match(r"^TR[ABDG][VJC]")
    g = g[:, keep].copy()
    sc.tl.rank_genes_groups(g, "Vd_family", method="wilcoxon", pts=True)  # Vd1 vs Vd2 and back
    sig = lambda grp: sc.get.rank_genes_groups_df(g, group=grp).query(
        "pvals_adj < 0.05 and logfoldchanges > 0 and pct_nz_group > 0.2")
    up = sig("Vd1").sort_values("logfoldchanges", ascending=False).head(25)
    dn = sig("Vd2").sort_values("logfoldchanges", ascending=False).head(25)
    de = sc.get.rank_genes_groups_df(g, group="Vd1")
    de.to_csv(f"{OUTDIR}/DE_Vd1_vs_Vd2_10x.csv", index=False)
    genes = list(up.names[:15]) + list(dn.names[:15])
    sc.pl.dotplot(g, genes, groupby="Vd_family", standard_scale="var", show=False,
                  title="Vδ1 vs Vδ2 (10x) — top DE genes (TCR genes excluded)")
    plt.savefig(f"{OUTDIR}/dotplot_DE_Vd1_vs_Vd2_10x.png", dpi=150, bbox_inches="tight")
    plt.close()
    return up, dn


if __name__ == "__main__":
    det = []
    # ── 10x ───────────────────────────────────────────────────────────────────
    tnk = score(sc.read_h5ad("data/azizi_10x_TNK.h5ad"))
    o = tnk.obs
    o["group"] = group_10x(o)
    gd = sc.read_h5ad("data/azizi_10x_gdT.h5ad")
    det.append(detection_table(gd, "10x"))

    keep = ~o["group"].isin(["excluded", "αβ other", "γδ Vδ_undetermined"])
    means = o[keep].groupby("group")[SIG_COLS].mean()
    order = ["γδ Vδ1", "γδ Vδ2", "γδ Vδ3", "αβ CD8", "αβ CD4", "αβ Treg"]
    means = means.loc[[g for g in order if g in means.index]]
    n = o[keep]["group"].value_counts()
    heatmap(means, "10x: signatures by group (mean score)", "heatmap_groups_10x.png", n)
    means.assign(n=n).to_csv(f"{OUTDIR}/signature_means_groups_10x.csv")
    print("── 10x: mean signature score per group ──")
    print(means.round(3).assign(n=n).to_string())

    res, d = vd1_vs_vd2(o)
    res.to_csv(f"{OUTDIR}/Vd1_vs_Vd2_signatures_10x.csv", index=False)
    print("\n── 10x: Vδ1 vs Vδ2 ──")
    print(res[["signature", "mean_Vd1", "mean_Vd2", "diff_Vd1_minus_Vd2", "p_adj",
               "patients_same_direction"]].round(4).to_string(index=False))
    violins_by_patient(d, "violins_Vd1_vs_Vd2_by_patient_10x.png")

    gd.obs = gd.obs.drop(columns=[c for c in gd.obs if c.startswith("sig_")]).join(o[SIG_COLS])
    cl = gd.obs.groupby("gd_leiden", observed=True)[SIG_COLS].mean()
    comp = pd.crosstab(gd.obs["gd_leiden"], gd.obs["Vd_family"])
    comp.columns = comp.columns.astype(str); comp.index = comp.index.astype(str)
    cl.index = cl.index.astype(str)
    heatmap(cl, "10x γδ subclusters: signatures", "heatmap_gd_subclusters_10x.png",
            gd.obs["gd_leiden"].value_counts())
    cl.join(comp).to_csv(f"{OUTDIR}/gd_subclusters_signatures_10x.csv")
    print("\n── 10x γδ subclusters: composition ──\n" + comp.to_string())
    sc.pl.umap(gd, color=["Vd_family", "patient"] + SIG_COLS, ncols=4, cmap="RdBu_r",
               vcenter=0, show=False, frameon=False)
    plt.savefig(f"{OUTDIR}/umap_gd_signatures_10x.png", dpi=150, bbox_inches="tight")
    plt.close()

    up, dn = de_vd1_vd2(gd)
    print("\nDE up in Vδ1:", ", ".join(up.names[:20]))
    print("DE up in Vδ2:", ", ".join(dn.names[:20]))

    tnk.write_h5ad("data/azizi_10x_TNK.h5ad", compression="gzip")
    gd.write_h5ad("data/azizi_10x_gdT.h5ad", compression="gzip")
    o[["patient", "group", "gd_subset", "is_gdT"] + SIG_COLS].rename_axis("cell").to_csv(
        "report/data/10x_signatures.csv.gz")

    # ── inDrop ────────────────────────────────────────────────────────────────
    tnk = score(sc.read_h5ad("data/azizi_indrop_TNK.h5ad"))
    o = tnk.obs
    gd = sc.read_h5ad("data/azizi_indrop_gdT.h5ad")
    det.append(detection_table(gd, "indrop"))
    o["group"] = np.where(o["is_gdT"], "γδ", np.where(o["gd_call"] == "non_gd", "resto T/NK", "excluded"))
    pb = (o[o.group != "excluded"].groupby(["patient", "tissue", "group"], observed=True)
            .agg(n=("group", "size"), **{s: (s, "mean") for s in SIG_COLS}).reset_index())
    pb = pb[pb.n >= MIN_CELLS]
    pb.to_csv(f"{OUTDIR}/pseudobulk_patient_tissue_indrop.csv", index=False)
    print("\n── inDrop: γδ pseudobulk (patient × tissue, ≥10 cells) ──")
    print(pb[pb.group == "γδ"].round(3).to_string(index=False))

    tis = (o[o.is_gdT].groupby("tissue", observed=True)[SIG_COLS].mean())
    heatmap(tis, "inDrop γδ: signatures by tissue (Exhaustion/Residency unreliable)",
            "heatmap_gd_tissue_indrop.png", o[o.is_gdT]["tissue"].value_counts())
    gd.obs = gd.obs.drop(columns=[c for c in gd.obs if c.startswith("sig_")]).join(o[SIG_COLS])
    tnk.write_h5ad("data/azizi_indrop_TNK.h5ad", compression="gzip")
    gd.write_h5ad("data/azizi_indrop_gdT.h5ad", compression="gzip")
    o[["patient", "tissue", "group", "is_gdT"] + SIG_COLS].rename_axis("cell").to_csv(
        "report/data/indrop_signatures.csv.gz")

    det = pd.concat(det)
    det.to_csv(f"{OUTDIR}/signature_gene_detection.csv", index=False)
    det.to_csv("report/data/signature_gene_detection.csv", index=False)
    print("Done.")
