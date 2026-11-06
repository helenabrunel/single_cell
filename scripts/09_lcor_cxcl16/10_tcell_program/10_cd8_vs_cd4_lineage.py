"""
10_cd8_vs_cd4_lineage.py
==========================
Differentiates CD8 vs CD4 T cells using CellTypist's OWN label semantics
(not a custom CD8A/B-vs-CD4 marker threshold) -- reuses the per-cell
CellTypist predictions already saved by 09_celltypist_annotation_allT.py
(run on ALL T cells: CD4+CD8+Treg together), so CellTypist is not rerun.

LINEAGE MAP (from the Immune_All_Low label vocabulary actually observed)
---------------------------------------------------------------------------
CD8 lineage: Tem/Trm cytotoxic T cells, Trm cytotoxic T cells,
  Tem/Temra cytotoxic T cells, Tcm/Naive cytotoxic T cells, CD8a/a,
  CD8a/b(entry)
CD4 lineage: Tcm/Naive helper T cells, Tem/Effector helper T cells,
  Follicular helper T cells, Type 1 helper T cells, Type 17 helper T cells,
  Memory CD4+ cytotoxic T cells (explicitly CD4+ despite "cytotoxic" name)
Treg: Regulatory T cells, Treg(diff)  -- reported but not a focus here
Other_T (unconventional, excluded from the CD8/CD4 comparison): Cycling T
  cells, MAIT cells, gamma-delta T cells (+ CRTAM+ variant), NKT cells,
  T(agonist), thymocyte states
Non_T: everything else (NK/B/myeloid/stromal CellTypist calls) -- marker-
  based T-cell gating in this project is imperfect, as already noted;
  excluded here.

For each lineage (CD8, CD4) separately: per-patient TRM_fraction and mean
Exhaustion_score, correlated against LCOR_raw and LCOR_OE_malignant score
(Bassez-style scatter, same as 06/08), in both datasets -- this is the
direct test of whether any pattern is CD8-specific or also present in CD4.

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/10_cd8_vs_cd4_lineage/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/10_cd8_vs_cd4_lineage"
SRC = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/09_celltypist_annotation_allT"
os.makedirs(OUTDIR, exist_ok=True)

CD8_LABELS = {"Tem/Trm cytotoxic T cells", "Trm cytotoxic T cells", "Tem/Temra cytotoxic T cells",
              "Tcm/Naive cytotoxic T cells", "CD8a/a", "CD8a/b(entry)"}
CD4_LABELS = {"Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Follicular helper T cells",
              "Type 1 helper T cells", "Type 17 helper T cells", "Memory CD4+ cytotoxic T cells"}
TREG_LABELS = {"Regulatory T cells", "Treg(diff)"}
TRM_LABELS = {"Trm cytotoxic T cells", "Tem/Trm cytotoxic T cells"}


def lineage_of(label):
    if label in CD8_LABELS:
        return "CD8"
    if label in CD4_LABELS:
        return "CD4"
    if label in TREG_LABELS:
        return "Treg"
    return None  # Other_T / Non_T -- excluded from this comparison


subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}


def bassez_plot(d, xcol, ycol, xlabel, ylabel, title, fname):
    dd = d.dropna(subset=[xcol, ycol])
    if len(dd) < 4:
        print(f"  SKIP {fname}: n={len(dd)}")
        return
    x, y = dd[xcol].values, dd[ycol].values
    if np.std(y) == 0:
        print(f"  SKIP {fname}: {ycol} is constant in this subset (structurally undefined, e.g. TRM label set doesn't exist in this lineage)")
        return
    r, p = stats.pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(7, 5.3))
    slope, intercept, _, _, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    yhat = intercept + slope * xs
    n = len(x); dof = n - 2
    x_mean, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid ** 2) / dof) if dof > 0 else np.nan
    t_val = stats.t.ppf(0.975, dof) if dof > 0 else np.nan
    ci = t_val * s_err * np.sqrt(1 / n + (xs - x_mean) ** 2 / sxx) if dof > 0 else np.zeros_like(xs)
    ax.fill_between(xs, yhat - ci, yhat + ci, color="gray", alpha=0.3, linewidth=0)
    ax.plot(xs, yhat, color="black", lw=1.5)
    for st, color in subtype_colors.items():
        m = dd["subtype"].values == st
        if m.sum() == 0:
            continue
        ax.scatter(x[m], y[m], color=color, s=70, label=f"{st} (n={m.sum()})", edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(0.02, 0.97, f"R = {r:.2f}\np-value = {p:.2g}", transform=ax.transAxes, fontsize=12, va="top", ha="left")
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="Subtype", loc="best", fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png  (R={r:.2f}, p={p:.3g}, n={n})")


for name in ["Visvader", "Klughammer"]:
    print(f"\n=== {name} ===")
    df = pd.read_csv(f"{SRC}/{name}_percell_labels.csv")
    df["lineage"] = df["label"].map(lineage_of)
    print(f"  Lineage cell counts:\n{df['lineage'].value_counts(dropna=False).to_string()}")

    # per-patient lineage proportions (of all cells assigned a T/Treg lineage)
    assigned = df.dropna(subset=["lineage"])
    lineage_prop = pd.crosstab(assigned["sample"], assigned["lineage"], normalize="index")
    lineage_prop.to_csv(f"{OUTDIR}/{name}_persample_lineage_proportions.csv")

    master = pd.read_csv(f"LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/06_scatterplots/{name.lower()}_master_data.csv")

    for lineage in ["CD8", "CD4"]:
        d_lin = df[df["lineage"] == lineage]
        patient_stats = d_lin.groupby("sample").agg(
            n_cells=("label", "size"), TRM_fraction=("is_trm", "mean"),
            mean_Exhaustion_score=("Exhaustion_score", "mean"),
        ).reset_index()
        patient_stats.to_csv(f"{OUTDIR}/{name}_{lineage}_persample_TRM_exhaustion.csv", index=False)

        merged = master.merge(patient_stats, on="sample", how="inner")
        merged = merged[merged["n_cells"] >= 10]
        print(f"\n  --- {lineage} lineage (n_cells>=10 patients: {len(merged)}) ---")

        for xcol, xlabel in [("LCOR_raw_pseudobulk", "LCOR in malignant cells (log1p CP10K)"),
                              ("mean_LCOR_OE_score", "LCOR_OE signature score (malignant cells)")]:
            bassez_plot(merged, xcol, "TRM_fraction", xlabel, f"Fraction of {lineage} T cells = TRM (CellTypist)",
                        f"{name}: {xlabel} vs {lineage} TRM fraction", f"{name}_{lineage}_{xcol}_vs_TRM_fraction")
            bassez_plot(merged, xcol, "mean_Exhaustion_score", xlabel,
                        f"{lineage} T-cell exhaustion score (PDCD1/LAG3/HAVCR2/TIGIT/CTLA4/TOX/ENTPD1)",
                        f"{name}: {xlabel} vs {lineage} exhaustion score", f"{name}_{lineage}_{xcol}_vs_Exhaustion_score")

print("\nDONE")
