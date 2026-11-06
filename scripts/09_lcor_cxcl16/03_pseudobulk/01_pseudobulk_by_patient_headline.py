"""
01_pseudobulk_by_patient_headline.py
====================================
Per-patient PSEUDOBULK of LCOR vs CXCL16, restricted to malignant
(epithelial) cells, correlated directly across patients.

WHY THIS SCRIPT EXISTS
-----------------------
Scripts 14-16 established that the cell-to-cell LCOR-CXCL16 correlation
within malignant cells is essentially null once sequencing depth is
properly accounted for (15: within-patient depth-offset RR~1; 16: the
apparent "multiple lines" are a discrete-UMI/depth artifact, not
subclones). Script 17 then moved to the CXCL16(Malignant)->CXCR6(T_cell)
ligand-receptor interaction and correlated its per-patient LIANA score
against LCOR pseudobulk -- but that correlation (rho=-0.63) collapsed
once subtype was controlled for, i.e. it was confounded by subtype, not
a LCOR effect.

There is a genuine gap between these two approaches:
  - Cell-to-cell (14-16): every malignant cell is one data point. Very
    high n, but dominated by dropout/depth noise per cell, and it cannot
    distinguish "LCOR represses CXCL16 transcription in tumor cells" from
    "LCOR is elevated in patients whose tumor cells happen to also express
    more CXCL16" -- those are different biological claims.
  - LIANA interaction score (17): patient-level, but the outcome is a
    ligand-receptor ENGAGEMENT score (CXCL16 in malignant cells x CXCR6 in
    T cells), which folds in T-cell composition/CXCR6 -- confounded by
    subtype as shown.
This script fills the missing middle: patient-level PSEUDOBULK of LCOR and
CXCL16 BOTH measured in the same compartment (malignant cells only, no
T-cell/CXCR6 term), correlated directly across patients. This is the
single-cell analogue of the bulk RNA-seq LCOR-CXCL16 test already run (6
public datasets, weak/inconsistent, ns in TNBC) but restricted to the
tumor-cell compartment only, which bulk cannot isolate.

PSEUDOBULK METHOD
-----------------
Per patient (sample), for each gene:
  pseudobulk_CP10K = 1e4 * sum(raw_counts) / sum(total_counts)
i.e. counts summed across all malignant cells of that patient, divided by
the summed library size of those same cells -- the standard "sum then
normalize" pseudobulk (same logic as CP10K per-cell normalization, just
applied to the patient-level aggregate instead of one cell). This is used
as the PRIMARY metric because arithmetic-summing raw counts is unbiased,
unlike averaging per-cell logcounts (a geometric-mean-like quantity that
systematically understates true expression for zero-inflated data -- see
19_cxcl16_lcor_linear_contribution.py docstring).
log1p(pseudobulk_CP10K) is used for correlation/plotting (variance
stabilization; same transform as the object's own 'logcounts' layer).

As a cross-check, the MEAN-LOGCOUNTS pseudobulk (mean of each cell's own
logcounts value, the method 01_liana_interaction_vs_lcor.py used for LCOR) is also computed and
correlated in parallel -- if the two methods disagree, that itself is
informative about how sensitive the result is to pseudobulking method.

DEPTH-CONFOUND CHECK
---------------------
Given how much of 14_lcor_cxcl16_hexbin.py, 02_depth_artifact_cell_level.py, 03_depth_artifact_diagnostic_hexbin.py turned out to be a depth artifact at the
cell level, the same concern applies at the patient level: if deeper
patients have both higher LCOR and higher CXCL16 pseudobulk expression
simply because deeper sequencing detects more of everything, that alone
would produce a spurious positive correlation. This is checked by (a)
correlating each patient's median per-cell depth against each gene's
pseudobulk value, and (b) an explicit depth-adjusted correlation: each
gene's log1p(pseudobulk_CP10K) is regressed on log(median patient depth),
and the residuals of LCOR vs CXCL16 are correlated (Spearman) -- this
directly asks "conditional on patient sequencing depth, does higher LCOR
still track higher CXCL16, patient to patient?"

CELL-COUNT CONFOUND CHECK
--------------------------
Separate from sequencing depth: does the NUMBER of malignant cells behind
a patient's pseudobulk (a precision/power issue, not a depth issue) drive
or mask the result? Checked by (a) correlating n_cells against each
gene's pseudobulk value directly, and (b) a sensitivity analysis --
recomputing the LCOR-CXCL16 correlation after progressively restricting
to patients with more cells (n_cells >= 20/50/100/500). If noise from
low-n_cells patients were masking a real signal, the correlation should
strengthen as weaker-powered patients are dropped; if it doesn't (or
weakens further), that argues against cell count as the explanation for
the null result.

MIN_CELLS = 20 malignant cells per patient (same threshold as 01_liana_interaction_vs_lcor.py).

Input:  compartment_epithelial.h5ad
Output: LCOR-CXCL16/03_pseudobulk/figures/
    persample_pseudobulk.csv
    correlation_table.csv
    ncell_sensitivity.csv
    fig_pseudobulk_scatter.png       (CP10K method, all + per subtype)
    fig_pseudobulk_scatter_meanlog.png (mean-logcounts method, cross-check)
    fig_depth_confound_check.png
    fig_ncell_confound_check.png
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/03_pseudobulk/scripts/01_pseudobulk_by_patient_headline.py
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Config ───────────────────────────────────────────────────────────────────
H5AD_FILE   = "compartment_epithelial.h5ad"
OUTDIR      = "LCOR-CXCL16/03_pseudobulk/figures"
GENE_X, GENE_Y = "LCOR", "CXCL16"
MIN_CELLS   = 20
SUBTYPES    = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(f" 01 -- {GENE_X} vs {GENE_Y} -- PATIENT PSEUDOBULK, malignant cells only")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}   Patients: {adata.obs['sample'].nunique()}")

missing = [g for g in (GENE_X, GENE_Y) if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

idx = [adata.var_names.get_loc(g) for g in (GENE_X, GENE_Y)]
raw = adata.layers["counts"][:, idx]
raw = np.asarray(raw.todense()) if issparse(raw) else np.asarray(raw)
logc = adata.layers["logcounts"][:, idx] if "logcounts" in adata.layers else adata.X[:, idx]
logc = np.asarray(logc.todense()) if issparse(logc) else np.asarray(logc)

obs = pd.DataFrame({
    "sample":  adata.obs["sample"].astype(str).values,
    "subtype": adata.obs["subtype"].astype(str).values,
    "total_counts": adata.obs["total_counts"].values.astype(float),
    f"{GENE_X}_raw": raw[:, 0].astype(float),
    f"{GENE_Y}_raw": raw[:, 1].astype(float),
    f"{GENE_X}_log": logc[:, 0],
    f"{GENE_Y}_log": logc[:, 1],
})
del adata

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PER-PATIENT PSEUDOBULK
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building per-patient pseudobulk (CP10K sum-then-normalize + mean-logcounts)...")
rows = []
for s, g in obs.groupby("sample"):
    n = len(g)
    if n < MIN_CELLS:
        continue
    lib_size = g["total_counts"].sum()
    row = {
        "sample": s, "subtype": g["subtype"].iloc[0], "n_cells": n,
        "median_depth": g["total_counts"].median(),
    }
    for gene in (GENE_X, GENE_Y):
        sum_raw = g[f"{gene}_raw"].sum()
        cp10k = 1e4 * sum_raw / lib_size
        row[f"{gene}_pseudobulk_CP10K"] = cp10k
        row[f"{gene}_log1p_CP10K"] = np.log1p(cp10k)
        row[f"{gene}_mean_logcounts"] = g[f"{gene}_log"].mean()
        row[f"{gene}_pct_detected"] = 100 * (g[f"{gene}_raw"] > 0).mean()
    rows.append(row)

df = pd.DataFrame(rows)
df["subtype"] = pd.Categorical(df["subtype"], categories=SUBTYPES, ordered=True)
df = df.sort_values(["subtype", "sample"]).reset_index(drop=True)
df.to_csv(f"{OUTDIR}/persample_pseudobulk.csv", index=False)
print(f"  Patients retained (>={MIN_CELLS} malignant cells): {len(df)} / {obs['sample'].nunique()}")
print(f"  Saved: {OUTDIR}/persample_pseudobulk.csv")
print("\n  Per-subtype patient counts:")
print(df["subtype"].value_counts().reindex(SUBTYPES).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# CORRELATIONS -- CP10K pseudobulk (primary) + mean-logcounts (cross-check)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Correlations: LCOR vs CXCL16 pseudobulk, across patients")
print("="*65)

def corr_group(d, xcol, ycol):
    if len(d) < 4:
        return dict(n=len(d), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_rho=np.nan, spearman_p=np.nan)
    r, p = stats.pearsonr(d[xcol], d[ycol])
    rho, p_s = stats.spearmanr(d[xcol], d[ycol])
    return dict(n=len(d), pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=p_s)

corr_rows = []
for method, xcol, ycol in [
    ("CP10K_log1p", f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K"),
    ("mean_logcounts", f"{GENE_X}_mean_logcounts", f"{GENE_Y}_mean_logcounts"),
]:
    res = corr_group(df, xcol, ycol)
    corr_rows.append({"method": method, "group": "All", **res})
    print(f"\n  [{method}] All patients (n={res['n']}): "
          f"Pearson r={res['pearson_r']:.3f} (p={res['pearson_p']:.3g}), "
          f"Spearman rho={res['spearman_rho']:.3f} (p={res['spearman_p']:.3g})")
    for st in SUBTYPES:
        d_st = df[df["subtype"] == st]
        res_st = corr_group(d_st, xcol, ycol)
        corr_rows.append({"method": method, "group": st, **res_st})
        print(f"    {st} (n={res_st['n']}): Pearson r={res_st['pearson_r']:.3f} "
              f"(p={res_st['pearson_p']:.3g}), Spearman rho={res_st['spearman_rho']:.3f} "
              f"(p={res_st['spearman_p']:.3g})")

# ══════════════════════════════════════════════════════════════════════════════
# DEPTH-CONFOUND CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Depth-confound check")
print("="*65)

df["log_median_depth"] = np.log(df["median_depth"])
depth_rows = []
for gene, col in [(GENE_X, f"{GENE_X}_log1p_CP10K"), (GENE_Y, f"{GENE_Y}_log1p_CP10K")]:
    rho, p = stats.spearmanr(df["log_median_depth"], df[col])
    depth_rows.append({"gene": gene, "spearman_rho_vs_depth": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient median depth: Spearman rho={rho:.3f}, p={p:.3g}")

# Depth-adjusted residual correlation: regress each gene's log1p(CP10K) on
# log(median depth) with simple OLS, correlate residuals (Spearman).
def resid_vs_depth(y):
    slope, intercept, r, p, se = stats.linregress(df["log_median_depth"], y)
    return y - (intercept + slope * df["log_median_depth"])

df[f"{GENE_X}_resid"] = resid_vs_depth(df[f"{GENE_X}_log1p_CP10K"])
df[f"{GENE_Y}_resid"] = resid_vs_depth(df[f"{GENE_Y}_log1p_CP10K"])
rho_adj, p_adj = stats.spearmanr(df[f"{GENE_X}_resid"], df[f"{GENE_Y}_resid"])
print(f"\n  DEPTH-ADJUSTED (residuals after regressing each gene on log median depth):")
print(f"    Spearman(LCOR resid, CXCL16 resid) = {rho_adj:.3f}, p={p_adj:.3g}, n={len(df)}")
corr_rows.append({"method": "CP10K_log1p_depth_adjusted_resid", "group": "All",
                   "n": len(df), "pearson_r": np.nan, "pearson_p": np.nan,
                   "spearman_rho": rho_adj, "spearman_p": p_adj})

df_corr = pd.DataFrame(corr_rows)

# ══════════════════════════════════════════════════════════════════════════════
# CELL-COUNT CONFOUND CHECK -- does the number of malignant cells contributing
# to a patient's pseudobulk (a precision/power issue, not a depth issue) drive
# or mask the LCOR-CXCL16 relationship? Two checks: (a) does n_cells itself
# correlate with either gene's pseudobulk value, and (b) does the LCOR-CXCL16
# correlation strengthen as we restrict to better-powered (higher n_cells)
# patients -- if noise from low-n_cells patients were hiding a real signal,
# it should emerge/strengthen here; if it doesn't, that's evidence against
# cell-count being the reason the correlation is null.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Cell-count confound check")
print("="*65)

ncell_rows = []
for gene, col in [(GENE_X, f"{GENE_X}_log1p_CP10K"), (GENE_Y, f"{GENE_Y}_log1p_CP10K")]:
    rho, p = stats.spearmanr(df["n_cells"], df[col])
    ncell_rows.append({"gene": gene, "spearman_rho_vs_ncells": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient n_cells: Spearman rho={rho:.3f}, p={p:.3g}")

print("\n  LCOR-CXCL16 correlation, restricted to better-powered patients "
      "(rising min. n_cells threshold):")
NCELL_THRESHOLDS = [MIN_CELLS, 50, 100, 500]
sensitivity_rows = []
for min_n in NCELL_THRESHOLDS:
    d = df[df["n_cells"] >= min_n]
    res = corr_group(d, f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K")
    sensitivity_rows.append({"min_n_cells": min_n, **res})
    print(f"    n_cells>={min_n}: n_patients={res['n']}, Spearman rho={res['spearman_rho']:.3f}, "
          f"p={res['spearman_p']:.3g}")
df_ncell_sensitivity = pd.DataFrame(sensitivity_rows)
df_ncell_sensitivity.to_csv(f"{OUTDIR}/ncell_sensitivity.csv", index=False)
print(f"  Saved: {OUTDIR}/ncell_sensitivity.csv")

ncells_strengthen = (abs(sensitivity_rows[-1]["spearman_rho"]) > abs(sensitivity_rows[0]["spearman_rho"]))
print(f"\n  Correlation {'STRENGTHENS' if ncells_strengthen else 'WEAKENS'} in the best-powered "
      f"patients (n_cells>={NCELL_THRESHOLDS[0]}: rho={sensitivity_rows[0]['spearman_rho']:.3f} "
      f"-> n_cells>={NCELL_THRESHOLDS[-1]}: rho={sensitivity_rows[-1]['spearman_rho']:.3f}) -- "
      f"{'consistent with cell-count noise masking a real signal' if ncells_strengthen else 'the opposite of what a cell-count-driven artifact would predict'}.")

df_corr.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)
print(f"\n  Saved: {OUTDIR}/correlation_table.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- CP10K pseudobulk scatter, All + per subtype
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building figures...")

def scatter_grid(xcol, ycol, xlabel, ylabel, fname, suptitle):
    fig, axes = plt.subplots(1, 1 + len(SUBTYPES), figsize=(5.2 * (1 + len(SUBTYPES)), 5))
    res_all = corr_group(df, xcol, ycol)
    ax = axes[0]
    for st, color in SUBTYPE_COLORS.items():
        d = df[df["subtype"] == st]
        ax.scatter(d[xcol], d[ycol], color=color, label=st, s=55, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_title(f"All (n={res_all['n']})\nPearson r={res_all['pearson_r']:.2f} (p={res_all['pearson_p']:.3g})\n"
                 f"Spearman rho={res_all['spearman_rho']:.2f} (p={res_all['spearman_p']:.3g})",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, title="Subtype")
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25)

    for i, st in enumerate(SUBTYPES):
        ax = axes[i + 1]
        d = df[df["subtype"] == st]
        res_st = corr_group(d, xcol, ycol)
        ax.scatter(d[xcol], d[ycol], color=SUBTYPE_COLORS[st], s=55, alpha=0.9, edgecolor="white", linewidth=0.6)
        if len(d) >= 2:
            try:
                z = np.polyfit(d[xcol], d[ycol], 1)
                xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color=SUBTYPE_COLORS[st], lw=1.2, alpha=0.6, ls="--")
            except Exception:
                pass
        ax.set_title(f"{st} (n={res_st['n']})\nPearson r={res_st['pearson_r']:.2f} (p={res_st['pearson_p']:.3g})\n"
                     f"Spearman rho={res_st['spearman_rho']:.2f} (p={res_st['spearman_p']:.3g})",
                     fontsize=9, fontweight="bold", color=SUBTYPE_COLORS[st])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTDIR}/{fname}.png")

scatter_grid(f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K",
             f"{GENE_X} pseudobulk (log1p CP10K)", f"{GENE_Y} pseudobulk (log1p CP10K)",
             "fig_pseudobulk_scatter",
             f"Per-patient pseudobulk, malignant cells only -- {GENE_X} vs {GENE_Y} (CP10K sum-then-normalize)")

scatter_grid(f"{GENE_X}_mean_logcounts", f"{GENE_Y}_mean_logcounts",
             f"{GENE_X} pseudobulk (mean logcounts/cell)", f"{GENE_Y} pseudobulk (mean logcounts/cell)",
             "fig_pseudobulk_scatter_meanlog",
             f"Per-patient pseudobulk (mean-logcounts method, cross-check vs 01_liana_interaction_vs_lcor.py) -- {GENE_X} vs {GENE_Y}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- depth confound check
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

for ax, gene, col in zip(axes[:2], (GENE_X, GENE_Y), (f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K")):
    for st, color in SUBTYPE_COLORS.items():
        d = df[df["subtype"] == st]
        ax.scatter(d["median_depth"], d[col], color=color, label=st, s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
    rho = depth_rows[0]["spearman_rho_vs_depth"] if gene == GENE_X else depth_rows[1]["spearman_rho_vs_depth"]
    p_ = depth_rows[0]["p"] if gene == GENE_X else depth_rows[1]["p"]
    ax.set_xscale("log")
    ax.set_xlabel("Patient median per-cell depth (total_counts)", fontsize=8.5)
    ax.set_ylabel(f"{gene} pseudobulk (log1p CP10K)", fontsize=8.5)
    ax.set_title(f"{gene} vs patient depth\nSpearman rho={rho:.2f}, p={p_:.3g}", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)

ax = axes[2]
for st, color in SUBTYPE_COLORS.items():
    d = df[df["subtype"] == st]
    ax.scatter(d[f"{GENE_X}_resid"], d[f"{GENE_Y}_resid"], color=color, label=st, s=50, alpha=0.9, edgecolor="white", linewidth=0.6)
ax.set_xlabel(f"{GENE_X} residual (depth-adjusted)", fontsize=8.5)
ax.set_ylabel(f"{GENE_Y} residual (depth-adjusted)", fontsize=8.5)
ax.set_title(f"Depth-adjusted: {GENE_X} vs {GENE_Y} residuals\nSpearman rho={rho_adj:.2f}, p={p_adj:.3g}, n={len(df)}",
             fontsize=9, fontweight="bold")
ax.axhline(0, color="gray", lw=0.8, alpha=0.5); ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
ax.legend(fontsize=7.5, title="Subtype")
ax.grid(alpha=0.25)

fig.suptitle("Depth-confound check: patient sequencing depth vs pseudobulk, and depth-adjusted residual correlation",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_depth_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_depth_confound_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 -- cell-count confound check
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

for ax, gene, col in zip(axes[:2], (GENE_X, GENE_Y), (f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K")):
    for st, color in SUBTYPE_COLORS.items():
        d = df[df["subtype"] == st]
        ax.scatter(d["n_cells"], d[col], color=color, label=st, s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
    row = next(r for r in ncell_rows if r["gene"] == gene)
    ax.set_xscale("log")
    ax.set_xlabel("Malignant cells contributing to this patient's pseudobulk", fontsize=8.5)
    ax.set_ylabel(f"{gene} pseudobulk (log1p CP10K)", fontsize=8.5)
    ax.set_title(f"{gene} vs n_cells\nSpearman rho={row['spearman_rho_vs_ncells']:.2f}, p={row['p']:.3g}",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, title="Subtype")

ax = axes[2]
ax.plot(df_ncell_sensitivity["min_n_cells"], df_ncell_sensitivity["spearman_rho"],
        "o-", color="#377EB8", lw=2, markersize=7)
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
for _, r in df_ncell_sensitivity.iterrows():
    ax.annotate(f"n={int(r['n'])}", (r["min_n_cells"], r["spearman_rho"]),
                textcoords="offset points", xytext=(0, 8), fontsize=7.5, ha="center")
ax.set_xscale("log")
ax.set_xlabel("Min. n_cells threshold (patients below excluded)", fontsize=8.5)
ax.set_ylabel(f"Spearman rho ({GENE_X} vs {GENE_Y})", fontsize=8.5)
ax.set_title("Does the correlation strengthen in\nbetter-powered (higher n_cells) patients?",
             fontsize=9, fontweight="bold")
ax.grid(alpha=0.25)

fig.suptitle("Cell-count confound check: patient n_cells vs pseudobulk, and sensitivity to a stricter n_cells threshold",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_ncell_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_ncell_confound_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
all_cp10k = df_corr[(df_corr.method == "CP10K_log1p") & (df_corr.group == "All")].iloc[0]
tnbc_cp10k = df_corr[(df_corr.method == "CP10K_log1p") & (df_corr.group == "TNBC")].iloc[0]
all_meanlog = df_corr[(df_corr.method == "mean_logcounts") & (df_corr.group == "All")].iloc[0]

readme = [
    f"01 -- {GENE_X} vs {GENE_Y} -- per-patient pseudobulk, malignant cells only",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS / WHAT GAP IT FILLS",
    "-"*70,
    "Scripts 14-16 found the LCOR-CXCL16 cell-to-cell correlation within",
    "malignant cells is mostly a depth artifact. Script 17's LIANA-based",
    "CXCL16->CXCR6 interaction score correlated with LCOR pseudobulk",
    "(rho=-0.63) but collapsed once subtype was controlled for. This script",
    "is the missing middle: a direct per-patient PSEUDOBULK correlation of",
    "LCOR against CXCL16, BOTH measured only in the malignant (epithelial)",
    "compartment -- no T-cell/CXCR6 term, no cell-level dropout noise. This",
    "is the single-cell analogue of the bulk RNA-seq LCOR-CXCL16 test (6",
    "public datasets, weak/inconsistent, ns in TNBC), but restricted to the",
    "tumor-cell compartment that bulk cannot isolate.",
    "",
    "PSEUDOBULK METHOD",
    "-"*70,
    "  Primary: pseudobulk_CP10K = 1e4 * sum(raw counts) / sum(total_counts),",
    "  summed across all malignant cells of a patient -- arithmetic, not the",
    "  geometric-mean-like mean-of-logcounts (see 19's docstring for why that",
    "  matters for zero-inflated data). log1p-transformed for correlation.",
    "  Cross-check: mean-logcounts pseudobulk (01_liana_interaction_vs_lcor.py's method for LCOR),",
    "  computed in parallel -- see correlation_table.csv, method column.",
    f"  MIN_CELLS = {MIN_CELLS} malignant cells/patient "
    f"({len(df)}/{obs['sample'].nunique()} patients retained).",
    "",
    "RESULTS -- headline correlations (CP10K pseudobulk, primary method)",
    "-"*70,
    f"  All patients (n={all_cp10k['n']}): Pearson r={all_cp10k['pearson_r']:.3f} "
    f"(p={all_cp10k['pearson_p']:.3g}), Spearman rho={all_cp10k['spearman_rho']:.3f} "
    f"(p={all_cp10k['spearman_p']:.3g})",
    f"  TNBC only (n={tnbc_cp10k['n']}): Pearson r={tnbc_cp10k['pearson_r']:.3f} "
    f"(p={tnbc_cp10k['pearson_p']:.3g}), Spearman rho={tnbc_cp10k['spearman_rho']:.3f} "
    f"(p={tnbc_cp10k['spearman_p']:.3g})",
    f"  Cross-check (mean-logcounts method), All (n={all_meanlog['n']}): "
    f"Pearson r={all_meanlog['pearson_r']:.3f} (p={all_meanlog['pearson_p']:.3g}), "
    f"Spearman rho={all_meanlog['spearman_rho']:.3f} (p={all_meanlog['spearman_p']:.3g})",
    "  Full breakdown (All/ER/HER2/TNBC x both methods): correlation_table.csv",
    "",
    "DEPTH-CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient median depth: Spearman rho={r['spearman_rho_vs_depth']:.3f}, p={r['p']:.3g}"
    for r in depth_rows
] + [
    f"  Depth-adjusted residual correlation (LCOR resid vs CXCL16 resid, both",
    f"  regressed on log median patient depth first): Spearman rho={rho_adj:.3f}, p={p_adj:.3g}",
    "  (If this drops much relative to the raw pseudobulk correlation above,",
    "   part of the raw signal is patient-level depth, not biology -- same",
    "   logic as 02_depth_artifact_cell_level.py's within-patient depth-offset check, applied here",
    "   across patients instead of across cells.)",
    "",
    "CELL-COUNT CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient n_cells: Spearman rho={r['spearman_rho_vs_ncells']:.3f}, p={r['p']:.3g}"
    for r in ncell_rows
] + [
    "  Sensitivity to a stricter n_cells threshold (does the correlation",
    "  strengthen in better-powered patients?):",
] + [
    f"    n_cells>={int(r['min_n_cells'])}: n_patients={int(r['n'])}, Spearman rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.3g}"
    for _, r in df_ncell_sensitivity.iterrows()
] + [
    f"  -> Correlation {'STRENGTHENS' if ncells_strengthen else 'WEAKENS'} at higher n_cells thresholds",
    "     -- " + ("consistent with cell-count noise masking a real signal (would need"
                  " follow-up)." if ncells_strengthen else
                  "the OPPOSITE of what a cell-count-driven artifact would predict:"
                  " if low-n_cells noise were hiding a real relationship, restricting"
                  " to the best-powered patients should reveal/strengthen it, not"
                  " weaken it further. Argues against cell count as the explanation"
                  " for the null result."),
    "",
    "OUTPUT FILES",
    "-"*70,
    "  persample_pseudobulk.csv   -- per-patient pseudobulk (both methods), n_cells, depth, subtype",
    "  correlation_table.csv      -- Pearson/Spearman, by method x group (All/ER/HER2/TNBC)",
    "  ncell_sensitivity.csv      -- correlation at each n_cells threshold",
    "  fig_pseudobulk_scatter.png          -- CP10K method (primary)",
    "  fig_pseudobulk_scatter_meanlog.png  -- mean-logcounts method (cross-check vs 01_liana_interaction_vs_lcor.py)",
    "  fig_depth_confound_check.png        -- depth vs pseudobulk, + depth-adjusted residual scatter",
    "  fig_ncell_confound_check.png        -- n_cells vs pseudobulk, + threshold sensitivity",
    "",
    "CAVEATS",
    "-"*70,
    f"  - n={len(df)} patients: even a real effect can fail to reach conventional",
    "    significance, and any single 'significant' result at this n should be",
    "    treated cautiously pending replication.",
    "  - Correlational, cross-sectional; cannot establish causality/direction",
    "    (does not distinguish LCOR->CXCL16 repression from confounding by an",
    "    unmeasured patient/tumor-level factor that shifts both).",
    "  - CP10K pseudobulk uses each patient's OWN malignant-cell library size",
    "    sum as the normalization denominator -- appropriate for cross-patient",
    "    comparison, but does not correct for compositional differences within",
    "    the malignant compartment (e.g. proliferative vs quiescent subclone",
    "    mix) the way scaling to a shared reference set might.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Patients: {len(df)} (All) / TNBC n={tnbc_cp10k['n']}
  All (CP10K):   Pearson r={all_cp10k['pearson_r']:.3f} (p={all_cp10k['pearson_p']:.3g})  Spearman rho={all_cp10k['spearman_rho']:.3f} (p={all_cp10k['spearman_p']:.3g})
  TNBC (CP10K):  Pearson r={tnbc_cp10k['pearson_r']:.3f} (p={tnbc_cp10k['pearson_p']:.3g})  Spearman rho={tnbc_cp10k['spearman_rho']:.3f} (p={tnbc_cp10k['spearman_p']:.3g})
  Depth-adjusted residual rho: {rho_adj:.3f} (p={p_adj:.3g})
  n_cells sensitivity (n_cells>={NCELL_THRESHOLDS[0]} -> >={NCELL_THRESHOLDS[-1]}): rho={sensitivity_rows[0]['spearman_rho']:.3f} -> {sensitivity_rows[-1]['spearman_rho']:.3f} ({'strengthens' if ncells_strengthen else 'weakens'})
  Output: {OUTDIR}/
============================================================
""")
