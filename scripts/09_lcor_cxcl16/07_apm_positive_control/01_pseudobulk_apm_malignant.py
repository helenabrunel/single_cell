"""
01_pseudobulk_apm_malignant.py
===============================
POSITIVE-CONTROL check, malignant (epithelial) compartment: per-patient
PSEUDOBULK of LCOR vs an "APM score" (Antigen Processing Machinery meta-gene:
TAP1+TAP2+TAPBP+HLA-A+HLA-B+HLA-C+PSMB8+PSMB9+B2M raw counts summed per
cell), correlated across patients. Direct analogue of
03_pseudobulk/01_pseudobulk_by_patient_headline.py (LCOR vs CXCL16, same
compartment, same method), which found that test null (Spearman rho=-0.11,
n=34, p=0.53; TNBC rho=0.24, n=8, p=0.57).

WHY THIS SCRIPT EXISTS
-----------------------
The myeloid-compartment LCOR-CXCL16 null (05_myeloid) raised a
methodological question: is this "no biological
relationship" or "the method can't detect a real relationship in this
sparse single-cell data"? This is a POSITIVE CONTROL -- a gene set
already known (from the lab's own prior mouse/culture work) to be
regulated by LCOR -- run through the identical pipeline. If the pipeline
recovers the known LCOR-APM relationship despite the same sparsity/depth
issues that afflict LCOR-CXCL16, that validates the method and makes the
CXCL16 null more credible as a real absence of relationship, not a false
negative from underpowering. If the positive control ALSO comes back
null, that points to a power/method limitation instead.

APM gene set (compact "APM pura" set, as specified by the lab after review):
TAP1, TAP2 (peptide transporter), TAPBP (tapasin, peptide-loading complex),
HLA-A/HLA-B/HLA-C (classical MHC-I), PSMB8/PSMB9 (immunoproteasome
subunits), B2M (beta-2-microglobulin). All 9 genes confirmed present in
var_names of both h5ad files used in this project.

METHOD
------
Identical to 03_pseudobulk/01, except GENE_Y is a META-GENE: instead of a
single gene's raw UMI count per cell, APM_raw = row-sum of raw counts
across all 9 APM genes per cell. This meta-gene is then pseudobulked
exactly like any single gene (sum across a patient's malignant cells,
divide by that patient's total library size -> CP10K, log1p for
correlation). Summing counts across 9 genes also gives APM a much higher
per-cell detection rate than any single gene (reported below), which is
itself part of the positive-control logic: if a less sparse readout still
fails to correlate with LCOR, sparsity of the single-gene CXCL16 readout
is a weaker excuse for its null result.

Same depth- and cell-count-confound checks as 03_pseudobulk/01.

Input:  compartment_epithelial.h5ad
Output: LCOR-CXCL16/07_apm_positive_control/figures/01_pseudobulk_apm_malignant/
    persample_pseudobulk.csv, correlation_table.csv, ncell_sensitivity.csv,
    fig_pseudobulk_scatter.png, fig_depth_confound_check.png,
    fig_ncell_confound_check.png, README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/07_apm_positive_control/scripts/01_pseudobulk_apm_malignant.py
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
OUTDIR      = "LCOR-CXCL16/07_apm_positive_control/figures/01_pseudobulk_apm_malignant"
GENE_X      = "LCOR"
APM_GENES   = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
GENE_Y      = "APM_score"
MIN_CELLS   = 20
SUBTYPES    = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(f" 01 -- {GENE_X} vs {GENE_Y} (positive control) -- PATIENT PSEUDOBULK, malignant cells")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}   Patients: {adata.obs['sample'].nunique()}")

missing = [g for g in [GENE_X] + APM_GENES if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

idx_x = adata.var_names.get_loc(GENE_X)
idx_apm = [adata.var_names.get_loc(g) for g in APM_GENES]

raw_x = adata.layers["counts"][:, idx_x]
raw_x = np.asarray(raw_x.todense()).ravel() if issparse(raw_x) else np.asarray(raw_x).ravel()
raw_apm_matrix = adata.layers["counts"][:, idx_apm]
raw_apm_matrix = np.asarray(raw_apm_matrix.todense()) if issparse(raw_apm_matrix) else np.asarray(raw_apm_matrix)
raw_apm = raw_apm_matrix.sum(axis=1)

log_x = adata.layers["logcounts"][:, idx_x] if "logcounts" in adata.layers else adata.X[:, idx_x]
log_x = np.asarray(log_x.todense()).ravel() if issparse(log_x) else np.asarray(log_x).ravel()

pct_detected_per_gene = {g: 100 * (raw_apm_matrix[:, i] > 0).mean() for i, g in enumerate(APM_GENES)}
pct_detected_x = 100 * (raw_x > 0).mean()
pct_detected_apm = 100 * (raw_apm > 0).mean()
print(f"\n  Per-cell detection rate, {GENE_X}: {pct_detected_x:.1f}%")
print(f"  Per-cell detection rate, {GENE_Y} (>=1 of 9 genes detected): {pct_detected_apm:.1f}%")
print("  Per-gene detection rates (individual APM genes):")
for g, pct in pct_detected_per_gene.items():
    print(f"    {g}: {pct:.1f}%")

obs = pd.DataFrame({
    "sample":  adata.obs["sample"].astype(str).values,
    "subtype": adata.obs["subtype"].astype(str).values,
    "total_counts": adata.obs["total_counts"].values.astype(float),
    f"{GENE_X}_raw": raw_x.astype(float),
    f"{GENE_Y}_raw": raw_apm.astype(float),
    f"{GENE_X}_log": log_x,
})
del adata

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PER-PATIENT PSEUDOBULK
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building per-patient pseudobulk (CP10K sum-then-normalize)...")
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
        row[f"{gene}_pct_detected"] = 100 * (g[f"{gene}_raw"] > 0).mean()
    rows.append(row)

df = pd.DataFrame(rows)
df["subtype"] = pd.Categorical(df["subtype"], categories=SUBTYPES, ordered=True)
df = df.sort_values(["subtype", "sample"]).reset_index(drop=True)
df.to_csv(f"{OUTDIR}/persample_pseudobulk.csv", index=False)
print(f"  Patients retained (>={MIN_CELLS} malignant cells): {len(df)} / {obs['sample'].nunique()}")
print(f"  Saved: {OUTDIR}/persample_pseudobulk.csv")

# ══════════════════════════════════════════════════════════════════════════════
# CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f" Correlations: {GENE_X} vs {GENE_Y} pseudobulk, across patients")
print("="*65)

def corr_group(d, xcol, ycol):
    if len(d) < 4:
        return dict(n=len(d), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_rho=np.nan, spearman_p=np.nan)
    r, p = stats.pearsonr(d[xcol], d[ycol])
    rho, p_s = stats.spearmanr(d[xcol], d[ycol])
    return dict(n=len(d), pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=p_s)

xcol, ycol = f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K"
corr_rows = []
res_all = corr_group(df, xcol, ycol)
corr_rows.append({"group": "All", **res_all})
print(f"\n  All patients (n={res_all['n']}): Pearson r={res_all['pearson_r']:.3f} "
      f"(p={res_all['pearson_p']:.3g}), Spearman rho={res_all['spearman_rho']:.3f} "
      f"(p={res_all['spearman_p']:.3g})")
for st in SUBTYPES:
    d_st = df[df["subtype"] == st]
    res_st = corr_group(d_st, xcol, ycol)
    corr_rows.append({"group": st, **res_st})
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
for gene, col in [(GENE_X, xcol), (GENE_Y, ycol)]:
    rho, p = stats.spearmanr(df["log_median_depth"], df[col])
    depth_rows.append({"gene": gene, "spearman_rho_vs_depth": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient median depth: Spearman rho={rho:.3f}, p={p:.3g}")

def resid_vs_depth(y):
    slope, intercept, r, p, se = stats.linregress(df["log_median_depth"], y)
    return y - (intercept + slope * df["log_median_depth"])

df[f"{GENE_X}_resid"] = resid_vs_depth(df[xcol])
df[f"{GENE_Y}_resid"] = resid_vs_depth(df[ycol])
rho_adj, p_adj = stats.spearmanr(df[f"{GENE_X}_resid"], df[f"{GENE_Y}_resid"])
print(f"\n  DEPTH-ADJUSTED: Spearman(LCOR resid, APM resid) = {rho_adj:.3f}, p={p_adj:.3g}, n={len(df)}")
corr_rows.append({"group": "All_depth_adjusted_resid", "n": len(df),
                   "pearson_r": np.nan, "pearson_p": np.nan,
                   "spearman_rho": rho_adj, "spearman_p": p_adj})

# ══════════════════════════════════════════════════════════════════════════════
# CELL-COUNT CONFOUND CHECK + sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Cell-count confound check")
print("="*65)
ncell_rows = []
for gene, col in [(GENE_X, xcol), (GENE_Y, ycol)]:
    rho, p = stats.spearmanr(df["n_cells"], df[col])
    ncell_rows.append({"gene": gene, "spearman_rho_vs_ncells": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient n_cells: Spearman rho={rho:.3f}, p={p:.3g}")

NCELL_THRESHOLDS = [MIN_CELLS, 50, 100, 500]
sensitivity_rows = []
for min_n in NCELL_THRESHOLDS:
    d = df[df["n_cells"] >= min_n]
    res = corr_group(d, xcol, ycol)
    sensitivity_rows.append({"min_n_cells": min_n, **res})
    print(f"    n_cells>={min_n}: n_patients={res['n']}, Spearman rho={res['spearman_rho']:.3f}, "
          f"p={res['spearman_p']:.3g}")
df_ncell_sensitivity = pd.DataFrame(sensitivity_rows)
df_ncell_sensitivity.to_csv(f"{OUTDIR}/ncell_sensitivity.csv", index=False)

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)
print(f"\n  Saved: {OUTDIR}/correlation_table.csv, {OUTDIR}/ncell_sensitivity.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building figures...")

fig, axes = plt.subplots(1, 1 + len(SUBTYPES), figsize=(5.2 * (1 + len(SUBTYPES)), 5))
ax = axes[0]
for st, color in SUBTYPE_COLORS.items():
    d = df[df["subtype"] == st]
    ax.scatter(d[xcol], d[ycol], color=color, label=st, s=55, alpha=0.85, edgecolor="white", linewidth=0.6)
ax.set_title(f"All (n={res_all['n']})\nPearson r={res_all['pearson_r']:.2f} (p={res_all['pearson_p']:.3g})\n"
             f"Spearman rho={res_all['spearman_rho']:.2f} (p={res_all['spearman_p']:.3g})",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=8, title="Subtype")
ax.set_xlabel(f"{GENE_X} pseudobulk (log1p CP10K)", fontsize=9)
ax.set_ylabel(f"{GENE_Y} pseudobulk (log1p CP10K)", fontsize=9)
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
    ax.set_xlabel(f"{GENE_X} pseudobulk (log1p CP10K)", fontsize=9)
    ax.grid(alpha=0.25)
fig.suptitle(f"Per-patient pseudobulk, malignant cells -- {GENE_X} vs {GENE_Y} (positive control)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_pseudobulk_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_pseudobulk_scatter.png")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
for ax, gene, col in zip(axes[:2], (GENE_X, GENE_Y), (xcol, ycol)):
    for st, color in SUBTYPE_COLORS.items():
        d = df[df["subtype"] == st]
        ax.scatter(d["median_depth"], d[col], color=color, label=st, s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
    row = next(r for r in depth_rows if r["gene"] == gene)
    ax.set_xscale("log")
    ax.set_xlabel("Patient median per-cell depth (total_counts)", fontsize=8.5)
    ax.set_ylabel(f"{gene} pseudobulk (log1p CP10K)", fontsize=8.5)
    ax.set_title(f"{gene} vs patient depth\nSpearman rho={row['spearman_rho_vs_depth']:.2f}, p={row['p']:.3g}",
                 fontsize=9, fontweight="bold")
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
fig.suptitle("Depth-confound check (positive control): patient depth vs pseudobulk, depth-adjusted residuals",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_depth_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_depth_confound_check.png")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
for ax, gene, col in zip(axes[:2], (GENE_X, GENE_Y), (xcol, ycol)):
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
ax.set_title("Sensitivity to n_cells threshold", fontsize=9, fontweight="bold")
ax.grid(alpha=0.25)
fig.suptitle("Cell-count confound check (positive control)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_ncell_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_ncell_confound_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
tnbc_row = df_corr[df_corr.group == "TNBC"].iloc[0]
readme = [
    f"01 -- {GENE_X} vs {GENE_Y} (POSITIVE CONTROL) -- per-patient pseudobulk, malignant cells",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "Positive-control check run after the",
    "myeloid LCOR-CXCL16 null: is the method able to detect a relationship",
    "already known from the lab's own prior work (LCOR-APM), under the same",
    "sparsity/depth conditions as the LCOR-CXCL16 test? Direct analogue of",
    "03_pseudobulk/01 (LCOR vs CXCL16, malignant, null: rho=-0.11, n=34,",
    "p=0.53), same methodology, GENE_Y replaced by an APM meta-gene.",
    "",
    "APM GENE SET (\"APM pura\" as specified)",
    "-"*70,
    "  " + ", ".join(APM_GENES),
    "  Per-cell raw count = sum of raw UMI counts across all 9 genes.",
    "",
    "PER-CELL DETECTION RATES",
    "-"*70,
    f"  {GENE_X}: {pct_detected_x:.1f}%",
    f"  {GENE_Y} (>=1 of 9 genes detected): {pct_detected_apm:.1f}%",
] + [f"    {g}: {pct:.1f}%" for g, pct in pct_detected_per_gene.items()] + [
    "",
    "RESULTS -- headline correlation",
    "-"*70,
    f"  All patients (n={res_all['n']}): Pearson r={res_all['pearson_r']:.3f} "
    f"(p={res_all['pearson_p']:.3g}), Spearman rho={res_all['spearman_rho']:.3f} "
    f"(p={res_all['spearman_p']:.3g})",
    f"  TNBC only (n={tnbc_row['n']}): Spearman rho={tnbc_row['spearman_rho']:.3f} "
    f"(p={tnbc_row['spearman_p']:.3g})",
    "  Full ER/HER2/TNBC breakdown: correlation_table.csv",
    "",
    "DEPTH-CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient median depth: Spearman rho={r['spearman_rho_vs_depth']:.3f}, p={r['p']:.3g}"
    for r in depth_rows
] + [
    f"  Depth-adjusted residual correlation: Spearman rho={rho_adj:.3f}, p={p_adj:.3g}",
    "",
    "CELL-COUNT CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient n_cells: Spearman rho={r['spearman_rho_vs_ncells']:.3f}, p={r['p']:.3g}"
    for r in ncell_rows
] + [
    "  Sensitivity to n_cells threshold:",
] + [
    f"    n_cells>={int(r['min_n_cells'])}: n_patients={int(r['n'])}, Spearman rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.3g}"
    for _, r in df_ncell_sensitivity.iterrows()
] + [
    "",
    "INTERPRETATION",
    "-"*70,
    "If this correlation is clearly non-null (and survives depth adjustment),",
    "it validates that the pseudobulk pipeline CAN detect a real LCOR-linked",
    "relationship at this sample size/sparsity -- making the LCOR-CXCL16 null",
    "more credible as a genuine absence of relationship. If this positive",
    "control ALSO comes back null, that is evidence the method/power, not",
    "biology, may be limiting what can be concluded from the CXCL16 tests.",
    "",
    "OUTPUT FILES",
    "-"*70,
    "  persample_pseudobulk.csv, correlation_table.csv, ncell_sensitivity.csv",
    "  fig_pseudobulk_scatter.png, fig_depth_confound_check.png, fig_ncell_confound_check.png",
    "",
    "CAVEATS",
    "-"*70,
    f"  - n={len(df)} patients; same small-n caveats as the rest of this project.",
    "  - The APM-LCOR relationship this tests against is taken as a given from",
    "    the lab's own prior (mouse/culture) work, not independently verified",
    "    here in bulk human RNA-seq first.",
    "  - Correlational; a positive result here validates method sensitivity,",
    "    it does not by itself validate direction/causality of LCOR->APM.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Patients: {len(df)} (All) / TNBC n={tnbc_row['n']}
  All: Pearson r={res_all['pearson_r']:.3f} (p={res_all['pearson_p']:.3g})  Spearman rho={res_all['spearman_rho']:.3f} (p={res_all['spearman_p']:.3g})
  TNBC: Spearman rho={tnbc_row['spearman_rho']:.3f} (p={tnbc_row['spearman_p']:.3g})
  Depth-adjusted residual rho: {rho_adj:.3f} (p={p_adj:.3g})
  Output: {OUTDIR}/
============================================================
""")
