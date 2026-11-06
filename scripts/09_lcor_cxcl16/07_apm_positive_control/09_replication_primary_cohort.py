"""
09_replication_regner.py
==========================
Second independent-dataset replication check (after 07_replication_
klughammer.py) for the LCOR-APM positive control and the ESR1->GREB1/PGR
method-sanity check, using Regner et al. (GSE243526, PMC11872555),
already available in this environment (gse243526_processed.h5ad).

DATASET DIFFERENCES FROM VISVADER/KLUGHAMMER (handled below)
---------------------------------------------------------------
- No precomputed malignant/immune compartment split -- built here with a
  simple marker-based winner-takes-all call: epithelial score (mean
  logcounts of EPCAM/KRT8/KRT18/KRT19) vs PTPRC logcounts, malignant =
  epithelial score wins.
- Mixes tumor, normal tissue, and cell lines -- restricted here to
  tissue=="primary breast tumor" and is_cell_line==False.
- subtype categories differ: "HR+/HER2-" (~ER-equivalent, best-powered
  group) and "TNBC" (only 1 patient here -- reported but not a real test).

Output: LCOR-CXCL16/07_apm_positive_control/figures/09_replication_regner/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, warnings, datetime
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "/home/hbrunel/single_cell/Regner/gse243526_processed.h5ad"
OUTDIR    = "LCOR-CXCL16/07_apm_positive_control/figures/09_replication_regner"
APM_GENES = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
EPI_MARKERS = ["EPCAM", "KRT8", "KRT18", "KRT19"]
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(" 09 -- REPLICATION in Regner/GSE243526 (independent atlas)")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells (raw): {adata.n_obs:,}")

mask_tumor = (adata.obs["tissue"].astype(str) == "primary breast tumor") & (~adata.obs["is_cell_line"].astype(bool))
adata = adata[mask_tumor].copy()
print(f"  Cells (primary tumor, non-cell-line): {adata.n_obs:,}, {adata.obs['sample_id'].nunique()} samples, {adata.obs['patient'].nunique()} patients")

needed = ["LCOR", "ESR1", "GREB1", "PGR"] + APM_GENES + EPI_MARKERS + ["PTPRC"]
missing = [g for g in needed if g not in adata.var_names]
if missing:
    raise SystemExit(f"Missing genes: {missing}")

idx_of = {g: adata.var_names.get_loc(g) for g in needed}
def get_log(g):
    col = adata.layers["logcounts"][:, idx_of[g]]
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
def get_raw(g):
    col = adata.layers["counts"][:, idx_of[g]]
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

# ============================================================================
# Malignant-cell call: epithelial marker score vs PTPRC, winner-takes-all
# ============================================================================
epi_score = np.mean(np.column_stack([get_log(g) for g in EPI_MARKERS]), axis=1)
ptprc = get_log("PTPRC")
is_malignant = (epi_score > 0) & (epi_score > ptprc)
print(f"\n  Malignant call (epithelial score > PTPRC, epithelial score > 0): {is_malignant.sum():,} / {adata.n_obs:,} ({100*is_malignant.mean():.1f}%)")

readme = [
    "09 -- REPLICATION in Regner/GSE243526 (independent atlas, PMC11872555)",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*70,
    "",
    "MALIGNANT-CELL CALL",
    "-"*70,
    "  No precomputed compartment split in this dataset -- built here:",
    "  epithelial score = mean logcounts(EPCAM,KRT8,KRT18,KRT19), malignant",
    "  = epithelial score > 0 AND > PTPRC logcounts (winner-takes-all).",
    f"  Malignant cells: {is_malignant.sum():,} / {adata.n_obs:,} ({100*is_malignant.mean():.1f}%), "
    f"restricted to primary tumor, non-cell-line.",
]

raw_lcor = get_raw("LCOR")
raw_esr1 = get_raw("ESR1")
raw_greb1 = get_raw("GREB1")
raw_pgr = get_raw("PGR")
raw_apm = np.column_stack([get_raw(g) for g in APM_GENES]).sum(axis=1)

df = pd.DataFrame({
    "LCOR_raw": raw_lcor.astype(int),
    "ESR1_raw": raw_esr1.astype(int),
    "GREB1_raw": raw_greb1.astype(int),
    "PGR_raw": raw_pgr.astype(int),
    "APM_raw": raw_apm.astype(int),
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "subtype": adata.obs["subtype"].astype(str).values,
    "sample": adata.obs["sample_id"].astype(str).values,
}, index=adata.obs_names)
df["log_total_counts"] = np.log(df["total_counts"])
df_mal = df[is_malignant].copy()
del adata

print(f"\n  Subtype counts (malignant cells): {df_mal['subtype'].value_counts().to_dict()}")
readme += ["", f"  Subtype counts (malignant cells): {df_mal['subtype'].value_counts().to_dict()}"]

def fit_glm(d, x_raw_col, y_raw_col, label):
    d = d.copy()
    d["log1p_X"] = np.log1p(d[x_raw_col])
    poisson_pooled = smf.glm(
        f"{y_raw_col} ~ log1p_X", data=d, family=sm.families.Poisson(),
        offset=d["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["sample"]})
    disp = poisson_pooled.pearson_chi2 / poisson_pooled.df_resid
    family = sm.families.NegativeBinomial() if disp > 1.5 else sm.families.Poisson()
    model_name = "Negative Binomial" if disp > 1.5 else "Poisson"

    within = smf.glm(
        f"{y_raw_col} ~ log1p_X + C(sample)", data=d, family=family,
        offset=d["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["sample"]})
    rr = np.exp(within.params["log1p_X"])
    ci = np.exp(within.conf_int().loc["log1p_X"].values)
    p = within.pvalues["log1p_X"]
    print(f"\n  [{label}] n_samples={d['sample'].nunique()}, n_cells={len(d):,}, dispersion={disp:.2f} -> {model_name}")
    print(f"    WITHIN-SAMPLE RR={rr:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p={p:.2e}")
    return dict(label=label, n_samples=d["sample"].nunique(), n_cells=len(d),
                model=model_name, rr=rr, ci_lo=ci[0], ci_hi=ci[1], p=p)

print("\n" + "="*65)
print(" Check 1: LCOR -> APM (all malignant cells)")
print("="*65)
r_apm = fit_glm(df_mal, "LCOR_raw", "APM_raw", "LCOR->APM (all malignant cells)")

print("\n" + "="*65)
print(" Check 2: ESR1 -> GREB1/PGR (HR+/HER2- malignant cells)")
print("="*65)
df_hr = df_mal[df_mal["subtype"] == "HR+/HER2-"]
r_greb1 = fit_glm(df_hr, "ESR1_raw", "GREB1_raw", "ESR1->GREB1 (HR+/HER2-)")
r_pgr   = fit_glm(df_hr, "ESR1_raw", "PGR_raw", "ESR1->PGR (HR+/HER2-)")

results = [r_apm, r_greb1, r_pgr]
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUTDIR}/results.csv", index=False)

readme += [
    "",
    "RESULTS (within-sample depth-offset GLM, cluster-robust SE by sample_id)",
    "-"*70,
] + [
    f"  {r['label']}: n_samples={r['n_samples']}, n_cells={r['n_cells']:,}, "
    f"RR={r['rr']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}], p={r['p']:.2e} ({r['model']})"
    for r in results
]

apm_negative = r_apm["ci_hi"] < 1
apm_positive = r_apm["ci_lo"] > 1
esr1_positive = (r_greb1["ci_lo"] > 1) and (r_pgr["ci_lo"] > 1)
readme += [
    "",
    "INTERPRETATION",
    "-"*70,
    f"  LCOR->APM direction: {'NEGATIVE (replicates Visvader-style reversal)' if apm_negative else ('POSITIVE (replicates Klughammer/expected direction)' if apm_positive else 'NULL/inconclusive')}",
    f"  ESR1->GREB1/PGR (method sanity check): {'confirms expected positive direction' if esr1_positive else 'does NOT confirm expected direction -- flag for review'}",
]

fig, ax = plt.subplots(figsize=(7, 3.2))
labels = [r["label"] for r in results]
rrs = [r["rr"] for r in results]
los = [r["ci_lo"] for r in results]
his = [r["ci_hi"] for r in results]
ys = list(range(len(results)))[::-1]
for y, rr, lo, hi in zip(ys, rrs, los, his):
    color = "#E41A1C" if lo > 1 else ("#377EB8" if hi < 1 else "gray")
    ax.plot([lo, hi], [y, y], color=color, lw=2)
    ax.scatter([rr], [y], color=color, s=70, zorder=3)
    ax.text(rr, y + 0.15, f"RR={rr:.2f} [{lo:.2f}-{hi:.2f}]", ha="center", fontsize=8)
ax.axvline(1, color="black", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Within-sample rate ratio (depth-offset GLM)", fontsize=8.5)
ax.set_ylim(-0.6, len(results) - 0.4)
ax.set_title("Regner/GSE243526 (independent atlas) replication check", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_replication.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUTDIR}/fig_replication.png")

readme += ["", "OUTPUT FILES", "-"*70, "  results.csv, fig_replication.png"]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  LCOR->APM:    RR={r_apm['rr']:.3f} [{r_apm['ci_lo']:.3f}-{r_apm['ci_hi']:.3f}], p={r_apm['p']:.2e}
  ESR1->GREB1:  RR={r_greb1['rr']:.3f} [{r_greb1['ci_lo']:.3f}-{r_greb1['ci_hi']:.3f}], p={r_greb1['p']:.2e}
  ESR1->PGR:    RR={r_pgr['rr']:.3f} [{r_pgr['ci_lo']:.3f}-{r_pgr['ci_hi']:.3f}], p={r_pgr['p']:.2e}
  Output: {OUTDIR}/
============================================================
""")
