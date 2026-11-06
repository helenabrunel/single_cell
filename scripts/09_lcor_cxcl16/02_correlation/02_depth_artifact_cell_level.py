"""
02_depth_artifact_cell_level.py
===============================
Depth-adjusted re-analysis of the LCOR x CXCL16 relationship in malignant
(epithelial) cells, broken down by PATIENT (sample) instead of subtype.

BACKGROUND / WHY THIS SCRIPT EXISTS
------------------------------------
14_lcor_cxcl16_hexbin.py found a strong positive correlation (Pearson
r~0.63) between LCOR and CXCL16 logcounts, but only after restricting to
the ~3% of cells where BOTH genes are detected ("codetected-only"/"sin
ceros"). That result was flagged as purely descriptive because:
  (a) restricting to codetected cells conditions on the outcome itself
      (collider-type selection), discarding 97% of cells, and
  (b) scRNA-seq detection (dropout) is itself driven by sequencing depth,
      so two genes can look "correlated" simply because deeper cells
      detect more of everything.
Visual inspection also suggested several distinct linear trends WITHIN
the same subtype (e.g. within ER), which is a classic signature of
between-patient (sample) heterogeneity in multi-patient scRNA-seq, not
necessarily biological subclones.

APPROACH
--------
Same logic as 07f_TIM3_hypoxia_depth_adjusted.py: model raw CXCL16 UMI
counts directly (not the codetected-only subset, not a binarized
detected/not) with sequencing depth as a model OFFSET (log total_counts),
using ALL epithelial cells. This avoids discarding 97% of cells and lets
depth act as an "exposure" rather than a hard filter.

  CXCL16_raw ~ log1p(LCOR_raw) + C(subtype) + offset(log(total_counts))   [pooled]
  CXCL16_raw ~ log1p(LCOR_raw) + C(sample)  + offset(log(total_counts))   [within-patient]

Comparing the pooled vs within-patient (sample fixed-effects) coefficient
directly tests the "several lines per subtype = patient effect" question:
if the LCOR effect shrinks/disappears once patient fixed effects absorb
each tumor's own baseline level and depth, the pooled correlation was
mostly a BETWEEN-patient artifact, not within-tumor co-variation.

Additionally: per-patient naive (codetected-only) Pearson r, forest-plotted
and sorted by patient (not subtype), to visualise the heterogeneity
directly and check whether it tracks patient-level sequencing depth.

Output: figures/15_LCOR_CXCL16_depth_patient/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, warnings, datetime
from scipy.sparse import issparse
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE   = "compartment_epithelial.h5ad"
OUTDIR      = "LCOR-CXCL16/02_correlacion/figures/02_depth_artifact_cell_level"
GENE_X, GENE_Y = "LCOR", "CXCL16"
MIN_CODET_PER_SAMPLE = 15
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

print("="*65)
print(f" 02 -- {GENE_X} x {GENE_Y} -- DEPTH-OFFSET REGRESSION, BY PATIENT")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}")

idx = [adata.var_names.get_loc(g) for g in (GENE_X, GENE_Y)]
raw = adata.layers["counts"][:, idx]
raw = np.asarray(raw.todense()) if issparse(raw) else np.asarray(raw)
log = adata.layers["logcounts"][:, idx] if "logcounts" in adata.layers else adata.X[:, idx]
log = np.asarray(log.todense()) if issparse(log) else np.asarray(log)

df = pd.DataFrame({
    f"{GENE_X}_raw": raw[:, 0].astype(int),
    f"{GENE_Y}_raw": raw[:, 1].astype(int),
    f"{GENE_X}_log": log[:, 0],
    f"{GENE_Y}_log": log[:, 1],
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "subtype": adata.obs["subtype"].astype(str).values,
    "sample":  adata.obs["sample"].astype(str).values,
}, index=adata.obs_names)
df["log_total_counts"] = np.log(df["total_counts"])
df["codet"] = (df[f"{GENE_X}_raw"] > 0) & (df[f"{GENE_Y}_raw"] > 0)

del adata
print(f"  Codetected (both>0): {df['codet'].sum():,} ({100*df['codet'].mean():.2f}%)")

readme = [
    f"02 -- {GENE_X} x {GENE_Y} -- depth-offset regression, by patient",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "WHY THIS REPLACES/EXTENDS THE 14_hexbin CODETECTED-ONLY RESULT",
    "-"*60,
    "14_lcor_cxcl16_hexbin.py found Pearson r~0.63 restricting to the ~3%",
    "of cells where both genes are detected. That restriction discards 97%",
    "of cells and conditions on the outcome (collider-type selection);",
    "also, within a subtype the scatter showed several distinct trends,",
    "suggestive of patient-level heterogeneity rather than one biological",
    "relationship. This script (a) models raw counts with a depth OFFSET",
    "instead of filtering to codetected cells, and (b) compares a pooled",
    "model to a within-patient (sample fixed-effects) model to test",
    "whether the association survives once each patient's own baseline",
    "level/depth is absorbed.",
]

# ============================================================================
# Depth vs codetection sanity check
# ============================================================================
med_codet   = df.loc[df["codet"], "total_counts"].median()
med_rest    = df.loc[~df["codet"], "total_counts"].median()
print(f"\n  Median total_counts: codetected={med_codet:,.0f}  rest={med_rest:,.0f}")
readme += [
    "",
    "DEPTH vs CODETECTION",
    "-"*60,
    f"  Median total_counts, codetected cells: {med_codet:,.0f}",
    f"  Median total_counts, rest of cells:    {med_rest:,.0f}",
    "(If codetected cells are systematically deeper, part of the",
    " codetected-only correlation could reflect depth rather than biology.)",
]

# ============================================================================
# Naive per-patient breakdown (codetected-only Pearson r), sorted by patient
# ============================================================================
print(f"\n  Naive per-patient Pearson r (codetected-only, n>={MIN_CODET_PER_SAMPLE})...")
rows = []
for s, g in df[df["codet"]].groupby("sample"):
    if len(g) < MIN_CODET_PER_SAMPLE:
        continue
    r, p = stats.pearsonr(g[f"{GENE_X}_log"], g[f"{GENE_Y}_log"])
    rows.append({"sample": s, "subtype": g["subtype"].iloc[0], "n_codet": len(g),
                 "r": r, "p": p, "med_depth": df.loc[df["sample"] == s, "total_counts"].median()})
df_persample = pd.DataFrame(rows).sort_values("r")
df_persample.to_csv(f"{OUTDIR}/persample_naive_correlation.csv", index=False)
print(df_persample.to_string(index=False))

depth_r_corr, depth_r_p = stats.spearmanr(df_persample["med_depth"], df_persample["r"])
print(f"\n  Per-patient r vs per-patient median depth: Spearman rho={depth_r_corr:.3f}, p={depth_r_p:.3f}")

readme += [
    "",
    "PER-PATIENT NAIVE CORRELATION (codetected-only, replaces subtype cut)",
    "-"*60,
    f"  {len(df_persample)} patients with >={MIN_CODET_PER_SAMPLE} codetected cells",
    f"  r ranges from {df_persample['r'].min():.2f} to {df_persample['r'].max():.2f}",
    f"  Spearman(per-patient r, per-patient median depth) = {depth_r_corr:.3f}, p={depth_r_p:.3f}",
    "(A significant relationship here would mean the patient-to-patient",
    " differences in slope track sequencing depth rather than being",
    " independent biological variation.)",
]

# ============================================================================
# Depth-offset Poisson / NB regression -- POOLED (subtype-adjusted)
# ============================================================================
print("\n  Fitting depth-offset GLM (pooled, subtype-adjusted)...")
model_df = df.copy()
model_df["subtype"] = pd.Categorical(model_df["subtype"], categories=["ER", "HER2", "TNBC"])
model_df["log1p_LCOR"] = np.log1p(model_df[f"{GENE_X}_raw"])

poisson_pooled = smf.glm(
    f"{GENE_Y}_raw ~ log1p_LCOR + C(subtype)",
    data=model_df, family=sm.families.Poisson(),
    offset=model_df["log_total_counts"],
).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})

disp = poisson_pooled.pearson_chi2 / poisson_pooled.df_resid
print(f"  Poisson dispersion (Pearson chi2/df): {disp:.3f}")
family = sm.families.NegativeBinomial() if disp > 1.5 else sm.families.Poisson()
model_name = "Negative Binomial" if disp > 1.5 else "Poisson"
print(f"  -> using {model_name} GLM")

pooled = smf.glm(
    f"{GENE_Y}_raw ~ log1p_LCOR + C(subtype)",
    data=model_df, family=family,
    offset=model_df["log_total_counts"],
).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})

pooled_coef = pooled.params["log1p_LCOR"]
pooled_p    = pooled.pvalues["log1p_LCOR"]
pooled_rr   = np.exp(pooled_coef)
pooled_ci   = np.exp(pooled.conf_int().loc["log1p_LCOR"].values)
print(f"\n  POOLED (subtype-adjusted): RR={pooled_rr:.3f} [{pooled_ci[0]:.3f}-{pooled_ci[1]:.3f}], p={pooled_p:.2e}")

# ============================================================================
# Depth-offset regression -- WITHIN-PATIENT (sample fixed effects)
# ============================================================================
print("\n  Fitting depth-offset GLM (sample fixed effects, within-patient)...")
within = smf.glm(
    f"{GENE_Y}_raw ~ log1p_LCOR + C(sample)",
    data=model_df, family=family,
    offset=model_df["log_total_counts"],
).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})

within_coef = within.params["log1p_LCOR"]
within_p    = within.pvalues["log1p_LCOR"]
within_rr   = np.exp(within_coef)
within_ci   = np.exp(within.conf_int().loc["log1p_LCOR"].values)
print(f"  WITHIN-PATIENT (sample FE): RR={within_rr:.3f} [{within_ci[0]:.3f}-{within_ci[1]:.3f}], p={within_p:.2e}")

readme += [
    "",
    f"DEPTH-ADJUSTED RESULT ({model_name} GLM, offset=log(total_counts),",
    "cluster-robust SE by sample; RR = rate ratio per unit log1p(LCOR raw count))",
    "-"*60,
    f"  POOLED (+ subtype covariate):    RR={pooled_rr:.3f} [{pooled_ci[0]:.3f}-{pooled_ci[1]:.3f}], p={pooled_p:.2e}",
    f"  WITHIN-PATIENT (+ sample FE):    RR={within_rr:.3f} [{within_ci[0]:.3f}-{within_ci[1]:.3f}], p={within_p:.2e}",
    "",
    "Interpretation: the pooled model still mixes between-patient and",
    "within-patient variation (e.g. one tumor simply expressing more of",
    "both genes than another). The within-patient model absorbs each",
    "patient's own baseline/depth via sample fixed effects, so it only",
    "uses variation ACROSS CELLS OF THE SAME TUMOR. If within-patient RR",
    "is close to pooled RR and still significant, the LCOR-CXCL16 link",
    "holds cell-to-cell inside the same tumor (stronger evidence). If it",
    "shrinks toward 1 / loses significance, the pooled correlation was",
    "largely a between-patient (batch-like) artifact -- consistent with",
    "the 'several lines within one subtype' pattern seen visually.",
]

# ============================================================================
# FIGURE 1 -- per-patient forest plot of naive correlation (replaces subtype cut)
# ============================================================================
print("\n  Generating figures...")
fig, ax = plt.subplots(figsize=(7, max(4, 0.28*len(df_persample))))
colors = [SUBTYPE_COLORS.get(s, "gray") for s in df_persample["subtype"]]
ypos = np.arange(len(df_persample))
ax.barh(ypos, df_persample["r"], color=colors, height=0.7)
ax.axvline(0, color="black", lw=1)
ax.axvline(0.633, color="black", ls="--", lw=1, alpha=0.6, label="pooled r=0.633 (all patients)")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(df_persample["sample"], df_persample["n_codet"])], fontsize=6.5)
ax.set_xlabel(f"Pearson r ({GENE_X} vs {GENE_Y}, codetected cells only)", fontsize=9)
ax.set_title("Per-patient correlation -- several distinct 'lines' resolve\ninto patient-to-patient heterogeneity",
             fontsize=10, fontweight="bold")
from matplotlib.patches import Patch
legend_elems = [Patch(color=c, label=st) for st, c in SUBTYPE_COLORS.items()]
ax.legend(handles=legend_elems + [ax.lines[-1]], fontsize=7.5, loc="lower right")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_persample_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig1_persample_forest.png")

# ============================================================================
# FIGURE 2 -- pooled vs within-patient depth-adjusted rate ratio
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 3.2))
labels = ["Pooled\n(+ subtype)", "Within-patient\n(+ sample FE)"]
rrs = [pooled_rr, within_rr]
los = [pooled_ci[0], within_ci[0]]
his = [pooled_ci[1], within_ci[1]]
ps  = [pooled_p, within_p]
ys = [1, 0]
for y, rr, lo, hi in zip(ys, rrs, los, his):
    color = "#E41A1C" if lo > 1 else ("#377EB8" if hi < 1 else "gray")
    ax.plot([lo, hi], [y, y], color=color, lw=2)
    ax.scatter([rr], [y], color=color, s=60, zorder=3)
    ax.text(rr, y + 0.18, f"RR={rr:.2f} [{lo:.2f}-{hi:.2f}]", ha="center", fontsize=8)
ax.axvline(1, color="black", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel(f"Rate ratio per unit log1p({GENE_X} raw count)\n({model_name} GLM, depth-offset)", fontsize=8.5)
ax.set_ylim(-0.6, 1.6)
ax.set_title(f"{GENE_X}->{GENE_Y}: pooled vs within-patient", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_pooled_vs_within_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig2_pooled_vs_within_patient.png")

# ============================================================================
# Save tables + README
# ============================================================================
df.drop(columns=["codet"]).to_csv(f"{OUTDIR}/cell_level_data.csv")
print(f"  Saved: {OUTDIR}/cell_level_data.csv")

with open(f"{OUTDIR}/model_summary_pooled_{model_name.replace(' ','')}.txt", "w") as f:
    f.write(pooled.summary().as_text())
with open(f"{OUTDIR}/model_summary_within_patient_{model_name.replace(' ','')}.txt", "w") as f:
    f.write(within.summary().as_text())
print(f"  Saved model summaries")

readme += [
    "",
    "OUTPUT FILES",
    "-"*60,
    "  README.txt",
    "  cell_level_data.csv                    -- per-cell raw/log counts, depth, subtype, sample",
    "  persample_naive_correlation.csv        -- per-patient codetected-only Pearson r",
    f"  model_summary_pooled_{model_name.replace(' ','')}.txt",
    f"  model_summary_within_patient_{model_name.replace(' ','')}.txt",
    "  fig1_persample_forest.png              -- per-patient r, replaces the subtype-level cut",
    "  fig2_pooled_vs_within_patient.png       -- depth-adjusted RR, pooled vs within-patient",
    "",
    "NOTE ON SCOPE",
    "-"*60,
    "This still cannot establish causality or rule out all confounds (e.g.",
    "cell state/quality beyond raw depth), but it directly addresses the",
    "two caveats raised on the codetected-only hexbin result: (1) it uses",
    "all cells via a depth offset instead of discarding 97% of them, and",
    "(2) it separates between-patient from within-patient signal, which is",
    "the natural explanation for the multiple visual 'lines' seen within",
    "a single subtype.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Naive codetected-only (14_hexbin, all cells): r=0.633
  Pooled depth-adjusted ({model_name}):         RR={pooled_rr:.2f} [{pooled_ci[0]:.2f}-{pooled_ci[1]:.2f}], p={pooled_p:.2e}
  Within-patient depth-adjusted:                RR={within_rr:.2f} [{within_ci[0]:.2f}-{within_ci[1]:.2f}], p={within_p:.2e}
  Per-patient r range: {df_persample['r'].min():.2f} to {df_persample['r'].max():.2f}
  Output: {OUTDIR}/
============================================================
""")
