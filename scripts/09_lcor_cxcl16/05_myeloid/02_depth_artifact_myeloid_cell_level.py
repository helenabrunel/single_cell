"""
02_depth_artifact_myeloid_cell_level.py
========================================
Depth-adjusted cell-level LCOR x CXCL16 test, restricted to MYELOID cells
(Macro_M1 + Macro_M2 + Dendritic combined). Direct myeloid analogue of
02_correlacion/02_depth_artifact_cell_level.py, which ran the same test in
malignant (epithelial) cells and found the pooled correlation was largely a
between-patient artifact (within-patient rate ratio ~1, n.s.).

WHY THIS SCRIPT EXISTS
-----------------------
01_pseudobulk_myeloid_headline.py (this folder) gives the patient-level
answer. This script complements it at the cell level, using the same
depth-offset GLM approach as 02_correlacion/02_depth_artifact_cell_level.py
instead of a naive/codetected-only correlation -- both genes are detected
much more often in myeloid cells (CXCL16 ~40-47%, LCOR ~14-17%) than in
malignant cells (CXCL16 13.7%, LCOR 15.5%), so the depth/dropout artifact
that dominated the malignant-cell analysis is expected to matter less here,
but is still checked directly rather than assumed away.

APPROACH
--------
Model raw CXCL16 UMI counts with sequencing depth as a model OFFSET
(log total_counts), using ALL myeloid cells (no codetected-only filter):

  CXCL16_raw ~ log1p(LCOR_raw) + C(subtype) + offset(log(total_counts))   [pooled]
  CXCL16_raw ~ log1p(LCOR_raw) + C(sample)  + offset(log(total_counts))   [within-patient]

As in 02_correlacion/02, comparing pooled vs within-patient (sample
fixed-effects) coefficients tests whether an apparent relationship is
within-tumor co-variation or a between-patient artifact (e.g. one tumor's
myeloid infiltrate simply expressing more of both genes than another's).

Output: LCOR-CXCL16/05_myeloid/figures/02_depth_artifact_myeloid_cell_level/
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

H5AD_FILE   = "compartment_immune.h5ad"
OUTDIR      = "LCOR-CXCL16/05_myeloid/figures/02_depth_artifact_myeloid_cell_level"
GENE_X, GENE_Y = "LCOR", "CXCL16"
MIN_CODET_PER_SAMPLE = 15
MYELOID_GROUPS = ["Macro_M1", "Macro_M2", "Dendritic"]
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

IMMUNE_CELL_TYPES = {
    "CD8_T": "score_CD8_T", "CD4_T": "score_CD4_T", "Treg": "score_Treg", "NK": "score_NK",
    "Macro_M1": "score_Macro_M1", "Macro_M2": "score_Macro_M2", "B_cell": "score_B_cell",
    "Plasmablast": "score_Plasmablast", "Dendritic": "score_Dendritic", "Mast": "score_Mast",
}


def assign_immune_type(obs):
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items() if col in obs.columns}
    score_matrix = obs[[col for col in cols_present.values()]].values
    cell_types = list(cols_present.keys())
    best_idx = score_matrix.argmax(axis=1)
    best_score = score_matrix.max(axis=1)
    return np.array([cell_types[i] if s > 0 else "Unknown"
                      for i, s in zip(best_idx, best_score)])


print("="*65)
print(f" 02 -- {GENE_X} x {GENE_Y} -- DEPTH-OFFSET REGRESSION, myeloid cells, by patient")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}")

celltype = assign_immune_type(adata.obs)
is_myeloid = np.isin(celltype, MYELOID_GROUPS)
print(f"  Myeloid cells ({'+'.join(MYELOID_GROUPS)}): {is_myeloid.sum():,} / {adata.n_obs:,}")

idx = [adata.var_names.get_loc(g) for g in (GENE_X, GENE_Y)]
raw = adata.layers["counts"][:, idx]
raw = np.asarray(raw.todense()) if issparse(raw) else np.asarray(raw)
log = adata.layers["logcounts"][:, idx] if "logcounts" in adata.layers else adata.X[:, idx]
log = np.asarray(log.todense()) if issparse(log) else np.asarray(log)

df_full = pd.DataFrame({
    f"{GENE_X}_raw": raw[:, 0].astype(int),
    f"{GENE_Y}_raw": raw[:, 1].astype(int),
    f"{GENE_X}_log": log[:, 0],
    f"{GENE_Y}_log": log[:, 1],
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "subtype": adata.obs["subtype"].astype(str).values,
    "sample":  adata.obs["sample"].astype(str).values,
}, index=adata.obs_names)
del adata
df = df_full[is_myeloid].copy()
df["log_total_counts"] = np.log(df["total_counts"])
df["codet"] = (df[f"{GENE_X}_raw"] > 0) & (df[f"{GENE_Y}_raw"] > 0)

print(f"  Codetected (both>0), myeloid cells: {df['codet'].sum():,} ({100*df['codet'].mean():.2f}%)")

readme = [
    f"02 -- {GENE_X} x {GENE_Y} -- depth-offset regression, myeloid cells, by patient",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}, restricted to {'+'.join(MYELOID_GROUPS)}",
    "="*60,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*60,
    "Myeloid analogue of 02_correlacion/02_depth_artifact_cell_level.py,",
    "which found the malignant-compartment cell-level correlation was",
    "largely a between-patient artifact once depth and patient identity",
    "were accounted for. CXCL16 (and LCOR) are detected much more often in",
    "myeloid cells than in malignant cells, so the same depth-offset /",
    "within-patient approach is applied here to check whether that changes",
    "the conclusion.",
]

# ============================================================================
# Depth vs codetection sanity check
# ============================================================================
med_codet = df.loc[df["codet"], "total_counts"].median()
med_rest  = df.loc[~df["codet"], "total_counts"].median()
print(f"\n  Median total_counts: codetected={med_codet:,.0f}  rest={med_rest:,.0f}")
readme += [
    "",
    "DEPTH vs CODETECTION",
    "-"*60,
    f"  Median total_counts, codetected cells: {med_codet:,.0f}",
    f"  Median total_counts, rest of cells:    {med_rest:,.0f}",
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
df_persample = pd.DataFrame(rows).sort_values("r") if rows else pd.DataFrame(
    columns=["sample", "subtype", "n_codet", "r", "p", "med_depth"])
df_persample.to_csv(f"{OUTDIR}/persample_naive_correlation.csv", index=False)
print(df_persample.to_string(index=False) if len(df_persample) else "  (no patient meets the codetection threshold)")

if len(df_persample) >= 3:
    depth_r_corr, depth_r_p = stats.spearmanr(df_persample["med_depth"], df_persample["r"])
else:
    depth_r_corr, depth_r_p = np.nan, np.nan
print(f"\n  Per-patient r vs per-patient median depth: Spearman rho={depth_r_corr:.3f}, p={depth_r_p:.3g}"
      if len(df_persample) >= 3 else "\n  Too few patients for per-patient r vs depth check.")

readme += [
    "",
    "PER-PATIENT NAIVE CORRELATION (codetected-only)",
    "-"*60,
    f"  {len(df_persample)} patients with >={MIN_CODET_PER_SAMPLE} codetected myeloid cells",
]
if len(df_persample):
    readme += [
        f"  r ranges from {df_persample['r'].min():.2f} to {df_persample['r'].max():.2f}",
        f"  Spearman(per-patient r, per-patient median depth) = {depth_r_corr:.3f}, p={depth_r_p:.3g}",
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
    "Interpretation: same logic as 02_correlacion/02_depth_artifact_cell_level.py.",
    "If within-patient RR stays close to the pooled RR and significant, the",
    "LCOR-CXCL16 link holds cell-to-cell inside the same tumor's myeloid",
    "infiltrate. If it shrinks toward 1 / loses significance, the pooled",
    "estimate is largely a between-patient artifact.",
]

# ============================================================================
# FIGURE 1 -- per-patient forest plot of naive correlation
# ============================================================================
print("\n  Generating figures...")
if len(df_persample):
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28*len(df_persample))))
    colors = [SUBTYPE_COLORS.get(s, "gray") for s in df_persample["subtype"]]
    ypos = np.arange(len(df_persample))
    ax.barh(ypos, df_persample["r"], color=colors, height=0.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(df_persample["sample"], df_persample["n_codet"])], fontsize=6.5)
    ax.set_xlabel(f"Pearson r ({GENE_X} vs {GENE_Y}, codetected myeloid cells only)", fontsize=9)
    ax.set_title("Per-patient correlation, myeloid cells", fontsize=10, fontweight="bold")
    from matplotlib.patches import Patch
    legend_elems = [Patch(color=c, label=st) for st, c in SUBTYPE_COLORS.items()]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="lower right")
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
ax.set_title(f"{GENE_X}->{GENE_Y}, myeloid cells: pooled vs within-patient", fontsize=10, fontweight="bold")
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
    "  cell_level_data.csv                    -- per-cell raw/log counts, depth, subtype, sample (myeloid only)",
    "  persample_naive_correlation.csv        -- per-patient codetected-only Pearson r",
    f"  model_summary_pooled_{model_name.replace(' ','')}.txt",
    f"  model_summary_within_patient_{model_name.replace(' ','')}.txt",
    "  fig1_persample_forest.png",
    "  fig2_pooled_vs_within_patient.png",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Codetected: {df['codet'].sum():,} ({100*df['codet'].mean():.2f}%) of {len(df):,} myeloid cells
  Pooled depth-adjusted ({model_name}):         RR={pooled_rr:.2f} [{pooled_ci[0]:.2f}-{pooled_ci[1]:.2f}], p={pooled_p:.2e}
  Within-patient depth-adjusted:                RR={within_rr:.2f} [{within_ci[0]:.2f}-{within_ci[1]:.2f}], p={within_p:.2e}
  Output: {OUTDIR}/
============================================================
""")
