"""
04_depth_artifact_apm_myeloid_cell_level.py
=============================================
POSITIVE-CONTROL check, MYELOID compartment, cell level: depth-offset GLM
of LCOR vs the APM meta-gene. Myeloid analogue of
02_depth_artifact_apm_malignant_cell_level.py, and direct positive-control
counterpart of 05_myeloid/02_depth_artifact_myeloid_cell_level.py (LCOR vs
CXCL16, myeloid cells, null: pooled RR=1.03 n.s., within-patient RR=1.02
n.s.).

APPROACH (identical to 02_depth_artifact_apm_malignant_cell_level.py)
------------------------------------------------------------------------
  APM_raw ~ log1p(LCOR_raw) + C(subtype) + offset(log(total_counts))   [pooled]
  APM_raw ~ log1p(LCOR_raw) + C(sample)  + offset(log(total_counts))   [within-patient]

Output: LCOR-CXCL16/07_apm_positive_control/figures/04_depth_artifact_apm_myeloid_cell_level/
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
OUTDIR      = "LCOR-CXCL16/07_apm_positive_control/figures/04_depth_artifact_apm_myeloid_cell_level"
GENE_X      = "LCOR"
APM_GENES   = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
GENE_Y      = "APM_score"
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
print(f" 04 -- {GENE_X} x {GENE_Y} (positive control) -- DEPTH-OFFSET REGRESSION, myeloid, by patient")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}")

celltype = assign_immune_type(adata.obs)
is_myeloid = np.isin(celltype, MYELOID_GROUPS)
print(f"  Myeloid cells ({'+'.join(MYELOID_GROUPS)}): {is_myeloid.sum():,} / {adata.n_obs:,}")

idx_x = adata.var_names.get_loc(GENE_X)
idx_apm = [adata.var_names.get_loc(g) for g in APM_GENES]
raw_x = adata.layers["counts"][:, idx_x]
raw_x = np.asarray(raw_x.todense()).ravel() if issparse(raw_x) else np.asarray(raw_x).ravel()
raw_apm_matrix = adata.layers["counts"][:, idx_apm]
raw_apm_matrix = np.asarray(raw_apm_matrix.todense()) if issparse(raw_apm_matrix) else np.asarray(raw_apm_matrix)
raw_apm = raw_apm_matrix.sum(axis=1)
log_x = adata.layers["logcounts"][:, idx_x] if "logcounts" in adata.layers else adata.X[:, idx_x]
log_x = np.asarray(log_x.todense()).ravel() if issparse(log_x) else np.asarray(log_x).ravel()
log_apm = np.log1p(raw_apm)

df_full = pd.DataFrame({
    f"{GENE_X}_raw": raw_x.astype(int),
    f"{GENE_Y}_raw": raw_apm.astype(int),
    f"{GENE_X}_log": log_x,
    f"{GENE_Y}_log": log_apm,
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
    f"04 -- {GENE_X} x {GENE_Y} (POSITIVE CONTROL) -- depth-offset regression, myeloid, by patient",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}, restricted to {'+'.join(MYELOID_GROUPS)}",
    "="*60,
    "",
    "APM GENE SET",
    "-"*60,
    "  " + ", ".join(APM_GENES),
]

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

readme += [
    "",
    "PER-PATIENT NAIVE CORRELATION (codetected-only)",
    "-"*60,
    f"  {len(df_persample)} patients with >={MIN_CODET_PER_SAMPLE} codetected myeloid cells",
]
if len(df_persample):
    readme += [f"  r ranges from {df_persample['r'].min():.2f} to {df_persample['r'].max():.2f}"]

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
]

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
    ax.set_title("Per-patient correlation, myeloid cells (positive control)", fontsize=10, fontweight="bold")
    from matplotlib.patches import Patch
    legend_elems = [Patch(color=c, label=st) for st, c in SUBTYPE_COLORS.items()]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig1_persample_forest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTDIR}/fig1_persample_forest.png")

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
ax.set_title(f"{GENE_X}->{GENE_Y} (positive control), myeloid: pooled vs within-patient", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_pooled_vs_within_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig2_pooled_vs_within_patient.png")

df.drop(columns=["codet"]).to_csv(f"{OUTDIR}/cell_level_data.csv")
with open(f"{OUTDIR}/model_summary_pooled_{model_name.replace(' ','')}.txt", "w") as f:
    f.write(pooled.summary().as_text())
with open(f"{OUTDIR}/model_summary_within_patient_{model_name.replace(' ','')}.txt", "w") as f:
    f.write(within.summary().as_text())
print(f"  Saved: {OUTDIR}/cell_level_data.csv + model summaries")

readme += [
    "",
    "OUTPUT FILES",
    "-"*60,
    "  README.txt, cell_level_data.csv, persample_naive_correlation.csv",
    f"  model_summary_pooled_{model_name.replace(' ','')}.txt",
    f"  model_summary_within_patient_{model_name.replace(' ','')}.txt",
    "  fig1_persample_forest.png, fig2_pooled_vs_within_patient.png",
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
