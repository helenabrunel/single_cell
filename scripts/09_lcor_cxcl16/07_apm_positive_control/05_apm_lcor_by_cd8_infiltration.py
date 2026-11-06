"""
05_apm_lcor_by_cd8_infiltration.py
====================================
Follow-up to the APM positive control (01-04): 02/04 found LCOR and the APM
meta-gene are NEGATIVELY associated at the cell level, within patient, in
both malignant and myeloid compartments -- the OPPOSITE direction to the
lab's own mouse/culture experiments, where LCOR positively regulates APM.

HYPOTHESIS TESTED HERE
-----------------------
This observational, cross-sectional reversal could reflect immune-mediated
selection rather than a genuinely reversed regulatory mechanism: in vivo,
malignant cells presenting more antigen (high APM) are more visible to CD8
T cells and may be preferentially eliminated. If a tumor has substantial
CD8 infiltration, this selection pressure has more opportunity to act,
which could produce a negative LCOR-APM association among the SURVIVING
malignant cells even if LCOR causally promotes APM at the moment of
transcription (as in the lab's controlled experiments). If a tumor has
little CD8 infiltration, that selection pressure is largely absent, so the
underlying (positive) regulatory relationship should be less distorted.

Prediction: the negative LCOR-APM relationship (within-patient RR) should
be MORE negative (further below 1) in patients with HIGH CD8 T-cell
infiltration than in patients with LOW CD8 infiltration.

This is exploratory/correlational -- it cannot prove the mechanism, only
check whether the pattern is consistent with it.

METHOD
------
1. Per-patient CD8 T-cell fraction among immune cells (winner-takes-all
   typing, identical to 05_myeloid), from compartment_immune.h5ad.
2. Patients split into top/bottom TERCILE of CD8 fraction (same
   infiltration framework as the earlier TCGA bulk analysis).
3. Malignant-cell LCOR vs APM depth-offset GLM (identical model to
   02_depth_artifact_apm_malignant_cell_level.py), refit separately within
   the High-CD8 and Low-CD8 patient subsets (within-patient, sample FE).
4. Compare the two within-patient RRs.

Output: LCOR-CXCL16/07_apm_positive_control/figures/05_apm_lcor_by_cd8_infiltration/
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

IMMUNE_H5AD     = "compartment_immune.h5ad"
MALIGNANT_H5AD  = "compartment_epithelial.h5ad"
OUTDIR          = "LCOR-CXCL16/07_apm_positive_control/figures/05_apm_lcor_by_cd8_infiltration"
GENE_X          = "LCOR"
APM_GENES       = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
GENE_Y          = "APM_score"
os.makedirs(OUTDIR, exist_ok=True)

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
print(" 05 -- LCOR x APM by CD8 infiltration tercile (immune-selection check)")
print("="*65)

# ============================================================================
# STEP 1 -- per-patient CD8 T-cell fraction (immune compartment)
# ============================================================================
print(f"\n  Loading {IMMUNE_H5AD}...")
adata_imm = sc.read_h5ad(IMMUNE_H5AD)
celltype = assign_immune_type(adata_imm.obs)
imm_obs = pd.DataFrame({
    "sample": adata_imm.obs["sample"].astype(str).values,
    "celltype": celltype,
})
del adata_imm

cd8_frac = imm_obs.groupby("sample").apply(
    lambda g: (g["celltype"] == "CD8_T").mean()
).rename("cd8_fraction").reset_index()
n_immune = imm_obs.groupby("sample").size().rename("n_immune_cells").reset_index()
cd8_frac = cd8_frac.merge(n_immune, on="sample")
print(f"  Patients with immune-compartment data: {len(cd8_frac)}")
print(cd8_frac.sort_values("cd8_fraction").to_string(index=False))

# Tercile split
q1, q2 = cd8_frac["cd8_fraction"].quantile([1/3, 2/3])
print(f"\n  CD8 fraction tercile cutoffs: {q1:.4f} / {q2:.4f}")
cd8_frac["cd8_group"] = np.select(
    [cd8_frac["cd8_fraction"] <= q1, cd8_frac["cd8_fraction"] >= q2],
    ["Low", "High"], default="Mid"
)
print(cd8_frac["cd8_group"].value_counts().to_string())
cd8_frac.to_csv(f"{OUTDIR}/persample_cd8_infiltration.csv", index=False)

readme = [
    "05 -- LCOR x APM by CD8 infiltration tercile (immune-selection check)",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "02/04_depth_artifact_apm_*_cell_level.py found LCOR-APM NEGATIVELY",
    "associated within-patient, in both malignant and myeloid cells -- the",
    "OPPOSITE direction from the lab's own mouse/culture experiments (LCOR",
    "positively regulates APM). Hypothesis tested here: in vivo immune",
    "selection (high-APM malignant cells preferentially killed by CD8 T",
    "cells) could produce this reversal in cross-sectional human tumor",
    "data, and should be more pronounced in tumors with more CD8",
    "infiltration (more opportunity for selection to have acted).",
    "Prediction: within-patient LCOR->APM RR further below 1 in High-CD8",
    "patients than in Low-CD8 patients.",
    "",
    "CD8 INFILTRATION (compartment_immune.h5ad, winner-takes-all typing)",
    "-"*70,
    f"  Tercile cutoffs: {q1:.4f} / {q2:.4f}",
] + [f"  {row.cd8_group}: {row.sample} (CD8 frac={row.cd8_fraction:.3f}, n_immune={row.n_immune_cells})"
     for row in cd8_frac.sort_values(["cd8_group", "cd8_fraction"]).itertuples()]

# ============================================================================
# STEP 2 -- malignant-cell LCOR + APM raw counts (identical to script 02)
# ============================================================================
print(f"\n  Loading {MALIGNANT_H5AD}...")
adata = sc.read_h5ad(MALIGNANT_H5AD)
idx_x = adata.var_names.get_loc(GENE_X)
idx_apm = [adata.var_names.get_loc(g) for g in APM_GENES]
raw_x = adata.layers["counts"][:, idx_x]
raw_x = np.asarray(raw_x.todense()).ravel() if issparse(raw_x) else np.asarray(raw_x).ravel()
raw_apm_matrix = adata.layers["counts"][:, idx_apm]
raw_apm_matrix = np.asarray(raw_apm_matrix.todense()) if issparse(raw_apm_matrix) else np.asarray(raw_apm_matrix)
raw_apm = raw_apm_matrix.sum(axis=1)

df = pd.DataFrame({
    f"{GENE_X}_raw": raw_x.astype(int),
    f"{GENE_Y}_raw": raw_apm.astype(int),
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "subtype": adata.obs["subtype"].astype(str).values,
    "sample":  adata.obs["sample"].astype(str).values,
}, index=adata.obs_names)
df["log_total_counts"] = np.log(df["total_counts"])
del adata

df = df.merge(cd8_frac[["sample", "cd8_group", "cd8_fraction"]], on="sample", how="left")
print(f"  Malignant cells with a CD8-infiltration group assigned: {df['cd8_group'].notna().sum():,} / {len(df):,}")
print(df.groupby("cd8_group")["sample"].nunique().to_string())

readme += [
    "",
    "MALIGNANT CELLS PER CD8 GROUP",
    "-"*70,
] + [f"  {grp}: {n:,} patients, {cnt:,} malignant cells"
     for grp, n, cnt in [
         (g, df[df.cd8_group == g]["sample"].nunique(), (df.cd8_group == g).sum())
         for g in ["Low", "Mid", "High"]
     ]]

# ============================================================================
# STEP 3 -- depth-offset GLM, refit within High-CD8 and Low-CD8 subsets
# ============================================================================
def fit_within_patient(d, label):
    d = d.copy()
    d["subtype"] = pd.Categorical(d["subtype"], categories=["ER", "HER2", "TNBC"])
    d["log1p_LCOR"] = np.log1p(d[f"{GENE_X}_raw"])

    poisson_pooled = smf.glm(
        f"{GENE_Y}_raw ~ log1p_LCOR + C(subtype)",
        data=d, family=sm.families.Poisson(),
        offset=d["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["sample"]})
    disp = poisson_pooled.pearson_chi2 / poisson_pooled.df_resid
    family = sm.families.NegativeBinomial() if disp > 1.5 else sm.families.Poisson()
    model_name = "Negative Binomial" if disp > 1.5 else "Poisson"

    within = smf.glm(
        f"{GENE_Y}_raw ~ log1p_LCOR + C(sample)",
        data=d, family=family,
        offset=d["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["sample"]})

    coef = within.params["log1p_LCOR"]
    p    = within.pvalues["log1p_LCOR"]
    rr   = np.exp(coef)
    ci   = np.exp(within.conf_int().loc["log1p_LCOR"].values)
    print(f"\n  [{label}] n_patients={d['sample'].nunique()}, n_cells={len(d):,}, dispersion={disp:.2f} -> {model_name}")
    print(f"    WITHIN-PATIENT RR={rr:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p={p:.2e}")
    return dict(label=label, n_patients=d["sample"].nunique(), n_cells=len(d),
                model=model_name, rr=rr, ci_lo=ci[0], ci_hi=ci[1], p=p)

print("\n" + "="*65)
print(" Depth-offset GLM (within-patient), refit per CD8 infiltration group")
print("="*65)
results = []
for grp in ["Low", "High"]:
    d_grp = df[df["cd8_group"] == grp]
    results.append(fit_within_patient(d_grp, f"{grp}-CD8"))
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUTDIR}/rr_by_cd8_group.csv", index=False)

more_negative_in_high = res_df.loc[res_df.label == "High-CD8", "rr"].values[0] < res_df.loc[res_df.label == "Low-CD8", "rr"].values[0]

readme += [
    "",
    "DEPTH-ADJUSTED WITHIN-PATIENT RESULT, BY CD8 GROUP",
    "-"*70,
] + [
    f"  {r['label']}: n_patients={r['n_patients']}, n_cells={r['n_cells']:,}, "
    f"RR={r['rr']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}], p={r['p']:.2e} ({r['model']})"
    for r in results
] + [
    "",
    f"  Prediction (immune-selection hypothesis): High-CD8 RR further below 1",
    f"  than Low-CD8 RR.",
    f"  Observed: {'CONSISTENT -- High-CD8 RR is more negative (further below 1)' if more_negative_in_high else 'NOT CONSISTENT -- High-CD8 RR is not further below 1 than Low-CD8'}",
]

# ============================================================================
# FIGURE -- RR by CD8 group
# ============================================================================
print("\n  Generating figure...")
fig, ax = plt.subplots(figsize=(6, 3))
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
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel(f"Within-patient rate ratio, LCOR->APM\n(depth-offset GLM)", fontsize=8.5)
ax.set_ylim(-0.6, len(results) - 0.4)
ax.set_title("LCOR->APM within-patient RR, by CD8 infiltration tercile\n(immune-selection hypothesis check)",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_rr_by_cd8_group.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig_rr_by_cd8_group.png")

readme += [
    "",
    "OUTPUT FILES",
    "-"*70,
    "  persample_cd8_infiltration.csv  -- per-patient CD8 fraction + tercile group",
    "  rr_by_cd8_group.csv             -- within-patient RR per group",
    "  fig_rr_by_cd8_group.png",
    "",
    "CAVEATS",
    "-"*70,
    "  - Exploratory, post-hoc, correlational -- cannot confirm the immune-",
    "    selection mechanism, only check whether the pattern is consistent.",
    "  - Splitting an already-small cohort (34 patients) into terciles leaves",
    "    ~11 patients per group -- read the comparison as suggestive, not",
    "    confirmatory.",
    "  - Mid tercile excluded from this comparison (matches the earlier TCGA",
    "    infiltration methodology).",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Low-CD8:  RR={results[0]['rr']:.3f} [{results[0]['ci_lo']:.3f}-{results[0]['ci_hi']:.3f}], p={results[0]['p']:.2e} (n={results[0]['n_patients']} patients)
  High-CD8: RR={results[1]['rr']:.3f} [{results[1]['ci_lo']:.3f}-{results[1]['ci_hi']:.3f}], p={results[1]['p']:.2e} (n={results[1]['n_patients']} patients)
  Immune-selection prediction: {'CONSISTENT' if more_negative_in_high else 'NOT CONSISTENT'}
  Output: {OUTDIR}/
============================================================
""")
