"""
06_negative_control_esr1_greb1.py
===================================
METHODOLOGICAL DIAGNOSTIC (not about LCOR): 02/04_depth_artifact_apm_*_
cell_level.py found LCOR-APM negatively associated within-patient, the
OPPOSITE direction to the lab's own experiments. Before trusting or
distrusting ANY result from this cell-level pipeline (CXCL16 included), we
need to know: is the sign reversal specific to LCOR/APM biology, or is the
within-patient depth-offset GLM design itself prone to inverting known
relationships in this dataset?

This script answers that with a gene pair that has NOTHING to do with LCOR,
APM, or immune selection, and whose direction is about as well-established
as biology gets: ESR1 (estrogen receptor) and GREB1, a canonical direct
ESR1 transcriptional target (estrogen-bound ESR1 activates GREB1
transcription). More ESR1 -> more GREB1 is expected with high confidence,
restricted to ER+ tumors (where ESR1 signaling is active). PGR
(progesterone receptor, also a canonical ESR1 target) is run in parallel
as a second check.

INTERPRETATION
--------------
- If this comes out RR>1 (positive, as expected): the within-patient
  depth-offset GLM pipeline is NOT systematically flipping signs -- the
  LCOR-APM reversal is something specific to that biology, unexplained,
  but not a sign the whole pipeline (including the CXCL16 null) is
  untrustworthy.
- If this ALSO comes out inverted/null despite being about as textbook an
  ER+ relationship as exists: that is evidence of a systematic issue with
  this analysis design in this dataset, and every result from this
  pipeline (including the CXCL16 nulls) needs to be treated with much more
  caution.

METHOD
------
Identical model to 02_depth_artifact_apm_malignant_cell_level.py, restricted
to subtype=="ER" cells (ESR1 signaling not active/relevant in HER2/TNBC):
  GENE_Y_raw ~ log1p(ESR1_raw) + C(sample) + offset(log(total_counts))

Output: LCOR-CXCL16/07_apm_positive_control/figures/06_negative_control_esr1_greb1/
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
OUTDIR      = "LCOR-CXCL16/07_apm_positive_control/figures/06_negative_control_esr1_greb1"
GENE_X      = "ESR1"
TARGETS     = ["GREB1", "PGR"]
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(f" 06 -- METHOD DIAGNOSTIC -- {GENE_X} vs {TARGETS}, ER+ malignant cells, by patient")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
is_er = adata.obs["subtype"].astype(str).values == "ER"
print(f"  ER+ malignant cells: {is_er.sum():,} / {adata.n_obs:,}")

genes = [GENE_X] + TARGETS
idx = [adata.var_names.get_loc(g) for g in genes]
raw = adata.layers["counts"][:, idx]
raw = np.asarray(raw.todense()) if issparse(raw) else np.asarray(raw)

df_full = pd.DataFrame(raw, columns=[f"{g}_raw" for g in genes], index=adata.obs_names)
df_full["total_counts"] = adata.obs["total_counts"].values.astype(float)
df_full["sample"] = adata.obs["sample"].astype(str).values
df_full["subtype"] = adata.obs["subtype"].astype(str).values
del adata
df = df_full[is_er].copy()
df["log_total_counts"] = np.log(df["total_counts"])
df["log1p_ESR1"] = np.log1p(df[f"{GENE_X}_raw"])

readme = [
    f"06 -- METHOD DIAGNOSTIC -- {GENE_X} vs {'/'.join(TARGETS)}, ER+ malignant cells",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "The APM positive control (01-04) found LCOR-APM negatively associated",
    "within-patient -- opposite to the lab's own experiments. Before trusting",
    "or distrusting ANY within-patient cell-level result from this pipeline",
    "(CXCL16 null included), we test whether the design itself can recover a",
    "textbook, unambiguous, immune-unrelated relationship: ESR1 -> GREB1/PGR",
    "(canonical direct ESR1 targets), in ER+ malignant cells only.",
    "",
    f"n = {is_er.sum():,} ER+ malignant cells, {df['sample'].nunique()} patients",
]

results = []
for target in TARGETS:
    print(f"\n{'='*65}")
    print(f" {GENE_X} -> {target}")
    print("="*65)
    model_df = df.copy()

    poisson_pooled = smf.glm(
        f"{target}_raw ~ log1p_ESR1",
        data=model_df, family=sm.families.Poisson(),
        offset=model_df["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})
    disp = poisson_pooled.pearson_chi2 / poisson_pooled.df_resid
    family = sm.families.NegativeBinomial() if disp > 1.5 else sm.families.Poisson()
    model_name = "Negative Binomial" if disp > 1.5 else "Poisson"
    print(f"  Dispersion: {disp:.2f} -> {model_name}")

    pooled = smf.glm(
        f"{target}_raw ~ log1p_ESR1",
        data=model_df, family=family,
        offset=model_df["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})
    pooled_rr = np.exp(pooled.params["log1p_ESR1"])
    pooled_ci = np.exp(pooled.conf_int().loc["log1p_ESR1"].values)
    pooled_p = pooled.pvalues["log1p_ESR1"]
    print(f"  POOLED: RR={pooled_rr:.3f} [{pooled_ci[0]:.3f}-{pooled_ci[1]:.3f}], p={pooled_p:.2e}")

    within = smf.glm(
        f"{target}_raw ~ log1p_ESR1 + C(sample)",
        data=model_df, family=family,
        offset=model_df["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})
    within_rr = np.exp(within.params["log1p_ESR1"])
    within_ci = np.exp(within.conf_int().loc["log1p_ESR1"].values)
    within_p = within.pvalues["log1p_ESR1"]
    print(f"  WITHIN-PATIENT: RR={within_rr:.3f} [{within_ci[0]:.3f}-{within_ci[1]:.3f}], p={within_p:.2e}")

    results.append(dict(target=target, model=model_name,
                         pooled_rr=pooled_rr, pooled_ci_lo=pooled_ci[0], pooled_ci_hi=pooled_ci[1], pooled_p=pooled_p,
                         within_rr=within_rr, within_ci_lo=within_ci[0], within_ci_hi=within_ci[1], within_p=within_p))

    readme += [
        "",
        f"{GENE_X} -> {target} ({model_name} GLM, depth-offset, cluster-robust SE by sample)",
        "-"*70,
        f"  POOLED:          RR={pooled_rr:.3f} [{pooled_ci[0]:.3f}-{pooled_ci[1]:.3f}], p={pooled_p:.2e}",
        f"  WITHIN-PATIENT:  RR={within_rr:.3f} [{within_ci[0]:.3f}-{within_ci[1]:.3f}], p={within_p:.2e}",
    ]

res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUTDIR}/results.csv", index=False)

# ============================================================================
# FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.2))
labels, rrs, los, his = [], [], [], []
for r in results:
    labels.append(f"{GENE_X}->{r['target']}\n(within-patient)")
    rrs.append(r["within_rr"]); los.append(r["within_ci_lo"]); his.append(r["within_ci_hi"])
ys = list(range(len(results)))[::-1]
for y, rr, lo, hi in zip(ys, rrs, los, his):
    color = "#E41A1C" if lo > 1 else ("#377EB8" if hi < 1 else "gray")
    ax.plot([lo, hi], [y, y], color=color, lw=2)
    ax.scatter([rr], [y], color=color, s=70, zorder=3)
    ax.text(rr, y + 0.15, f"RR={rr:.2f} [{lo:.2f}-{hi:.2f}]", ha="center", fontsize=8)
ax.axvline(1, color="black", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Within-patient rate ratio (depth-offset GLM)", fontsize=8.5)
ax.set_ylim(-0.6, len(results) - 0.4)
ax.set_title(f"Method diagnostic: {GENE_X} -> canonical targets, ER+ malignant cells\n(expected: RR>1)",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_esr1_targets_within_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUTDIR}/fig_esr1_targets_within_patient.png")

all_positive_expected = all(r["within_ci_lo"] > 1 for r in results)
readme += [
    "",
    "INTERPRETATION",
    "-"*70,
    f"  Expected direction (canonical biology): RR>1 for both targets.",
    f"  Observed: {'CONFIRMS expected direction for both targets (RR>1, significant) -- the within-patient depth-offset GLM design is NOT systematically inverting known relationships. The LCOR-APM reversal (01-04) is therefore specific to that biology, not a generic pipeline artifact.' if all_positive_expected else 'Does NOT cleanly confirm the expected direction for at least one target -- raises doubt about whether this within-patient design reliably recovers known relationships in this dataset at all, which would also undercut confidence in the LCOR-CXCL16 null.'}",
    "",
    "OUTPUT FILES",
    "-"*70,
    "  results.csv, fig_esr1_targets_within_patient.png",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================""")
for r in results:
    print(f"  {GENE_X}->{r['target']}: within-patient RR={r['within_rr']:.3f} [{r['within_ci_lo']:.3f}-{r['within_ci_hi']:.3f}], p={r['within_p']:.2e}")
print(f"  Expected direction confirmed for both: {all_positive_expected}")
print(f"  Output: {OUTDIR}/")
print("============================================================")
