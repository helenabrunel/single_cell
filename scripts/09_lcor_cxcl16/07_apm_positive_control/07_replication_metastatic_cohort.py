"""
07_replication_klughammer.py
==============================
Is the LCOR-APM sign reversal (01-04) specific to the Visvader/Pal et al.
atlas, or does it replicate in an INDEPENDENT breast cancer scRNA-seq
dataset? 06 already showed the within-patient depth-offset GLM design
itself is not a generic sign-flipper (ESR1->GREB1/PGR recovers the correct
direction cleanly in Visvader) -- this script instead asks whether the
LCOR-APM finding itself is a property of this one atlas.

Independent dataset: Klughammer et al. (HTAPP breast cancer scRNA-seq,
malignant cells only, already available in this environment as
klughammer_malignant_processed.h5ad), 19 donors, ER/HER2/TNBC.

Runs BOTH checks from this project in this second dataset:
  1. LCOR -> APM meta-gene (TAP1, TAP2, TAPBP, HLA-A/B/C, PSMB8, PSMB9, B2M),
     within-patient depth-offset GLM, all malignant cells.
  2. ESR1 -> GREB1/PGR (method-sanity check), ER+ cells only.

Output: LCOR-CXCL16/07_apm_positive_control/figures/07_replication_klughammer/
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

H5AD_FILE = "/home/hbrunel/single_cell/klughammer_malignant_processed.h5ad"
OUTDIR    = "LCOR-CXCL16/07_apm_positive_control/figures/07_replication_klughammer"
APM_GENES = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(" 07 -- REPLICATION in Klughammer (independent atlas): LCOR->APM, ESR1->targets")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}   Donors: {adata.obs['donor_id'].nunique()}")

# Map gene symbol -> Ensembl ID via raw.var['feature_name']
sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
needed = ["LCOR", "ESR1", "GREB1", "PGR"] + APM_GENES
missing = [g for g in needed if g not in sym2ens.index]
if missing:
    raise SystemExit(f"  Gene(s) not found in feature_name: {missing}")
ens_map = {g: sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0] for g in needed}

raw_var_index = pd.Index(adata.raw.var_names)
idx_of = {g: raw_var_index.get_loc(ens_map[g]) for g in needed}

def get_raw(gene):
    col = adata.raw.X[:, idx_of[gene]]
    col = np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
    return col

raw_lcor = get_raw("LCOR")
raw_apm_matrix = np.column_stack([get_raw(g) for g in APM_GENES])
raw_apm = raw_apm_matrix.sum(axis=1)
raw_esr1 = get_raw("ESR1")
raw_greb1 = get_raw("GREB1")
raw_pgr = get_raw("PGR")

# sanity check: raw counts should be non-negative integers
assert np.allclose(raw_lcor, np.round(raw_lcor)), "raw.X does not look like integer counts"

df = pd.DataFrame({
    "LCOR_raw": raw_lcor.astype(int),
    "APM_raw": raw_apm.astype(int),
    "ESR1_raw": raw_esr1.astype(int),
    "GREB1_raw": raw_greb1.astype(int),
    "PGR_raw": raw_pgr.astype(int),
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "subtype": adata.obs["subtype"].astype(str).values,
    "sample": adata.obs["donor_id"].astype(str).values,
}, index=adata.obs_names)
df["log_total_counts"] = np.log(df["total_counts"])
del adata

print(f"  Subtype counts: {df['subtype'].value_counts().to_dict()}")

readme = [
    "07 -- REPLICATION in Klughammer (independent breast cancer scRNA-seq atlas)",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "Tests whether the LCOR-APM sign reversal found in Visvader (01-04) is",
    "specific to that one atlas or replicates independently. Uses the",
    "Klughammer/HTAPP malignant-cell atlas (19 donors) already available in",
    "this environment, and reruns both this project's positive control",
    "(LCOR->APM) and its method-sanity check (ESR1->GREB1/PGR) here.",
    "",
    f"n = {len(df):,} malignant cells, {df['sample'].nunique()} donors",
]

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
    print(f"\n  [{label}] n_patients={d['sample'].nunique()}, n_cells={len(d):,}, dispersion={disp:.2f} -> {model_name}")
    print(f"    WITHIN-PATIENT RR={rr:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p={p:.2e}")
    return dict(label=label, n_patients=d["sample"].nunique(), n_cells=len(d),
                model=model_name, rr=rr, ci_lo=ci[0], ci_hi=ci[1], p=p)

print("\n" + "="*65)
print(" Check 1: LCOR -> APM (replication of this project's positive control)")
print("="*65)
r_apm = fit_glm(df, "LCOR_raw", "APM_raw", "LCOR->APM (all malignant cells)")

print("\n" + "="*65)
print(" Check 2: ESR1 -> GREB1/PGR (replication of the method-sanity check)")
print("="*65)
df_er = df[df["subtype"] == "ER"]
r_greb1 = fit_glm(df_er, "ESR1_raw", "GREB1_raw", "ESR1->GREB1 (ER+ cells)")
r_pgr   = fit_glm(df_er, "ESR1_raw", "PGR_raw", "ESR1->PGR (ER+ cells)")

results = [r_apm, r_greb1, r_pgr]
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUTDIR}/results.csv", index=False)

apm_replicates_negative = r_apm["ci_hi"] < 1
esr1_replicates_positive = (r_greb1["ci_lo"] > 1) and (r_pgr["ci_lo"] > 1)

readme += [
    "",
    "RESULTS (within-patient depth-offset GLM, cluster-robust SE by donor)",
    "-"*70,
] + [
    f"  {r['label']}: n_patients={r['n_patients']}, n_cells={r['n_cells']:,}, "
    f"RR={r['rr']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}], p={r['p']:.2e} ({r['model']})"
    for r in results
] + [
    "",
    "INTERPRETATION",
    "-"*70,
    f"  LCOR->APM negative reversal replicates in Klughammer: "
    f"{'YES -- RR<1 confirmed independently, same direction as Visvader (biological, not a one-dataset artifact)' if apm_replicates_negative else 'NO -- does not replicate here, consistent with a Visvader-specific artifact rather than general biology'}",
    f"  ESR1->GREB1/PGR positive control replicates in Klughammer: "
    f"{'YES -- method sanity check holds up independently too' if esr1_replicates_positive else 'NO -- unexpected, would need further investigation of this dataset'}",
]

# ============================================================================
# FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
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
ax.set_xlabel("Within-patient rate ratio (depth-offset GLM)", fontsize=8.5)
ax.set_ylim(-0.6, len(results) - 0.4)
ax.set_title("Klughammer (independent atlas) replication check", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_replication.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUTDIR}/fig_replication.png")

readme += [
    "",
    "OUTPUT FILES",
    "-"*70,
    "  results.csv, fig_replication.png",
]
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
