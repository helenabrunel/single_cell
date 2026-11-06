"""
08_lcor_esr1_relationship.py
==============================
Motivated by the lab's own paper (PMID 41289011, "Estrogen receptor
signaling drives immune evasion and immunotherapy resistance in HR+ breast
cancer"): ERalpha (ESR1) is reported to "hijack" LCOR, preventing LCOR's
normal function of inducing tumor immunogenicity (including antigen
presentation). This does not necessarily predict simple co-expression of
the two mRNAs, but it DOES predict that LCOR's positive effect on APM
(established in the lab's experiments, and replicated in the expected
direction in Klughammer, 07) should be BLUNTED OR REVERSED specifically
where ESR1/ER signaling is active -- i.e. in ER+ tumors/cells -- which
could explain why Visvader (composition/behavior unknown) shows the
reversed LCOR-APM relationship that Klughammer does not.

TWO CHECKS
----------
1. LCOR vs ESR1 direct relationship (within-patient depth-offset GLM,
   same design as throughout this project), in Visvader and in Klughammer,
   both pooled-with-subtype-covariate (all cells) and restricted to ER+
   cells only (continuous ESR1 variation within HR+ tumors).
2. THE KEY TEST: does the LCOR->APM relationship (01-04, 07) differ by
   subtype/ESR1 status? Refit the LCOR->APM within-patient GLM separately
   in ER+ cells vs non-ER+ (HER2+TNBC) cells, in both Visvader and
   Klughammer. If the "ERalpha hijacks LCOR" mechanism is at play, LCOR->APM
   should be more negative/blunted in ER+ cells than in non-ER+ cells.

Output: LCOR-CXCL16/07_apm_positive_control/figures/08_lcor_esr1_relationship/
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

OUTDIR = "LCOR-CXCL16/07_apm_positive_control/figures/08_lcor_esr1_relationship"
APM_GENES = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
os.makedirs(OUTDIR, exist_ok=True)

readme = [
    "08 -- LCOR x ESR1 relationship, and subtype-stratified LCOR->APM",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "="*70,
    "",
    "MOTIVATION",
    "-"*70,
    "Lab's paper (PMID 41289011): ERalpha 'hijacks' LCOR, blocking LCOR's",
    "normal induction of tumor immunogenicity/antigen presentation. Tests",
    "here: (1) direct LCOR-ESR1 relationship; (2) whether LCOR->APM is more",
    "negative/blunted specifically in ER+ (high-ESR1-signaling) cells,",
    "consistent with the hijacking mechanism, in both Visvader and the",
    "independent Klughammer atlas.",
]

def fit_within(d, x_raw, y_raw, label, subtype_covariate=False):
    d = d.copy()
    d["log1p_X"] = np.log1p(d[x_raw])
    formula_pooled = f"{y_raw} ~ log1p_X" + (" + C(subtype)" if subtype_covariate else "")
    poisson_pooled = smf.glm(
        formula_pooled, data=d, family=sm.families.Poisson(),
        offset=d["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["sample"]})
    disp = poisson_pooled.pearson_chi2 / poisson_pooled.df_resid
    family = sm.families.NegativeBinomial() if disp > 1.5 else sm.families.Poisson()
    model_name = "Negative Binomial" if disp > 1.5 else "Poisson"

    within = smf.glm(
        f"{y_raw} ~ log1p_X + C(sample)", data=d, family=family,
        offset=d["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["sample"]})
    rr = np.exp(within.params["log1p_X"])
    ci = np.exp(within.conf_int().loc["log1p_X"].values)
    p = within.pvalues["log1p_X"]
    print(f"  [{label}] n_patients={d['sample'].nunique()}, n_cells={len(d):,}, dispersion={disp:.2f} -> {model_name}")
    print(f"    WITHIN-PATIENT RR={rr:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p={p:.2e}")
    return dict(label=label, n_patients=d["sample"].nunique(), n_cells=len(d),
                model=model_name, rr=rr, ci_lo=ci[0], ci_hi=ci[1], p=p)


def load_dataset(name, h5ad_path, sample_col, ensembl_lookup=False):
    print(f"\n  Loading {name}: {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    genes = ["LCOR", "ESR1"] + APM_GENES
    if ensembl_lookup:
        sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
        ens_map = {g: (sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]) for g in genes}
        raw_var_index = pd.Index(adata.raw.var_names)
        idx_of = {g: raw_var_index.get_loc(ens_map[g]) for g in genes}
        def get_raw(g):
            col = adata.raw.X[:, idx_of[g]]
            return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
        total_counts = adata.obs["total_counts"].values.astype(float)
    else:
        idx = {g: adata.var_names.get_loc(g) for g in genes}
        def get_raw(g):
            col = adata.layers["counts"][:, idx[g]]
            return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
        total_counts = adata.obs["total_counts"].values.astype(float)

    raw_lcor = get_raw("LCOR")
    raw_esr1 = get_raw("ESR1")
    raw_apm = np.column_stack([get_raw(g) for g in APM_GENES]).sum(axis=1)

    df = pd.DataFrame({
        "LCOR_raw": raw_lcor.astype(int),
        "ESR1_raw": raw_esr1.astype(int),
        "APM_raw": raw_apm.astype(int),
        "total_counts": total_counts,
        "subtype": adata.obs["subtype"].astype(str).values,
        "sample": adata.obs[sample_col].astype(str).values,
    }, index=adata.obs_names)
    df["log_total_counts"] = np.log(df["total_counts"])
    del adata
    return df


all_results = []

for name, path, sample_col, ens_lookup in [
    ("Visvader", "compartment_epithelial.h5ad", "sample", False),
    ("Klughammer", "/home/hbrunel/single_cell/klughammer_malignant_processed.h5ad", "donor_id", True),
]:
    df = load_dataset(name, path, sample_col, ens_lookup)
    readme += ["", f"=== {name} ===", "-"*70]

    print(f"\n{'='*65}\n {name}: LCOR vs ESR1\n{'='*65}")
    r1 = fit_within(df, "ESR1_raw", "LCOR_raw", f"{name}: ESR1->LCOR (all cells, +subtype cov)", subtype_covariate=True)
    df_er = df[df["subtype"] == "ER"]
    r2 = fit_within(df_er, "ESR1_raw", "LCOR_raw", f"{name}: ESR1->LCOR (ER+ only)")
    readme += [
        f"  LCOR vs ESR1, all cells (+subtype covariate): RR={r1['rr']:.3f} [{r1['ci_lo']:.3f}-{r1['ci_hi']:.3f}], p={r1['p']:.2e}",
        f"  LCOR vs ESR1, ER+ only:                       RR={r2['rr']:.3f} [{r2['ci_lo']:.3f}-{r2['ci_hi']:.3f}], p={r2['p']:.2e}",
    ]
    all_results += [r1, r2]

    print(f"\n{'='*65}\n {name}: LCOR -> APM, ER+ vs non-ER+ (hijacking test)\n{'='*65}")
    df_nonER = df[df["subtype"] != "ER"]
    r3 = fit_within(df_er, "LCOR_raw", "APM_raw", f"{name}: LCOR->APM, ER+ cells")
    r4 = fit_within(df_nonER, "LCOR_raw", "APM_raw", f"{name}: LCOR->APM, non-ER+ (HER2+TNBC) cells")
    readme += [
        f"  LCOR->APM, ER+ cells:     RR={r3['rr']:.3f} [{r3['ci_lo']:.3f}-{r3['ci_hi']:.3f}], p={r3['p']:.2e}",
        f"  LCOR->APM, non-ER+ cells: RR={r4['rr']:.3f} [{r4['ci_lo']:.3f}-{r4['ci_hi']:.3f}], p={r4['p']:.2e}",
        f"  Hijacking prediction (LCOR->APM more negative in ER+ than non-ER+): "
        f"{'CONSISTENT' if r3['rr'] < r4['rr'] else 'NOT CONSISTENT'}",
    ]
    all_results += [r3, r4]

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/results.csv", index=False)

# ============================================================================
# FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 0.5 * len(all_results) + 1))
labels = [r["label"] for r in all_results]
rrs = [r["rr"] for r in all_results]
los = [r["ci_lo"] for r in all_results]
his = [r["ci_hi"] for r in all_results]
ys = list(range(len(all_results)))[::-1]
for y, rr, lo, hi in zip(ys, rrs, los, his):
    color = "#E41A1C" if lo > 1 else ("#377EB8" if hi < 1 else "gray")
    ax.plot([lo, hi], [y, y], color=color, lw=2)
    ax.scatter([rr], [y], color=color, s=60, zorder=3)
    ax.text(rr, y + 0.15, f"RR={rr:.2f}", ha="center", fontsize=7.5)
ax.axvline(1, color="black", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Within-patient rate ratio (depth-offset GLM)", fontsize=8.5)
ax.set_title("LCOR-ESR1 relationship, and LCOR->APM by ER status", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_lcor_esr1_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUTDIR}/fig_lcor_esr1_summary.png")

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")
print("\nDONE")
