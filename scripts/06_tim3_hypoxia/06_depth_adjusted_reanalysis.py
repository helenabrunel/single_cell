"""
07f_TIM3_hypoxia_depth_adjusted.py
===================================
Depth-adjusted re-analysis of the TIM3 (HAVCR2) x hypoxia relationship
in epithelial cells.

BACKGROUND / WHY THIS SCRIPT EXISTS
------------------------------------
07d/07e defined two cell populations by thresholding *normalized* HAVCR2
expression (G1: HAVCR2>1.5 & hypoxia<0  vs  G2: 0<HAVCR2<=1.5 & hypoxia>0)
and ran DEG (Wilcoxon) + GSEA between them.

Diagnostics revealed this comparison is confounded with sequencing depth,
not just underpowered:
  - G1 median total_counts = 1,108   G2 median total_counts = 10,006
  - G1/G2 total_counts ranges barely overlap (1/102 G1 cells fall in G2's
    depth range) -> depth and group membership are ~collinear ->
    unidentifiable with any covariate adjustment on that group definition.
  - 95.5% of HAVCR2+ cells have exactly 1 raw UMI. The ">1.5" normalized
    threshold is mostly just selecting shallow cells (a single UMI
    normalizes to a much larger value in a low-depth cell).
  - Redefining "TIM3-high" by raw UMI (>=2) leaves only 20 cells overall,
    5 with hypoxia<0 -- not enough for DEG/GSEA.

Also: cells with HAVCR2 raw count = 0 are NOT necessarily biological
negatives -- HAVCR2 is extremely lowly expressed (0.4% detection), so most
zeros are dropout (transcript present, not sampled), especially in
shallow cells. A binary "detected vs not" model still treats every zero
the same regardless of how much depth "had a chance" to catch the
transcript.

APPROACH
--------
Model raw HAVCR2 UMI counts directly with sequencing depth as a model
OFFSET (log(total_counts)), the same logic used for library-size
normalization in bulk RNA-seq (edgeR/DESeq2 offsets). This treats depth
as an exposure: a zero in a 50,000-count cell is much stronger evidence
of true low expression than a zero in a 500-count cell, and the
Poisson/NB likelihood reflects that automatically -- unlike a hard
detected/not-detected threshold.

  HAVCR2_raw ~ hypoxia_z + subtype + offset(log(total_counts))

Cluster-robust (sandwich) SEs by sample to account for repeated
measurements within patients.

Sensitivity checks:
  - Naive (unadjusted) comparison for reference
  - Restriction to higher-depth cells only (dropout-robustness check)
  - Negative Binomial vs Poisson (overdispersion check)

Output: figures/07f_depth_adjusted/
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

H5AD_FILE = "compartment_epithelial.h5ad"
OUTDIR    = "figures/07f_depth_adjusted"
os.makedirs(OUTDIR, exist_ok=True)

GO_0001666 = list(dict.fromkeys([
    "HIF1A","EPAS1","HIF3A","ARNT","ARNT2","VEGFA","VEGFB","VEGFC","VEGFD","PGF",
    "LDHA","LDHB","PFKL","PFKM","PFKFB3","PFKFB4","SLC2A1","SLC2A3","HK1","HK2","ENO1","ENO2",
    "PDK1","PDK3","BNIP3","BNIP3L","CA9","CA12","ANGPT1","ANGPT2","ANGPTL4","EPO","EPOR",
    "NOS2","NOS3","HMOX1","HMOX2","ADM","ADM2","SERPINE1","SERPINE2","FN1","COL1A1","COL4A1",
    "MMP2","MMP9","MMP14","CXCR4","CXCL12","CXCL8","TNF","IL1A","IL1B","IL6","PTGS2",
    "TF","TFRC","FTH1","FTL","EGLN1","EGLN2","EGLN3","VHL","P4HA1","P4HA2","PLOD1","PLOD2","DDIT4","DDIT3",
    "ATF4","ATF3","SLC7A11","GPI","ALDOA","TPI1","GAPDH","PGK1","PGAM1","PKM","BCL2","BCL2L1","BAX","BAK1",
    "NOTCH1","NOTCH2","DLL4","JAG1","KDR","FLT1","TWIST1","SNAI1","SNAI2","ZEB1","VIM","CDH2",
    "MET","HGF","TGFB1","TGFB2","TGFBR1","MYC","PTEN","PIK3CA","AKT1","MTOR","MAPK1","MAPK3","STAT3","JAK1","JAK2",
    "NFE2L2","SOD1","SOD2","CAT","GPX1","GPX4",
]))

print("="*65)
print(" 07f -- TIM3 x HYPOXIA -- DEPTH-OFFSET COUNT REGRESSION")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE, backed="r")
print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

hyp_genes = [g for g in GO_0001666 if g in adata.var_names]
genes_needed = list(dict.fromkeys(hyp_genes + ["HAVCR2"]))
print(f"  Hypoxia genes found: {len(hyp_genes)}/{len(GO_0001666)}")

sub = adata[:, genes_needed].to_memory()
lognorm = sub.layers["logcounts"]
lognorm = np.asarray(lognorm.todense()) if issparse(lognorm) else np.asarray(lognorm)
raw = sub.layers["counts"]
raw = np.asarray(raw.todense()) if issparse(raw) else np.asarray(raw)

expr = pd.DataFrame(lognorm, columns=genes_needed, index=sub.obs_names)
raw_df = pd.DataFrame(raw, columns=genes_needed, index=sub.obs_names)

raw_hyp = expr[hyp_genes].mean(axis=1).values
hyp_z = (raw_hyp - raw_hyp.mean()) / (raw_hyp.std() + 1e-8)

obs = adata.obs.copy()
df = pd.DataFrame({
    "HAVCR2_raw": raw_df["HAVCR2"].values,
    "hypoxia_z": hyp_z,
    "total_counts": obs["total_counts"].values.astype(float),
    "subtype": obs["subtype"].astype(str).values,
    "sample": obs["sample"].astype(str).values,
}, index=sub.obs_names)
df["log_total_counts"] = np.log(df["total_counts"])
df["HAVCR2_detected"] = (df["HAVCR2_raw"] > 0).astype(int)

del adata, sub
print(f"\n  Total epithelial cells: {len(df):,}")
print(f"  HAVCR2+ (raw>0): {df['HAVCR2_detected'].sum()} ({100*df['HAVCR2_detected'].mean():.2f}%)")

hyp_depth_r, hyp_depth_p = stats.spearmanr(df["total_counts"], df["hypoxia_z"])
print(f"\n  NOTE: hypoxia_z is itself correlated with total_counts "
      f"(Spearman r={hyp_depth_r:.3f}, p={hyp_depth_p:.2e})")
print(f"  The depth offset below adjusts HAVCR2 counts for depth, but does not")
print(f"  remove this hypoxia_z-depth relationship -- see README for interpretation.")

readme = [
    "07f -- TIM3 (HAVCR2) x Hypoxia -- depth-offset count regression",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "WHY THIS REPLACES THE 07d/07e G1-vs-G2 DEG/GSEA DESIGN",
    "-"*60,
    "G1 (TIM3high/Hyp-low, HAVCR2>1.5 & hyp<0) vs G2 (TIM3low/Hyp-high,",
    "0<HAVCR2<=1.5 & hyp>0) is confounded with sequencing depth almost",
    "to the point of collinearity:",
    "  G1 median total_counts = 1,108   G2 median total_counts = 10,006",
    "  Only 1/102 G1 cells overlaps G2's depth range at all.",
    "95.5% of HAVCR2+ cells have exactly 1 raw UMI -- the >1.5 normalized",
    "threshold is mostly selecting shallow cells (a single UMI normalizes",
    "to a much larger value when total_counts is small), not real",
    "differences in TIM3 transcript abundance.",
    "",
    "Redefining 'TIM3-high' by raw UMI (>=2) leaves only 20 cells total",
    "(5 with hypoxia<0) -- not enough statistical power for DEG/GSEA.",
    "",
    "Cells with HAVCR2 raw=0 are not necessarily biological negatives --",
    "HAVCR2 detection is only ~0.4%, so most zeros are dropout,",
    "especially in shallow cells. A raw-count model with sequencing depth",
    "as an OFFSET (not a covariate on a binarized outcome) is the",
    "statistically appropriate way to handle this: it treats a zero in a",
    "deep cell as stronger evidence of true absence than a zero in a",
    "shallow cell, rather than discarding that distinction.",
    "",
    "MODEL",
    "-"*60,
    "HAVCR2_raw ~ hypoxia_z + subtype + offset(log(total_counts))",
    "Poisson / Negative Binomial GLM, cluster-robust SE by sample.",
    "",
    "IMPORTANT CAVEAT: hypoxia_z is itself correlated with total_counts",
    f"(Spearman r={hyp_depth_r:.3f}, p={hyp_depth_p:.2e}). The depth offset",
    "below corrects HAVCR2 counts for depth, but does NOT remove this",
    "hypoxia_z-depth relationship. This means depth is a potential",
    "confound on both sides of the test (it affects both how reliably",
    "HAVCR2 is detected AND the hypoxia score itself), so the",
    "depth-adjusted coefficient should still be read as 'associated with',",
    "not proven causal. Worth double-checking whether score_genes() has a",
    "residual depth-dependence upstream (e.g. via its control-gene",
    "background) before treating this as final.",
]

# ============================================================================
# Naive (unadjusted) reference: original Mann-Whitney framing
# ============================================================================
print("\n  Naive (unadjusted) comparison, for reference...")
pos = df["HAVCR2_detected"] == 1
u, p_naive = stats.mannwhitneyu(df.loc[pos, "hypoxia_z"], df.loc[~pos, "hypoxia_z"],
                                 alternative="two-sided")
n1, n2 = pos.sum(), (~pos).sum()
rbc_naive = (2 * u) / (n1 * n2) - 1  # >0 means HAVCR2+ has higher hypoxia_z
print(f"  Mann-Whitney hypoxia_z: HAVCR2+ (n={n1}) vs HAVCR2- (n={n2})")
print(f"  rank-biserial r = {rbc_naive:.4f}   p = {p_naive:.3e}")

readme += [
    "",
    "NAIVE (UNADJUSTED) REFERENCE",
    "-"*60,
    f"Mann-Whitney hypoxia_z: HAVCR2+ (n={n1}) vs HAVCR2- (n={n2})",
    f"  rank-biserial r = {rbc_naive:.4f}   p = {p_naive:.3e}",
    "(This is the comparison originally reported -- reproduced here so",
    " the depth-adjusted result below can be directly compared to it.)",
]

# ============================================================================
# Depth-offset Poisson / NB regression
# ============================================================================
print("\n  Fitting depth-offset Poisson GLM...")
model_df = df.copy()
model_df["subtype"] = pd.Categorical(model_df["subtype"], categories=["ER","HER2","TNBC"])

poisson_model = smf.glm(
    "HAVCR2_raw ~ hypoxia_z + C(subtype)",
    data=model_df,
    family=sm.families.Poisson(),
    offset=model_df["log_total_counts"],
).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})

print(poisson_model.summary())

# Overdispersion check
pearson_chi2 = poisson_model.pearson_chi2
dof = poisson_model.df_resid
disp = pearson_chi2 / dof
print(f"\n  Poisson dispersion (Pearson chi2/df): {disp:.3f}  "
      f"({'overdispersed -> fit NB' if disp > 1.5 else 'roughly OK for Poisson'})")

nb_model = None
if disp > 1.5:
    print("  Fitting Negative Binomial GLM (overdispersion detected)...")
    nb_model = smf.glm(
        "HAVCR2_raw ~ hypoxia_z + C(subtype)",
        data=model_df,
        family=sm.families.NegativeBinomial(),
        offset=model_df["log_total_counts"],
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["sample"]})
    print(nb_model.summary())

main_model = nb_model if nb_model is not None else poisson_model
model_name = "Negative Binomial" if nb_model is not None else "Poisson"

hyp_coef = main_model.params["hypoxia_z"]
hyp_se   = main_model.bse["hypoxia_z"]
hyp_p    = main_model.pvalues["hypoxia_z"]
hyp_rr   = np.exp(hyp_coef)
hyp_ci   = np.exp(main_model.conf_int().loc["hypoxia_z"].values)

print(f"\n  DEPTH-ADJUSTED RESULT ({model_name} GLM, offset=log(total_counts)):")
print(f"  hypoxia_z coefficient: {hyp_coef:.4f} (SE={hyp_se:.4f}), p={hyp_p:.3e}")
print(f"  Rate ratio per 1-SD hypoxia_z: {hyp_rr:.3f}  (95% CI {hyp_ci[0]:.3f}-{hyp_ci[1]:.3f})")

readme += [
    "",
    f"DEPTH-ADJUSTED RESULT ({model_name} GLM, offset=log(total_counts),",
    "cluster-robust SE by sample)",
    "-"*60,
    f"  hypoxia_z coefficient: {hyp_coef:.4f} (SE={hyp_se:.4f}), p={hyp_p:.3e}",
    f"  Rate ratio per 1-SD hypoxia_z: {hyp_rr:.3f}  "
    f"(95% CI {hyp_ci[0]:.3f}-{hyp_ci[1]:.3f})",
    f"  Poisson dispersion check: {disp:.3f} "
    f"({'-> used NB due to overdispersion' if nb_model is not None else '-> Poisson adequate'})",
    "",
    "Interpretation: rate ratio > 1 means HAVCR2 expression (per unit",
    "sequencing depth) increases with hypoxia score, i.e. supports the",
    "original hypothesis even after removing the depth confound.",
    "Rate ratio ~ 1 or CI crossing 1 means the naive result above was",
    "likely driven by the depth artifact, not real TIM3-hypoxia biology.",
]

# ============================================================================
# Sensitivity: restrict to higher-depth cells (dropout-robustness check)
# ============================================================================
print("\n  Sensitivity check: restricting to cells with total_counts >= median...")
med_depth = df["total_counts"].median()
deep_df = model_df[model_df["total_counts"] >= med_depth].copy()
print(f"  Cells retained: {len(deep_df):,} / {len(df):,}  "
      f"(HAVCR2+: {(deep_df['HAVCR2_raw']>0).sum()})")

deep_model = smf.glm(
    "HAVCR2_raw ~ hypoxia_z + C(subtype)",
    data=deep_df,
    family=sm.families.Poisson() if nb_model is None else sm.families.NegativeBinomial(),
    offset=deep_df["log_total_counts"],
).fit(cov_type="cluster", cov_kwds={"groups": deep_df["sample"]})

deep_coef = deep_model.params["hypoxia_z"]
deep_p    = deep_model.pvalues["hypoxia_z"]
deep_rr   = np.exp(deep_coef)
deep_ci   = np.exp(deep_model.conf_int().loc["hypoxia_z"].values)
print(f"  High-depth-only rate ratio: {deep_rr:.3f} (95% CI {deep_ci[0]:.3f}-{deep_ci[1]:.3f}), p={deep_p:.3e}")

readme += [
    "",
    "SENSITIVITY CHECK -- high-depth cells only (total_counts >= median)",
    "-"*60,
    f"  Cells retained: {len(deep_df):,} / {len(df):,}",
    f"  Rate ratio: {deep_rr:.3f} (95% CI {deep_ci[0]:.3f}-{deep_ci[1]:.3f}), p={deep_p:.3e}",
    "(If this agrees in direction/magnitude with the full-data result",
    " above, the depth-offset model is not being driven by residual",
    " dropout noise in the shallowest cells.)",
]

# ============================================================================
# FIGURE 1 -- depth confound documentation (why we moved away from G1/G2)
# ============================================================================
print("\n  Generating figures...")

havcr = expr["HAVCR2"].values
g1 = (df["HAVCR2_raw"].values > 0) & (havcr > 1.5) & (hyp_z < 0)
g2 = (df["HAVCR2_raw"].values > 0) & (havcr > 0) & (havcr <= 1.5) & (hyp_z > 0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Why the original G1-vs-G2 design was replaced: depth confound",
             fontsize=11, fontweight="bold")

ax = axes[0]
ax.hist(np.log10(df.loc[g1, "total_counts"]), bins=25, alpha=0.65,
        color="#E41A1C", label=f"G1 TIM3high/Hyp-low (n={g1.sum()})")
ax.hist(np.log10(df.loc[g2, "total_counts"]), bins=25, alpha=0.65,
        color="#377EB8", label=f"G2 TIM3low/Hyp-high (n={g2.sum()})")
ax.set_xlabel("log10(total_counts)", fontsize=9)
ax.set_ylabel("Number of cells", fontsize=9)
ax.set_title("Original threshold-based groups:\ndepth ranges barely overlap", fontsize=9.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

ax = axes[1]
one_umi = df["HAVCR2_raw"] == 1
ax.scatter(df.loc[one_umi, "total_counts"], expr.loc[one_umi, "HAVCR2"],
           s=6, alpha=0.4, c="#4C72B0")
ax.axhline(1.5, color="black", ls="--", lw=1, alpha=0.6, label="TIM3-high threshold (1.5)")
ax.set_xscale("log")
ax.set_xlabel("total_counts (log scale)", fontsize=9)
ax.set_ylabel("Normalized HAVCR2 (logcounts)", fontsize=9)
ax.set_title("Cells with the SAME 1 raw UMI:\nnormalized value is ~entirely depth-driven",
              fontsize=9.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_depth_confound_diagnosis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig1_depth_confound_diagnosis.png")

# ============================================================================
# FIGURE 2 -- naive vs depth-adjusted effect size comparison (forest plot)
# ============================================================================
naive_rr_approx = np.exp(np.arctanh(np.clip(rbc_naive, -0.999, 0.999)))  # not a real RR; use r for its own axis instead
# Better: show rank-biserial r (naive) and rate ratio (adjusted) on separate normalized panels

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("HAVCR2 ~ hypoxia_z: naive vs depth-adjusted", fontsize=11, fontweight="bold")

ax = axes[0]
ax.barh([0], [rbc_naive], color="#E41A1C" if rbc_naive > 0 else "#377EB8", height=0.5)
ax.axvline(0, color="black", lw=1)
ax.set_yticks([0])
ax.set_yticklabels([f"Naive Mann-Whitney\nHAVCR2+ (n={n1}) vs HAVCR2- (n={n2})"], fontsize=8)
ax.set_xlabel("Rank-biserial r\n(hypoxia_z, HAVCR2+ vs HAVCR2-)", fontsize=8.5)
ax.text(rbc_naive, 0.32, f"r={rbc_naive:.3f}\np={p_naive:.1e}", ha="center", fontsize=8)
ax.set_xlim(-1, 1)
ax.grid(axis="x", alpha=0.25)

ax = axes[1]
labels = [f"Full data\n(n={len(df):,})", f"High-depth only\n(n={len(deep_df):,})"]
rrs  = [hyp_rr, deep_rr]
los  = [hyp_ci[0], deep_ci[0]]
his  = [hyp_ci[1], deep_ci[1]]
ys = [1, 0]
for y, rr, lo, hi in zip(ys, rrs, los, his):
    color = "#E41A1C" if lo > 1 else ("#377EB8" if hi < 1 else "gray")
    ax.plot([lo, hi], [y, y], color=color, lw=2)
    ax.scatter([rr], [y], color=color, s=60, zorder=3)
ax.axvline(1, color="black", ls="--", lw=1)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(f"Rate ratio per 1-SD hypoxia_z\n({model_name} GLM, depth-offset)",
              fontsize=8.5)
ax.set_ylim(-0.9, 2.05)
ax.grid(axis="x", alpha=0.25)
for y, rr, lo, hi, p_ in zip(ys, rrs, los, his, [hyp_p, deep_p]):
    ax.text(rr, y + 0.28, f"RR={rr:.2f} [{lo:.2f}-{hi:.2f}]\np={p_:.1e}",
            ha="center", fontsize=7.5, zorder=5)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color="#E41A1C", lw=2, label="CI entirely >1 (higher HAVCR2 rate with hypoxia)"),
    Line2D([0],[0], color="#377EB8", lw=2, label="CI entirely <1 (lower HAVCR2 rate with hypoxia)"),
    Line2D([0],[0], color="gray", lw=2, label="CI crosses 1 (not significant)"),
]
ax.legend(handles=legend_elements, fontsize=6.5, loc="upper center",
          bbox_to_anchor=(0.5, 1.28), framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_naive_vs_adjusted.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig2_naive_vs_adjusted.png")

# ============================================================================
# FIGURE 3 -- predicted HAVCR2 rate vs hypoxia_z at fixed depth
# ============================================================================
depth_ref = df["total_counts"].median()
hyp_range = np.linspace(df["hypoxia_z"].quantile(0.02), df["hypoxia_z"].quantile(0.98), 100)
pred_rows = []
for st in ["ER", "HER2", "TNBC"]:
    pred_df = pd.DataFrame({
        "hypoxia_z": hyp_range,
        "subtype": pd.Categorical([st]*100, categories=["ER","HER2","TNBC"]),
        "log_total_counts": np.log(depth_ref),
    })
    pred = main_model.get_prediction(pred_df, offset=pred_df["log_total_counts"])
    summ = pred.summary_frame(alpha=0.05)
    pred_rows.append(pd.DataFrame({
        "hypoxia_z": hyp_range, "subtype": st,
        "mean": summ["mean"].values,
        "lo": summ["mean_ci_lower"].values,
        "hi": summ["mean_ci_upper"].values,
    }))
pred_all = pd.concat(pred_rows)

fig, ax = plt.subplots(figsize=(7, 5))
colors = {"ER": "#4C72B0", "HER2": "#DD8452", "TNBC": "#55A868"}
for st in ["ER", "HER2", "TNBC"]:
    d = pred_all[pred_all["subtype"] == st]
    ax.plot(d["hypoxia_z"], d["mean"], color=colors[st], label=st, lw=2)
    ax.fill_between(d["hypoxia_z"], d["lo"], d["hi"], color=colors[st], alpha=0.15)
ax.set_xlabel("Hypoxia score (z-normalised, GO:0001666)", fontsize=10)
ax.set_ylabel(f"Predicted HAVCR2 UMI count\n(at median depth = {depth_ref:,.0f} total counts)", fontsize=10)
ax.set_title(f"Depth-adjusted HAVCR2~hypoxia relationship ({model_name} GLM)\n"
             f"hypoxia_z rate ratio = {hyp_rr:.2f} [{hyp_ci[0]:.2f}-{hyp_ci[1]:.2f}], p={hyp_p:.1e}",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, title="Subtype")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig3_predicted_rate_by_hypoxia.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig3_predicted_rate_by_hypoxia.png")

# ============================================================================
# Save tables + README
# ============================================================================
df.to_csv(f"{OUTDIR}/cell_level_data.csv")
print(f"  Saved: {OUTDIR}/cell_level_data.csv")

with open(f"{OUTDIR}/model_summary_{model_name.replace(' ','')}.txt", "w") as f:
    f.write(main_model.summary().as_text())
print(f"  Saved: {OUTDIR}/model_summary_{model_name.replace(' ','')}.txt")

readme += [
    "",
    "OUTPUT FILES",
    "-"*60,
    "  README.txt",
    "  cell_level_data.csv               -- per-cell HAVCR2, hypoxia_z, depth, subtype",
    f"  model_summary_{model_name.replace(' ','')}.txt      -- full GLM output",
    "  fig1_depth_confound_diagnosis.png -- why G1/G2 (07d/07e) was abandoned",
    "  fig2_naive_vs_adjusted.png        -- naive vs depth-adjusted effect size",
    "  fig3_predicted_rate_by_hypoxia.png-- depth-adjusted dose-response curve",
    "",
    "NOTE ON SCOPE",
    "-"*60,
    "This model answers: 'does HAVCR2 expression, properly accounting for",
    "sequencing depth, associate with hypoxia score?' It does NOT replace",
    "the pathway-level DEG/GSEA from 07d/07e -- that requires two",
    "depth-comparable cell populations, and none could be constructed from",
    "this gene given its extreme sparsity (95.5% of positive cells = 1 UMI).",
    "A pathway-level answer would need either much deeper sequencing of",
    "HAVCR2+ cells, or an orthogonal depth-independent readout (e.g. TIM3",
    "protein/IHC, CITE-seq, or pseudobulk on sorted populations).",
]

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Naive (unadjusted):     r={rbc_naive:.3f}, p={p_naive:.2e}
  Depth-adjusted ({model_name}): RR={hyp_rr:.2f} [{hyp_ci[0]:.2f}-{hyp_ci[1]:.2f}], p={hyp_p:.2e}
  High-depth sensitivity: RR={deep_rr:.2f} [{deep_ci[0]:.2f}-{deep_ci[1]:.2f}], p={deep_p:.2e}
  Output: {OUTDIR}/
============================================================
""")
