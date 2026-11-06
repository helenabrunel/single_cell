"""
07j_TIM3_hypoxia_depth_checks.py
===================================
Recreates the original hexbin/scatter plots (HAVCR2 vs hypoxia score,
HAVCR2+ epithelial cells, by subtype) alongside a depth-corrected version,
to show directly why the original plots looked like two populations.

BACKGROUND
----------
The very first hexbins (figures/07_hypoxia_TIM3/hexbin_Epithelial_Global_
hypoxia_vs_TIM3_pos_only.png) showed a significant negative correlation
between normalized HAVCR2 and hypoxia_z within HAVCR2+ cells (r=-0.334,
p=5e-13, n=445). This looked like two populations (TIM3-high/hypoxia-low
vs TIM3-low/hypoxia-high) and became the basis for the G1/G2 design used
in 07d-07g.

We now know both plotted axes are independently confounded with per-cell
sequencing depth (total_counts):
  - Normalized HAVCR2 is inflated in shallow cells (single-UMI dilution
    effect, see figures/07f_depth_adjusted/).
  - hypoxia_z is itself positively correlated with total_counts (r=0.65).

A shallow cell therefore gets BOTH an inflated HAVCR2 value AND a
deflated hypoxia_z -- mechanically producing a negative correlation
between the two variables that has nothing to do with epithelial TIM3
biology. This script makes that visible: naive plot vs depth-corrected
plot, side by side, by subtype.

DEPTH CORRECTION
-----------------
  - Hypoxia axis: residuals of OLS(hypoxia_z ~ log(total_counts)),
    fit on all epithelial cells.
  - HAVCR2 axis: Pearson residuals from a Negative Binomial GLM of
    HAVCR2_raw ~ offset(log(total_counts)) (intercept-only -- population
    average rate curve), fit on all epithelial cells.
  Both models are fit on the full epithelial population so the
  depth-rate relationship is well estimated, then residuals are
  extracted for the HAVCR2+ cells being plotted.

Output: figures/TIM3_Hipoxia_depth_checks/
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
OUTDIR = "figures/TIM3_Hipoxia_depth_checks"
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
print(" 07j -- TIM3 x HYPOXIA -- NAIVE vs DEPTH-CORRECTED HEXBINS")
print("="*65)

readme = [
    "07j -- Naive vs depth-corrected TIM3(HAVCR2) x hypoxia relationship",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "WHY THIS EXISTS",
    "-"*60,
    "The original hexbins (figures/07_hypoxia_TIM3/) showed a significant",
    "negative correlation between normalized HAVCR2 and hypoxia_z within",
    "HAVCR2+ epithelial cells (r=-0.334, p=5e-13, n=445), which looked like",
    "two populations and motivated the G1/G2 design (07d-07g). Both plotted",
    "variables are independently confounded with per-cell sequencing depth",
    "(HAVCR2 via single-UMI dilution in normalization; hypoxia_z via its",
    "own residual depth-dependence, r=0.65 with total_counts). A shallow",
    "cell gets an inflated HAVCR2 value AND a deflated hypoxia_z at the",
    "same time -- mechanically producing a negative correlation between",
    "them that isn't about TIM3/hypoxia biology.",
    "",
    "DEPTH CORRECTION APPLIED",
    "-"*60,
    "  Hypoxia axis : residuals of OLS(hypoxia_z ~ log(total_counts))",
    "  HAVCR2 axis  : Pearson residuals, NB GLM HAVCR2_raw ~ offset(log(total_counts))",
    "  Both models fit on all epithelial cells; residuals shown for HAVCR2+ cells.",
]

# ── Load ─────────────────────────────────────────────────────────────────────
print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

print("\n  Scoring hypoxia (GO:0001666)...")
hyp_genes = [g for g in GO_0001666 if g in adata.var_names]
sc.tl.score_genes(adata, hyp_genes, score_name="score_hypoxia", use_raw=False)
raw_hyp = adata.obs["score_hypoxia"].values.astype(float)
hypoxia_z = (raw_hyp - raw_hyp.mean()) / (raw_hyp.std() + 1e-8)
print(f"  {len(hyp_genes)}/{len(GO_0001666)} genes found")

x = adata[:, "HAVCR2"].X
havcr_norm = np.asarray(x.todense()).flatten() if issparse(x) else np.asarray(x).flatten()
raw = adata.layers["counts"][:, adata.var_names.get_loc("HAVCR2")]
havcr_raw = np.asarray(raw.todense()).flatten() if issparse(raw) else np.asarray(raw).flatten()

total_counts = adata.obs["total_counts"].values.astype(float)
log_total_counts = np.log(total_counts)
subtype = adata.obs["subtype"].astype(str).values

df_all = pd.DataFrame({
    "hypoxia_z": hypoxia_z,
    "havcr_norm": havcr_norm,
    "havcr_raw": havcr_raw,
    "log_total_counts": log_total_counts,
    "subtype": subtype,
})

# ── Fit depth models on ALL epithelial cells ───────────────────────────────
print("\n  Fitting depth-correction models on all epithelial cells...")

hyp_model = smf.ols("hypoxia_z ~ log_total_counts", data=df_all).fit()
df_all["hypoxia_z_resid"] = hyp_model.resid
print(f"  Hypoxia depth model: R2={hyp_model.rsquared:.3f}, "
      f"coef(log_total_counts)={hyp_model.params['log_total_counts']:.4f}")

havcr_model = smf.glm("havcr_raw ~ 1", data=df_all,
                       family=sm.families.NegativeBinomial(),
                       offset=df_all["log_total_counts"]).fit()
pred = havcr_model.predict(df_all, offset=df_all["log_total_counts"])
alpha = havcr_model.family.alpha if hasattr(havcr_model.family, "alpha") else 1.0
var = pred + alpha * pred**2
df_all["havcr_pearson_resid"] = (df_all["havcr_raw"] - pred) / np.sqrt(var)
print(f"  HAVCR2 depth model (NB, offset=log total_counts): "
      f"intercept={havcr_model.params['Intercept']:.4f}")

readme += [
    "",
    "DEPTH MODEL FITS (all epithelial cells)",
    "-"*60,
    f"  hypoxia_z ~ log(total_counts): R2={hyp_model.rsquared:.4f}, "
    f"coef={hyp_model.params['log_total_counts']:.4f}, "
    f"p={hyp_model.pvalues['log_total_counts']:.2e}",
    f"  HAVCR2_raw ~ offset(log total_counts): intercept={havcr_model.params['Intercept']:.4f}",
]

pos = df_all["havcr_raw"] > 0
df_pos = df_all[pos].copy()
print(f"\n  HAVCR2+ cells: {len(df_pos)}")

# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════
def make_panel_plot(df, xcol, ycol, xlabel, ylabel, title, fname):
    groups = [("All HAVCR2+", df)]
    for st in ["ER", "HER2", "TNBC"]:
        groups.append((f"{st} HAVCR2+", df[df["subtype"] == st]))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    colors = {"All HAVCR2+": "#4C72B0", "ER HAVCR2+": "#2ca02c",
              "HER2 HAVCR2+": "#9467bd", "TNBC HAVCR2+": "#d62728"}

    stats_rows = []
    for ax, (label, d) in zip(axes, groups):
        if len(d) < 3:
            ax.set_title(f"{label} (n={len(d)})", fontsize=10)
            continue
        r, p = stats.pearsonr(d[xcol], d[ycol])
        stats_rows.append({"panel": label, "r": r, "p": p, "n": len(d)})
        ax.scatter(d[xcol], d[ycol], s=14, alpha=0.6, color=colors[label],
                   edgecolors="white", linewidths=0.3)
        ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
        z = np.polyfit(d[xcol], d[ycol], 1)
        xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="black", ls="--", lw=1.2, alpha=0.7)
        ax.set_title(f"{label} (n={len(d)})\nr={r:.3f}, p={p:.2e}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTDIR}/{fname}")
    return pd.DataFrame(stats_rows)

print("\n  Generating naive plot (as originally computed)...")
stats_naive = make_panel_plot(
    df_pos, "hypoxia_z", "havcr_norm",
    "Hypoxia score (z)", "HAVCR2 expression (normalized)",
    "NAIVE -- Epithelial_Global -- Hypoxia vs HAVCR2 (HAVCR2+ cells only)\n"
    "Not corrected for sequencing depth",
    "hexbin_naive_by_subtype.png"
)

print("\n  Generating depth-corrected plot...")
stats_corr = make_panel_plot(
    df_pos, "hypoxia_z_resid", "havcr_pearson_resid",
    "Hypoxia score residual\n(depth-adjusted)", "HAVCR2 Pearson residual\n(depth-adjusted)",
    "DEPTH-CORRECTED -- Epithelial_Global -- Hypoxia vs HAVCR2 (HAVCR2+ cells only)\n"
    "Both axes adjusted for per-cell sequencing depth",
    "hexbin_depth_corrected_by_subtype.png"
)

# ── Summary comparison table ────────────────────────────────────────────────
comparison = stats_naive.merge(stats_corr, on=["panel", "n"], suffixes=("_naive", "_corrected"))
comparison = comparison[["panel", "n", "r_naive", "p_naive", "r_corrected", "p_corrected"]]
comparison.to_csv(f"{OUTDIR}/correlation_comparison.csv", index=False)
print(f"\n  Saved: {OUTDIR}/correlation_comparison.csv")
print(comparison.to_string(index=False))

readme += [
    "",
    "CORRELATION COMPARISON -- naive vs depth-corrected",
    "-"*60,
]
for _, row in comparison.iterrows():
    readme.append(
        f"  {row['panel']:16s} n={int(row['n']):4d}  "
        f"naive r={row['r_naive']:+.3f} (p={row['p_naive']:.2e})  "
        f"->  corrected r={row['r_corrected']:+.3f} (p={row['p_corrected']:.2e})"
    )

readme += [
    "",
    "OUTPUT FILES",
    "-"*60,
    "  README.txt",
    "  hexbin_naive_by_subtype.png          -- original-style plot, not depth-corrected",
    "  hexbin_depth_corrected_by_subtype.png -- same cells, both axes depth-adjusted",
    "  correlation_comparison.csv           -- r/p side by side, naive vs corrected",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE -- 07j depth checks
============================================================
  Output: {OUTDIR}/
============================================================
""")
