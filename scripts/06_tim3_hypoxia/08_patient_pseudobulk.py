"""
07h_pseudobulk_TIM3_hypoxia.py
================================
Patient-level (pseudobulk) re-analysis of the TIM3 (HAVCR2) x hypoxia
relationship in epithelial cells.

WHY: 07d/07e/07g compared two THRESHOLD-DEFINED cell populations and found
that split is confounded with per-cell sequencing depth almost to the
point of collinearity (see figures/07f_depth_adjusted/). 07f fixed this
for a single gene (HAVCR2) using a depth-offset regression at the cell
level, and the signal reappeared in high-depth cells.

This script sidesteps the per-cell depth problem entirely by aggregating
to the PATIENT level (pseudobulk): summing raw counts across all
epithelial cells within each sample. Sample-level library sizes differ
by orders of magnitude less than single-cell library sizes, and are
handled with a standard size-factor offset (the same logic bulk RNA-seq
tools use) -- this is not the same normalization that caused the
per-cell artifact.

Two analyses:
  1. PRIMARY: does a sample's TIM3 burden (fraction of epithelial cells
     HAVCR2+, and pseudobulk HAVCR2 CPM) correlate with that sample's
     overall hypoxia signature score, across ~34 patients? This mirrors
     the original hypothesis at the level of the patient rather than
     the cell.
  2. SECONDARY (bonus): genome-wide pseudobulk association -- for every
     gene, regress its pseudobulk count against each sample's HAVCR2+
     fraction (continuous), with library size as an offset (Negative
     Binomial GLM). Produces a real gene-level result + GSEA, at
     patient-level resolution.

Output: figures/07h_pseudobulk/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, warnings, datetime, gc
from scipy.sparse import issparse, csr_matrix
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "compartment_epithelial.h5ad"
OUTDIR    = "figures/07h_pseudobulk"
GMT_DIR   = "gmt_files"
os.makedirs(OUTDIR, exist_ok=True)

GSEA_MIN_SIZE  = 15
GSEA_MAX_SIZE  = 500
GSEA_PERMUTATIONS = 1000
FDR_THRESHOLD  = 0.25
TOP_N          = 20

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
print(" 07h -- TIM3 x HYPOXIA -- PSEUDOBULK (PATIENT-LEVEL) ANALYSIS")
print("="*65)

readme = [
    "07h -- TIM3 (HAVCR2) x Hypoxia -- pseudobulk (patient-level) analysis",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "WHY THIS ANALYSIS",
    "-"*60,
    "07d/07e/07g compared two threshold-defined single-cell populations and",
    "found the split confounded with per-cell sequencing depth (see",
    "figures/07f_depth_adjusted/). This script aggregates to the PATIENT",
    "level instead: pseudobulk = sum of raw counts across all epithelial",
    "cells in a sample. Sample-level library sizes vary far less than",
    "single-cell library sizes and are handled with a standard size-factor",
    "offset -- this is the same normalization logic bulk RNA-seq tools use,",
    "not the per-cell normalization that caused the earlier artifact.",
]

# ── Load ─────────────────────────────────────────────────────────────────────
print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

counts = adata.layers["counts"]
if not issparse(counts):
    counts = csr_matrix(counts)

samples_arr = adata.obs["sample"].astype(str).values
subtype_arr = adata.obs["subtype"].astype(str).values
sample_list = sorted(pd.unique(samples_arr))
sample_idx  = {s: i for i, s in enumerate(sample_list)}
n_samples   = len(sample_list)
print(f"  Samples: {n_samples}")

# ── Build pseudobulk matrix (samples x genes) via indicator matmul ────────────
print("\n  Aggregating pseudobulk counts per sample...")
row_idx = np.arange(adata.n_obs)
col_idx = np.array([sample_idx[s] for s in samples_arr])
indicator = csr_matrix((np.ones(adata.n_obs), (row_idx, col_idx)),
                        shape=(adata.n_obs, n_samples))
pseudobulk = np.asarray((indicator.T @ counts).todense())  # n_samples x n_genes
print(f"  Pseudobulk matrix: {pseudobulk.shape}")

gene_names = adata.var_names.tolist()
havcr_idx  = gene_names.index("HAVCR2")

n_cells_per_sample = np.asarray(indicator.sum(axis=0)).flatten()
library_size = pseudobulk.sum(axis=1)

# HAVCR2 detected (raw>0) fraction per sample
havcr_raw_cell = np.asarray(counts[:, havcr_idx].todense()).flatten()
havcr_pos_cell = (havcr_raw_cell > 0).astype(float)
n_havcr_pos_per_sample = np.asarray(indicator.T @ havcr_pos_cell).flatten()
frac_havcr_pos = n_havcr_pos_per_sample / n_cells_per_sample

havcr_cpm = pseudobulk[:, havcr_idx] / library_size * 1e6

# Sample-level subtype (constant within sample)
subtype_per_sample = pd.Series(subtype_arr, index=samples_arr).groupby(level=0).first().reindex(sample_list).values

# Pseudobulk hypoxia score: mean log1p(CPM) across the GO:0001666 panel
hyp_genes = [g for g in GO_0001666 if g in gene_names]
hyp_idx = [gene_names.index(g) for g in hyp_genes]
cpm_matrix = pseudobulk / library_size[:, None] * 1e6
hypoxia_pseudobulk = np.log1p(cpm_matrix[:, hyp_idx]).mean(axis=1)

meta = pd.DataFrame({
    "sample": sample_list,
    "subtype": subtype_per_sample,
    "n_cells": n_cells_per_sample,
    "library_size": library_size,
    "HAVCR2_CPM": havcr_cpm,
    "frac_HAVCR2_pos": frac_havcr_pos,
    "hypoxia_pseudobulk": hypoxia_pseudobulk,
})
meta.to_csv(f"{OUTDIR}/sample_metadata.csv", index=False)
print(f"  Saved: {OUTDIR}/sample_metadata.csv")
print(f"\n  {hyp_genes.__len__()}/{len(GO_0001666)} hypoxia genes found")
print(f"  Samples with >=1 HAVCR2+ cell: {(meta['frac_HAVCR2_pos']>0).sum()}/{n_samples}")

readme += [
    "",
    "SAMPLE SUMMARY",
    "-"*60,
    f"  Samples: {n_samples}",
    f"  Samples with >=1 HAVCR2+ epithelial cell: {(meta['frac_HAVCR2_pos']>0).sum()}",
    f"  Median epithelial cells/sample: {np.median(n_cells_per_sample):.0f}",
    f"  Median library size/sample: {np.median(library_size):,.0f}",
]

# ══════════════════════════════════════════════════════════════════════════
# PRIMARY: sample-level correlation, TIM3 burden vs hypoxia score
# ══════════════════════════════════════════════════════════════════════════
print("\n  Primary analysis: TIM3 burden vs hypoxia score across patients...")

use = meta[meta["frac_HAVCR2_pos"] > 0].copy()  # only samples with detectable HAVCR2
print(f"  Using {len(use)} samples with detectable HAVCR2")

rho_frac, p_frac = stats.spearmanr(use["frac_HAVCR2_pos"], use["hypoxia_pseudobulk"])
rho_cpm,  p_cpm  = stats.spearmanr(use["HAVCR2_CPM"], use["hypoxia_pseudobulk"])
print(f"  Spearman (frac HAVCR2+ vs hypoxia): rho={rho_frac:.3f}, p={p_frac:.3e}")
print(f"  Spearman (HAVCR2 CPM vs hypoxia):   rho={rho_cpm:.3f}, p={p_cpm:.3e}")

# Linear model adjusting for subtype
use["subtype"] = pd.Categorical(use["subtype"], categories=["ER","HER2","TNBC"])
lm = smf.ols("hypoxia_pseudobulk ~ frac_HAVCR2_pos + C(subtype)", data=use).fit()
coef = lm.params["frac_HAVCR2_pos"]
pval = lm.pvalues["frac_HAVCR2_pos"]
ci   = lm.conf_int().loc["frac_HAVCR2_pos"].values
print(f"  Linear model (adjusted for subtype): coef={coef:.3f} "
      f"(95% CI {ci[0]:.3f}-{ci[1]:.3f}), p={pval:.3e}")

readme += [
    "",
    "PRIMARY RESULT -- TIM3 burden vs hypoxia score across patients",
    "-"*60,
    f"  Samples used (detectable HAVCR2): {len(use)} / {n_samples}",
    f"  Spearman, frac HAVCR2+ cells vs pseudobulk hypoxia score: "
    f"rho={rho_frac:.3f}, p={p_frac:.3e}",
    f"  Spearman, HAVCR2 CPM vs pseudobulk hypoxia score: "
    f"rho={rho_cpm:.3f}, p={p_cpm:.3e}",
    f"  Linear model (hypoxia ~ frac_HAVCR2_pos + subtype): "
    f"coef={coef:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}), p={pval:.3e}",
    "  Positive coef/rho = higher TIM3 burden associated with higher",
    "  pseudobulk hypoxia score, consistent with the original hypothesis.",
]

# Figure: scatter, frac HAVCR2+ vs hypoxia, coloured by subtype
fig, ax = plt.subplots(figsize=(7, 5.5))
colors = {"ER": "#4C72B0", "HER2": "#DD8452", "TNBC": "#55A868"}
for st in ["ER", "HER2", "TNBC"]:
    d = use[use["subtype"] == st]
    if len(d) == 0:
        continue
    ax.scatter(d["frac_HAVCR2_pos"] * 100, d["hypoxia_pseudobulk"],
               s=70, alpha=0.85, color=colors[st], edgecolors="white",
               linewidths=0.6, label=f"{st} (n={len(d)})")

# overall regression line (unadjusted, for visual reference only)
z = np.polyfit(use["frac_HAVCR2_pos"] * 100, use["hypoxia_pseudobulk"], 1)
xs = np.linspace((use["frac_HAVCR2_pos"] * 100).min(), (use["frac_HAVCR2_pos"] * 100).max(), 50)
ax.plot(xs, np.polyval(z, xs), color="black", ls="--", lw=1.2, alpha=0.6,
        label="Overall trend (not adjusted for subtype)")

ax.set_xlabel("% HAVCR2+ epithelial cells (per patient)", fontsize=10)
ax.set_ylabel("Pseudobulk hypoxia score\n(mean log1p CPM, GO:0001666 panel)", fontsize=10)
ax.set_title(
    f"TIM3 vs hypoxia at the patient level (n={len(use)} patients)\n"
    f"Spearman rho={rho_frac:.2f}, p={p_frac:.3f}  |  "
    f"adjusted for subtype: p={pval:.3f}",
    fontsize=10.5, fontweight="bold"
)
ax.legend(fontsize=8.5, loc="best")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_TIM3_vs_hypoxia_by_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig1_TIM3_vs_hypoxia_by_patient.png")

del adata
gc.collect()

# ══════════════════════════════════════════════════════════════════════════
# SECONDARY: genome-wide pseudobulk DE (NB GLM, library-size offset)
# ══════════════════════════════════════════════════════════════════════════
print("\n  Secondary analysis: genome-wide pseudobulk DE vs TIM3 burden...")

burden_z = (meta["frac_HAVCR2_pos"].values - meta["frac_HAVCR2_pos"].values.mean()) / \
           (meta["frac_HAVCR2_pos"].values.std() + 1e-8)
subtype_cat = pd.Categorical(meta["subtype"].values, categories=["ER", "HER2", "TNBC"])
log_lib = np.log(library_size)

# Filter to genes detected (raw count > 0) in >= 10 of 34 samples
detected_per_gene = (pseudobulk > 0).sum(axis=0)
gene_mask = detected_per_gene >= 10
gene_idx_keep = np.where(gene_mask)[0]
print(f"  Genes detected in >=10/{n_samples} samples: {len(gene_idx_keep)} / {len(gene_names)}")

rows = []
warnings.filterwarnings("ignore")
for gi in gene_idx_keep:
    y = pseudobulk[:, gi]
    df = pd.DataFrame({"y": y, "burden": burden_z, "subtype": subtype_cat, "logoffset": log_lib})
    try:
        m = smf.glm("y ~ burden + C(subtype)", data=df,
                     family=sm.families.NegativeBinomial(), offset=df["logoffset"]).fit()
        rows.append({
            "gene": gene_names[gi],
            "coef": m.params.get("burden", np.nan),
            "se": m.bse.get("burden", np.nan),
            "pval": m.pvalues.get("burden", np.nan),
        })
    except Exception:
        continue

deg_pb = pd.DataFrame(rows).dropna(subset=["pval"])
from scipy.stats import rankdata
n = len(deg_pb)
ranks = rankdata(deg_pb["pval"])
deg_pb["pval_adj"] = np.minimum(1, deg_pb["pval"] * n / ranks)
deg_pb = deg_pb.sort_values("pval")
deg_pb.to_csv(f"{OUTDIR}/pseudobulk_DEG_vs_TIM3_burden.csv", index=False)
n_sig = (deg_pb["pval_adj"] < 0.05).sum()
n_up  = ((deg_pb["pval_adj"] < 0.05) & (deg_pb["coef"] > 0)).sum()
n_down = ((deg_pb["pval_adj"] < 0.05) & (deg_pb["coef"] < 0)).sum()
print(f"  Fit {len(deg_pb)} genes. Significant (FDR<0.05): {n_sig} "
      f"({n_up} positively, {n_down} negatively associated with TIM3 burden)")
print(f"  Saved: {OUTDIR}/pseudobulk_DEG_vs_TIM3_burden.csv")

readme += [
    "",
    "SECONDARY RESULT -- genome-wide pseudobulk association with TIM3 burden",
    "-"*60,
    "Model: gene_count ~ frac_HAVCR2_pos (z-scored) + subtype, "
    "offset=log(library_size), Negative Binomial GLM, one fit per gene.",
    f"  Genes tested: {len(deg_pb)} (detected in >=10/{n_samples} samples)",
    f"  Significant (FDR<0.05): {n_sig}  "
    f"({n_up} positively, {n_down} negatively associated with TIM3 burden)",
    "  Positive coef = gene increases with TIM3 burden across patients.",
    "  NOTE: n=34 patients is modest for genome-wide testing -- treat this",
    "  as a patient-level complement to the single-gene depth-adjusted",
    "  result (07f_depth_adjusted), not as a definitive gene list.",
]

# Volcano plot
fig, ax = plt.subplots(figsize=(8, 6))
deg_pb["nlp"] = -np.log10(deg_pb["pval_adj"].clip(1e-300))
up   = (deg_pb["pval_adj"] < 0.05) & (deg_pb["coef"] > 0)
down = (deg_pb["pval_adj"] < 0.05) & (deg_pb["coef"] < 0)
rest = ~(up | down)
ax.scatter(deg_pb.loc[rest, "coef"], deg_pb.loc[rest, "nlp"],
            c="#CCCCCC", s=5, alpha=0.5, label=f"Not significant ({rest.sum()})")
ax.scatter(deg_pb.loc[up, "coef"], deg_pb.loc[up, "nlp"],
            c="#E41A1C", s=12, alpha=0.8, label=f"+ with TIM3 ({up.sum()})")
ax.scatter(deg_pb.loc[down, "coef"], deg_pb.loc[down, "nlp"],
            c="#377EB8", s=12, alpha=0.8, label=f"- with TIM3 ({down.sum()})")
for _, row in pd.concat([deg_pb[up].nlargest(8, "nlp"), deg_pb[down].nlargest(8, "nlp")]).iterrows():
    ax.text(row["coef"], row["nlp"] + 0.1, row["gene"], fontsize=6.5, ha="center")
ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, alpha=0.6, label="FDR = 0.05")
ax.set_xlabel("Coefficient (association with per-patient fraction of TIM3+ cells)", fontsize=9.5)
ax.set_ylabel("-log10(FDR)", fontsize=10)
ax.set_title(
    f"Pseudobulk: gene-TIM3 association at the patient level (n={n_samples})\n"
    f"Adjusted for subtype and library size",
    fontsize=10.5, fontweight="bold"
)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_pseudobulk_volcano.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig2_pseudobulk_volcano.png")

# ── GSEA on pseudobulk ranking ─────────────────────────────────────────────
print("\n  GSEA on pseudobulk ranking...")
import gseapy as gp

pval_r = deg_pb["pval"].values.copy()
pval_r[pval_r == 0] = pval_r[pval_r > 0].min()
metric = np.sign(deg_pb["coef"].values) * (-np.log10(pval_r))
rnk = pd.Series(metric, index=deg_pb["gene"].values).sort_values(ascending=False)

def run_gsea_pb(rnk, gmt_path, label):
    res = gp.prerank(rnk=rnk, gene_sets=gmt_path, min_size=GSEA_MIN_SIZE,
                      max_size=GSEA_MAX_SIZE, permutation_num=GSEA_PERMUTATIONS,
                      outdir=None, seed=42, verbose=False, threads=4)
    df = res.res2d.copy()
    df = df.rename(columns={"Term": "pathway", "NOM p-val": "pval",
                             "FDR q-val": "FDR", "Lead_genes": "leading_genes"})
    df["NES"] = df["NES"].astype(float)
    df["FDR"] = df["FDR"].astype(float)
    df["pathway_clean"] = df["pathway"].str.replace(
        r"^HALLMARK_|^GOBP_|^GO_BIOLOGICAL_PROCESS_", "", regex=True
    ).str.replace("_", " ").str.title()
    n_sig = (df["FDR"] < FDR_THRESHOLD).sum()
    print(f"  GSEA {label}: {len(df)} pathways, {n_sig} significant (FDR<{FDR_THRESHOLD})")
    df.to_csv(f"{OUTDIR}/GSEA_{label}_pseudobulk.csv", index=False)
    return df

def bubbleplot_pb(gsea_df, label, top_n):
    df = gsea_df.copy()
    sig = df[df["FDR"] < FDR_THRESHOLD].copy()
    if sig.empty:
        sig = df.nsmallest(top_n, "FDR").copy()
        print(f"  No pathways significant at FDR<{FDR_THRESHOLD} for {label} -- showing top {top_n}")
    n_pos = (sig["NES"] > 0).sum()
    n_neg = (sig["NES"] < 0).sum()
    g1 = sig[sig["NES"] > 0].nlargest(top_n // 2, "NES")
    g2 = sig[sig["NES"] < 0].nsmallest(top_n // 2, "NES")
    plot_df = pd.concat([g1, g2]).drop_duplicates().sort_values("NES")
    if plot_df.empty:
        print(f"  No pathways to plot for {label}"); return
    plot_df["name"] = plot_df["pathway_clean"].str[:55]
    fdr_clip = plot_df["FDR"].clip(1e-10, 1)
    sizes = 40 + 200 * (-np.log10(fdr_clip) / (-np.log10(fdr_clip).max() + 1e-8))
    colors = ["#E41A1C" if n > 0 else "#377EB8" for n in plot_df["NES"]]
    fig, ax = plt.subplots(figsize=(11, max(5.5, len(plot_df) * 0.42 + 0.8)))
    fig.suptitle(
        f"{label} GSEA pseudobulk (patient-level, n={n_samples}) -- association with TIM3+ fraction\n"
        f"{n_pos} pathway(s) positively associated, {n_neg} negatively (FDR<{FDR_THRESHOLD})",
        fontsize=10, fontweight="bold")
    ax.scatter(plot_df["NES"], range(len(plot_df)), c=colors, s=sizes, alpha=0.88,
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["name"], fontsize=8.5)
    ax.set_xlabel("NES  (positive = associated with higher per-patient fraction of TIM3+ cells)", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    for i, row in enumerate(plot_df.itertuples()):
        fdr_str = f"FDR={row.FDR:.3f}" if row.FDR >= 0.001 else "FDR<0.001"
        x_off = -0.08 if row.NES >= 0 else 0.08
        ax.text(row.NES + x_off, i, fdr_str, va="center",
                ha="right" if row.NES >= 0 else "left", fontsize=6.5, color="#444",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E41A1C", markersize=10,
               label=f"+ associated with TIM3+, n={n_pos}" if n_pos else "+ associated: none at this FDR"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#377EB8", markersize=10,
               label=f"- associated with TIM3+, n={n_neg}" if n_neg else "- associated: none at this FDR"),
    ], fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fname = f"{OUTDIR}/bubbleplot_{label.replace(' ', '_')}_pseudobulk.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

df_hall_pb = df_gobp_pb = None
gmt_h = os.path.join(GMT_DIR, "HALLMARK.gmt")
gmt_g = os.path.join(GMT_DIR, "GOBP.gmt")
if os.path.exists(gmt_h):
    df_hall_pb = run_gsea_pb(rnk, gmt_h, "HALLMARK")
    bubbleplot_pb(df_hall_pb, "HALLMARK", TOP_N)
if os.path.exists(gmt_g):
    df_gobp_pb = run_gsea_pb(rnk, gmt_g, "GOBP")
    bubbleplot_pb(df_gobp_pb, "GO BP", TOP_N)

if df_hall_pb is not None:
    readme += [
        "",
        "GSEA (pseudobulk ranking) -- HALLMARK",
        "-"*60,
        f"  Significant (FDR<{FDR_THRESHOLD}): {(df_hall_pb['FDR']<FDR_THRESHOLD).sum()} / {len(df_hall_pb)}",
    ]
if df_gobp_pb is not None:
    readme += [
        "",
        "GSEA (pseudobulk ranking) -- GO BP",
        "-"*60,
        f"  Significant (FDR<{FDR_THRESHOLD}): {(df_gobp_pb['FDR']<FDR_THRESHOLD).sum()} / {len(df_gobp_pb)}",
    ]

readme += [
    "",
    "OUTPUT FILES",
    "-"*60,
    "  README.txt",
    "  sample_metadata.csv               -- per-patient pseudobulk summary",
    "  fig1_TIM3_vs_hypoxia_by_patient.png -- primary result (scatter, by subtype)",
    "  pseudobulk_DEG_vs_TIM3_burden.csv -- genome-wide gene-level association",
    "  fig2_pseudobulk_volcano.png       -- volcano of the above",
    "  GSEA_HALLMARK_pseudobulk.csv, GSEA_GOBP_pseudobulk.csv",
    "  bubbleplot_HALLMARK_pseudobulk.png, bubbleplot_GO_BP_pseudobulk.png",
]

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE -- 07h pseudobulk
============================================================
  Primary: frac HAVCR2+ vs hypoxia, rho={rho_frac:.3f}, p={p_frac:.3e}
           (adjusted for subtype: p={pval:.3e})
  Secondary: {len(deg_pb)} genes tested, {n_sig} significant (FDR<0.05)
  Output: {OUTDIR}/
============================================================
""")
