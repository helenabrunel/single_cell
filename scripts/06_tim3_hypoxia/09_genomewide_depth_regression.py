"""
07i_genomewide_depth_regression.py
====================================
Genome-wide, single-cell-resolution version of the 07f depth-offset
regression. Instead of comparing two threshold-defined cell populations
(confounded with sequencing depth, see figures/07f_depth_adjusted/) or
aggregating to patient level (figures/07h_pseudobulk/, loses single-cell
resolution), this regresses EVERY gene's raw count against the hypoxia
score directly, with each cell's own sequencing depth as a model offset.

Model per gene:
  gene_raw_count ~ hypoxia_z + subtype, offset = log(total_counts)
  Negative Binomial GLM, cluster-robust SE by patient sample.

This is the same logic as 07f (proven to recover a real, depth-robust
signal for HAVCR2 itself), scaled to all genes -- giving a genuine
volcano + GSEA at single-cell resolution, without the group-vs-depth
collinearity problem that broke 07d/07e/07g.

Genes filtered to those detected in >=100 cells (~21k genes) to keep
runtime reasonable (~80 min for the full run) and avoid unstable fits on
near-undetected genes.

Output: figures/07i_genomewide_regression/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, warnings, datetime, gc, time
from scipy.sparse import issparse
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "compartment_epithelial.h5ad"
OUTDIR    = "figures/07i_genomewide_regression"
GMT_DIR   = "gmt_files"
os.makedirs(OUTDIR, exist_ok=True)

MIN_CELLS_DETECTED = 100
GSEA_MIN_SIZE  = 15
GSEA_MAX_SIZE  = 500
GSEA_PERMUTATIONS = 1000
FDR_THRESHOLD  = 0.25
TOP_N = 20

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

t_start = time.time()
print("="*65)
print(" 07i -- GENOME-WIDE DEPTH-OFFSET REGRESSION vs HYPOXIA")
print("="*65)

readme = [
    "07i -- Genome-wide depth-offset regression vs hypoxia score",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "MODEL",
    "-"*60,
    "gene_raw_count ~ hypoxia_z + subtype, offset = log(total_counts)",
    "Negative Binomial GLM, cluster-robust SE by patient sample.",
    "One fit per gene, across all epithelial cells (all subtypes).",
    "",
    f"Genes filtered to those detected (raw>0) in >={MIN_CELLS_DETECTED} cells.",
    "This replaces the G1-vs-G2 threshold design of 07d/07e/07g, which was",
    "confounded with sequencing depth almost to the point of collinearity",
    "(see figures/07f_depth_adjusted/). Every cell contributes according to",
    "its own depth here, instead of being sorted into a depth-biased group.",
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

subtype = pd.Categorical(adata.obs["subtype"].astype(str).values, categories=["ER", "HER2", "TNBC"])
sample  = adata.obs["sample"].astype(str).values
log_total_counts = np.log(adata.obs["total_counts"].values.astype(float))

counts = adata.layers["counts"]
if not issparse(counts):
    from scipy.sparse import csr_matrix
    counts = csr_matrix(counts)

gene_names = adata.var_names.tolist()
if "n_cells_by_counts" in adata.var.columns:
    n_cells_detected = adata.var["n_cells_by_counts"].values
else:
    n_cells_detected = np.asarray((counts > 0).sum(axis=0)).flatten()

gene_mask = n_cells_detected >= MIN_CELLS_DETECTED
gene_idx_keep = np.where(gene_mask)[0]
print(f"\n  Genes with >= {MIN_CELLS_DETECTED} cells detected: {len(gene_idx_keep)} / {len(gene_names)}")
readme.append(f"  Genes tested: {len(gene_idx_keep)}")

base_df = pd.DataFrame({
    "hypoxia_z": hypoxia_z,
    "subtype": subtype,
    "sample": sample,
    "log_total_counts": log_total_counts,
})

del adata
gc.collect()

# ── Per-gene regression loop ────────────────────────────────────────────────
print(f"\n  Fitting {len(gene_idx_keep)} Negative Binomial GLMs "
      f"(offset=log total_counts, cluster-robust SE by sample)...")
print("  This will take a while -- progress every 2000 genes.")

rows = []
n_fail = 0
t_loop = time.time()
for k, gi in enumerate(gene_idx_keep):
    y = np.asarray(counts[:, gi].todense()).flatten()
    df = base_df.copy()
    df["y"] = y
    try:
        m = smf.glm("y ~ hypoxia_z + C(subtype)", data=df,
                     family=sm.families.NegativeBinomial(),
                     offset=df["log_total_counts"]).fit(
                     cov_type="cluster", cov_kwds={"groups": df["sample"]})
        rows.append({
            "gene": gene_names[gi],
            "coef": m.params.get("hypoxia_z", np.nan),
            "se": m.bse.get("hypoxia_z", np.nan),
            "pval": m.pvalues.get("hypoxia_z", np.nan),
        })
    except Exception:
        n_fail += 1
        continue

    if (k + 1) % 2000 == 0:
        elapsed = time.time() - t_loop
        rate = (k + 1) / elapsed
        eta_min = (len(gene_idx_keep) - (k + 1)) / rate / 60
        print(f"    {k+1}/{len(gene_idx_keep)} genes fit "
              f"({elapsed/60:.1f} min elapsed, ETA {eta_min:.1f} min, {n_fail} failed)")

print(f"\n  Done fitting. {len(rows)} genes converged, {n_fail} failed "
      f"({(time.time()-t_loop)/60:.1f} min)")

deg = pd.DataFrame(rows).dropna(subset=["pval"])
n = len(deg)
ranks = rankdata(deg["pval"])
deg["pval_adj"] = np.minimum(1, deg["pval"] * n / ranks)
deg = deg.sort_values("pval")
deg.to_csv(f"{OUTDIR}/genomewide_hypoxia_regression.csv", index=False)

n_sig = (deg["pval_adj"] < 0.05).sum()
n_up  = ((deg["pval_adj"] < 0.05) & (deg["coef"] > 0)).sum()
n_down = ((deg["pval_adj"] < 0.05) & (deg["coef"] < 0)).sum()
print(f"  Significant (FDR<0.05): {n_sig} ({n_up} positively, {n_down} negatively "
      f"associated with hypoxia)")
print(f"  Saved: {OUTDIR}/genomewide_hypoxia_regression.csv")

readme += [
    "",
    "RESULTS",
    "-"*60,
    f"  Genes converged: {len(deg)} (failed to converge: {n_fail})",
    f"  Significant (FDR<0.05): {n_sig}  "
    f"({n_up} positively, {n_down} negatively associated with hypoxia)",
    "  Positive coef = gene expression increases with hypoxia score,",
    "  independent of each cell's own sequencing depth.",
]

# ── Volcano ──────────────────────────────────────────────────────────────────
print("\n  Generating figures...")
fig, ax = plt.subplots(figsize=(8, 6))
deg["nlp"] = -np.log10(deg["pval_adj"].clip(1e-300))
up = (deg["pval_adj"] < 0.05) & (deg["coef"] > 0)
down = (deg["pval_adj"] < 0.05) & (deg["coef"] < 0)
rest = ~(up | down)
ax.scatter(deg.loc[rest, "coef"], deg.loc[rest, "nlp"], c="#CCCCCC", s=4, alpha=0.4,
           rasterized=True, label=f"Not significant ({rest.sum()})")
ax.scatter(deg.loc[up, "coef"], deg.loc[up, "nlp"], c="#E41A1C", s=8, alpha=0.7,
           label=f"+ with hypoxia ({up.sum()})")
ax.scatter(deg.loc[down, "coef"], deg.loc[down, "nlp"], c="#377EB8", s=8, alpha=0.7,
           label=f"- with hypoxia ({down.sum()})")
for _, row in pd.concat([deg[up].nlargest(10, "nlp"), deg[down].nlargest(10, "nlp")]).iterrows():
    ax.text(row["coef"], row["nlp"] + 0.15, row["gene"], fontsize=6, ha="center")
ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, alpha=0.6, label="FDR = 0.05")
ax.set_xlabel("Coefficient (association with hypoxia score, adjusted for depth and subtype)",
              fontsize=9)
ax.set_ylabel("-log10(FDR)", fontsize=10)
ax.set_title(
    f"Genome-wide regression: gene ~ hypoxia + subtype, offset=depth\n"
    f"Epithelial cells, all subtypes (n={len(deg):,} genes tested)",
    fontsize=10, fontweight="bold"
)
ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/volcano_genomewide.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/volcano_genomewide.png")

# ── GSEA ─────────────────────────────────────────────────────────────────────
print("\n  Running GSEA on regression ranking...")
import gseapy as gp

pval_r = deg["pval"].values.copy()
pval_r[pval_r == 0] = pval_r[pval_r > 0].min()
metric = np.sign(deg["coef"].values) * (-np.log10(pval_r))
rnk = pd.Series(metric, index=deg["gene"].values).sort_values(ascending=False)

def run_gsea_gw(rnk, gmt_path, label):
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
    df.to_csv(f"{OUTDIR}/GSEA_{label}_genomewide.csv", index=False)
    return df

def bubbleplot_gw(gsea_df, label, top_n):
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
    colors = ["#E41A1C" if x > 0 else "#377EB8" for x in plot_df["NES"]]
    fig, ax = plt.subplots(figsize=(11, max(5.5, len(plot_df) * 0.42 + 0.8)))
    fig.suptitle(
        f"{label} GSEA (genome-wide regression, single-cell resolution) -- association with hypoxia\n"
        f"{n_pos} positive pathway(s), {n_neg} negative (FDR<{FDR_THRESHOLD})",
        fontsize=10, fontweight="bold")
    ax.scatter(plot_df["NES"], range(len(plot_df)), c=colors, s=sizes, alpha=0.88,
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["name"], fontsize=8.5)
    ax.set_xlabel("NES  (positive = associated with higher hypoxia score)", fontsize=9)
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
               label=f"+ associated with hypoxia, n={n_pos}" if n_pos else "+ associated: none at this FDR"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#377EB8", markersize=10,
               label=f"- associated with hypoxia, n={n_neg}" if n_neg else "- associated: none at this FDR"),
    ], fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fname = f"{OUTDIR}/bubbleplot_{label.replace(' ', '_')}_genomewide.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

gmt_h = os.path.join(GMT_DIR, "HALLMARK.gmt")
gmt_g = os.path.join(GMT_DIR, "GOBP.gmt")
df_hall_gw = df_gobp_gw = None
if os.path.exists(gmt_h):
    df_hall_gw = run_gsea_gw(rnk, gmt_h, "HALLMARK")
    bubbleplot_gw(df_hall_gw, "HALLMARK", TOP_N)
if os.path.exists(gmt_g):
    df_gobp_gw = run_gsea_gw(rnk, gmt_g, "GOBP")
    bubbleplot_gw(df_gobp_gw, "GO BP", TOP_N)

if df_hall_gw is not None:
    readme += ["", "GSEA -- HALLMARK", "-"*60,
               f"  Significant (FDR<{FDR_THRESHOLD}): {(df_hall_gw['FDR']<FDR_THRESHOLD).sum()} / {len(df_hall_gw)}"]
if df_gobp_gw is not None:
    readme += ["", "GSEA -- GO BP", "-"*60,
               f"  Significant (FDR<{FDR_THRESHOLD}): {(df_gobp_gw['FDR']<FDR_THRESHOLD).sum()} / {len(df_gobp_gw)}"]

readme += [
    "",
    "OUTPUT FILES", "-"*60,
    "  README.txt",
    "  genomewide_hypoxia_regression.csv -- per-gene coef/p/FDR vs hypoxia",
    "  volcano_genomewide.png",
    "  GSEA_HALLMARK_genomewide.csv, GSEA_GOBP_genomewide.csv",
    "  bubbleplot_HALLMARK_genomewide.png, bubbleplot_GO_BP_genomewide.png",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE -- 07i genome-wide regression
============================================================
  Genes tested: {len(deg):,}
  Significant (FDR<0.05): {n_sig} ({n_up} up, {n_down} down)
  Total runtime: {(time.time()-t_start)/60:.1f} min
  Output: {OUTDIR}/
============================================================
""")
