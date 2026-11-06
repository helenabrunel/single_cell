"""
07g_DEA_GSEA_filtered.py — DEG + GSEA pre-ranked, RETRY with gene filtering
G1: TIM3high/Hypoxia-low  vs  G2: TIM3low/Hypoxia-high (epithelial cells)

This is a retry of 07e_DEA_GSEA.py. That version's volcano plot showed
~8,120 genes (29% of all genes tested) piled at an artificially extreme
log2FC (-20 to -30, many at an identical value) -- a pseudocount-floor
artifact: G1 is small (n=82) and most genes are simply undetected there,
so scanpy's log2((mean_G1+eps)/(mean_G2+eps)) saturates at the pseudocount
regardless of real biology.

FIX APPLIED HERE: before running DEG, drop genes that aren't reliably
detected in BOTH groups (>=3 cells expressing in G1 AND >=3 in G2 --
standard scanpy filter_genes(min_cells=3) convention, applied per-group
instead of dataset-wide). This removes the floor-saturated genes from the
DEG table and, critically, from the GSEA ranking metric they were
distorting.

IMPORTANT CAVEAT (see figures/07f_depth_adjusted/README.txt for full
detail): this filtering fixes the pseudocount-floor artifact, but NOT the
deeper problem that G1 and G2 are confounded with sequencing depth almost
to the point of collinearity (G1 median total_counts=1,108 vs G2=10,006;
only 1/102 G1 cells overlaps G2's depth range). That confound cannot be
removed by gene filtering -- it would require redefining the groups
entirely, which isn't feasible here since raw-UMI-based regrouping leaves
only 20 cells. Every figure below carries an explicit warning banner
about this; treat results as exploratory, not confirmatory.
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, gc, urllib.request, gzip, datetime
from scipy.sparse import issparse
import gseapy as gp

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
H5AD_FILE  = "compartment_epithelial.h5ad"
OUTDIR     = "figures/07g_DEA_GSEA_filtered"
GMT_DIR    = "gmt_files"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(GMT_DIR, exist_ok=True)

TIM3_THRESHOLD    = 1.5
HYPOXIA_ZERO      = 0.0
MIN_N             = 30
MIN_CELLS_PER_GROUP = 3   # gene must be detected in >=3 cells in G1 AND >=3 in G2
GSEA_PERMUTATIONS = 1000
GSEA_MIN_SIZE     = 15
GSEA_MAX_SIZE     = 500
FDR_THRESHOLD     = 0.25
TOP_N_HALLMARK    = 20
TOP_N_GOBP        = 30

DEPTH_CAVEAT = (
    "CAVEAT: G1/G2 are confounded with sequencing depth (median total_counts\n"
    "1,108 vs 10,006; depth ranges barely overlap) -- gene filtering fixes a\n"
    "separate pseudocount artifact but NOT this confound. See 07f_depth_adjusted/. Exploratory only."
)

GO_0001666 = list(dict.fromkeys([
    "HIF1A","EPAS1","HIF3A","ARNT","VEGFA","VEGFB","VEGFC","PGF",
    "LDHA","LDHB","PFKL","PFKM","PFKFB3","SLC2A1","SLC2A3","HK1","HK2",
    "ENO1","ENO2","PDK1","PDK3","BNIP3","BNIP3L","CA9","CA12",
    "ANGPT1","ANGPT2","ANGPTL4","EPO","EPOR","NOS2","NOS3",
    "HMOX1","HMOX2","ADM","SERPINE1","SERPINE2","FN1","COL1A1","COL4A1",
    "MMP2","MMP9","MMP14","CXCR4","CXCL12","CXCL8","TNF","IL1A","IL1B",
    "IL6","PTGS2","TF","TFRC","FTH1","FTL","EGLN1","EGLN2","EGLN3","VHL",
    "P4HA1","P4HA2","PLOD1","PLOD2","DDIT4","DDIT3","ATF4","ATF3",
    "SLC7A11","GPI","ALDOA","TPI1","GAPDH","PGK1","PGAM1","PKM",
    "BCL2","BCL2L1","BAX","BAK1","NOTCH1","NOTCH2","DLL4","JAG1",
    "KDR","FLT1","TWIST1","SNAI1","SNAI2","ZEB1","VIM","CDH2",
    "MET","HGF","TGFB1","TGFB2","TGFBR1","MYC","PTEN","PIK3CA",
    "AKT1","MTOR","MAPK1","MAPK3","STAT3","JAK1","JAK2",
    "NFE2L2","SOD1","SOD2","CAT","GPX1","GPX4",
]))

GMT_URLS = {
    "HALLMARK": [
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt",
        "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020",
    ],
    "GOBP": [
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/c5.go.bp.v2023.2.Hs.symbols.gmt",
        "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023",
    ],
}
GMT_MANUAL = {
    "HALLMARK": "https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H",
    "GOBP"    : "https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C5",
}


# ── GMT helpers ───────────────────────────────────────────────────────────────
def download_gmt(name):
    cache = os.path.join(GMT_DIR, f"{name}.gmt")
    if os.path.exists(cache):
        print(f"  Using cached {name}: {cache}")
        return cache
    for url in GMT_URLS[name]:
        try:
            print(f"  Downloading {name} from {url[:70]}...")
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                content = r.read()
            if content[:2] == b'\x1f\x8b':
                content = gzip.decompress(content)
            with open(cache, "wb") as f:
                f.write(content)
            print(f"  Downloaded {name} ({len(content)//1024} KB)")
            return cache
        except Exception as e:
            print(f"  Failed: {e}")
    print(f"\n  Could not download {name}.")
    print(f"  Download manually: {GMT_MANUAL[name]}")
    print(f"  Save as: {cache}")
    return None


# ── Group definition ──────────────────────────────────────────────────────────
def define_groups(obs, readme):
    havcr   = obs["HAVCR2_expr"].values.astype(float)
    hypoxia = obs["score_hypoxia_z"].values.astype(float)
    pos     = havcr > 0

    readme.append(f"\nTotal HAVCR2+ cells: {pos.sum()}")
    readme.append(f"HAVCR2 range (pos): {havcr[pos].min():.3f} - {havcr[pos].max():.3f}")
    readme.append(f"Hypoxia z range: {hypoxia.min():.3f} - {hypoxia.max():.3f}")

    g1 = pos & (havcr > TIM3_THRESHOLD) & (hypoxia < HYPOXIA_ZERO)
    g2 = pos & (havcr > 0) & (havcr <= TIM3_THRESHOLD) & (hypoxia > HYPOXIA_ZERO)
    method = "predefined"
    t1, t_lo, t_hi = TIM3_THRESHOLD, HYPOXIA_ZERO, HYPOXIA_ZERO

    if g1.sum() < MIN_N or g2.sum() < MIN_N:
        print(f"  predefined thresholds give G1={g1.sum()}, G2={g2.sum()} -- using percentile fallback")
        t1   = np.percentile(havcr[pos], 75)
        t_lo = np.percentile(hypoxia[pos], 33)
        t_hi = np.percentile(hypoxia[pos], 67)
        g1 = pos & (havcr > t1) & (hypoxia < t_lo)
        g2 = pos & (havcr > 0) & (havcr <= t1) & (hypoxia > t_hi)
        method = "percentile-fallback (p75/p33/p67)"

    readme += [
        f"\nGroup definition: {method}",
        f"  HAVCR2 threshold      : {t1:.3f}",
        f"  Hypoxia low threshold : {t_lo:.3f}",
        f"  Hypoxia high threshold: {t_hi:.3f}",
        f"  G1 (TIM3high/Hyp-low) : n={g1.sum()}",
        f"  G2 (TIM3low/Hyp-high) : n={g2.sum()}",
    ]
    if "subtype" in obs.columns:
        for lbl, mask in [("G1", g1), ("G2", g2)]:
            readme.append(f"  {lbl} subtypes: " +
                ", ".join(f"{st}={(mask & (obs['subtype']==st)).sum()}"
                          for st in ["ER","HER2","TNBC"]))

    print(f"  G1 (TIM3high/Hyp-low): n={g1.sum()}")
    print(f"  G2 (TIM3low/Hyp-high): n={g2.sum()}")
    return g1, g2, method


# ── Gene filter (fixes the pseudocount-floor artifact) ─────────────────────────
def filter_genes_reliable(adata, g1_mask, g2_mask, min_cells, readme):
    """
    Keep only genes detected (raw count>0) in >=min_cells cells in BOTH
    G1 and G2. Genes failing this are exactly the ones that saturated the
    logFC pseudocount floor in 07e (undetected in the small G1 group).
    """
    raw = adata.layers["counts"]
    raw1 = raw[g1_mask]
    raw2 = raw[g2_mask]
    nnz1 = np.asarray((raw1 > 0).sum(axis=0)).flatten()
    nnz2 = np.asarray((raw2 > 0).sum(axis=0)).flatten()
    keep = (nnz1 >= min_cells) & (nnz2 >= min_cells)
    n_before, n_after = adata.n_vars, keep.sum()
    print(f"  Gene filter (>= {min_cells} cells detected in BOTH G1 and G2): "
          f"{n_before} -> {n_after} genes kept ({n_before - n_after} dropped)")
    readme += [
        f"\nGENE FILTER (fixes pseudocount-floor artifact from 07e)",
        "-"*30,
        f"  Criterion: gene detected in >={min_cells} cells in G1 AND >={min_cells} cells in G2",
        f"  Genes tested in 07e (unfiltered): {n_before}",
        f"  Genes retained here: {n_after}  ({n_before - n_after} dropped)",
    ]
    return adata[:, keep].copy()


# ── DEG ───────────────────────────────────────────────────────────────────────
def run_deg(adata, g1_mask, g2_mask):
    sub = adata[g1_mask | g2_mask].copy()
    sub.obs["group"] = "G2"
    sub.obs.loc[g1_mask[g1_mask | g2_mask], "group"] = "G1"
    sc.tl.rank_genes_groups(
        sub, groupby="group", groups=["G1"],
        reference="G2", method="wilcoxon",
        use_raw=False, tie_correct=True
    )
    df = sc.get.rank_genes_groups_df(sub, group="G1")
    df = df.rename(columns={
        "names":"gene","logfoldchanges":"log2FC",
        "pvals":"pval","pvals_adj":"pval_adj","scores":"score"
    }).sort_values("log2FC", ascending=False)
    n_up   = ((df["pval_adj"]<0.05) & (df["log2FC"]>0.25)).sum()
    n_down = ((df["pval_adj"]<0.05) & (df["log2FC"]<-0.25)).sum()
    print(f"  DEG: {n_up} up in G1, {n_down} up in G2 (FDR<0.05, |logFC|>0.25)")
    return df


def make_rnk(deg_df):
    pval = deg_df["pval"].values.copy()
    pval[pval == 0] = pval[pval > 0].min()
    metric = np.sign(deg_df["log2FC"].values) * (-np.log10(pval))
    rnk = pd.Series(metric, index=deg_df["gene"].values).sort_values(ascending=False)
    return rnk


# ── GSEA ─────────────────────────────────────────────────────────────────────
def run_gsea(rnk, gmt_path, label):
    res = gp.prerank(
        rnk=rnk, gene_sets=gmt_path,
        min_size=GSEA_MIN_SIZE, max_size=GSEA_MAX_SIZE,
        permutation_num=GSEA_PERMUTATIONS,
        outdir=None, seed=42, verbose=False, threads=4,
    )
    df = res.res2d.copy()
    df = df.rename(columns={
        "Term":"pathway","NOM p-val":"pval",
        "FDR q-val":"FDR","Lead_genes":"leading_genes"
    })
    df["NES"]  = df["NES"].astype(float)
    df["FDR"]  = df["FDR"].astype(float)
    df["pathway_clean"] = df["pathway"].str.replace(
        r"^HALLMARK_|^GOBP_|^GO_BIOLOGICAL_PROCESS_","",regex=True
    ).str.replace("_"," ").str.title()
    n_sig = (df["FDR"] < FDR_THRESHOLD).sum()
    print(f"  GSEA {label}: {len(df)} pathways, {n_sig} significant (FDR<{FDR_THRESHOLD})")
    df.to_csv(f"{OUTDIR}/GSEA_{label}.csv", index=False)
    print(f"  Saved: {OUTDIR}/GSEA_{label}.csv")
    return df


# ── Volcano plot ──────────────────────────────────────────────────────────────
def volcano(deg_df, g1_n, g2_n, n_genes_tested):
    df = deg_df.copy()
    pval = df["pval_adj"].values.copy()
    pval[pval == 0] = pval[pval > 0].min()
    df["nlp"] = -np.log10(pval)
    up   = (df["pval_adj"]<0.05) & (df["log2FC"]>0.25)
    down = (df["pval_adj"]<0.05) & (df["log2FC"]<-0.25)
    rest = ~(up | down)

    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.scatter(df.loc[rest,"log2FC"], df.loc[rest,"nlp"],
               c="#CCCCCC", s=4, alpha=0.4, rasterized=True,
               label=f"Not significant ({rest.sum()})")
    ax.scatter(df.loc[up,"log2FC"], df.loc[up,"nlp"],
               c="#E41A1C", s=10, alpha=0.75, label=f"Up G1 ({up.sum()})")
    ax.scatter(df.loc[down,"log2FC"], df.loc[down,"nlp"],
               c="#377EB8", s=10, alpha=0.75, label=f"Up G2 ({down.sum()})")

    for direction, mask in [(1, up), (-1, down)]:
        for _, row in df[mask].nlargest(10,"nlp").iterrows():
            ax.text(row["log2FC"]+0.05*direction, row["nlp"]+0.1,
                    row["gene"], fontsize=6,
                    ha="left" if direction>0 else "right")

    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, alpha=0.6,
               label="p_adj = 0.05")
    ax.axvline(0.25,  color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(-0.25, color="gray", ls="--", lw=0.8, alpha=0.6,
               label="|log2FC| = 0.25")
    ax.set_xlabel("log2FC  (G1 / G2)  --  genes filtered to those reliably\n"
                  f"detected in both groups ({n_genes_tested:,} of 28,178 tested)",
                  fontsize=9.5)
    ax.set_ylabel("-log10(p_adj)", fontsize=10)
    ax.set_title(
        f"DEG (filtered): TIM3high/Hyp-low (G1, n={g1_n}) vs TIM3low/Hyp-high (G2, n={g2_n})\n"
        f"Epithelial cells, all subtypes",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper left")

    fig.text(0.5, 0.005, DEPTH_CAVEAT, ha="center", va="bottom", fontsize=7.5,
              color="#8A2E00",
              bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#E67E22", pad=0.5))

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(f"{OUTDIR}/volcano_DEG.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTDIR}/volcano_DEG.png")


# ── Bubble plot ───────────────────────────────────────────────────────────────
def bubbleplot(gsea_df, label, top_n, g1_n, g2_n):
    df = gsea_df.copy()
    sig = df[df["FDR"] < FDR_THRESHOLD].copy()
    if sig.empty:
        sig = df.nsmallest(top_n, "FDR").copy()
        print(f"  No significant pathways at FDR<{FDR_THRESHOLD} -- showing top {top_n}")

    g1_up = sig[sig["NES"] > 0].nlargest(top_n // 2, "NES")
    g2_up = sig[sig["NES"] < 0].nsmallest(top_n // 2, "NES")
    plot_df = pd.concat([g1_up, g2_up]).drop_duplicates().sort_values("NES")

    if plot_df.empty:
        print(f"  No pathways to plot for {label}"); return

    n_g1_sig = (sig["NES"] > 0).sum()
    n_g2_sig = (sig["NES"] < 0).sum()

    plot_df = plot_df.copy()
    plot_df["name"] = plot_df["pathway_clean"].str[:55]
    fdr_clip    = plot_df["FDR"].clip(1e-10, 1)
    neg_log_fdr = -np.log10(fdr_clip)
    sizes  = 40 + 200 * (neg_log_fdr / (neg_log_fdr.max() + 1e-8))
    colors = ["#E41A1C" if n > 0 else "#377EB8" for n in plot_df["NES"]]

    fig, ax = plt.subplots(figsize=(12, max(6.8, len(plot_df) * 0.44 + 0.8)))
    fig.suptitle(
        f"{label} GSEA (filtered DEG input) — TIM3high/Hyp-low (G1, n={g1_n}) vs "
        f"TIM3low/Hyp-high (G2, n={g2_n})\n"
        f"Epithelial cells, all subtypes  |  FDR threshold = {FDR_THRESHOLD}  |  "
        f"{n_g1_sig} pathway(s) enriched in G1, {n_g2_sig} in G2 (FDR<{FDR_THRESHOLD})",
        fontsize=10, fontweight="bold"
    )

    ax.scatter(plot_df["NES"], range(len(plot_df)),
               c=colors, s=sizes, alpha=0.88,
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["name"], fontsize=9)
    ax.set_xlabel(
        "Normalised Enrichment Score (NES)\n"
        "positive = enriched in G1 (TIM3high/Hyp-low)  |  "
        "negative = enriched in G2 (TIM3low/Hyp-high)",
        fontsize=9
    )
    ax.grid(axis="x", alpha=0.25)

    # FDR annotations -- placed on the side facing NES=0 (center), not the
    # outer edge, so labels for points near the axis limits don't collide
    # with the pathway-name row labels on the far left. White background so
    # they stay legible even when large (highly significant) bubbles overlap.
    for i, row in enumerate(plot_df.itertuples()):
        fdr_str = f"FDR={row.FDR:.3f}" if row.FDR >= 0.001 else "FDR<0.001"
        x_off = -0.10 if row.NES >= 0 else 0.10
        ax.text(row.NES + x_off, i, fdr_str,
                va="center", ha="right" if row.NES >= 0 else "left",
                fontsize=6.5, color="#444", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75))

    # Legend -- only show categories that actually have plotted points, so an
    # empty side (e.g. 0 pathways enriched in G1) isn't implied to have data
    from matplotlib.lines import Line2D
    legend_handles = []
    if n_g1_sig > 0:
        legend_handles.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#E41A1C",
                               markersize=10, label=f"Enriched in G1 (TIM3high/Hyp-low), n={n_g1_sig}"))
    else:
        legend_handles.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#E41A1C",
                               markersize=10, alpha=0.3, label="Enriched in G1: none at this FDR"))
    if n_g2_sig > 0:
        legend_handles.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#377EB8",
                               markersize=10, label=f"Enriched in G2 (TIM3low/Hyp-high), n={n_g2_sig}"))
    else:
        legend_handles.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#377EB8",
                               markersize=10, alpha=0.3, label="Enriched in G2: none at this FDR"))
    legend_handles.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
                           markersize=7,  label="Size = -log10(FDR)"))
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.9)

    fig.text(0.5, 0.005, DEPTH_CAVEAT, ha="center", va="bottom", fontsize=7.5,
              color="#8A2E00",
              bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#E67E22", pad=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = f"{OUTDIR}/bubbleplot_{label.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 07d — TIM3 x HYPOXIA EPITHELIAL — DEG + GSEA")
print("="*65)

readme = [
    "07d - DEG + GSEA: TIM3high/Hypoxia-low vs TIM3low/Hypoxia-high",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*55,
    "\nGROUP DEFINITIONS",
    "-"*30,
    "G1: HAVCR2 > threshold  AND  hypoxia_z < threshold_lo",
    "G2: 0 < HAVCR2 <= threshold  AND  hypoxia_z > threshold_hi",
    "Only HAVCR2+ cells. All subtypes combined (ER+HER2+TNBC).",
    f"\nDEG: Wilcoxon rank-sum (scanpy rank_genes_groups)",
    f"GSEA ranking metric: sign(log2FC) x -log10(pval)",
    f"GSEA permutations: {GSEA_PERMUTATIONS}",
    f"GSEA FDR threshold: {FDR_THRESHOLD}",
]

# Load
print(f"\n  Loading {H5AD_FILE}...")
if not os.path.exists(H5AD_FILE):
    print(f"  File not found"); exit(1)
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")
if "logcounts" in adata.layers:
    adata.X = adata.layers["logcounts"].copy()

# Hypoxia score
print("\n  Scoring hypoxia...")
hyp_genes = [g for g in GO_0001666 if g in adata.var_names]
sc.tl.score_genes(adata, hyp_genes, score_name="score_hypoxia", use_raw=False)
raw = adata.obs["score_hypoxia"].values.astype(float)
adata.obs["score_hypoxia_z"] = (raw - raw.mean()) / (raw.std() + 1e-8)
print(f"  GO:0001666: {len(hyp_genes)}/{len(GO_0001666)} genes")

# HAVCR2
if "HAVCR2" not in adata.var_names:
    print("  HAVCR2 not found"); exit(1)
x = adata[:,"HAVCR2"].X
adata.obs["HAVCR2_expr"] = np.asarray(x.todense()).flatten() \
                            if issparse(x) else np.asarray(x).flatten()
n_pos = (adata.obs["HAVCR2_expr"] > 0).sum()
print(f"  HAVCR2+: {n_pos} ({100*n_pos/adata.n_obs:.2f}%)")

# Groups
print("\n  Defining groups...")
g1_mask, g2_mask, method = define_groups(adata.obs, readme)
g1_n, g2_n = g1_mask.sum(), g2_mask.sum()

# Gene filter -- fixes the pseudocount-floor artifact from 07e
print("\n  Filtering genes (fix for pseudocount-floor artifact)...")
adata_filt = filter_genes_reliable(adata, g1_mask, g2_mask, MIN_CELLS_PER_GROUP, readme)

# DEG
print("\n  Running DEG...")
deg_df = run_deg(adata_filt, g1_mask, g2_mask)
deg_df.to_csv(f"{OUTDIR}/DEG_G1_vs_G2.csv", index=False)
print(f"  Saved: {OUTDIR}/DEG_G1_vs_G2.csv")

readme += [
    f"\nDEG RESULTS", "-"*30,
    f"  Genes tested: {len(deg_df)}",
    f"  Up in G1 (FDR<0.05, logFC>0.25): {((deg_df['pval_adj']<0.05) & (deg_df['log2FC']>0.25)).sum()}",
    f"  Up in G2 (FDR<0.05, logFC<-0.25): {((deg_df['pval_adj']<0.05) & (deg_df['log2FC']<-0.25)).sum()}",
    f"  Top G1 genes: {', '.join(deg_df[deg_df['log2FC']>0].head(10)['gene'].tolist())}",
    f"  Top G2 genes: {', '.join(deg_df[deg_df['log2FC']<0].head(10)['gene'].tolist())}",
]

rnk = make_rnk(deg_df)
print(f"  Ranked {len(rnk)} genes")

# GMT + GSEA
print("\n  Downloading gene sets...")
gmt_h = download_gmt("HALLMARK")
gmt_g = download_gmt("GOBP")

df_hall = df_gobp = None
if gmt_h:
    print("\n  GSEA HALLMARK...")
    df_hall = run_gsea(rnk, gmt_h, "HALLMARK")
if gmt_g:
    print("\n  GSEA GO BP...")
    df_gobp = run_gsea(rnk, gmt_g, "GOBP")

# Figures
print("\n  Generating figures...")
volcano(deg_df, g1_n, g2_n, adata_filt.n_vars)
if df_hall is not None:
    bubbleplot(df_hall, "HALLMARK", TOP_N_HALLMARK, g1_n, g2_n)
if df_gobp is not None:
    bubbleplot(df_gobp, "GO BP", TOP_N_GOBP, g1_n, g2_n)

# README
readme += [
    f"\nOUTPUT FILES", "-"*30,
    "  README_groups.txt",
    "  DEG_G1_vs_G2.csv",
    "  GSEA_HALLMARK.csv",
    "  GSEA_GOBP.csv",
    "  volcano_DEG.png",
    "  bubbleplot_HALLMARK.png",
    "  bubbleplot_GO_BP.png",
    f"\nINTERPRETATION", "-"*30,
    "  NES > 0 = enriched in G1 (TIM3high/Hypoxia-low)",
    "  NES < 0 = enriched in G2 (TIM3low/Hypoxia-high)",
    f"  FDR threshold: {FDR_THRESHOLD} (standard for GSEA)",
    "",
    "STILL UNRESOLVED -- READ BEFORE INTERPRETING",
    "-"*30,
    "Gene filtering above removes the pseudocount-floor artifact (the",
    "~8,120-gene cluster at log2FC -20 to -30 seen in 07e's volcano), but",
    "G1 vs G2 remain confounded with sequencing depth (median total_counts",
    "1,108 vs 10,006; ranges barely overlap). This cannot be fixed by gene",
    "filtering. Treat any remaining DEG/GSEA signal here as exploratory --",
    "see figures/07f_depth_adjusted/README.txt for the depth-aware analysis",
    "and figures/07f_depth_adjusted/fig1_depth_confound_diagnosis.png for",
    "the supporting evidence.",
]
with open(f"{OUTDIR}/README_groups.txt","w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README_groups.txt")

del adata; gc.collect()

print(f"""
============================================================
 DONE
============================================================
  G1 (TIM3high / Hypoxia-low) : n={g1_n}
  G2 (TIM3low  / Hypoxia-high): n={g2_n}
  Method: {method}
  Output: {OUTDIR}/
============================================================
""")
