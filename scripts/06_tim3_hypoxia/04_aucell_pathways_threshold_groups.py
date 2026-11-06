"""
07d_TIM3_hypoxia_epithelial_GSEA.py
=====================================
AUCell-based pathway analysis comparing two epithelial populations:
  Group 1: TIM3high / Hypoxia-low  (HAVCR2 > t1 AND hypoxia_score < 0)
  Group 2: TIM3low  / Hypoxia-high (0 < HAVCR2 ≤ t1 AND hypoxia_score > 0)

Both groups restricted to HAVCR2+ epithelial cells across all subtypes.

Rationale
---------
HAVCR2 is expressed in only ~0.4% of epithelial cells — grouping all
subtypes together (Epithelial_Global) maximises statistical power.
AUCell scores each cell individually, avoiding pseudobulk aggregation
issues with small n.

Genesets
--------
1. HALLMARK (MSigDB H collection) — 50 curated hallmark gene sets
2. GO Biological Process — mechanistic resolution

Downloaded automatically from MSigDB FTP or Enrichr API.
If download fails, place GMT files manually (see README output).

Group definition strategy
--------------------------
1. Try predefined thresholds: HAVCR2 > 1.5 AND hypoxia < 0 (G1)
                                0 < HAVCR2 ≤ 1.5 AND hypoxia > 0 (G2)
2. If either group n < MIN_N, fall back to percentile-based thresholds
3. All thresholds and n documented in README_groups.txt

Output (figures/07d_GSEA/)
--------------------------
  README_groups.txt            — thresholds, n, group definitions
  dotplot_HALLMARK.png         — top pathways HALLMARK
  dotplot_GOBP.png             — top pathways GO BP
  barplot_HALLMARK.png         — signed effect size barplot
  barplot_GOBP.png             — signed effect size barplot
  table_HALLMARK.csv           — full results
  table_GOBP.csv               — full results

Run from: ~/single_cell/Visvader/
  python3 07d_TIM3_hypoxia_epithelial_GSEA.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings, gc, urllib.request, gzip, shutil, datetime
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
H5AD_FILE   = "compartment_epithelial.h5ad"   # Epithelial_Global
OUTDIR      = "figures/07d_GSEA"
GMT_DIR     = "gmt_files"                      # local cache for GMT files
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(GMT_DIR, exist_ok=True)

# predefined thresholds (fallback to percentile if n < MIN_N)
TIM3_THRESHOLD  = 1.5     # HAVCR2 expression cutoff
HYPOXIA_ZERO    = 0.0     # hypoxia score cutoff (z-normalised)
MIN_N           = 30      # minimum cells per group
AUCELL_PERCENTILE = 5     # AUCell: top % genes threshold

# GO BP: limit to terms with between MIN_GENES and MAX_GENES
GO_MIN_GENES = 15
GO_MAX_GENES = 500

# Top pathways to show in figures
TOP_N_HALLMARK = 20
TOP_N_GOBP     = 30

# GO:0001666 genes (same as script 07, used for hypoxia scoring)
GO_0001666 = list(dict.fromkeys([
    "HIF1A","EPAS1","HIF3A","ARNT","ARNT2",
    "VEGFA","VEGFB","VEGFC","VEGFD","PGF",
    "LDHA","LDHB","PFKL","PFKM","PFKFB3","PFKFB4",
    "SLC2A1","SLC2A3","HK1","HK2","ENO1","ENO2",
    "PDK1","PDK3","BNIP3","BNIP3L","CA9","CA12",
    "ANGPT1","ANGPT2","ANGPTL4","EPO","EPOR",
    "NOS2","NOS3","HMOX1","HMOX2","ADM","ADM2",
    "SERPINE1","SERPINE2","FN1","COL1A1","COL4A1",
    "MMP2","MMP9","MMP14","CXCR4","CXCL12","CXCL8",
    "TNF","IL1A","IL1B","IL6","PTGS2",
    "TF","TFRC","FTH1","FTL",
    "EGLN1","EGLN2","EGLN3","VHL",
    "P4HA1","P4HA2","PLOD1","PLOD2","DDIT4","DDIT3",
    "ATF4","ATF3","SLC7A11","GPI","ALDOA","TPI1",
    "GAPDH","PGK1","PGAM1","PKM",
    "BCL2","BCL2L1","BAX","BAK1",
    "NOTCH1","NOTCH2","DLL4","JAG1",
    "KDR","FLT1","TWIST1","SNAI1","SNAI2","ZEB1","VIM","CDH2",
    "MET","HGF","TGFB1","TGFB2","TGFBR1",
    "MYC","PTEN","PIK3CA","AKT1","MTOR",
    "MAPK1","MAPK3","STAT3","JAK1","JAK2",
    "NFE2L2","SOD1","SOD2","CAT","GPX1","GPX4",
]))

IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B"
]


# ══════════════════════════════════════════════════════════════════════════════
# GMT DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

GMT_SOURCES = {
    "HALLMARK": [
        # MSigDB FTP (most reliable)
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt",
        # Enrichr mirror
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


def download_gmt(name, cache_path):
    """Try each URL in order; return True if successful."""
    for url in GMT_SOURCES[name]:
        try:
            print(f"  Trying {url[:70]}...")
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                content = r.read()
            # Handle gzip
            if content[:2] == b'\x1f\x8b':
                content = gzip.decompress(content)
            with open(cache_path, "wb") as f:
                f.write(content)
            print(f"  ✓ Downloaded {name} ({len(content)//1024} KB)")
            return True
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    return False


def parse_gmt(path):
    """Parse GMT file → dict {geneset_name: [genes]}."""
    genesets = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # parts[1] = description/URL, skip
            genes = [g for g in parts[2:] if g]
            if genes:
                genesets[name] = genes
    return genesets


def get_genesets(name):
    """Load geneset library, downloading if needed."""
    cache = os.path.join(GMT_DIR, f"{name}.gmt")
    if not os.path.exists(cache):
        print(f"\n  Downloading {name} gene sets...")
        ok = download_gmt(name, cache)
        if not ok:
            print(f"\n  ⚠️  Could not download {name} automatically.")
            print(f"  Please download manually from:")
            print(f"  {GMT_MANUAL[name]}")
            print(f"  and save as: {cache}")
            return None
    gs = parse_gmt(cache)
    print(f"  ✓ {name}: {len(gs)} gene sets loaded")
    return gs


# ══════════════════════════════════════════════════════════════════════════════
# AUCELL IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def aucell_score(expr_matrix, gene_names, gene_set, percentile=AUCELL_PERCENTILE):
    """
    AUCell-style score for one gene set.
    
    For each cell: rank all genes by expression (descending),
    compute AUC of recovery curve for genes in the set
    within the top `percentile`% of expressed genes.
    
    Returns array of AUC scores (one per cell).
    """
    set_genes  = [g for g in gene_set if g in gene_names]
    if len(set_genes) < 3:
        return None

    gene_idx   = {g: i for i, g in enumerate(gene_names)}
    set_idx    = np.array([gene_idx[g] for g in set_genes])
    n_cells    = expr_matrix.shape[0]
    n_genes    = expr_matrix.shape[1]
    n_top      = max(1, int(n_genes * percentile / 100))

    scores = np.zeros(n_cells)

    # Process in chunks to avoid memory issues
    chunk = 500
    for start in range(0, n_cells, chunk):
        end   = min(start + chunk, n_cells)
        block = expr_matrix[start:end]
        if issparse(block):
            block = np.asarray(block.todense())

        # Rank genes per cell (descending) — argsort of negated values
        ranked = np.argsort(-block, axis=1)  # shape: (chunk, n_genes)

        for ci in range(end - start):
            top_genes  = set(ranked[ci, :n_top])
            n_set_top  = sum(1 for g in set_idx if g in top_genes)
            # AUC approximation: fraction of set genes in top
            scores[start + ci] = n_set_top / len(set_idx)

    return scores


def run_aucell(adata, genesets, min_genes=5, max_genes=None):
    """
    Run AUCell for all genesets on adata.
    Returns DataFrame (cells × genesets).
    """
    gene_names = adata.var_names.tolist()
    X = adata.X

    results = {}
    n = len(genesets)
    for i, (gs_name, gs_genes) in enumerate(genesets.items()):
        # Size filter
        present = [g for g in gs_genes if g in gene_names]
        if len(present) < min_genes:
            continue
        if max_genes and len(present) > max_genes:
            continue

        scores = aucell_score(X, gene_names, gs_genes)
        if scores is not None:
            results[gs_name] = scores

        if (i+1) % 100 == 0:
            print(f"    {i+1}/{n} gene sets scored...")

    return pd.DataFrame(results, index=adata.obs_names)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def define_groups(obs, readme_lines):
    """
    Define Group 1 (TIM3high/Hypoxia-low) and Group 2 (TIM3low/Hypoxia-high).
    
    Strategy:
      1. predefined thresholds: HAVCR2 > 1.5 AND hypoxia < 0 (G1)
                        0 < HAVCR2 ≤ 1.5 AND hypoxia > 0 (G2)
      2. If n < MIN_N in either group, try percentile fallback:
         G1: HAVCR2 > p75 AND hypoxia < p33
         G2: 0 < HAVCR2 ≤ p75 AND hypoxia > p67
    """
    havcr  = obs["HAVCR2_expr"].values.astype(float)
    hypoxia = obs["score_hypoxia_z"].values.astype(float)

    # Only HAVCR2+ cells
    pos_mask = havcr > 0
    n_pos    = pos_mask.sum()

    readme_lines.append(f"\nTotal HAVCR2+ epithelial cells: {n_pos}")
    readme_lines.append(f"HAVCR2 range (pos cells): "
                        f"{havcr[pos_mask].min():.3f} – "
                        f"{havcr[pos_mask].max():.3f}")
    readme_lines.append(f"Hypoxia score range: "
                        f"{hypoxia.min():.3f} – {hypoxia.max():.3f}")

    # ── Attempt 1: predefined thresholds ──────────────────────────────────────────
    g1 = pos_mask & (havcr > TIM3_THRESHOLD) & (hypoxia < HYPOXIA_ZERO)
    g2 = pos_mask & (havcr > 0) & (havcr <= TIM3_THRESHOLD) & \
         (hypoxia > HYPOXIA_ZERO)

    threshold_label = "predefined"
    t1_val = TIM3_THRESHOLD
    t_hyp  = HYPOXIA_ZERO

    if g1.sum() >= MIN_N and g2.sum() >= MIN_N:
        readme_lines.append(f"\nGroup definition: {threshold_label}")
        readme_lines.append(f"  HAVCR2 threshold : {t1_val}")
        readme_lines.append(f"  Hypoxia threshold: {t_hyp} (z-score)")
    else:
        print(f"  ⚠️  predefined thresholds give G1={g1.sum()}, G2={g2.sum()} "
              f"(min={MIN_N}) — trying percentile fallback")

        # ── Attempt 2: percentile fallback ──────────────────────────────
        havcr_pos  = havcr[pos_mask]
        hypoxia_pos = hypoxia[pos_mask]

        t1_val = np.percentile(havcr_pos, 75)
        t_hyp_lo = np.percentile(hypoxia_pos, 33)
        t_hyp_hi = np.percentile(hypoxia_pos, 67)

        g1 = pos_mask & (havcr > t1_val) & (hypoxia < t_hyp_lo)
        g2 = pos_mask & (havcr > 0) & (havcr <= t1_val) & \
             (hypoxia > t_hyp_hi)

        threshold_label = "percentile-fallback (p75/p33/p67)"
        readme_lines.append(f"\nGroup definition: {threshold_label}")
        readme_lines.append(f"  HAVCR2 threshold (p75)     : {t1_val:.3f}")
        readme_lines.append(f"  Hypoxia low threshold (p33): {t_hyp_lo:.3f}")
        readme_lines.append(f"  Hypoxia high threshold (p67): {t_hyp_hi:.3f}")

        if g1.sum() < MIN_N or g2.sum() < MIN_N:
            print(f"  ⚠️  Percentile fallback also gives small groups "
                  f"G1={g1.sum()}, G2={g2.sum()} — proceeding anyway")
            readme_lines.append(f"  ⚠️  WARNING: small groups, interpret with caution")

    readme_lines.append(f"\n  Group 1 (TIM3high / Hypoxia-low) : n = {g1.sum()}")
    readme_lines.append(f"  Group 2 (TIM3low  / Hypoxia-high): n = {g2.sum()}")

    # Subtype composition of each group
    if "subtype" in obs.columns:
        for label, mask in [("G1", g1), ("G2", g2)]:
            readme_lines.append(f"\n  {label} subtype composition:")
            for st in ["ER","HER2","TNBC"]:
                n_st = (mask & (obs["subtype"]==st)).sum()
                if n_st > 0:
                    readme_lines.append(f"    {st}: {n_st}")

    print(f"  ✓ G1 (TIM3high/Hyp-low): n={g1.sum()}  "
          f"G2 (TIM3low/Hyp-high): n={g2.sum()}")
    print(f"  Threshold method: {threshold_label}")

    return g1, g2, threshold_label, t1_val, t_hyp


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON + FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def compare_groups(aucell_df, g1_idx, g2_idx, label):
    """
    Mann-Whitney U test + rank-biserial r for each pathway.
    Returns DataFrame sorted by effect size.
    """
    rows = []
    for pathway in aucell_df.columns:
        s1 = aucell_df.loc[g1_idx, pathway].values
        s2 = aucell_df.loc[g2_idx, pathway].values

        if len(s1) < 3 or len(s2) < 3:
            continue
        if s1.std() == 0 and s2.std() == 0:
            continue

        stat, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        n1, n2  = len(s1), len(s2)
        rbc     = 1 - (2 * stat) / (n1 * n2)  # rank-biserial r
        # rbc > 0 means G1 > G2 (TIM3high/Hyp-low has higher score)

        rows.append({
            "pathway"    : pathway,
            "mean_G1"    : round(s1.mean(), 5),
            "mean_G2"    : round(s2.mean(), 5),
            "delta"      : round(s1.mean() - s2.mean(), 5),
            "rbc"        : round(rbc, 4),
            "p_value"    : p,
            "n_G1"       : n1,
            "n_G2"       : n2,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # FDR correction (Benjamini-Hochberg)
    from scipy.stats import rankdata
    n = len(df)
    df = df.sort_values("p_value").reset_index(drop=True)
    ranks = rankdata(df["p_value"])
    df["p_adj"] = np.minimum(1, df["p_value"] * n / ranks)
    df["significant"] = df["p_adj"] < 0.05
    df = df.sort_values("rbc", ascending=False)

    out_csv = f"{OUTDIR}/table_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ {out_csv}  ({df['significant'].sum()} significant pathways)")
    return df


def dotplot(df, label, top_n, title, fname):
    """
    Dotplot: x=rbc (effect size), y=pathway,
    dot size=−log10(p_adj), colour=direction (G1 vs G2).
    """
    sig = df[df["significant"]].copy()
    if sig.empty:
        # Fall back to top by p-value
        sig = df.head(top_n).copy()
        sig["significant"] = False

    # Select top_n: split between enriched in G1 and G2
    g1_up = sig[sig["rbc"] > 0].head(top_n // 2)
    g2_up = sig[sig["rbc"] < 0].head(top_n // 2)
    plot_df = pd.concat([g1_up, g2_up]).drop_duplicates()
    plot_df = plot_df.sort_values("rbc")

    if plot_df.empty:
        print(f"  ⚠️  No pathways to plot for {label}")
        return

    # Clean pathway names
    plot_df = plot_df.copy()
    plot_df["name"] = plot_df["pathway"].str.replace(
        r"^HALLMARK_|^GOBP_|^GO_", "", regex=True
    ).str.replace("_", " ").str.title().str[:60]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.38)))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    neg_log_p = -np.log10(plot_df["p_adj"].clip(1e-300))
    colors    = ["#E41A1C" if r > 0 else "#377EB8"
                 for r in plot_df["rbc"]]
    sizes     = 30 + 120 * (neg_log_p / (neg_log_p.max() + 1e-8))

    ax.scatter(plot_df["rbc"], range(len(plot_df)),
               c=colors, s=sizes, alpha=0.85, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["name"], fontsize=8)
    ax.set_xlabel("Rank-biserial r\n(positive = enriched in TIM3high/Hyp-low)",
                  fontsize=9)
    ax.grid(axis="x", alpha=0.25)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#E41A1C", markersize=9,
               label="Higher in G1 (TIM3high / Hyp-low)"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#377EB8", markersize=9,
               label="Higher in G2 (TIM3low / Hyp-high)"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="gray", markersize=6,
               label="Dot size = −log₁₀(p_adj)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # Annotate n
    ax.text(0.02, 0.01,
            f"G1 n={plot_df['n_G1'].iloc[0]}  G2 n={plot_df['n_G2'].iloc[0]}",
            transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/{fname}")


def barplot_signed(df, label, top_n, title, fname):
    """Signed barplot: rbc for top significant pathways."""
    sig = df[df["significant"]].copy()
    if sig.empty:
        sig = df.head(top_n).copy()

    g1_up = sig[sig["rbc"] > 0].head(top_n // 2)
    g2_up = sig[sig["rbc"] < 0].tail(top_n // 2)
    plot_df = pd.concat([g1_up, g2_up]).sort_values("rbc")

    if plot_df.empty:
        return

    plot_df = plot_df.copy()
    plot_df["name"] = plot_df["pathway"].str.replace(
        r"^HALLMARK_|^GOBP_|^GO_", "", regex=True
    ).str.replace("_", " ").str.title().str[:55]

    colors = ["#E41A1C" if r > 0 else "#377EB8"
              for r in plot_df["rbc"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.38)))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    ax.barh(range(len(plot_df)), plot_df["rbc"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["name"], fontsize=8)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Rank-biserial r", fontsize=9)
    ax.grid(axis="x", alpha=0.25)

    # p_adj annotations
    for i, row in enumerate(plot_df.itertuples()):
        pstr = f"p={row.p_adj:.2e}" if row.p_adj >= 1e-300 else "p<1e-300"
        x_pos = row.rbc + 0.005 if row.rbc >= 0 else row.rbc - 0.005
        ha    = "left" if row.rbc >= 0 else "right"
        ax.text(x_pos, i, pstr, va="center", ha=ha, fontsize=6)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/{fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 07d — TIM3×HYPOXIA EPITHELIAL — PATHWAY ANALYSIS (AUCell)")
print("="*65)

readme_lines = [
    "07d — TIM3high/Hypoxia-low vs TIM3low/Hypoxia-high",
    f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*55,
    "\nGROUP DEFINITIONS",
    "-"*30,
    "Group 1 (G1): TIM3high / Hypoxia-low",
    "  Cells with HAVCR2 > threshold AND hypoxia_score < 0",
    "Group 2 (G2): TIM3low / Hypoxia-high",
    "  Cells with 0 < HAVCR2 ≤ threshold AND hypoxia_score > 0",
    "\nOnly HAVCR2+ cells included (HAVCR2 > 0).",
    "All subtypes combined (ER + HER2 + TNBC) to maximise n.",
]

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(H5AD_FILE):
    print(f"  ⚠️  {H5AD_FILE} not found")
    exit(1)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

if "logcounts" in adata.layers:
    adata.X = adata.layers["logcounts"].copy()

# ── Score hypoxia ─────────────────────────────────────────────────────────────
print("\n  Scoring hypoxia (GO:0001666)...")
hyp_genes = [g for g in GO_0001666 if g in adata.var_names]
print(f"  {len(hyp_genes)}/{len(GO_0001666)} genes found")
sc.tl.score_genes(adata, hyp_genes, score_name="score_hypoxia", use_raw=False)
raw_hyp = adata.obs["score_hypoxia"].values.astype(float)
adata.obs["score_hypoxia_z"] = (raw_hyp - raw_hyp.mean()) / (raw_hyp.std() + 1e-8)

# ── Extract HAVCR2 ─────────────────────────────────────────────────────────────
print("  Extracting HAVCR2...")
if "HAVCR2" in adata.var_names:
    x = adata[:, "HAVCR2"].X
    adata.obs["HAVCR2_expr"] = np.asarray(x.todense()).flatten() \
                                if issparse(x) else np.asarray(x).flatten()
else:
    print("  ⚠️  HAVCR2 not found in var_names")
    exit(1)

n_pos = (adata.obs["HAVCR2_expr"] > 0).sum()
print(f"  HAVCR2+: {n_pos} cells ({100*n_pos/adata.n_obs:.2f}%)")

# ── Define groups ─────────────────────────────────────────────────────────────
print("\n  Defining groups...")
g1_mask, g2_mask, thresh_method, t1, t_hyp = define_groups(
    adata.obs, readme_lines
)

adata_g1 = adata[g1_mask].copy()
adata_g2 = adata[g2_mask].copy()

# ── Download gene sets ────────────────────────────────────────────────────────
print("\n  Loading gene sets...")
hallmark_gs = get_genesets("HALLMARK")
gobp_gs     = get_genesets("GOBP")

readme_lines.append(f"\nGENE SETS")
readme_lines.append("-"*30)
if hallmark_gs:
    readme_lines.append(f"HALLMARK: {len(hallmark_gs)} gene sets")
if gobp_gs:
    readme_lines.append(f"GO BP: {len(gobp_gs)} gene sets "
                        f"(filtered {GO_MIN_GENES}–{GO_MAX_GENES} genes)")

# ── AUCell scoring ────────────────────────────────────────────────────────────
# Score on the combined G1+G2 subset (more efficient)
combined_mask = g1_mask | g2_mask
adata_sub = adata[combined_mask].copy()
g1_idx = adata_sub.obs_names[g1_mask[combined_mask]]
g2_idx = adata_sub.obs_names[g2_mask[combined_mask]]

results_hall = None
results_gobp = None

if hallmark_gs:
    print(f"\n  Running AUCell — HALLMARK ({len(hallmark_gs)} sets)...")
    results_hall = run_aucell(adata_sub, hallmark_gs)
    print(f"  ✓ Scored {results_hall.shape[1]} HALLMARK gene sets")

if gobp_gs:
    print(f"\n  Running AUCell — GO BP ({len(gobp_gs)} sets, "
          f"filtering {GO_MIN_GENES}–{GO_MAX_GENES} genes)...")
    results_gobp = run_aucell(adata_sub, gobp_gs,
                              min_genes=GO_MIN_GENES,
                              max_genes=GO_MAX_GENES)
    print(f"  ✓ Scored {results_gobp.shape[1]} GO BP gene sets")

# ── Statistical comparison ────────────────────────────────────────────────────
print("\n  Comparing groups...")

df_hall = df_gobp = None

if results_hall is not None:
    df_hall = compare_groups(results_hall, g1_idx, g2_idx, "HALLMARK")
    if not df_hall.empty:
        n_sig = df_hall["significant"].sum()
        readme_lines.append(f"\nRESULTS — HALLMARK")
        readme_lines.append(f"  Significant pathways (FDR<0.05): {n_sig}")
        readme_lines.append(f"  Enriched in G1 (TIM3high/Hyp-low): "
                            f"{(df_hall['rbc']>0 & df_hall['significant']).sum()}")
        readme_lines.append(f"  Enriched in G2 (TIM3low/Hyp-high): "
                            f"{(df_hall['rbc']<0 & df_hall['significant']).sum()}")

if results_gobp is not None:
    df_gobp = compare_groups(results_gobp, g1_idx, g2_idx, "GOBP")
    if not df_gobp.empty:
        n_sig = df_gobp["significant"].sum()
        readme_lines.append(f"\nRESULTS — GO BP")
        readme_lines.append(f"  Significant pathways (FDR<0.05): {n_sig}")

# ── Figures ───────────────────────────────────────────────────────────────────
print("\n  Generating figures...")

g1_n = g1_mask.sum()
g2_n = g2_mask.sum()

if df_hall is not None and not df_hall.empty:
    dotplot(df_hall, "HALLMARK", TOP_N_HALLMARK,
            f"HALLMARK — TIM3high/Hyp-low (G1, n={g1_n}) vs "
            f"TIM3low/Hyp-high (G2, n={g2_n})\nEpithelial cells, all subtypes",
            "dotplot_HALLMARK.png")
    barplot_signed(df_hall, "HALLMARK", TOP_N_HALLMARK,
                   f"HALLMARK — signed effect size (rank-biserial r)\n"
                   f"G1=TIM3high/Hyp-low (n={g1_n})  G2=TIM3low/Hyp-high (n={g2_n})",
                   "barplot_HALLMARK.png")

if df_gobp is not None and not df_gobp.empty:
    dotplot(df_gobp, "GOBP", TOP_N_GOBP,
            f"GO BP — TIM3high/Hyp-low (G1, n={g1_n}) vs "
            f"TIM3low/Hyp-high (G2, n={g2_n})\nEpithelial cells, all subtypes",
            "dotplot_GOBP.png")
    barplot_signed(df_gobp, "GOBP", TOP_N_GOBP,
                   f"GO BP — signed effect size (rank-biserial r)\n"
                   f"G1=TIM3high/Hyp-low (n={g1_n})  G2=TIM3low/Hyp-high (n={g2_n})",
                   "barplot_GOBP.png")

# ── Save README ───────────────────────────────────────────────────────────────
readme_lines += [
    f"\nOUTPUT FILES",
    "-"*30,
    f"  README_groups.txt       — this file",
    f"  table_HALLMARK.csv      — full HALLMARK results",
    f"  table_GOBP.csv          — full GO BP results",
    f"  dotplot_HALLMARK.png",
    f"  dotplot_GOBP.png",
    f"  barplot_HALLMARK.png",
    f"  barplot_GOBP.png",
    f"\nINTERPRETATION",
    "-"*30,
    f"  Rank-biserial r > 0 = pathway enriched in G1 (TIM3high/Hyp-low)",
    f"  Rank-biserial r < 0 = pathway enriched in G2 (TIM3low/Hyp-high)",
    f"  FDR correction: Benjamini-Hochberg",
]

readme_path = f"{OUTDIR}/README_groups.txt"
with open(readme_path, "w") as f:
    f.write("\n".join(readme_lines))
print(f"\n  ✓ {readme_path}")

del adata, adata_sub, adata_g1, adata_g2
gc.collect()

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Output: {OUTDIR}/
  Groups:
    G1 (TIM3high / Hypoxia-low): n={g1_n}
    G2 (TIM3low  / Hypoxia-high): n={g2_n}
  Method: {thresh_method}
  See README_groups.txt for full details
══════════════════════════════════════════════════════════════
""")
