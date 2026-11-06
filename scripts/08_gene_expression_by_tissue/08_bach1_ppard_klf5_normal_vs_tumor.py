"""
21_bach1_ppard_klf5_normal_vs_tumor.py
========================================
BACH1 / PPARD / KLF5 expression: normal breast tissue vs primary tumor
(ER, HER2, TNBC pooled), Visvader cohort (GSE161529).

The standard pipeline (01_filter_tumor.py onward) DROPS subtype=="Normal"
at the very first step, so it cannot be reused here. This script goes
back to the raw object and builds an independent normal+tumor branch,
using the same QC/normalization/integration recipe as 02_qc_filter_normalize.py
/ 03_split_integrate.py for methodological consistency.

Cell selection
--------------
tissue == "Total" AND subtype in {Normal, ER, HER2, TNBC}
  - "Epi" (epithelial-sorted) dropped: different sorting protocol than
    Normal/tumor "Total" samples, not comparable via a simple violin/UMAP.
  - "LN" (lymph-node metastasis) dropped: not normal or primary tissue.
  - BRCA1_mutant dropped: ambiguous carrier status (project convention,
    see memory project_analysis_state).

Normalization (validity of the comparison)
-------------------------------------------
normalize_total(1e4) + log1p, exactly as in 02_qc_filter_normalize.py.
This is per-cell library-size normalization -- it corrects for sequencing
depth differences between the Normal and Tumor groups (and between
samples/patients) BEFORE any comparison is made, which is what makes the
Normal-vs-Tumor violin plots valid. Expression plots (UMAP color, violins)
use this UNSCALED log-normalized matrix (layers["logcounts"]) so magnitudes
stay biologically interpretable. The z-scored/clipped matrix
(sc.pp.scale, max_value=10) is used ONLY as PCA input for the embedding,
never for reporting expression values -- scaling would destroy the
between-group magnitude comparison the user asked for.

Batch handling: HVG selection uses batch_key=sample; PCA -> Harmony
(batch=sample) -> neighbors -> UMAP, matching 03_split_integrate.py.

OUTPUTS
-------
figures/21_BACH1_PPARD_KLF5_normal_vs_tumor/
  fig_umap_combined.png     UMAP panels side-by-side: Normal/Tumor, subtype,
                             BACH1/PPARD/KLF5 expression (monochromatic)
  fig_violin_normal_tumor.png   violins, Normal vs Tumor (pooled)
  fig_violin_by_subtype.png     violins, Normal/ER/HER2/TNBC
  stats_normal_vs_tumor.csv
  stats_by_subtype.csv

Run from: ~/single_cell/Visvader/   (takes ~30-60 min, background-safe)
  python3 21_bach1_ppard_klf5_normal_vs_tumor.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sstats
import os, warnings, gc, sys, traceback

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.n_jobs = 8

RAW_FILE = "GSE161529_cellwhisperer.h5ad"
OUTDIR   = "figures/21_BACH1_PPARD_KLF5_normal_vs_tumor"
os.makedirs(OUTDIR, exist_ok=True)

CKPT_QC = "_ckpt_21_qc_filtered.h5ad"

GENES = ["BACH1", "PPARD", "KLF5"]

KEEP_SUBTYPES = ["Normal", "ER", "HER2", "TNBC"]
KEEP_TISSUE   = ["Total"]

GROUP_COLORS = {"Normal": "#4C72B0", "Tumor": "#C44E52"}
SUBTYPE_COLORS = {"Normal": "#4C72B0", "ER": "#4DAF4A",
                   "HER2": "#984EA3", "TNBC": "#E41A1C"}

MIN_GENES, MIN_COUNTS, MAX_MT, MIN_CELLS = 200, 500, 30, 3
N_HVG, N_PCS = 3000, 40


def log(msg):
    print(msg, flush=True)


def main():
    if os.path.exists(CKPT_QC):
        log("=" * 70)
        log(f" Resuming from checkpoint: {CKPT_QC}")
        log("=" * 70)
        adata = sc.read_h5ad(CKPT_QC)
        log(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    else:
        # ── 1. LOAD + FILTER (backed mode: never materialize the full
        #      425k-cell raw matrix, only the ~307k-cell subset we keep) ──
        log("=" * 70)
        log(" 1. LOAD + FILTER (tissue=Total, subtype in Normal/ER/HER2/TNBC)")
        log("=" * 70)
        adata_backed = sc.read_h5ad(RAW_FILE, backed="r")
        log(f"  Raw: {adata_backed.n_obs:,} cells x {adata_backed.n_vars:,} genes")

        mask = (adata_backed.obs["subtype"].isin(KEEP_SUBTYPES) &
                adata_backed.obs["tissue"].isin(KEEP_TISSUE))
        adata = adata_backed[mask].to_memory()
        del adata_backed; gc.collect()
        log(f"  After tissue/subtype filter: {adata.n_obs:,} cells")
        log(adata.obs["subtype"].value_counts().to_string())

        # ── 2. GENE SYMBOLS ─────────────────────────────────────────────
        log("\n" + "=" * 70)
        log(" 2. ENSEMBL -> SYMBOL")
        log("=" * 70)
        adata.var["ensembl_id"] = adata.var_names.tolist()
        symbols = [s if (isinstance(s, str) and s.strip() and not s.startswith("ENSG")) else e
                   for s, e in zip(adata.var["gene_name"], adata.var["ensembl_id"])]
        seen, uniq = {}, []
        for s in symbols:
            if s in seen:
                seen[s] += 1
                uniq.append(f"{s}_{seen[s]}")
            else:
                seen[s] = 0
                uniq.append(s)
        adata.var["symbol"] = uniq
        adata.var_names = uniq
        adata.var_names_make_unique()
        for g in GENES:
            assert g in adata.var_names, f"{g} not found after symbol conversion!"
        log(f"  Confirmed present: {GENES}")

        # ── 3. QC METRICS + FILTER ───────────────────────────────────────
        log("\n" + "=" * 70)
        log(" 3. QC FILTER")
        log("=" * 70)
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                    log1p=False, inplace=True)

        p99_per_sample = adata.obs.groupby("sample")["n_genes_by_counts"].quantile(0.99)
        sample_str = adata.obs["sample"].astype(str)
        p99_map = p99_per_sample.copy(); p99_map.index = p99_map.index.astype(str)
        mask_max = (adata.obs["n_genes_by_counts"] <= sample_str.map(p99_map)).values

        qc_mask = (
            (adata.obs["n_genes_by_counts"] >= MIN_GENES) & mask_max &
            (adata.obs["total_counts"] >= MIN_COUNTS) &
            (adata.obs["pct_counts_mt"] <= MAX_MT)
        )
        n_before = adata.n_obs
        adata = adata[qc_mask].copy()
        sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
        log(f"  Cells: {n_before:,} -> {adata.n_obs:,}")
        log(adata.obs["subtype"].value_counts().to_string())
        log(f"  Samples: {adata.obs['sample'].nunique()}")

        # tissue_group: Normal vs Tumor (pooled ER/HER2/TNBC)
        adata.obs["tissue_group"] = np.where(adata.obs["subtype"] == "Normal", "Normal", "Tumor")
        adata.obs["tissue_group"] = adata.obs["tissue_group"].astype("category")

        log(f"\n  Saving QC checkpoint (resume point if a later step fails)...")
        adata.write_h5ad(CKPT_QC)
        log(f"  ✓ {CKPT_QC}")

    # ── 4. NORMALIZE ─────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log(" 4. NORMALIZE  (normalize_total(1e4) + log1p)")
    log("=" * 70)
    # NOTE: no separate 'counts' layer here -- raw counts are still on disk
    # in CKPT_QC / RAW_FILE if ever needed, and keeping a duplicate in RAM
    # through this step nearly doubled peak memory for no downstream use.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = adata.X.astype(np.float32)
    adata.layers["logcounts"] = adata.X
    log("  Done. layers['logcounts'] holds the UNSCALED values used for all")
    log("  expression reporting (UMAP color + violins) below.")

    # ── 5. HVG + PCA + HARMONY + UMAP (embedding only) ──────────────────
    log("\n" + "=" * 70)
    log(" 5. HVG -> scale -> PCA -> Harmony -> neighbors -> UMAP")
    log("=" * 70)
    n_samples = adata.obs["sample"].nunique()
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat",
                                 batch_key="sample", subset=False)
    log(f"  HVGs: {adata.var['highly_variable'].sum():,}")

    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    ad_hvg.X = ad_hvg.layers["logcounts"].copy()
    sc.pp.scale(ad_hvg, max_value=10)
    sc.tl.pca(ad_hvg, svd_solver="arpack", n_comps=N_PCS)
    adata.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    adata.uns["pca"] = ad_hvg.uns["pca"]
    del ad_hvg; gc.collect()
    log("  PCA done")

    import harmonypy as hm
    log(f"  Running Harmony (batch=sample, {n_samples} samples)...")
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs[["sample"]],
                         "sample", max_iter_harmony=20, random_state=42, verbose=False)
    Z = ho.Z_corr
    if Z.shape[0] == N_PCS and Z.shape[1] == adata.n_obs:
        Z = Z.T
    adata.obsm["X_pca_harmony"] = Z
    log("  Harmony done")

    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=N_PCS, n_neighbors=15)
    log("  Computing UMAP...")
    sc.tl.umap(adata)
    log("  UMAP done")

    # adata.X was never touched by the HVG/PCA/Harmony block above (that
    # ran on the independent ad_hvg copy), so it's still the unscaled
    # log-normalized matrix -- no need to restore it from layers here.

    # ── 6. SAVE intermediate (so plots can be rebuilt without recompute) ─
    log("\n  Saving intermediate object...")
    adata.write_h5ad("visvader_normal_vs_tumor.h5ad")
    log("  ✓ visvader_normal_vs_tumor.h5ad")

    make_plots(adata)


def make_plots(adata):
    coords = adata.obsm["X_umap"]
    rng = np.random.RandomState(0)
    order = rng.permutation(adata.n_obs)

    # ── UMAP: tissue group + subtype + gene expression, one combined row ─
    log("\n  Plotting combined UMAP panel...")
    fig, axes = plt.subplots(1, 5, figsize=(30, 5.6))

    ax = axes[0]
    grp = adata.obs["tissue_group"].values[order]
    for g, c in GROUP_COLORS.items():
        m = grp == g
        ax.scatter(coords[order][m, 0], coords[order][m, 1], s=1.0, alpha=0.5,
                   color=c, label=f"{g} (n={m.sum():,})", rasterized=True)
    ax.legend(fontsize=8, markerscale=8, loc="best")
    ax.set_title("Normal vs Tumor"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[1]
    sub = adata.obs["subtype"].values[order]
    for s, c in SUBTYPE_COLORS.items():
        m = sub == s
        ax.scatter(coords[order][m, 0], coords[order][m, 1], s=1.0, alpha=0.5,
                   color=c, label=f"{s} (n={m.sum():,})", rasterized=True)
    ax.legend(fontsize=8, markerscale=8, loc="best")
    ax.set_title("Subtype"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    # gene expression panels: single-hue (monochromatic) colormap so they
    # read as a magnitude ramp rather than competing with the categorical
    # colors used in the two panels above
    GENE_CMAP = "Purples"
    for ax, gene in zip(axes[2:], GENES):
        expr = adata[:, gene].X
        expr = np.asarray(expr.todense()).ravel() if hasattr(expr, "todense") else np.asarray(expr).ravel()
        vmax = np.percentile(expr, 99) if expr.max() > 0 else 1
        plot_ord = np.argsort(expr)  # low first, so high-expressors draw on top
        sc_p = ax.scatter(coords[plot_ord, 0], coords[plot_ord, 1], c=expr[plot_ord],
                          s=1.0, alpha=0.8, cmap=GENE_CMAP, vmin=0, vmax=vmax, rasterized=True)
        plt.colorbar(sc_p, ax=ax, label="log1p(CP10K)")
        ax.set_title(f"{gene}  ({(expr>0).mean()*100:.1f}% expressing)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.suptitle(f"Visvader (GSE161529) -- Normal + primary tumor, tissue=Total, "
                 f"n={adata.n_obs:,}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_umap_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ {OUTDIR}/fig_umap_combined.png")

    # ── Violins + stats: Normal vs Tumor ────────────────────────────────
    log("  Violin plots + Mann-Whitney U (Normal vs Tumor)...")
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, gene in zip(axes, GENES):
        expr = np.asarray(adata[:, gene].X.todense()).ravel()
        vn = expr[(adata.obs["tissue_group"] == "Normal").values]
        vt = expr[(adata.obs["tissue_group"] == "Tumor").values]
        u, p = sstats.mannwhitneyu(vn, vt, alternative="two-sided")
        parts = ax.violinplot([vn, vt], showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], [GROUP_COLORS["Normal"], GROUP_COLORS["Tumor"]]):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"Normal\n(n={len(vn):,})", f"Tumor\n(n={len(vt):,})"])
        ax.set_ylabel("log1p(CP10K)")
        # genes are mostly-dropout (most cells at 0), so the KDE tail stretches
        # the auto ylim far past where the data actually sit -- clip to the
        # 99.5th percentile of the pooled data instead
        y_top = np.percentile(np.concatenate([vn, vt]), 97)
        y_top = y_top if y_top > 0 else max(vn.max(), vt.max(), 1e-6)
        ax.set_ylim(-0.08 * y_top, 1.15 * y_top)
        p_str = "p<1e-300" if p == 0 else f"p={p:.2e}"
        ax.set_title(f"{gene}\n{p_str}", fontsize=10)
        rows.append(dict(gene=gene, n_normal=len(vn), n_tumor=len(vt),
                          mean_normal=vn.mean(), mean_tumor=vt.mean(),
                          median_normal=np.median(vn), median_tumor=np.median(vt),
                          pct_expr_normal=100*(vn > 0).mean(), pct_expr_tumor=100*(vt > 0).mean(),
                          mannwhitney_U=u, pvalue=p))
    fig.suptitle("Visvader -- BACH1/PPARD/KLF5, Normal vs Tumor (pooled ER/HER2/TNBC)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_violin_normal_tumor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    stats_df = pd.DataFrame(rows)
    stats_df["pvalue_BH"] = sstats.false_discovery_control(stats_df["pvalue"]) if hasattr(sstats, "false_discovery_control") else stats_df["pvalue"]
    stats_df.to_csv(f"{OUTDIR}/stats_normal_vs_tumor.csv", index=False)
    log(f"  ✓ {OUTDIR}/fig_violin_normal_tumor.png")
    log(f"  ✓ {OUTDIR}/stats_normal_vs_tumor.csv")
    log(stats_df.to_string(index=False))

    # ── Violins + stats: by fine subtype ────────────────────────────────
    log("  Violin plots by subtype (Normal/ER/HER2/TNBC)...")
    subtype_order = ["Normal", "ER", "HER2", "TNBC"]
    rows2 = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, gene in zip(axes, GENES):
        expr = np.asarray(adata[:, gene].X.todense()).ravel()
        groups = [expr[(adata.obs["subtype"] == s).values] for s in subtype_order]
        kw_stat, kw_p = sstats.kruskal(*groups)
        parts = ax.violinplot(groups, showmedians=True, showextrema=False)
        for pc, s in zip(parts["bodies"], subtype_order):
            pc.set_facecolor(SUBTYPE_COLORS[s]); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(subtype_order)+1))
        ax.set_xticklabels([f"{s}\n(n={len(g):,})" for s, g in zip(subtype_order, groups)], fontsize=8)
        ax.set_ylabel("log1p(CP10K)")
        y_top = np.percentile(np.concatenate(groups), 97)
        y_top = y_top if y_top > 0 else max(g.max() for g in groups)
        ax.set_ylim(-0.08 * y_top, 1.15 * y_top)
        ax.set_title(f"{gene}\nKruskal-Wallis p={kw_p:.2e}", fontsize=10)
        for s, g in zip(subtype_order, groups):
            if s == "Normal":
                continue
            u, p = sstats.mannwhitneyu(groups[0], g, alternative="two-sided")
            rows2.append(dict(gene=gene, comparison=f"Normal_vs_{s}",
                              n_normal=len(groups[0]), n_group=len(g),
                              mean_normal=groups[0].mean(), mean_group=g.mean(),
                              mannwhitney_U=u, pvalue=p))
    fig.suptitle("Visvader -- BACH1/PPARD/KLF5 by subtype", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_violin_by_subtype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    stats_df2 = pd.DataFrame(rows2)
    if len(stats_df2):
        stats_df2["pvalue_BH"] = sstats.false_discovery_control(stats_df2["pvalue"]) if hasattr(sstats, "false_discovery_control") else stats_df2["pvalue"]
    stats_df2.to_csv(f"{OUTDIR}/stats_by_subtype.csv", index=False)
    log(f"  ✓ {OUTDIR}/fig_violin_by_subtype.png")
    log(f"  ✓ {OUTDIR}/stats_by_subtype.csv")

    log("\n" + "=" * 70)
    log(" DONE")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
