"""
03_bach1_ppard_klf5_normal_vs_tumor.py
========================================
BACH1 / PPARD / KLF5 expression: normal breast tissue vs primary tumor vs
breast cancer cell lines, Regner cohort (GSE243526).

Cell selection
--------------
obs['tissue'] in {"normal breast tissue", "primary breast tumor",
"breast cancer cell line"}. Cell lines present: HCC1143, MCF7, SUM149PT
(T47D RNA was never deposited in GSE243526_RAW.tar -- GEO lists a GSM for
it but only the ATAC data exists at the source; see memory
project_regner_dataset). Cell-line identity comes from obs['patient'],
which holds the line name directly for these rows (obs['subtype'] is only
the coarse "cell_line" label).

Normalization (validity of the comparison)
-------------------------------------------
Reuses layers["logcounts"] from 02_qc_filter.py, i.e.
normalize_total(1e4) + log1p computed on the full processed object.
This is valid to reuse on any subset: normalize_total rescales each cell
by ITS OWN total counts, so the value for a given cell does not depend on
which other cells are present in the object. All expression reporting
below (UMAP color + violins) uses this unscaled log-normalized matrix, so
magnitudes stay comparable between Normal, Tumor and CellLine groups.

UMAP embedding
---------------
The embedding already stored in gse243526_processed.h5ad was fit
JOINTLY with the cell lines and is explicitly documented as
"unintegrated -- QC only" (02_qc_filter.py); it is not reused here. A
fresh HVG -> scale -> PCA -> Harmony(batch=sample_id) -> neighbors -> UMAP
is computed on the Normal+Tumor+CellLine subset, matching the recipe used
for the Visvader companion script
(21_bach1_ppard_klf5_normal_vs_tumor.py). Caveat: Harmony is asked to
align 3 cell-line "batches" (1 each) against 16 patient-tissue batches --
cell lines are clonal/cultured and biologically very different from
primary tissue, so expect them to still separate out substantially in the
embedding even after correction. That's a real biological signal, not a
batch-correction failure, and is exactly why they're shown as their own
group/color rather than blended in.

OUTPUTS
-------
figures/03_BACH1_PPARD_KLF5_normal_vs_tumor/
  fig_umap_combined.png     UMAP panels side-by-side: Normal/Tumor/CellLine,
                             subtype (incl. per-line), BACH1/PPARD/KLF5
                             expression (monochromatic)
  fig_violin_by_group.png       violins, Normal vs Tumor vs CellLine
  fig_violin_by_subtype.png     violins, normal/HR+/HER2-eq/TNBC/HCC1143/MCF7/SUM149PT
  stats_by_group.csv
  stats_by_subtype.csv

Run from: ~/single_cell/Regner/   (takes ~15-30 min, background-safe)
  python3 03_bach1_ppard_klf5_normal_vs_tumor.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats
import os, warnings, gc, sys, traceback

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.n_jobs = 8

FILE   = "gse243526_processed.h5ad"
OUTDIR = "figures/03_BACH1_PPARD_KLF5_normal_vs_tumor"
os.makedirs(OUTDIR, exist_ok=True)

GENES = ["BACH1", "PPARD", "KLF5"]
KEEP_TISSUE = ["normal breast tissue", "primary breast tumor", "breast cancer cell line"]

GROUP_COLORS = {"Normal": "#4C72B0", "Tumor": "#C44E52", "CellLine": "#55A868"}
SUBTYPE_COLORS = {"normal": "#4C72B0", "HR+/HER2-": "#4DAF4A",
                   "HER2-equivocal": "#984EA3", "TNBC": "#E41A1C",
                   "HCC1143": "#FF7F00", "MCF7": "#A65628", "SUM149PT": "#999999"}
SUBTYPE_ORDER = ["normal", "HR+/HER2-", "HER2-equivocal", "TNBC",
                  "HCC1143", "MCF7", "SUM149PT"]

N_HVG, N_PCS = 3000, 30


def log(msg):
    print(msg, flush=True)


def main():
    # ── 1. LOAD (backed) + SUBSET ───────────────────────────────────────
    log("=" * 70)
    log(" 1. LOAD (backed) + SUBSET to Normal + Primary tumor + Cell lines")
    log("=" * 70)
    backed = sc.read_h5ad(FILE, backed="r")
    log(f"  Full object: {backed.n_obs:,} cells x {backed.n_vars:,} genes")
    mask = backed.obs["tissue"].isin(KEEP_TISSUE).values
    adata = backed[mask].to_memory()
    del backed; gc.collect()
    log(f"  Subset: {adata.n_obs:,} cells")
    log(adata.obs["tissue"].value_counts().to_string())
    log(adata.obs["subtype"].value_counts().to_string())

    for g in GENES:
        assert g in adata.var_names, f"{g} not found!"

    # Drop stale full-object derived fields (fit on cell lines + tissue together)
    for key in ["X_pca", "X_umap"]:
        adata.obsm.pop(key, None)
    for key in ["connectivities", "distances"]:
        adata.obsp.pop(key, None)
    for key in ["neighbors", "umap", "pca", "leiden_0.5"]:
        adata.uns.pop(key, None)

    adata.obs["tissue_group"] = np.select(
        [adata.obs["tissue"] == "normal breast tissue",
         adata.obs["tissue"] == "breast cancer cell line"],
        ["Normal", "CellLine"], default="Tumor"
    )
    adata.obs["tissue_group"] = adata.obs["tissue_group"].astype("category")

    # subtype: "normal" for the Normal group, per-line name (from obs['patient'])
    # for CellLine, existing IHC-based subtype for Tumor
    adata.obs["subtype"] = adata.obs["subtype"].astype(str)
    adata.obs.loc[adata.obs["tissue_group"] == "Normal", "subtype"] = "normal"
    is_cl = (adata.obs["tissue_group"] == "CellLine").values
    adata.obs.loc[is_cl, "subtype"] = adata.obs.loc[is_cl, "patient"].astype(str).values
    adata.obs["subtype"] = adata.obs["subtype"].astype("category")
    log(f"\n  tissue_group counts:\n{adata.obs['tissue_group'].value_counts().to_string()}")
    log(f"\n  subtype counts:\n{adata.obs['subtype'].value_counts().to_string()}")

    # ── 2. HVG + PCA + HARMONY + UMAP (embedding only) ──────────────────
    log("\n" + "=" * 70)
    log(" 2. HVG -> scale -> PCA -> Harmony -> neighbors -> UMAP")
    log("=" * 70)
    adata.X = adata.layers["logcounts"].copy()
    n_samples = adata.obs["sample_id"].nunique()
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat",
                                 batch_key="sample_id", subset=False)
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
    log(f"  Running Harmony (batch=sample_id, {n_samples} samples)...")
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs[["sample_id"]],
                         "sample_id", max_iter_harmony=20, random_state=42, verbose=False)
    Z = ho.Z_corr
    if Z.shape[0] == N_PCS and Z.shape[1] == adata.n_obs:
        Z = Z.T
    adata.obsm["X_pca_harmony"] = Z
    log("  Harmony done")

    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=N_PCS, n_neighbors=15)
    log("  Computing UMAP...")
    sc.tl.umap(adata)
    log("  UMAP done")

    # Restore unscaled log-normalized expression for plotting
    adata.X = adata.layers["logcounts"].copy()

    log("\n  Saving intermediate object...")
    adata.write_h5ad("regner_normal_vs_tumor.h5ad")
    log("  ✓ regner_normal_vs_tumor.h5ad")

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
        ax.scatter(coords[order][m, 0], coords[order][m, 1], s=1.5, alpha=0.5,
                   color=c, label=f"{g} (n={m.sum():,})", rasterized=True)
    ax.legend(fontsize=8, markerscale=6, loc="best")
    ax.set_title("Normal vs Tumor vs CellLine"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[1]
    sub = adata.obs["subtype"].values[order]
    for s, c in SUBTYPE_COLORS.items():
        m = sub == s
        if not m.any():
            continue
        ax.scatter(coords[order][m, 0], coords[order][m, 1], s=1.5, alpha=0.5,
                   color=c, label=f"{s} (n={m.sum():,})", rasterized=True)
    ax.legend(fontsize=7, markerscale=6, loc="best")
    ax.set_title("Subtype (IHC-based; HER2-equivocal = ambiguous)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    # gene expression panels: single-hue (monochromatic) colormap so they
    # read as a magnitude ramp rather than competing with the categorical
    # colors used in the two panels above
    GENE_CMAP = "Purples"
    for ax, gene in zip(axes[2:], GENES):
        expr = adata[:, gene].X
        expr = np.asarray(expr.todense()).ravel() if hasattr(expr, "todense") else np.asarray(expr).ravel()
        vmax = np.percentile(expr, 99) if expr.max() > 0 else 1
        plot_ord = np.argsort(expr)
        sc_p = ax.scatter(coords[plot_ord, 0], coords[plot_ord, 1], c=expr[plot_ord],
                          s=1.5, alpha=0.8, cmap=GENE_CMAP, vmin=0, vmax=vmax, rasterized=True)
        plt.colorbar(sc_p, ax=ax, label="log1p(CP10K)")
        ax.set_title(f"{gene}  ({(expr>0).mean()*100:.1f}% expressing)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.suptitle(f"Regner (GSE243526) -- Normal + primary tumor + cell lines, "
                 f"n={adata.n_obs:,}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_umap_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ {OUTDIR}/fig_umap_combined.png")

    # ── Violins + stats: Normal vs Tumor vs CellLine ─────────────────────
    log("  Violin plots + Kruskal-Wallis (Normal vs Tumor vs CellLine)...")
    group_order = ["Normal", "Tumor", "CellLine"]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, gene in zip(axes, GENES):
        expr = np.asarray(adata[:, gene].X.todense()).ravel()
        groups = [expr[(adata.obs["tissue_group"] == g).values] for g in group_order]
        kw_stat, kw_p = sstats.kruskal(*groups)
        parts = ax.violinplot(groups, showmedians=True, showextrema=False)
        for pc, g in zip(parts["bodies"], group_order):
            pc.set_facecolor(GROUP_COLORS[g]); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(group_order)+1))
        ax.set_xticklabels([f"{g}\n(n={len(v):,})" for g, v in zip(group_order, groups)])
        ax.set_ylabel("log1p(CP10K)")
        # genes are mostly-dropout (most cells at 0), so the KDE tail stretches
        # the auto ylim far past where the data actually sit -- clip to the
        # 97th percentile of the pooled data instead
        y_top = np.percentile(np.concatenate(groups), 97)
        y_top = y_top if y_top > 0 else max(v.max() for v in groups)
        ax.set_ylim(-0.08 * y_top, 1.15 * y_top)
        ax.set_title(f"{gene}\nKruskal-Wallis p={kw_p:.2e}", fontsize=10)
        for i, g in enumerate(group_order):
            for g2 in group_order[i+1:]:
                v1, v2 = groups[group_order.index(g)], groups[group_order.index(g2)]
                u, p = sstats.mannwhitneyu(v1, v2, alternative="two-sided")
                rows.append(dict(gene=gene, comparison=f"{g}_vs_{g2}",
                                  n1=len(v1), n2=len(v2), mean1=v1.mean(), mean2=v2.mean(),
                                  median1=np.median(v1), median2=np.median(v2),
                                  pct_expr1=100*(v1 > 0).mean(), pct_expr2=100*(v2 > 0).mean(),
                                  mannwhitney_U=u, pvalue=p))
    fig.suptitle("Regner -- BACH1/PPARD/KLF5, Normal vs Tumor vs CellLine",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_violin_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    stats_df = pd.DataFrame(rows)
    stats_df["pvalue_BH"] = sstats.false_discovery_control(stats_df["pvalue"]) if hasattr(sstats, "false_discovery_control") else stats_df["pvalue"]
    stats_df.to_csv(f"{OUTDIR}/stats_by_group.csv", index=False)
    log(f"  ✓ {OUTDIR}/fig_violin_by_group.png")
    log(f"  ✓ {OUTDIR}/stats_by_group.csv")
    log(stats_df.to_string(index=False))

    # ── Violins + stats: by fine subtype (incl. individual cell lines) ──
    log("  Violin plots by subtype...")
    subtype_order = [s for s in SUBTYPE_ORDER if (adata.obs["subtype"] == s).any()]
    rows2 = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, gene in zip(axes, GENES):
        expr = np.asarray(adata[:, gene].X.todense()).ravel()
        groups = [expr[(adata.obs["subtype"] == s).values] for s in subtype_order]
        kw_stat, kw_p = sstats.kruskal(*groups)
        parts = ax.violinplot(groups, showmedians=True, showextrema=False)
        for pc, s in zip(parts["bodies"], subtype_order):
            pc.set_facecolor(SUBTYPE_COLORS[s]); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(subtype_order)+1))
        ax.set_xticklabels([f"{s}\n(n={len(g):,})" for s, g in zip(subtype_order, groups)], fontsize=7)
        ax.set_ylabel("log1p(CP10K)")
        y_top = np.percentile(np.concatenate(groups), 97)
        y_top = y_top if y_top > 0 else max(g.max() for g in groups)
        ax.set_ylim(-0.08 * y_top, 1.15 * y_top)
        ax.set_title(f"{gene}\nKruskal-Wallis p={kw_p:.2e}", fontsize=10)
        for s, g in zip(subtype_order, groups):
            if s == "normal":
                continue
            u, p = sstats.mannwhitneyu(groups[0], g, alternative="two-sided")
            rows2.append(dict(gene=gene, comparison=f"normal_vs_{s}",
                              n_normal=len(groups[0]), n_group=len(g),
                              mean_normal=groups[0].mean(), mean_group=g.mean(),
                              mannwhitney_U=u, pvalue=p))
    fig.suptitle("Regner -- BACH1/PPARD/KLF5 by subtype "
                 "(HER2-equivocal = IHC-ambiguous; HCC1143/MCF7/SUM149PT = cell lines)",
                 fontsize=12, fontweight="bold")
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
