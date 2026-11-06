"""
02_qc_filter.py
================
QC, filtering and normalization on the Regner (GSE243526) raw object.

Steps:
  1. Load gse243526_raw.h5ad (from 01_load_samples.py)
  2. QC metrics: MT / ribosomal / hemoglobin gene percentages
  3. Pre-filter QC figures (overall + per sample)
  4. Filter cells:
       MIN_GENES  = 200
       MAX_GENES  = p99 per sample (handles cell lines' much higher
                    baseline complexity vs. patient tumors — a single
                    global cutoff would unfairly penalize them)
       MIN_COUNTS = 500
       MAX_MT     = 30%
     Filter genes:
       MIN_CELLS  = 3
  5. Normalize (normalize_total 1e4 + log1p) — raw counts kept in
     layers["counts"], log-normalized in layers["logcounts"]
  6. HVGs (batch_key=sample_name) + PCA + UMAP + Leiden — unintegrated,
     for QC inspection only (real integration comes in 03)

Notes:
  - No subtype (ER/HER2/TNBC) annotation exists for this dataset yet —
    grouping/coloring uses sample_name / patient / is_cell_line instead
  - Patient_11 flagged in 01_load_samples.py: notably low complexity
    (median 771 genes, 2,262 UMIs) — watch its cell count after filtering
  - Cell lines (HCC1143, MCF7, SUM149PT) have ~1.5-2x the genes/UMIs of
    patient tumors — expected (immortalized, no stromal/immune admixture)

Input:  gse243526_raw.h5ad
Output: gse243526_processed.h5ad

Run from: ~/single_cell/Regner/
  python3 02_qc_filter.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, gc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.n_jobs = 4

FILE       = "gse243526_raw.h5ad"
OUTFILE    = "gse243526_processed.h5ad"
OUTDIR     = "figures/02_qc"
SAMPLE_COL = "sample_name"
os.makedirs(OUTDIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD")
print("="*65)
adata = sc.read_h5ad(FILE)
print(adata)
print(f"\n  Cells   : {adata.n_obs:,}")
print(f"  Genes   : {adata.n_vars:,}")
print(f"  Samples : {adata.obs[SAMPLE_COL].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. QC METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. QC METRICS")
print("="*65)

adata.var["mt"]   = adata.var_names.str.upper().str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
adata.var["hb"]   = (adata.var_names.str.upper().str.startswith("HB") &
                     ~adata.var_names.str.upper().str.startswith("HBP"))

print(f"  MT genes  : {adata.var['mt'].sum()}")
print(f"  Ribo genes: {adata.var['ribo'].sum()}")
print(f"  HB genes  : {adata.var['hb'].sum()}")

if adata.var["mt"].sum() == 0:
    print("  WARNING: No MT genes found — check gene symbol conversion")

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"],
    percent_top=None, log1p=False, inplace=True
)

print(f"\n  QC summary:")
print(adata.obs[["n_genes_by_counts", "total_counts",
                  "pct_counts_mt", "pct_counts_ribo"]]
      .describe().round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRE-FILTER FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating pre-filter QC figures...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("QC metrics — before filtering (Regner / GSE243526)",
             fontsize=13, fontweight="bold")

for i, (metric, label, color) in enumerate([
    ("n_genes_by_counts", "N genes / cell",    "#4C72B0"),
    ("total_counts",      "Total UMIs / cell", "#DD8452"),
    ("pct_counts_mt",     "% MT counts",       "#C44E52"),
    ("pct_counts_ribo",   "% ribosomal",       "#8172B2"),
]):
    ax = axes[i // 3][i % 3]
    vals = adata.obs[metric]
    ax.hist(vals, bins=100, color=color, alpha=0.7, edgecolor="none")
    ax.axvline(vals.median(), color="black", ls="--", lw=1.5,
               label=f"Median: {vals.median():.1f}")
    if "mt" in metric:
        ax.axvline(30, color="red", ls=":", lw=1.5, label="cutoff=30%")
    elif "genes" in metric:
        ax.axvline(200, color="red", ls=":", lw=1.5, label="min=200")
    ax.set_title(label); ax.set_xlabel(label)
    ax.set_ylabel("N cells"); ax.legend(fontsize=8)

ax = axes[1][1]
ax.scatter(adata.obs["pct_counts_mt"], adata.obs["n_genes_by_counts"],
           s=0.3, alpha=0.15, c="#4C72B0", rasterized=True)
ax.axvline(30, color="red", ls="--", lw=1.5, label="MT cutoff")
ax.set_xlabel("% MT"); ax.set_ylabel("N genes")
ax.set_title("N genes vs % MT"); ax.legend(fontsize=8)

ax = axes[1][2]
sc_p = ax.scatter(adata.obs["total_counts"], adata.obs["n_genes_by_counts"],
                  s=0.3, alpha=0.15, c=adata.obs["pct_counts_mt"],
                  cmap="RdYlGn_r", vmax=30, rasterized=True)
plt.colorbar(sc_p, ax=ax, label="% MT")
ax.set_xlabel("Total UMIs"); ax.set_ylabel("N genes")
ax.set_title("N genes vs UMIs (color = % MT)")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/pre_filter_qc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/pre_filter_qc.png")

# Violin per sample (patients vs cell lines), like 01's overview but post-QC-metrics
order = (adata.obs.groupby(SAMPLE_COL, observed=True)["sample_id"]
         .first().sort_values().index.tolist())
order = sorted(order, key=lambda s: int(
    adata.obs.loc[adata.obs[SAMPLE_COL] == s, "sample_id"].iloc[0][3:]))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("QC metrics by sample — before filtering",
             fontsize=12, fontweight="bold")
for ax, col, title in zip(
    axes,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    ["Genes / cell", "UMI counts / cell", "% mitochondrial"],
):
    data = [adata.obs.loc[adata.obs[SAMPLE_COL] == s, col] for s in order]
    colors = ["#DD8452" if adata.obs.loc[adata.obs[SAMPLE_COL] == s,
              "is_cell_line"].iloc[0] else "#4C72B0" for s in order]
    parts = ax.violinplot(data, showmedians=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_title(title)
    if col == "total_counts":
        ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pre_filter_qc_by_sample.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/pre_filter_qc_by_sample.png  (blue=patient tumor, orange=cell line)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FILTER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 4. FILTER")
print("="*65)

MIN_GENES  = 200
MIN_COUNTS = 500
MAX_MT     = 30
MIN_CELLS  = 3

sample_str = adata.obs[SAMPLE_COL].astype(str)
p99_per_sample = adata.obs.groupby(sample_str)["n_genes_by_counts"].quantile(0.99)
mask_genes_max = (adata.obs["n_genes_by_counts"] <=
                  sample_str.map(p99_per_sample)).values
print(f"  MAX_GENES: p99 per sample "
      f"(range: {p99_per_sample.min():.0f}–{p99_per_sample.max():.0f})")

mask = (
    (adata.obs["n_genes_by_counts"] >= MIN_GENES) &
    mask_genes_max &
    (adata.obs["total_counts"]  >= MIN_COUNTS) &
    (adata.obs["pct_counts_mt"] <= MAX_MT)
)

n_before = adata.n_obs
print(f"\n  Filters applied:")
print(f"    n_genes  >= {MIN_GENES}      : "
      f"{(adata.obs['n_genes_by_counts'] < MIN_GENES).sum():,} removed")
print(f"    n_genes  <= p99 per sample : "
      f"{(~mask_genes_max).sum():,} removed")
print(f"    n_counts >= {MIN_COUNTS}      : "
      f"{(adata.obs['total_counts'] < MIN_COUNTS).sum():,} removed")
print(f"    pct_mt   <= {MAX_MT}%        : "
      f"{(adata.obs['pct_counts_mt'] > MAX_MT).sum():,} removed")

adata = adata[mask].copy()
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

print(f"\n  Cells : {n_before:,} -> {adata.n_obs:,} "
      f"(removed {n_before - adata.n_obs:,}, "
      f"{100 * (n_before - adata.n_obs) / n_before:.1f}%)")
print(f"  Genes : {adata.n_vars:,} (min_cells={MIN_CELLS})")

print(f"\n  Cells per sample after filtering:")
print(adata.obs.groupby(SAMPLE_COL, observed=True).size()
      .reindex(order).to_string())

# Post-filter QC figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("QC metrics — after filtering", fontsize=12, fontweight="bold")
for ax, (metric, label, color) in zip(axes, [
    ("n_genes_by_counts", "N genes / cell",    "#4C72B0"),
    ("total_counts",      "Total UMIs / cell", "#DD8452"),
    ("pct_counts_mt",     "% MT counts",       "#C44E52"),
]):
    vals = adata.obs[metric]
    ax.hist(vals, bins=100, color=color, alpha=0.7, edgecolor="none")
    ax.axvline(vals.median(), color="black", ls="--", lw=1.5,
               label=f"Median: {vals.median():.1f}")
    ax.set_title(label); ax.set_xlabel(label)
    ax.set_ylabel("N cells"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/post_filter_qc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/post_filter_qc.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 5. NORMALIZE")
print("="*65)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()

print(f"  normalize_total(1e4) + log1p applied")
print(f"  Raw counts -> layers['counts']")
print(f"  Log counts -> layers['logcounts']")


# ══════════════════════════════════════════════════════════════════════════════
# 6. HVGs + PCA + UMAP  (unintegrated — for QC inspection only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 6. HVGs + PCA + UMAP  (unintegrated, QC only)")
print("="*65)

sc.pp.highly_variable_genes(
    adata, n_top_genes=3000, flavor="seurat",
    batch_key=SAMPLE_COL, subset=False
)
n_hvg = adata.var["highly_variable"].sum()
print(f"  HVGs: {n_hvg:,} (flavor=seurat, batch_key={SAMPLE_COL})")

adata_hvg   = adata[:, adata.var["highly_variable"]].copy()
adata_hvg.X = adata_hvg.layers["logcounts"].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=50)
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.uns["pca"]    = adata_hvg.uns["pca"]
del adata_hvg; gc.collect()
print(f"  PCA (50 components)")

fig, ax = plt.subplots(figsize=(8, 4))
var_ratio = adata.uns["pca"]["variance_ratio"]
ax.plot(range(1, len(var_ratio) + 1), np.cumsum(var_ratio) * 100,
        marker="o", ms=3, color="#4C72B0")
ax.axhline(80, color="red", ls="--", lw=1, label="80% variance")
ax.set_xlabel("N PCA components")
ax.set_ylabel("Cumulative variance (%)")
ax.set_title("PCA variance explained")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/pca_variance.png")

N_PCS = 30
sc.pp.neighbors(adata, n_pcs=N_PCS, n_neighbors=15)
print("  Computing UMAP...")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added="leiden_0.5")
print(f"  UMAP  |  Leiden clusters: {adata.obs['leiden_0.5'].nunique()}")

coords = adata.obsm["X_umap"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("UMAP — post QC, unintegrated (Regner / GSE243526)",
             fontsize=12, fontweight="bold")

ax = axes[0]
cats = adata.obs[SAMPLE_COL].astype("category")
cmap = plt.cm.get_cmap("tab20", cats.nunique())
ax.scatter(coords[:, 0], coords[:, 1], c=cats.cat.codes,
           s=0.3, alpha=0.3, cmap=cmap, rasterized=True)
ax.set_title(f"Sample ({cats.nunique()} samples)")
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
for label, color in [("Patient tumor", "#4C72B0"), ("Cell line", "#DD8452")]:
    m = (adata.obs["is_cell_line"] == (label == "Cell line")).values
    ax.scatter(coords[m, 0], coords[m, 1], c=color, s=0.3, alpha=0.3,
               label=label, rasterized=True)
ax.legend(fontsize=8)
ax.set_title("Patient tumor vs cell line")
ax.set_xticks([]); ax.set_yticks([])

ax = axes[2]
sc_p = ax.scatter(coords[:, 0], coords[:, 1], c=adata.obs["n_genes_by_counts"],
                  s=0.3, alpha=0.3, cmap="viridis", rasterized=True)
plt.colorbar(sc_p, ax=ax, label="N genes")
ax.set_title("N genes per cell")
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUTDIR}/umap_unintegrated.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/umap_unintegrated.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 7. SAVE")
print("="*65)

adata.X = adata.layers["logcounts"].copy()
adata.write_h5ad(OUTFILE)

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Output  : {OUTFILE}
  Cells   : {adata.n_obs:,}
  Genes   : {adata.n_vars:,}
  Layers  : counts, logcounts
  Samples : {adata.obs[SAMPLE_COL].nunique()} ({(~adata.obs['is_cell_line']).sum():,} patient-tumor cells, {adata.obs['is_cell_line'].sum():,} cell-line cells)

  Figures : {OUTDIR}/
    pre_filter_qc.png
    pre_filter_qc_by_sample.png
    post_filter_qc.png
    pca_variance.png
    umap_unintegrated.png

  Next: decide integration strategy (Harmony, batch=sample_name) and
        write 03_split_integrate.py — likely separate patient tumors
        from cell lines before pooling
══════════════════════════════════════════════════════════════
""")
