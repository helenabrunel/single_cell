"""
02_qc_filter_normalize.py
=========================
QC, filtering and normalization on tumor-only raw object.

Steps:
  1. Convert Ensembl IDs → Gene Symbols (gene_name column or MyGeneInfo)
  2. Recalculate QC metrics with correct MT gene identification
  3. Filter low-quality cells and genes
  4. Normalize (log1p) — save raw counts in layers["counts"]
  5. HVGs + PCA + UMAP (unintegrated, for QC inspection only)

Parameters:
  MIN_GENES  = 200
  MAX_GENES  = p99 per sample
  MIN_COUNTS = 500
  MAX_MT     = 30%
  MIN_CELLS  = 3

Notes:
  - flavor="seurat" (seurat_v3 fails with large datasets due to LOESS singularity)
  - Scale only on HVG submatrix to avoid OOM
  - UMAP here is unintegrated — final UMAP is computed in script 03

Input:  GSE161529_tumor_raw.h5ad
Output: GSE161529_tumor_processed.h5ad

Run from: ~/single_cell/
  python3 02_qc_filter_normalize.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings, sys, gc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.n_jobs = 4

FILE    = "GSE161529_tumor_raw.h5ad"
OUTFILE = "GSE161529_tumor_processed.h5ad"
OUTDIR  = "figures/02_qc"
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}

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
print(f"  First var_names : {list(adata.var_names[:5])}")
print(f"  obs columns     : {list(adata.obs.columns)}")
print(f"  var columns     : {list(adata.var.columns)}")

sample_col  = next((c for c in ["sample","Sample","patient","batch"]
                    if c in adata.obs.columns), None)
subtype_col = "subtype"
print(f"\n  Sample col  : {sample_col}")
print(f"  Subtype col : {subtype_col}")
print(f"  Subtypes    : {dict(adata.obs[subtype_col].value_counts())}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ENSEMBL → GENE SYMBOL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. ENSEMBL → GENE SYMBOL")
print("="*65)

is_ensembl = adata.var_names[0].startswith("ENSG")
print(f"  Ensembl IDs detected: {is_ensembl}")

if is_ensembl:
    adata.var["ensembl_id"] = adata.var_names.tolist()

    # Option 1: gene_name column already present
    if "gene_name" in adata.var.columns:
        print("  Using existing 'gene_name' column")
        symbols = [
            s if (isinstance(s, str) and s.strip() and
                  not s.startswith("ENSG"))
            else e
            for s, e in zip(adata.var["gene_name"],
                            adata.var["ensembl_id"])
        ]

    # Option 2: saved mapping from previous run
    elif os.path.exists("ensembl_to_symbol_mapping.csv"):
        print("  Loading saved mapping: ensembl_to_symbol_mapping.csv")
        mapping = pd.read_csv("ensembl_to_symbol_mapping.csv",
                              index_col=0)["symbol"].to_dict()
        symbols = [mapping.get(e, e) for e in adata.var["ensembl_id"]]
        print(f"  Mapped: {sum(1 for s,e in zip(symbols, adata.var['ensembl_id']) if s!=e):,} genes")

    # Option 3: query MyGeneInfo
    else:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            print("  Querying MyGeneInfo (~30-60s)...")
            results = mg.querymany(
                adata.var_names.tolist(), scopes="ensembl.gene",
                fields="symbol", species="human",
                returnall=False, verbose=False
            )
            id2sym = {r.get("query",""): r.get("symbol","")
                      for r in results if r.get("symbol")}
            print(f"  Mapped: {len(id2sym):,}/{adata.n_vars:,} "
                  f"({100*len(id2sym)/adata.n_vars:.1f}%)")
            symbols = [id2sym.get(e, e) for e in adata.var["ensembl_id"]]
        except ImportError:
            print("  ERROR: mygene not installed — run: pip install mygene")
            sys.exit(1)

    # Make symbols unique
    seen = {}
    unique_syms = []
    for s in symbols:
        if s in seen:
            seen[s] += 1
            unique_syms.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            unique_syms.append(s)

    adata.var["symbol"] = unique_syms
    adata.var_names = unique_syms
    adata.var_names_make_unique()
    print(f"  Updated var_names: {list(adata.var_names[:5])}")

    # Save mapping for future use
    adata.var[["ensembl_id","symbol"]].to_csv(
        "ensembl_to_symbol_mapping.csv"
    )
    print(f"  ✓ Mapping saved: ensembl_to_symbol_mapping.csv")

else:
    print("  Gene symbols already present — skipping conversion")

# ══════════════════════════════════════════════════════════════════════════════
# 3. QC METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 3. QC METRICS")
print("="*65)

adata.var["mt"]   = adata.var_names.str.upper().str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS","RPL"))
adata.var["hb"]   = (adata.var_names.str.upper().str.startswith("HB") &
                     ~adata.var_names.str.upper().str.startswith("HBP"))

print(f"  MT genes  : {adata.var['mt'].sum()}")
print(f"  Ribo genes: {adata.var['ribo'].sum()}")
print(f"  HB genes  : {adata.var['hb'].sum()}")

if adata.var["mt"].sum() == 0:
    print("  ⚠️  No MT genes found — check that gene symbols were converted")

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt","ribo","hb"],
    percent_top=None, log1p=False, inplace=True
)

print(f"\n  QC summary:")
print(adata.obs[["n_genes_by_counts","total_counts",
                  "pct_counts_mt","pct_counts_ribo"]].describe().round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. PRE-FILTER FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating pre-filter QC figures...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("QC metrics — before filtering (ER / HER2 / TNBC)",
             fontsize=13, fontweight="bold")

for i, (metric, label, color) in enumerate([
    ("n_genes_by_counts", "N genes / cell",    "#4C72B0"),
    ("total_counts",      "Total UMIs / cell", "#DD8452"),
    ("pct_counts_mt",    "% MT counts",        "#C44E52"),
    ("pct_counts_ribo",  "% ribosomal",        "#8172B2"),
]):
    ax = axes[i//3][i%3]
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

# nGenes vs %MT scatter
ax = axes[1][1]
ax.scatter(adata.obs["pct_counts_mt"],
           adata.obs["n_genes_by_counts"],
           s=0.3, alpha=0.15, c="#4C72B0", rasterized=True)
ax.axvline(30, color="red", ls="--", lw=1.5, label="MT cutoff")
ax.set_xlabel("% MT"); ax.set_ylabel("N genes")
ax.set_title("N genes vs % MT"); ax.legend(fontsize=8)

# nGenes vs nUMIs colored by %MT
ax = axes[1][2]
sc_p = ax.scatter(adata.obs["total_counts"],
                  adata.obs["n_genes_by_counts"],
                  s=0.3, alpha=0.15,
                  c=adata.obs["pct_counts_mt"],
                  cmap="RdYlGn_r", vmax=30, rasterized=True)
plt.colorbar(sc_p, ax=ax, label="% MT")
ax.set_xlabel("Total UMIs"); ax.set_ylabel("N genes")
ax.set_title("N genes vs UMIs (color = % MT)")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/pre_filter_qc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUTDIR}/pre_filter_qc.png")

# Violin per subtype
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("QC metrics by subtype — before filtering",
             fontsize=12, fontweight="bold")
for ax, (metric, label) in zip(axes, [
    ("n_genes_by_counts", "N genes / cell"),
    ("total_counts",      "Total UMIs / cell"),
    ("pct_counts_mt",    "% MT counts"),
]):
    for subtype, color in SUBTYPE_COLORS.items():
        vals = adata.obs.loc[adata.obs[subtype_col]==subtype, metric]
        if len(vals) > 0:
            ax.hist(vals, bins=80, color=color, alpha=0.5,
                    label=subtype, density=True, edgecolor="none")
    ax.set_title(label); ax.set_xlabel(label)
    ax.set_ylabel("Density"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pre_filter_qc_by_subtype.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUTDIR}/pre_filter_qc_by_subtype.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FILTER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 5. FILTER")
print("="*65)

MIN_GENES  = 200
MIN_COUNTS = 500
MAX_MT     = 30
MIN_CELLS  = 3

# MAX_GENES: p99 per sample to handle variable sequencing depth
if sample_col and adata.obs[sample_col].nunique() > 1:
    p99_per_sample = adata.obs.groupby(sample_col)["n_genes_by_counts"] \
                              .quantile(0.99)
    # Convert to string to avoid Categorical comparison error
    sample_str     = adata.obs[sample_col].astype(str)
    p99_per_sample.index = p99_per_sample.index.astype(str)
    mask_genes_max = (adata.obs["n_genes_by_counts"] <=
                      sample_str.map(p99_per_sample)).values
    print(f"  MAX_GENES: p99 per sample "
          f"(range: {p99_per_sample.min():.0f}–{p99_per_sample.max():.0f})")
else:
    MAX_GENES      = int(np.percentile(adata.obs["n_genes_by_counts"], 99))
    mask_genes_max = adata.obs["n_genes_by_counts"] <= MAX_GENES
    print(f"  MAX_GENES: {MAX_GENES} (global p99)")

mask = (
    (adata.obs["n_genes_by_counts"] >= MIN_GENES) &
    mask_genes_max &
    (adata.obs["total_counts"]  >= MIN_COUNTS) &
    (adata.obs["pct_counts_mt"] <= MAX_MT)
)

n_before = adata.n_obs
print(f"\n  Filters applied:")
print(f"    n_genes  >= {MIN_GENES}        : "
      f"{(adata.obs['n_genes_by_counts']<MIN_GENES).sum():,} removed")
print(f"    n_genes  <= p99 per sample   : "
      f"{(~mask_genes_max).sum():,} removed")
print(f"    n_counts >= {MIN_COUNTS}       : "
      f"{(adata.obs['total_counts']<MIN_COUNTS).sum():,} removed")
print(f"    pct_mt   <= {MAX_MT}%          : "
      f"{(adata.obs['pct_counts_mt']>MAX_MT).sum():,} removed")

adata = adata[mask].copy()
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

print(f"\n  Cells : {n_before:,} → {adata.n_obs:,} "
      f"(removed {n_before-adata.n_obs:,}, "
      f"{100*(n_before-adata.n_obs)/n_before:.1f}%)")
print(f"  Genes : {adata.n_vars:,} (min_cells={MIN_CELLS})")
print(f"\n  Cells per subtype after filtering:")
print(adata.obs[subtype_col].value_counts().to_string())

# Post-filter QC figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("QC metrics — after filtering",
             fontsize=12, fontweight="bold")
for ax, (metric, label, color) in zip(axes, [
    ("n_genes_by_counts", "N genes / cell",    "#4C72B0"),
    ("total_counts",      "Total UMIs / cell", "#DD8452"),
    ("pct_counts_mt",    "% MT counts",        "#C44E52"),
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
print(f"  ✓ {OUTDIR}/post_filter_qc.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 6. NORMALIZE")
print("="*65)

adata.layers["counts"]    = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()

print(f"  normalize_total(1e4) + log1p applied")
print(f"  Raw counts   → layers['counts']")
print(f"  Log counts   → layers['logcounts']")

# ══════════════════════════════════════════════════════════════════════════════
# 7. HVGs + PCA + UMAP  (unintegrated — for QC inspection only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 7. HVGs + PCA + UMAP  (unintegrated, QC only)")
print("="*65)

n_samples = adata.obs[sample_col].nunique() if sample_col else 1
sc.pp.highly_variable_genes(
    adata, n_top_genes=3000, flavor="seurat",
    batch_key=sample_col if n_samples > 1 else None,
    subset=False
)
n_hvg = adata.var["highly_variable"].sum()
print(f"  HVGs: {n_hvg:,} (flavor=seurat, batch_key={sample_col})")

# Scale + PCA on HVG submatrix only (memory-efficient)
adata_hvg   = adata[:, adata.var["highly_variable"]].copy()
adata_hvg.X = adata_hvg.layers["logcounts"].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=50)
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.uns["pca"]    = adata_hvg.uns["pca"]
del adata_hvg; gc.collect()
print(f"  ✓ PCA (50 components)")

# PCA variance plot
fig, ax = plt.subplots(figsize=(8, 4))
var_ratio = adata.uns["pca"]["variance_ratio"]
ax.plot(range(1, len(var_ratio)+1), np.cumsum(var_ratio)*100,
        marker="o", ms=3, color="#4C72B0")
ax.axhline(80, color="red", ls="--", lw=1, label="80% variance")
ax.set_xlabel("N PCA components")
ax.set_ylabel("Cumulative variance (%)")
ax.set_title("PCA variance explained")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUTDIR}/pca_variance.png")

N_PCS = 30
sc.pp.neighbors(adata, n_pcs=N_PCS, n_neighbors=15)
print("  Computing UMAP (5-15 min)...")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added="leiden_0.5")
print(f"  ✓ UMAP  |  Leiden clusters: {adata.obs['leiden_0.5'].nunique()}")

# UMAP figure
coords = adata.obsm["X_umap"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("UMAP — post QC, unintegrated (ER / HER2 / TNBC)",
             fontsize=12, fontweight="bold")

ax = axes[0]
for st, color in SUBTYPE_COLORS.items():
    m = adata.obs[subtype_col] == st
    ax.scatter(coords[m,0], coords[m,1], c=color, s=0.3,
               alpha=0.4, label=st, rasterized=True)
ax.legend(handles=[mpatches.Patch(color=c, label=l)
                   for l, c in SUBTYPE_COLORS.items()], fontsize=9)
ax.set_title("Tumor subtype"); ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
if sample_col:
    cats = adata.obs[sample_col].astype("category")
    cmap = plt.cm.get_cmap("tab20", cats.nunique())
    ax.scatter(coords[:,0], coords[:,1], c=cats.cat.codes,
               s=0.3, alpha=0.3, cmap=cmap, rasterized=True)
ax.set_title(f"Sample ({adata.obs[sample_col].nunique()} samples)")
ax.set_xticks([]); ax.set_yticks([])

ax = axes[2]
sc_p = ax.scatter(coords[:,0], coords[:,1],
                  c=adata.obs["n_genes_by_counts"],
                  s=0.3, alpha=0.3, cmap="viridis", rasterized=True)
plt.colorbar(sc_p, ax=ax, label="N genes")
ax.set_title("N genes per cell"); ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUTDIR}/umap_unintegrated.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUTDIR}/umap_unintegrated.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 8. SAVE")
print("="*65)

# Reset X to logcounts before saving
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
  Subtypes: {sorted(adata.obs[subtype_col].unique())}

  Figures : {OUTDIR}/
    pre_filter_qc.png
    pre_filter_qc_by_subtype.png
    post_filter_qc.png
    pca_variance.png
    umap_unintegrated.png

  Next: python3 03_split_integrate.py
══════════════════════════════════════════════════════════════
""")
