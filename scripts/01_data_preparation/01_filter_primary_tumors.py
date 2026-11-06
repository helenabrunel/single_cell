"""
00_filter_tumor.py
==================
Filter raw object BEFORE QC to retain only primary tumor samples.

Removes:
  - subtype == "Normal"       (healthy tissue, not tumor)
  - subtype == "BRCA1_mutant" (excluded from this analysis)
  - tissue  == "LN"           (lymph node)

Keeps:
  - subtype in [ER, HER2, TNBC]
  - tissue  == "Total"

Input:  GSE161529_cellwhisperer.h5ad
Output: GSE161529_tumor_raw.h5ad

Run from: ~/single_cell/
  python3 00_filter_tumor.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

FILE    = "GSE161529_cellwhisperer.h5ad"
OUTFILE = "GSE161529_tumor_raw.h5ad"
OUTDIR  = "figures/00_filter"
os.makedirs(OUTDIR, exist_ok=True)

KEEP_SUBTYPES = ["ER", "HER2", "TNBC"]
KEEP_TISSUE   = ["Total"]
SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}

# ── Load ──────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(" 1. LOAD")
print("="*65)
adata = sc.read_h5ad(FILE)
print(adata)
print(f"\n  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")
print(f"  obs columns: {list(adata.obs.columns)}")

# ── Inspect before ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(" 2. METADATA BEFORE FILTERING")
print("="*65)
print(f"\n  Subtype counts:")
print(adata.obs["subtype"].value_counts().to_string())
print(f"\n  Tissue counts:")
print(adata.obs["tissue"].value_counts().to_string())
print(f"\n  Subtype × tissue:")
print(pd.crosstab(adata.obs["subtype"], adata.obs["tissue"]).to_string())

# ── Filter ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(" 3. FILTER")
print("="*65)

mask = (adata.obs["subtype"].isin(KEEP_SUBTYPES) &
        adata.obs["tissue"].isin(KEEP_TISSUE))

print(f"  Cells before : {adata.n_obs:,}")
print(f"  Removed Normal       : {(adata.obs['subtype']=='Normal').sum():,}")
print(f"  Removed BRCA1_mutant : {(adata.obs['subtype']=='BRCA1_mutant').sum():,}")
print(f"  Removed LN tissue    : {(adata.obs['tissue']=='LN').sum():,}")

adata = adata[mask].copy()
print(f"  Cells after  : {adata.n_obs:,}")
print(f"\n  Final subtype × tissue:")
print(pd.crosstab(adata.obs["subtype"], adata.obs["tissue"]).to_string())
print(f"\n  Cells per subtype:")
print(adata.obs["subtype"].value_counts().to_string())
print(f"\n  Samples per subtype:")
print(adata.obs.groupby("subtype")["sample"].nunique().to_string())

# ── Basic QC overview (raw counts) ────────────────────────────────────────────
print("\n" + "="*65)
print(" 4. RAW QC OVERVIEW")
print("="*65)

# MT genes — may be 0 if Ensembl IDs (will be fixed in script 02)
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
n_mt = adata.var["mt"].sum()
print(f"  MT genes detected: {n_mt}")
if n_mt == 0:
    print("  (Ensembl IDs — MT genes will be found after symbol "
          "conversion in script 02)")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"],
                            percent_top=None, log1p=False, inplace=True)
print(f"\n{adata.obs[['n_genes_by_counts','total_counts','pct_counts_mt']].describe().round(2).to_string()}")

# Distribution per subtype
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Raw QC metrics by subtype — tumor only (ER, HER2, TNBC)",
             fontsize=12, fontweight="bold")
for ax, (metric, label) in zip(axes, [
    ("n_genes_by_counts", "N genes / cell"),
    ("total_counts",      "Total UMIs / cell"),
    ("pct_counts_mt",    "% MT counts"),
]):
    for subtype, color in SUBTYPE_COLORS.items():
        vals = adata.obs.loc[adata.obs["subtype"]==subtype, metric]
        if len(vals) > 0:
            ax.hist(vals, bins=80, color=color, alpha=0.5,
                    label=subtype, density=True)
    ax.set_title(label); ax.set_xlabel(label); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/raw_qc_by_subtype.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✓ {OUTDIR}/raw_qc_by_subtype.png")

# N genes per sample
samples = adata.obs["sample"].unique()
fig, ax = plt.subplots(figsize=(max(12, len(samples)*0.25), 5))
bp = ax.boxplot(
    [adata.obs.loc[adata.obs["sample"]==s, "n_genes_by_counts"].values
     for s in samples],
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black", lw=1.5)
)
for patch, s in zip(bp["boxes"], samples):
    st = adata.obs.loc[adata.obs["sample"]==s, "subtype"].iloc[0]
    patch.set_facecolor(SUBTYPE_COLORS.get(st,"#AAAAAA"))
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(samples)+1))
ax.set_xticklabels(samples, rotation=90, fontsize=5)
ax.set_ylabel("N genes / cell")
ax.set_title("N genes per sample (colored by subtype)")
handles = [mpatches.Patch(color=c, label=l, alpha=0.7)
           for l, c in SUBTYPE_COLORS.items()]
ax.legend(handles=handles, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/raw_ngenes_by_sample.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUTDIR}/raw_ngenes_by_sample.png")

# ── Save ──────────────────────────────────────────────────────────────────────
adata.write_h5ad(OUTFILE)
print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Output  : {OUTFILE}
  Cells   : {adata.n_obs:,}
  Genes   : {adata.n_vars:,}
  Subtypes: {sorted(adata.obs['subtype'].unique())}

  Next: python3 02_qc_filter_normalize.py
══════════════════════════════════════════════════════════════
""")
