"""
01_load_samples.py
===================
Load the 19 raw scRNA-seq (10x, RNA-only) samples from GSE243526 and
build a single concatenated AnnData object.

Dataset: 15 breast cancer patients (some with 2 biopsies) + 4 cell
lines (HCC1143, MCF7, SUM149PT, T47D), paired scRNA+scATAC (10x
Multiome). Only the RNA side is loaded here — ATAC fragments were
excluded during extraction (too large, not needed for this analysis).

Each sample = 3 files (barcodes/features/matrix.mtx), standard
CellRanger output, identical gene reference across all 19 samples
(33,538 genes) confirmed by md5.

Steps:
  1. Discover samples in GSE243526_RNA/ from filenames
  2. Load each sample's mtx/barcodes/features, tag obs (patient,
     biopsy_code, is_cell_line), prefix barcodes to keep them unique
  3. Concatenate into one object (var identical across samples)
  4. Ensembl ID → gene symbol (var_names), mirrors Klughammer 01
  5. QC metrics (n_genes, total_counts, pct_counts_mt) — no filtering
     yet, that's for 02
  6. Save raw object + per-sample summary + QC overview figure

Input:  ../GSE243526_RNA/*.tsv.gz, *.mtx.gz
Output: gse243526_raw.h5ad, gse243526_ensembl_to_symbol.csv,
        gse243526_sample_summary.csv, figures/01_load_samples/

Run from: ~/single_cell/Regner/
  python3 01_load_samples.py
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import gzip
import os
import re
import warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

RNA_DIR = "../GSE243526_RNA"
OUTDIR  = "figures/01_load_samples"
OUTFILE = "gse243526_raw.h5ad"
os.makedirs(OUTDIR, exist_ok=True)

CELL_LINES = {"HCC1143", "MCF7", "SUM149PT", "T47D"}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def discover_samples(rna_dir):
    """Parse GSM id + sample name from *_RNA_barcodes.tsv.gz filenames."""
    pattern = re.compile(r"^(GSM\d+)_(.+)_RNA_barcodes\.tsv\.gz$")
    samples = []
    for fname in sorted(os.listdir(rna_dir)):
        m = pattern.match(fname)
        if m:
            samples.append({"gsm": m.group(1), "sample_name": m.group(2)})
    samples.sort(key=lambda s: int(s["gsm"][3:]))
    return samples


def split_patient_biopsy(sample_name):
    """'Patient_5_35A4AL' -> ('Patient_5', '35A4AL'); cell lines -> (name, None)."""
    if sample_name in CELL_LINES:
        return sample_name, None
    m = re.match(r"^(Patient_\d+)_(.+)$", sample_name)
    if m:
        return m.group(1), m.group(2)
    return sample_name, None


def load_10x_sample(rna_dir, gsm, sample_name):
    """Read one sample's barcodes/features/matrix.mtx (gzipped) into an AnnData."""
    prefix = f"{gsm}_{sample_name}_RNA"

    with gzip.open(os.path.join(rna_dir, f"{prefix}_matrix.mtx.gz"), "rb") as f:
        mat = scipy.io.mmread(f).tocsr().T.tocsr()  # cells x genes

    barcodes = pd.read_csv(
        os.path.join(rna_dir, f"{prefix}_barcodes.tsv.gz"),
        header=None, sep="\t")[0].values
    features = pd.read_csv(
        os.path.join(rna_dir, f"{prefix}_features.tsv.gz"),
        header=None, sep="\t",
        names=["ensembl_id", "symbol", "feature_type"])

    a = ad.AnnData(X=mat)
    a.var_names = features["ensembl_id"].values
    a.var["symbol"] = features["symbol"].values
    a.var["feature_type"] = features["feature_type"].values

    patient, biopsy_code = split_patient_biopsy(sample_name)
    a.obs_names = [f"{sample_name}_{bc}" for bc in barcodes]
    a.obs["sample_id"]   = gsm
    a.obs["sample_name"] = sample_name
    a.obs["patient"]     = patient
    a.obs["biopsy_code"] = biopsy_code
    a.obs["is_cell_line"] = sample_name in CELL_LINES
    return a


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD")
print("="*65)

samples = discover_samples(RNA_DIR)
print(f"  Samples found: {len(samples)}")

adatas = []
for s in samples:
    a = load_10x_sample(RNA_DIR, s["gsm"], s["sample_name"])
    print(f"    {s['gsm']}  {s['sample_name']:22s} "
          f"{a.n_obs:>6,} cells | {a.n_vars:,} genes")
    adatas.append(a)

# Sanity check: identical gene reference across samples (confirmed via md5
# beforehand) — concat should need no var reconciliation.
ref_vars = adatas[0].var_names
assert all((a.var_names == ref_vars).all() for a in adatas), \
    "Gene reference differs across samples — investigate before concatenating"

adata = ad.concat(adatas, join="outer", merge="same")
adata.var = adatas[0].var.copy()
del adatas

print(f"\n  Concatenated: {adata.n_obs:,} cells | {adata.n_vars:,} genes | "
      f"{adata.obs['sample_name'].nunique()} samples")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENSEMBL → GENE SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. ENSEMBL → GENE SYMBOLS")
print("="*65)

symbols = adata.var["symbol"].tolist()
seen = {}
unique_syms = []
for i, sym in enumerate(symbols):
    sym = str(sym).strip() if sym and str(sym) != "nan" else adata.var_names[i]
    if sym in seen:
        seen[sym] += 1
        unique_syms.append(f"{sym}_{seen[sym]}")
    else:
        seen[sym] = 0
        unique_syms.append(sym)

adata.var["ensembl_id"] = adata.var_names.tolist()
adata.var_names = unique_syms
adata.var_names_make_unique()
adata.var[["ensembl_id", "symbol"]].to_csv("gse243526_ensembl_to_symbol.csv")
print(f"  Symbols updated: {list(adata.var_names[:5])}")
print(f"  Mapping saved: gse243526_ensembl_to_symbol.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 3. QC METRICS (no filtering — informational only, see 02)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 3. QC METRICS")
print("="*65)

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

summary = adata.obs.groupby("sample_name", observed=True).agg(
    gsm=("sample_id", "first"),
    patient=("patient", "first"),
    is_cell_line=("is_cell_line", "first"),
    n_cells=("sample_name", "size"),
    median_genes=("n_genes_by_counts", "median"),
    median_counts=("total_counts", "median"),
    median_pct_mt=("pct_counts_mt", "median"),
).sort_values("gsm")
summary.to_csv("gse243526_sample_summary.csv")
print(summary.to_string())

# QC overview figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
order = summary.index.tolist()
for ax, col, title in zip(
    axes,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    ["Genes / cell", "UMI counts / cell", "% mitochondrial"],
):
    data = [adata.obs.loc[adata.obs["sample_name"] == s, col] for s in order]
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_title(title)
    if col == "total_counts":
        ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/qc_overview_by_sample.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure: {OUTDIR}/qc_overview_by_sample.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE
# ══════════════════════════════════════════════════════════════════════════════
adata.write_h5ad(OUTFILE)

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  {OUTFILE}
    {adata.n_obs:,} cells | {adata.n_vars:,} genes | {adata.obs['sample_name'].nunique()} samples
    Patients   : {adata.obs.loc[~adata.obs['is_cell_line'], 'patient'].nunique()}
    Cell lines : {sorted(adata.obs.loc[adata.obs['is_cell_line'], 'patient'].unique())}

  gse243526_ensembl_to_symbol.csv
  gse243526_sample_summary.csv
  {OUTDIR}/qc_overview_by_sample.png

  Next: define QC thresholds and write 02_qc_filter.py
        (raw counts only — no normalization/filtering applied yet)
══════════════════════════════════════════════════════════════
""")
