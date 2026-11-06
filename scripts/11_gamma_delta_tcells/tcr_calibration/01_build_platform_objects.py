"""
01_prepare_dataset.py
=====================
Build AnnData objects from GSE114727 (Azizi et al., Cell 2018 — breast
cancer immune atlas). The series contains TWO different experiments that
are kept as separate objects (different chemistry, depth and gene space):

  A) 10x 5' GEX + TCR (V(D)J)  — BC09, BC10, BC11 tumour, 5 libraries
     → azizi_10x_raw.h5ad
     TCR contigs are summarised per barcode into obs (TRA/TRB/TRG/TRD
     presence, V genes, CDR3, clonotype).
  B) inDrop CD45+ immune cells — BC01–BC08, tumour / normal / blood /
     lymph node, 56 libraries
     → azizi_indrop_raw.h5ad
     Each CSV only lists genes detected in that library, so the gene
     space is the union (missing = 0).

Both objects contain raw counts in X and layers["counts"], basic QC
metrics, and are QC-filtered with lenient thresholds (figures saved
before/after so thresholds can be revisited).

Input:  data/raw/GSM*  (extracted from GSE114727_RAW.tar)
Output: data/azizi_10x_raw.h5ad, data/azizi_indrop_raw.h5ad,
        figures/01_prepare/

Run from: ~/single_cell/Azizi/
  python3 01_prepare_dataset.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, os, re, gc, warnings
from scipy import sparse
from scipy.io import mmread

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

RAW    = "data/raw"
OUTDIR = "figures/01_prepare"
os.makedirs(OUTDIR, exist_ok=True)

# QC thresholds (lenient; inDrop is shallow)
QC = {
    "10x"   : {"min_genes": 200, "max_genes": 6000, "max_mt": 15},
    "indrop": {"min_genes": 150, "max_genes": 5000, "max_mt": 25},
}
MIN_CELLS_PER_GENE = 3

FNAME_RE = re.compile(r"(GSM\d+)_(BC\d+)_([A-Z]+)(\d[\d_]*)")


def parse_name(path):
    gsm, patient, tissue, rep = FNAME_RE.search(os.path.basename(path)).groups()
    return gsm, patient, tissue, rep


# ══════════════════════════════════════════════════════════════════════════════
# A) 10x
# ══════════════════════════════════════════════════════════════════════════════
def summarise_tcr(path):
    """One row per barcode: chain presence + main V gene/CDR3 per chain."""
    df = pd.read_csv(path)
    df = df[(df["is_cell"]) & (df["high_confidence"])]
    rows = {}
    for bc, g in df.groupby("barcode"):
        r = {"tcr_clonotype": g["raw_clonotype_id"].iloc[0]}
        for ch in ["TRA", "TRB", "TRG", "TRD"]:
            gc_ = g[g["chain"] == ch]
            prod = gc_[gc_["productive"].astype(str) == "True"]
            r[f"has_{ch}"] = len(gc_) > 0
            r[f"has_{ch}_productive"] = len(prod) > 0
            top = (prod if len(prod) else gc_).sort_values("umis", ascending=False)
            r[f"{ch}_v"]    = top["v_gene"].iloc[0] if len(top) else np.nan
            r[f"{ch}_cdr3"] = top["cdr3"].iloc[0]   if len(top) else np.nan
            r[f"{ch}_umis"] = int(gc_["umis"].sum())
        r["has_IG"] = g["chain"].isin(["IGH", "IGK", "IGL"]).any()
        rows[bc] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def load_10x():
    adatas = []
    for mtx in sorted(glob.glob(f"{RAW}/*_matrix.mtx.gz")):
        prefix = mtx.replace("_matrix.mtx.gz", "")
        gsm, patient, tissue, rep = parse_name(prefix)
        sample = f"{patient}_{tissue}{rep}"
        X = sparse.csr_matrix(mmread(mtx).T.astype(np.float32))
        genes = pd.read_csv(f"{prefix}_genes.tsv.gz", sep="\t", header=None,
                            names=["gene_ids", "symbol"])
        bcs = pd.read_csv(f"{prefix}_barcodes.tsv.gz", header=None)[0].values
        a = ad.AnnData(X=X)
        a.var_names = genes["symbol"].values
        a.var["gene_ids"] = genes["gene_ids"].values
        a.var_names_make_unique()
        a.obs_names = [f"{sample}_{b}" for b in bcs]
        a.obs["barcode"] = bcs
        a.obs["gsm"], a.obs["patient"], a.obs["tissue"] = gsm, patient, tissue
        a.obs["sample"] = sample

        # matching TCR file (same patient/tissue/rep)
        tcr_f = glob.glob(f"{RAW}/*_{patient}_{tissue}{rep}_filtered_contig_annotations.csv.gz")
        assert len(tcr_f) == 1, tcr_f
        tcr = summarise_tcr(tcr_f[0])
        tcr.index = [f"{sample}_{b}" for b in tcr.index]
        a.obs = a.obs.join(tcr, how="left")
        print(f"  10x {sample}: {a.n_obs} cells, TCR for "
              f"{a.obs['tcr_clonotype'].notna().sum()}")
        adatas.append(a)

    adata = ad.concat(adatas, join="inner", merge="same")
    for ch in ["TRA", "TRB", "TRG", "TRD"]:
        for c in [f"has_{ch}", f"has_{ch}_productive"]:
            adata.obs[c] = adata.obs[c].fillna(False).astype(bool)
    adata.obs["has_IG"] = adata.obs["has_IG"].fillna(False).astype(bool)
    # Cell Ranger gives clonotype "None" (→ NaN) when no productive chain
    adata.obs["has_clonotype"] = adata.obs["tcr_clonotype"].notna()
    adata.obs["has_tcr_contig"] = adata.obs[["has_TRA", "has_TRB"]].any(axis=1)
    adata.obs["platform"] = "10x_5p"
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# B) inDrop
# ══════════════════════════════════════════════════════════════════════════════
def load_indrop():
    adatas = []
    for f in sorted(glob.glob(f"{RAW}/*_counts.csv.gz")):
        gsm, patient, tissue, rep = parse_name(f)
        sample = f"{patient}_{tissue}{rep}"
        df = pd.read_csv(f, index_col=0).fillna(0)
        a = ad.AnnData(X=sparse.csr_matrix(df.values.astype(np.float32)))
        a.var_names = df.columns.astype(str)
        a.obs_names = [f"{sample}_{b}" for b in df.index.astype(str)]
        a.obs["gsm"], a.obs["patient"], a.obs["tissue"] = gsm, patient, tissue
        a.obs["sample"] = sample
        adatas.append(a)
        del df
    print(f"  inDrop: {len(adatas)} libraries")
    adata = ad.concat(adatas, join="outer", fill_value=0)
    adata.X = sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    adata.obs["platform"] = "inDrop"
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# QC
# ══════════════════════════════════════════════════════════════════════════════
def qc(adata, name, th):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, lines in zip(
        axs, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        [[th["min_genes"], th["max_genes"]], [], [th["max_mt"]]]):
        sc.pl.violin(adata, col, groupby="sample", ax=ax, show=False,
                     stripplot=False, rotation=90)
        for l in lines:
            ax.axhline(l, color="red", ls="--", lw=1)
        ax.tick_params(axis="x", labelsize=5)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/qc_{name}_before.png", dpi=150)
    plt.close()

    n0 = adata.n_obs
    keep = ((adata.obs["n_genes_by_counts"] >= th["min_genes"]) &
            (adata.obs["n_genes_by_counts"] <= th["max_genes"]) &
            (adata.obs["pct_counts_mt"] <= th["max_mt"]))
    adata = adata[keep].copy()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    adata.layers["counts"] = adata.X.copy()
    print(f"  {name}: {n0} → {adata.n_obs} cells, {adata.n_vars} genes")
    summ = adata.obs.groupby(["patient", "tissue"], observed=True).size()
    summ.to_csv(f"{OUTDIR}/cells_per_patient_tissue_{name}.csv")
    return adata


if __name__ == "__main__":
    print("Loading 10x ...")
    a10 = qc(load_10x(), "10x", QC["10x"])
    for c in a10.obs.columns:
        if a10.obs[c].dtype == object:
            a10.obs[c] = a10.obs[c].astype(str).replace("nan", np.nan)
    a10.write_h5ad("data/azizi_10x_raw.h5ad", compression="gzip")
    del a10; gc.collect()

    print("Loading inDrop ...")
    aid = qc(load_indrop(), "indrop", QC["indrop"])
    aid.write_h5ad("data/azizi_indrop_raw.h5ad", compression="gzip")
    print("Done.")
