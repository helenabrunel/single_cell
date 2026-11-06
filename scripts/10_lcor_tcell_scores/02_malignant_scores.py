"""
01_malignant_scores.py
========================
Per-patient, malignant-compartment-only scores for LCOR (raw), LCOR_OE
(signature), and CXCL16 (raw), in Visvader and Klughammer separately.

NON-INTEGRATED DATA ONLY: all values computed from raw UMI counts
(adata.layers["counts"] for Visvader; adata.raw.X via Ensembl-ID lookup
for Klughammer, since Klughammer's .X is Seurat-LogNormalize, not CP10K),
re-normalized here to log1p(CP10K) ourselves. Harmony/scVI/DRVI embeddings
present in these objects are NOT used for any expression value here --
only for the compartment/cell-type calls already baked into the input
files (assign_immune_type scores for Visvader; author_cell_type for
Klughammer), which is a categorical gate, not an expression value.

LCOR_OE score: scanpy sc.tl.score_genes (control-gene-corrected mean
expression), computed directly on each dataset's own log1p(CP10K) matrix
(use_raw=False, so it is unaffected by any .raw/.X normalization quirks).

Output: 02_malignant_scores/figures/{Visvader,Klughammer}_persample_malignant_scores.csv
"""
import scanpy as sc
import numpy as np
import pandas as pd
import json, os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "02_malignant_scores/figures"
os.makedirs(OUTDIR, exist_ok=True)

with open("01_signatures/figures/signatures.json") as f:
    SIGNATURES = json.load(f)
LCOR_OE_GENES = SIGNATURES["LCOR_OE_malignant"]


def get_symbol_ensembl_map(adata, genes):
    """Maps gene SYMBOL -> its Ensembl ID valid in adata.raw.var_names, via
    raw.var['feature_name']. ALWAYS goes through this mapping (never returns
    the bare symbol) because adata.raw is indexed by Ensembl ID even when the
    same symbol also happens to appear directly in adata.var_names (used for
    the lognorm .X) -- mixing the two up silently pulls the wrong column."""
    sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
    resolved = {}
    for g in genes:
        if g in sym2ens.index:
            ens = sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]
            if ens in adata.raw.var_names:
                resolved[g] = ens
    return resolved


def pseudobulk_log1p_cp10k(raw_counts, total_counts, groups):
    """sum(raw)/sum(total_counts) per group, log1p(1e4 * .) -- the project's
    standard pseudobulk method (arithmetic sum, not mean-of-logcounts)."""
    df = pd.DataFrame({"raw": raw_counts, "total": total_counts, "group": groups})
    g = df.groupby("group").agg(sum_raw=("raw", "sum"), sum_total=("total", "sum"))
    return np.log1p(1e4 * g["sum_raw"] / g["sum_total"])


def process_visvader():
    print("\n=== Visvader (compartment_epithelial.h5ad) ===")
    adata = sc.read_h5ad("/home/hbrunel/single_cell/Visvader/compartment_epithelial.h5ad")
    print(f"  Malignant cells: {adata.n_obs:,}, patients: {adata.obs['sample'].nunique()}")

    # raw counts for LCOR, CXCL16 (pseudobulk)
    idx_lcor = adata.var_names.get_loc("LCOR")
    idx_cxcl16 = adata.var_names.get_loc("CXCL16")
    raw = adata.layers["counts"]
    raw_lcor = np.asarray(raw[:, idx_lcor].todense()).ravel() if issparse(raw) else np.asarray(raw[:, idx_lcor]).ravel()
    raw_cxcl16 = np.asarray(raw[:, idx_cxcl16].todense()).ravel() if issparse(raw) else np.asarray(raw[:, idx_cxcl16]).ravel()
    total_counts = adata.obs["total_counts"].values.astype(float)
    sample = adata.obs["sample"].astype(str).values
    subtype = adata.obs["subtype"].astype(str).values

    LCOR_raw_pseudobulk = pseudobulk_log1p_cp10k(raw_lcor, total_counts, sample)
    CXCL16_raw_pseudobulk = pseudobulk_log1p_cp10k(raw_cxcl16, total_counts, sample)

    # LCOR_OE score: own log1p(CP10K) matrix (NOT .X as loaded -- rebuild to be explicit/safe)
    adata.X = adata.layers["logcounts"]  # already log1p(CP10K) in this project's prior pipeline
    genes_present = [g for g in LCOR_OE_GENES if g in adata.var_names]
    print(f"  LCOR_OE genes found: {len(genes_present)}/{len(LCOR_OE_GENES)}")
    sc.tl.score_genes(adata, genes_present, score_name="LCOR_OE_score", random_state=0, use_raw=False)
    lcor_oe_per_cell = pd.DataFrame({"sample": sample, "score": adata.obs["LCOR_OE_score"].values})
    LCOR_OE_score = lcor_oe_per_cell.groupby("sample")["score"].mean()

    n_cells = pd.Series(sample).value_counts()
    subtype_map = pd.Series(subtype, index=sample).groupby(level=0).first()

    out = pd.DataFrame({
        "sample": LCOR_raw_pseudobulk.index,
        "subtype": subtype_map.reindex(LCOR_raw_pseudobulk.index).values,
        "n_malignant": n_cells.reindex(LCOR_raw_pseudobulk.index).values,
        "LCOR_raw_pseudobulk": LCOR_raw_pseudobulk.values,
        "LCOR_OE_score": LCOR_OE_score.reindex(LCOR_raw_pseudobulk.index).values,
        "CXCL16_raw_pseudobulk": CXCL16_raw_pseudobulk.values,
    })
    out = out[out["n_malignant"] >= 20].reset_index(drop=True)
    out.to_csv(f"{OUTDIR}/Visvader_persample_malignant_scores.csv", index=False)
    print(f"  Saved: {OUTDIR}/Visvader_persample_malignant_scores.csv ({len(out)} patients)")
    return out


def process_klughammer():
    print("\n=== Klughammer (klughammer_Malignant.h5ad) ===")
    adata = sc.read_h5ad("/home/hbrunel/single_cell/Klughammer/klughammer_Malignant.h5ad")
    print(f"  Malignant cells: {adata.n_obs:,}, patients: {adata.obs['donor_id'].nunique()}")

    needed = ["LCOR", "CXCL16"] + LCOR_OE_GENES
    resolved = get_symbol_ensembl_map(adata, needed)
    missing = set(needed) - set(resolved.keys())
    print(f"  Genes resolved: {len(resolved)}/{len(needed)} (missing: {sorted(missing)})")

    def get_raw(symbol):
        col = adata.raw[:, resolved[symbol]].X
        return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

    raw_lcor = get_raw("LCOR")
    raw_cxcl16 = get_raw("CXCL16")
    total_counts = adata.obs["total_counts"].values.astype(float)
    sample = adata.obs["donor_id"].astype(str).values
    subtype = adata.obs["subtype"].astype(str).values

    LCOR_raw_pseudobulk = pseudobulk_log1p_cp10k(raw_lcor, total_counts, sample)
    CXCL16_raw_pseudobulk = pseudobulk_log1p_cp10k(raw_cxcl16, total_counts, sample)

    # LCOR_OE score: build a proper log1p(CP10K) AnnData from RAW counts (not
    # the Seurat-LogNormalize .X that ships with this file) -- uses the FULL
    # transcriptome (not just the 50 signature genes), because score_genes
    # needs a broad gene pool to sample expression-matched control genes from.
    genes_present_oe = [g for g in LCOR_OE_GENES if g in resolved]
    print(f"  LCOR_OE genes found: {len(genes_present_oe)}/{len(LCOR_OE_GENES)}")
    raw_mat_full = adata.raw.X
    raw_mat_full = raw_mat_full.tocsr() if issparse(raw_mat_full) else raw_mat_full
    feature_names = adata.raw.var["feature_name"].astype(str).values
    adata_oe = sc.AnnData(X=raw_mat_full, obs=adata.obs.copy(), var=pd.DataFrame(index=feature_names))
    adata_oe.var_names_make_unique()
    sc.pp.normalize_total(adata_oe, target_sum=1e4)
    sc.pp.log1p(adata_oe)
    sc.tl.score_genes(adata_oe, genes_present_oe, score_name="LCOR_OE_score", random_state=0, use_raw=False)
    lcor_oe_per_cell = pd.DataFrame({"sample": sample, "score": adata_oe.obs["LCOR_OE_score"].values})
    LCOR_OE_score = lcor_oe_per_cell.groupby("sample")["score"].mean()

    n_cells = pd.Series(sample).value_counts()
    subtype_map = pd.Series(subtype, index=sample).groupby(level=0).first()

    out = pd.DataFrame({
        "sample": LCOR_raw_pseudobulk.index,
        "subtype": subtype_map.reindex(LCOR_raw_pseudobulk.index).values,
        "n_malignant": n_cells.reindex(LCOR_raw_pseudobulk.index).values,
        "LCOR_raw_pseudobulk": LCOR_raw_pseudobulk.values,
        "LCOR_OE_score": LCOR_OE_score.reindex(LCOR_raw_pseudobulk.index).values,
        "CXCL16_raw_pseudobulk": CXCL16_raw_pseudobulk.values,
    })
    out = out[out["n_malignant"] >= 20].reset_index(drop=True)
    out.to_csv(f"{OUTDIR}/Klughammer_persample_malignant_scores.csv", index=False)
    print(f"  Saved: {OUTDIR}/Klughammer_persample_malignant_scores.csv ({len(out)} patients)")
    return out


vis = process_visvader()
klu = process_klughammer()
print("\nDONE")
