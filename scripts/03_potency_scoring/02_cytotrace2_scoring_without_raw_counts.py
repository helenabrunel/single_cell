"""
K02_cytotrace2.py
=================
Run CytoTRACE2 on all 20 Klughammer objects.

Since raw counts are not available, X (log-normalized) is reverted
to pseudo-counts WITHIN EACH BATCH using:
    pseudo_counts = round(expm1(X_batch) * scale_factor)

Key fix vs previous version that crashed:
    - NO full matrix densification at any point
    - Gene selection uses chunk-wise column sums (sparse)
    - log1p reversal done row-by-row within each batch
    - Peak RAM per batch ≈ batch_size × top_n_genes × 4 bytes = 200 MB

Memory-efficient approach:
    - Top 5000 genes selected chunk-wise (no full dense)
    - Batches of 10k cells
    - Reversal + TSV writing done per batch
    - Auto-resume from partial CSV

Output: cytotrace2_scores_klughammer_[name].csv

Run from: ~/single_cell/Klughammer/
    python3 K02_cytotrace2.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os, tempfile, warnings, gc
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

BATCH_SIZE   = 10_000
TOP_N_GENES  = 5_000
SCALE_FACTOR = 10_000

FILES = {
    "all_global"      : "klughammer_all_global.h5ad",
    "all_ER"          : "klughammer_all_ER.h5ad",
    "all_HER2"        : "klughammer_all_HER2.h5ad",
    "all_TNBC"        : "klughammer_all_TNBC.h5ad",
    "Malignant_Global": "klughammer_Malignant.h5ad",
    "Malignant_ER"    : "klughammer_Malignant_ER.h5ad",
    "Malignant_HER2"  : "klughammer_Malignant_HER2.h5ad",
    "Malignant_TNBC"  : "klughammer_Malignant_TNBC.h5ad",
    "Lymphoid_Global" : "klughammer_Lymphoid.h5ad",
    "Lymphoid_ER"     : "klughammer_Lymphoid_ER.h5ad",
    "Lymphoid_HER2"   : "klughammer_Lymphoid_HER2.h5ad",
    "Lymphoid_TNBC"   : "klughammer_Lymphoid_TNBC.h5ad",
    "Myeloid_Global"  : "klughammer_Myeloid.h5ad",
    "Myeloid_ER"      : "klughammer_Myeloid_ER.h5ad",
    "Myeloid_HER2"    : "klughammer_Myeloid_HER2.h5ad",
    "Myeloid_TNBC"    : "klughammer_Myeloid_TNBC.h5ad",
    "Stromal_Global"  : "klughammer_Stromal.h5ad",
    "Stromal_ER"      : "klughammer_Stromal_ER.h5ad",
    "Stromal_HER2"    : "klughammer_Stromal_HER2.h5ad",
    "Stromal_TNBC"    : "klughammer_Stromal_TNBC.h5ad",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def select_top_gene_indices(X, top_n, chunk_size=500):
    """
    Select top N expressed gene indices using chunk-wise column sums.
    Never materializes full dense matrix.
    """
    n_genes   = X.shape[1]
    gene_sums = np.zeros(n_genes, dtype=np.float64)

    if issparse(X):
        X_csc = X.tocsc()
        for start in range(0, n_genes, chunk_size):
            end = min(start + chunk_size, n_genes)
            gene_sums[start:end] = np.asarray(
                X_csc[:, start:end].sum(axis=0)
            ).flatten()
        del X_csc; gc.collect()
    else:
        gene_sums = np.asarray(X).sum(axis=0)

    return sorted(np.argsort(gene_sums)[::-1][:top_n])


def revert_and_write_tsv(X_batch, top_idx, gene_names, cell_names,
                          filepath, scale_factor):
    """
    For a batch:
      1. Subset to top genes
      2. Revert log1p → pseudo-counts row-by-row
      3. Filter zero genes
      4. Write TSV (genes × cells)

    Returns list of non-zero gene names.
    Peak RAM: batch_size × len(top_idx) × 4 bytes ≈ 200 MB for 10k × 5k
    """
    n_cells = X_batch.shape[0]
    n_top   = len(top_idx)

    # Subset to top genes and densify (only this batch × top genes)
    if issparse(X_batch):
        X_sub = X_batch[:, top_idx].toarray().astype(np.float32)
    else:
        X_sub = np.array(X_batch[:, top_idx], dtype=np.float32)

    # Revert log1p → pseudo-counts
    pseudo = np.round(np.expm1(X_sub) * scale_factor).astype(np.int32)
    del X_sub; gc.collect()

    # Filter zero-sum genes
    gene_sums = pseudo.sum(axis=0)
    nz_mask   = gene_sums > 0
    pseudo_nz = pseudo[:, nz_mask]
    genes_nz  = [gene_names[i] for i, keep in zip(top_idx, nz_mask) if keep]
    del pseudo; gc.collect()

    # Write TSV (genes × cells)
    with open(filepath, "w") as f:
        f.write("\t" + "\t".join(cell_names) + "\n")
        for gene_idx, gene in enumerate(genes_nz):
            row = [str(int(v)) for v in pseudo_nz[:, gene_idx]]
            f.write(gene + "\t" + "\t".join(row) + "\n")

    del pseudo_nz; gc.collect()
    return genes_nz


def run_cytotrace2_batched(adata, name, batch_size, top_n_genes):
    from cytotrace2_py.cytotrace2_py import cytotrace2 as ct2

    n_cells   = adata.n_obs
    n_batches = int(np.ceil(n_cells / batch_size))
    print(f"\n  {name}: {n_cells:,} cells → {n_batches} batches")
    print(f"  Peak RAM/batch ≈ "
          f"{batch_size * top_n_genes * 4 / 1e6:.0f} MB")

    # Filter out genes with NaN/float names (Ensembl conversion artifact)
    all_names   = adata.var_names.tolist()
    valid_mask  = np.array([
        isinstance(g, str) and g.strip() != "" and g.lower() != "nan"
        for g in all_names
    ])
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  Filtering {n_invalid:,} genes with NaN/invalid names")
    valid_idx        = np.where(valid_mask)[0]
    X_valid          = adata.X[:, valid_idx] if n_invalid > 0 else adata.X
    gene_names_valid = [all_names[i] for i in valid_idx]

    # Select top gene indices (chunk-wise, no dense)
    print(f"  Selecting top {top_n_genes:,} genes (chunk-wise)...")
    top_idx   = select_top_gene_indices(X_valid, top_n_genes)
    top_genes = gene_names_valid
    print(f"  ✓ {len(top_idx):,} gene indices selected")

    # Resume from partial
    partial_csv = f"cytotrace2_scores_klughammer_{name}_partial.csv"
    if os.path.exists(partial_csv):
        existing   = pd.read_csv(partial_csv, index_col=0)
        existing.index = existing.index.astype(str)
        done_cells = set(existing.index)
        all_results = [existing]
        print(f"  Resuming: {len(done_cells):,} cells already scored")
    else:
        done_cells  = set()
        all_results = []

    for batch_idx in range(n_batches):
        start      = batch_idx * batch_size
        end        = min(start + batch_size, n_cells)
        cell_names = adata.obs_names[start:end].tolist()

        if all(c in done_cells for c in cell_names):
            print(f"  Batch {batch_idx+1}/{n_batches}: skip (done)")
            continue

        print(f"\n  Batch {batch_idx+1}/{n_batches}: "
              f"{start:,}–{end:,} ({end-start:,} cells)")

        X_batch = X_valid[start:end]

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv  = os.path.join(tmpdir, "expression.txt")
            odir = os.path.join(tmpdir, "ct2_out")

            print(f"    Reverting log1p + writing TSV...")
            genes_nz = revert_and_write_tsv(
                X_batch, top_idx, top_genes, cell_names,
                tsv, SCALE_FACTOR
            )
            mb = os.path.getsize(tsv) / 1e6
            print(f"    TSV: {mb:.0f} MB | Non-zero genes: {len(genes_nz):,}")

            try:
                results = ct2(
                    input_path=tsv, species="human",
                    max_cores=2, seed=42,
                    disable_plotting=True, output_dir=odir
                )
                pred = results["pred_df"] \
                       if isinstance(results, dict) else results
                pred.index = pred.index.astype(str)

                sc_col = next(
                    (c for c in pred.columns
                     if "score" in c.lower() and "cytotrace" in c.lower()),
                    pred.columns[0]
                )
                pt_col = next(
                    (c for c in pred.columns
                     if any(k in c.lower()
                            for k in ["potency","category","class"])),
                    None
                )
                res = pd.DataFrame(
                    {"CytoTRACE2_Score": pred[sc_col].astype(float)},
                    index=pred.index
                )
                if pt_col:
                    res["CytoTRACE2_Potency"] = pred[pt_col]

                print(f"    ✓ {len(res):,} cells scored "
                      f"({res['CytoTRACE2_Score'].min():.3f}"
                      f"–{res['CytoTRACE2_Score'].max():.3f})")
                all_results.append(res)

                partial = pd.concat(all_results).drop_duplicates()
                partial.to_csv(partial_csv)
                print(f"    ✓ Progress: {len(partial):,}/{n_cells:,} "
                      f"({100*len(partial)/n_cells:.0f}%)")

            except Exception as e:
                print(f"    ⚠️  Batch {batch_idx+1} failed: {e}")
                import traceback; traceback.print_exc()
                all_results.append(pd.DataFrame(
                    {"CytoTRACE2_Score": np.nan}, index=cell_names
                ))

        del X_batch
        gc.collect()

    combined = pd.concat(all_results).drop_duplicates()
    combined  = combined.reindex(adata.obs_names)
    n_ok      = (~combined["CytoTRACE2_Score"].isna()).sum()
    print(f"\n  ✓ Final: {n_ok:,}/{n_cells:,} cells scored")

    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" K02 — CytoTRACE2 (Klughammer, memory-efficient)")
print("="*65)
print(f"  Batch size     : {BATCH_SIZE:,} cells")
print(f"  Top genes      : {TOP_N_GENES:,}")
print(f"  Pseudo-counts  : expm1(X_batch) × {SCALE_FACTOR:,}")
print(f"  Peak RAM/batch ≈ {BATCH_SIZE*TOP_N_GENES*4/1e6:.0f} MB")
print(f"  NO full dense matrix at any point")
print(f"  Auto-resume    : yes")
print(f"  Objects        : {len(FILES)}")

for name, fpath in FILES.items():
    out_csv = f"cytotrace2_scores_klughammer_{name}.csv"

    if not os.path.exists(fpath):
        print(f"\n  ⚠️  {fpath} not found — skipping {name}")
        continue

    if os.path.exists(out_csv):
        n = len(pd.read_csv(out_csv))
        print(f"\n  {name}: already done ({n:,} cells) — skipping")
        continue

    print(f"\n{'='*65}")
    print(f"  {name.upper()}")
    print(f"{'='*65}")

    adata  = sc.read_h5ad(fpath)
    print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

    scores = run_cytotrace2_batched(adata, name, BATCH_SIZE, TOP_N_GENES)
    scores.to_csv(out_csv)
    print(f"  ✓ {out_csv}")

    del adata, scores
    gc.collect()

print("""
══════════════════════════════════════════════════════════════
 DONE — cytotrace2_scores_klughammer_*.csv (20 files)
 Next: python3 K03_hexbin_correlation.py
══════════════════════════════════════════════════════════════
""")
