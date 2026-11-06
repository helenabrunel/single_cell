"""
cytotrace2_scoring.py
=====================
Runs CytoTRACE2 on all compartment objects:
  - 3 global  : compartment_epithelial/immune/stromal.h5ad
  - 9 per subtype: epithelial/immune/stromal_ER/HER2/TNBC.h5ad

Memory-efficient approach:
  - TSV written gene-by-gene from sparse matrix (no full dense matrix in RAM)
  - Batch size: 10k cells
  - Top 5000 expressed genes per batch
  - Auto-resume from partial CSV if crashed

Outputs: cytotrace2_scores_[object_name].csv

Run from: ~/single_cell/
  python3 cytotrace2_scoring.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os, tempfile, warnings, gc
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

BATCH_SIZE  = 10_000
TOP_N_GENES = 5_000

FILES = {
    # All cells (no compartment split)
    "All_Global": "all_global.h5ad",
    "All_ER"    : "all_ER.h5ad",
    "All_HER2"  : "all_HER2.h5ad",
    "All_TNBC"  : "all_TNBC.h5ad",
    # Epithelial
    "Epithelial_Global": "compartment_epithelial.h5ad",
    "Epithelial_ER"    : "epithelial_ER.h5ad",
    "Epithelial_HER2"  : "epithelial_HER2.h5ad",
    "Epithelial_TNBC"  : "epithelial_TNBC.h5ad",
    # Immune
    "Immune_Global": "compartment_immune.h5ad",
    "Immune_ER"    : "immune_ER.h5ad",
    "Immune_HER2"  : "immune_HER2.h5ad",
    "Immune_TNBC"  : "immune_TNBC.h5ad",
    # Stromal
    "Stromal_Global": "compartment_stromal.h5ad",
    "Stromal_ER"    : "stromal_ER.h5ad",
    "Stromal_HER2"  : "stromal_HER2.h5ad",
    "Stromal_TNBC"  : "stromal_TNBC.h5ad",
}


def write_sparse_tsv(X, gene_names, cell_names, filepath):
    """Write genes×cells TSV gene-by-gene from sparse matrix."""
    X_csc = X.tocsc() if issparse(X) else None
    with open(filepath, "w") as f:
        f.write("\t" + "\t".join(cell_names) + "\n")
        for gene_idx, gene in enumerate(gene_names):
            values = np.asarray(
                X_csc.getcol(gene_idx).todense()
            ).flatten() if X_csc is not None else X[:, gene_idx]
            row = [str(int(v)) if v == int(v) else f"{v:.4f}"
                   for v in values]
            f.write(gene + "\t" + "\t".join(row) + "\n")


def select_top_genes(X, gene_names, top_n):
    gene_sums = np.asarray(X.sum(axis=0)).flatten() if issparse(X) \
                else np.asarray(X).sum(axis=0)
    top_idx   = sorted(np.argsort(gene_sums)[::-1][:top_n])
    return X[:, top_idx], [gene_names[i] for i in top_idx]


def run_cytotrace2_batched(adata, name, batch_size, top_n_genes):
    from cytotrace2_py.cytotrace2_py import cytotrace2 as ct2

    n_cells   = adata.n_obs
    n_batches = int(np.ceil(n_cells / batch_size))
    print(f"\n  {name}: {n_cells:,} cells → {n_batches} batches")

    X_raw = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X_top, top_genes = select_top_genes(
        X_raw, adata.var_names.tolist(), top_n_genes
    )
    print(f"  Top {len(top_genes):,} genes selected")

    # Resume from partial if available
    partial_csv = f"cytotrace2_scores_{name}_partial.csv"
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

        X_batch = X_top[start:end]
        gene_sums  = np.asarray(X_batch.sum(axis=0)).flatten() \
                     if issparse(X_batch) else X_batch.sum(axis=0)
        nz         = gene_sums > 0
        X_nz       = X_batch[:, nz]
        genes_nz   = [g for g, keep in zip(top_genes, nz) if keep]
        print(f"    Non-zero genes: {len(genes_nz):,}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv  = os.path.join(tmpdir, "expression.txt")
            odir = os.path.join(tmpdir, "ct2_out")

            print(f"    Writing sparse TSV...")
            write_sparse_tsv(X_nz, genes_nz, cell_names, tsv)
            print(f"    TSV: {os.path.getsize(tsv)/1e6:.0f} MB")

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
                res = pd.DataFrame({"CytoTRACE2_Score":
                                    pred[sc_col].astype(float)},
                                   index=pred.index)
                if pt_col:
                    res["CytoTRACE2_Potency"] = pred[pt_col]

                print(f"    ✓ {len(res):,} cells "
                      f"(range {res['CytoTRACE2_Score'].min():.3f}"
                      f"–{res['CytoTRACE2_Score'].max():.3f})")
                all_results.append(res)

                # Save progress
                partial = pd.concat(all_results).drop_duplicates()
                partial.to_csv(partial_csv)
                print(f"    ✓ Progress saved: "
                      f"{len(partial):,}/{n_cells:,} "
                      f"({100*len(partial)/n_cells:.0f}%)")

            except Exception as e:
                print(f"    ⚠️  Batch {batch_idx+1} failed: {e}")
                import traceback; traceback.print_exc()
                all_results.append(pd.DataFrame(
                    {"CytoTRACE2_Score": np.nan}, index=cell_names
                ))

        del X_batch, X_nz
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
print(" CytoTRACE2 — MEMORY-EFFICIENT BATCHED SCORING")
print("="*65)
print(f"  Batch size      : {BATCH_SIZE:,} cells")
print(f"  Top genes       : {TOP_N_GENES:,}")
print(f"  Peak RAM/batch  ≈ {BATCH_SIZE*TOP_N_GENES*4/1e6:.0f} MB")
print(f"  Auto-resume     : yes")
print(f"  Objects to score: {len(FILES)}")

for name, fpath in FILES.items():
    out_csv = f"cytotrace2_scores_{name}.csv"

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
 DONE
══════════════════════════════════════════════════════════════
  cytotrace2_scores_Epithelial.csv
  cytotrace2_scores_Immune.csv
  cytotrace2_scores_Stromal.csv
  cytotrace2_scores_Epithelial_ER/HER2/TNBC.csv
  cytotrace2_scores_Immune_ER/HER2/TNBC.csv
  cytotrace2_scores_Stromal_ER/HER2/TNBC.csv

  Next: python3 04_scoring_hexbin.py
══════════════════════════════════════════════════════════════
""")
