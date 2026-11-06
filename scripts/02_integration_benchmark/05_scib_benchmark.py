"""
part0_scib_benchmark.py
========================
Runs the scib-metrics Benchmarker on the per-subtype objects produced by
part0_scib_prepare.py (figures/part0/scib_inputs/{Dataset}_{Subtype}.h5ad).

Embeddings compared per object: X_pca_fresh (Unintegrated) / X_harmony / X_DRVI
batch_key : patient (sample / sampleid)
label_key : bio_label — argmax of FINE_EPI marker scores, computed on
            unintegrated expression (independent of any embedding)

Run in the isolated 'scib-metrics' conda env (numpy>=2, incompatible with
the scanpy/scvi-tools/drvi/cytotrace2 stack used for prepare/training):
  /home/hbrunel/miniconda3/envs/scib-metrics/bin/python part0_scib_benchmark.py

Outputs:
  figures/part0/scib_inputs/{Dataset}_{Subtype}_scib_results.csv
  figures/part0/scib_inputs/{Dataset}_{Subtype}_scib_results.png
  figures/part0/scib_inputs/scib_summary_all_objects.csv   (combined table)
"""

import scanpy as sc
import pandas as pd
import os, glob, warnings
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

INDIR = "figures/part0/scib_inputs"

BATCH_COL_BY_DATASET = {
    "Visvader": "sample",
    "Klughammer": "sampleid",
}

EMBEDDINGS = ["X_pca_fresh", "X_harmony", "X_DRVI"]

all_summaries = []

files = sorted(glob.glob(os.path.join(INDIR, "*.h5ad")))
print(f"Found {len(files)} prepared objects:")
for f in files:
    print(" ", f)

for f in files:
    base = os.path.basename(f).replace(".h5ad", "")
    dataset_name, subtype = base.split("_", 1)
    batch_col = BATCH_COL_BY_DATASET[dataset_name]

    print(f"\n{'='*70}\n {base}\n{'='*70}")
    adata = sc.read_h5ad(f)
    print(f"  {adata.n_obs:,} cells | batch={batch_col} "
          f"({adata.obs[batch_col].nunique()} patients) | "
          f"embeddings={[e for e in EMBEDDINGS if e in adata.obsm]}")

    present_embeddings = [e for e in EMBEDDINGS if e in adata.obsm]
    if len(present_embeddings) < 2:
        print("  ⚠  Fewer than 2 embeddings present — skipping")
        continue

    bm = Benchmarker(
        adata,
        batch_key=batch_col,
        label_key="bio_label",
        embedding_obsm_keys=present_embeddings,
        pre_integrated_embedding_obsm_key="X_pca_fresh",
        bio_conservation_metrics=BioConservation(
            isolated_labels=True,
            nmi_ari_cluster_labels_leiden=False,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=True,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=True, ilisi_knn=True, kbet_per_label=True,
            graph_connectivity=True, pcr_comparison=True,
        ),
        n_jobs=4,
    )
    bm.benchmark()

    df = bm.get_results(min_max_scale=False)
    out_csv = os.path.join(INDIR, f"{base}_scib_results.csv")
    df.to_csv(out_csv)
    print(f"  ✓ {out_csv}")
    print(df.to_string())

    try:
        plot_dir = os.path.join(INDIR, f"_plot_{base}")
        os.makedirs(plot_dir, exist_ok=True)
        bm.plot_results_table(min_max_scale=False, show=False, save_dir=plot_dir)
        svg_path = os.path.join(plot_dir, "scib_results.svg")
        out_svg = os.path.join(INDIR, f"{base}_scib_results.svg")
        os.replace(svg_path, out_svg)
        os.rmdir(plot_dir)
        print(f"  ✓ {out_svg}")
    except Exception as e:
        print(f"  ⚠  plot failed: {e}")

    df_long = df.reset_index().rename(columns={"index": "Method"})
    df_long.insert(0, "object", base)
    all_summaries.append(df_long)

if all_summaries:
    combined = pd.concat(all_summaries, ignore_index=True)
    out = os.path.join(INDIR, "scib_summary_all_objects.csv")
    combined.to_csv(out, index=False)
    print(f"\n✓ Combined summary: {out}")

print("\nDONE")
