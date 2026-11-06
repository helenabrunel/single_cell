# Integration benchmark

Choose a batch-integration method for malignant cells and benchmark the candidates.

| Script | What it does |
|---|---|
| `01_compare_integration_methods.py` | Compares batch-integration methods on malignant and epithelial cells. |
| `02_integrate_malignant_by_subtype.py` | Integrates malignant cells independently for each subtype (ER, HER2, TNBC). |
| `03_plot_drvi_reconstruction_scores.py` | Plots DRVI latent-dimension reconstruction scores for each per-subtype model. |
| `04_scib_prepare.py` | Saves per-subtype objects with the embeddings needed for the scib-metrics benchmark. |
| `05_scib_benchmark.py` | Runs the scib-metrics Benchmarker on those objects (uses the scib-metrics environment). |
| `06_replot_main_panel.py` | Regenerates the main UMAP panels from saved objects, without recomputing. |
| `07_replot_all_signatures.py` | Regenerates the all-signature UMAP figures from saved objects. |
| `08_replot_per_signature.py` | Plots one figure per stemness signature, with integration methods as rows. |
| `09_replot_per_signature_panel.py` | Plots the main-panel layout once per stemness signature. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
