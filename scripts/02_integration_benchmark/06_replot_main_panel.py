"""
part0_replot_main_panel.py
==========================
Regenerate only the _umap_panel.png figures from saved processed h5ads.
Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 part0_replot_main_panel.py
"""
import sys
sys.path.insert(0, ".")
# Import the updated plotting functions from the main script
import importlib.util, types

# Load the module without executing the main loop
src = open("scripts/02_integration_benchmark/02_integrate_malignant_by_subtype.py").read()
# Replace the main loop with a no-op so only functions are defined
src = src[:src.index("\nfor dataset_name, cfg in DATASETS.items():")]
exec(compile(src, "scripts/02_integration_benchmark/02_integrate_malignant_by_subtype.py", "exec"), globals())

import scanpy as sc

DATASETS_REPLOT = {
    "Visvader"   : {
        **DATASETS["Visvader"],
        "path"   : "visvader_malignant_processed.h5ad",
        "compartment_col": None,
    },
    "Klughammer" : {
        **DATASETS["Klughammer"],
        "path"   : "klughammer_malignant_processed.h5ad",
        "compartment_col": None,
    },
}

for dataset_name, cfg in DATASETS_REPLOT.items():
    print(f"\n--- {dataset_name} ---")
    adata = sc.read_h5ad(cfg["path"])
    methods_done = [m for m in METHODS + ["scVI", "DRVI"]
                    if f"X_umap_{m}" in adata.obsm]
    print(f"  methods: {methods_done}")
    plot_main_umap_panel(adata, cfg, dataset_name, methods_done)

print("\nDone.")
