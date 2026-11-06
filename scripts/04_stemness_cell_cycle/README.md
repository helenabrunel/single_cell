# Stemness cell cycle

Find which stemness signatures capture developmental potency independently of the cell cycle.

| Script | What it does |
|---|---|
| `01_evaluate_signatures_vs_cell_cycle.py` | Evaluates the 9 stemness signatures against potency and cell cycle, and builds cell-cycle-free scores. |
| `02_plot_hexbin_cc_free.py` | Plots hexbins of stemness versus S score, before and after cell-cycle residualization. |
| `03_plot_partial_r_bars.py` | Plots the ranked bar chart of partial correlation with CytoTRACE2 given cell cycle. |
| `04_plot_per_patient.py` | Plots per-patient distributions of each metric and signature. |
| `05_plot_standalone_results.py` | Plots the standalone before/after and trade-off result figures. |
| `06_subtype_outlier_analysis.py` | Compares subtypes per patient and detects outlier patients. |
| `07_subtype_summary.py` | Combines subtype metrics and patient outliers into one summary figure. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
