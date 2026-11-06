# Tim3 hypoxia

Test the relationship between TIM3 (HAVCR2) and hypoxia, controlling for sequencing depth.

| Script | What it does |
|---|---|
| `01_hypoxia_tim3_overview.py` | First overview of the hypoxia signature (GO:0001666) and HAVCR2 across all objects. |
| `02_hypoxia_tim3_overview_positive_cells.py` | Later version of that overview, with an option to restrict hexbins to HAVCR2-positive cells. |
| `03_hypoxia_tim3_extended.py` | Adds further figure types to the hypoxia-by-TIM3 overview. |
| `04_aucell_pathways_threshold_groups.py` | AUCell pathway comparison of TIM3-high/hypoxia-low and TIM3-low/hypoxia-high epithelial cells. |
| `05_dea_gsea_threshold_groups.py` | Differential expression and pre-ranked GSEA between those two groups. |
| `06_depth_adjusted_reanalysis.py` | Re-analyzes the relationship adjusting for depth, after the groups proved confounded by sequencing depth. |
| `07_dea_gsea_filtered_genes.py` | Retries differential expression and GSEA with gene filtering. |
| `08_patient_pseudobulk.py` | Tests the relationship at patient level with pseudobulk. |
| `09_genomewide_depth_regression.py` | Runs the depth-offset regression genome-wide at single-cell resolution. |
| `10_naive_vs_depth_corrected_plots.py` | Plots the original hexbins beside depth-corrected versions to show the artifact. |
| `11_clean_analysis_metastatic.py` | The same clean analysis in metastatic malignant cells. |
| `11_clean_analysis_primary.py` | Clean, depth-corrected analysis of TIM3 and hypoxia in primary tumor epithelial cells. |
| `12_confident_positive_primary.py` | Sensitivity check that drops single-UMI HAVCR2-positive cells. |
| `13_clipped_violins_metastatic.py` | Replots the clean analysis with clipped violin axes (display only, metastases). |
| `13_clipped_violins_primary.py` | Replots the clean analysis with clipped violin axes (display only, primary tumors). |
| `14_reverse_comparison_metastatic.py` | Reverse test: compares percent HAVCR2-positive across hypoxia tertiles (metastases). |
| `14_reverse_comparison_primary.py` | Reverse test: compares percent HAVCR2-positive across hypoxia tertiles (primary tumors). |
| `15_reverse_quartiles_metastatic.py` | Sensitivity check of the reverse test using quartiles (metastases). |
| `15_reverse_quartiles_primary.py` | Sensitivity check of the reverse test using quartiles (primary tumors). |
| `16_collect_final_results.py` | Gathers the final TIM3 and hypoxia results into one self-contained folder. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
