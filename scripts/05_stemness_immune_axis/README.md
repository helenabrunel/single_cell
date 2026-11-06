# Stemness immune axis

Define and characterize the stemness-by-immune axis and the SCIE and NSI subpopulations.

| Script | What it does |
|---|---|
| `01_state_scoring_hexbin_tumor_compartments.py` | Scores cell states and runs hexbin density and cell-cycle quadrant analysis on tumor compartments. |
| `02_state_scoring_hexbin_metastatic_compartments.py` | Runs the same state scoring, hexbin and quadrant analysis on metastatic compartments. |
| `03_stem_immune_cell_cycle_corrected.py` | Repeats the stemness-by-immunogenicity analysis with cell-cycle correction. |
| `04_compare_stem_immune_metrics.py` | Compares metrics for the non-linear stem-immune relationship, beyond Pearson r. |
| `05_signature_immune_correlation.py` | Meta-analyzes per-patient correlation of each stemness signature with the immune consensus score. |
| `06_subpopulation_metrics.py` | Computes and meta-analyzes per-patient metrics for detecting the SCIE and NSI subpopulations. |
| `07_signature_selection.py` | Selects the stemness signature by rank stability across patients and thresholds. |
| `08_patient_hexbin.py` | Plots a stem-by-immune density hexbin per patient, with analysis thresholds overlaid. |
| `09_phenotype_classification.py` | Classifies cells into phenotypes with one per-patient threshold and reports their distribution. |
| `10_axis_signatures_umap.py` | Scores subtype-specific SCIE and NSI signatures and plots them on Harmony UMAPs. |
| `11_axis_signature_states.py` | Assigns each cell its dominant axis signature, per subtype. |
| `12_axis_signature_labels.py` | Labels each cell SCIE, NSI, SCI or NC from its two scores, one UMAP per subtype. |
| `13_axis_phenotype_umaps.py` | Plots four UMAPs per subtype: phenotype, patient, CytoTRACE2 potency and related annotations. |
| `14_stem_signatures_umap.py` | Plots fMaSC_over_aMaSC and LIM_MAMMARY_STEM_CELL_UP scores on UMAPs per subtype. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
