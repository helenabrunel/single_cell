# Lcor cxcl16

Investigate the LCOR, CXCL16 and CXCR6 relationship across cell types and patients.

## 01_exploratory

| Script | What it does |
|---|---|
| `01_exploratory/01_celltype_specificity_overview.py` | Shows where LCOR, CXCL16 and CXCR6 are expressed across compartments, cell types and subtypes. |
| `01_exploratory/02_tissue_contribution_by_celltype.py` | Measures each cell type's linear-scale contribution to tissue CXCL16 and LCOR signal. |
| `01_exploratory/03_malignant_structure_no_harmony.py` | Overview panel of the malignant compartment on the non-Harmony embedding. |

## 02_correlation

| Script | What it does |
|---|---|
| `02_correlation/01_compartment_hexbin_scan.py` | Uncorrected per-cell hexbin scan of the three genes across all cell groups and subtypes. |
| `02_correlation/02_depth_artifact_cell_level.py` | Depth-adjusted LCOR and CXCL16 analysis in malignant cells, per patient. |
| `02_correlation/03_depth_artifact_diagnostic_hexbin.py` | Per-patient hexbins of co-detected cells, grouped by subtype. |

## 03_pseudobulk

| Script | What it does |
|---|---|
| `03_pseudobulk/01_pseudobulk_by_patient_headline.py` | Correlates per-patient pseudobulk LCOR and CXCL16 in malignant cells. |

## 04_liana

| Script | What it does |
|---|---|
| `04_liana/01_liana_interaction_vs_lcor.py` | Tests the CXCL16 to CXCR6 interaction per patient with LIANA and relates it to LCOR. |

## 05_myeloid

| Script | What it does |
|---|---|
| `05_myeloid/01_pseudobulk_myeloid_headline.py` | Correlates per-patient pseudobulk LCOR and CXCL16 in myeloid cells. |
| `05_myeloid/02_depth_artifact_myeloid_cell_level.py` | Depth-adjusted cell-level test of LCOR and CXCL16 in myeloid cells. |

## 06_liana_within_patient

| Script | What it does |
|---|---|
| `06_liana_within_patient/01_liana_lcor_split_within_patient.py` | Exploratory LIANA comparison of LCOR-based cell splits within each patient. |

## 07_apm_positive_control

| Script | What it does |
|---|---|
| `07_apm_positive_control/01_pseudobulk_apm_malignant.py` | Positive control: per-patient pseudobulk of LCOR and the APM score in malignant cells. |
| `07_apm_positive_control/02_depth_artifact_apm_malignant_cell_level.py` | Positive control: cell-level depth-offset GLM of LCOR and APM in malignant cells. |
| `07_apm_positive_control/03_pseudobulk_apm_myeloid.py` | Positive control: per-patient pseudobulk of LCOR and APM in myeloid cells. |
| `07_apm_positive_control/04_depth_artifact_apm_myeloid_cell_level.py` | Positive control: cell-level depth-offset GLM of LCOR and APM in myeloid cells. |
| `07_apm_positive_control/05_apm_lcor_by_cd8_infiltration.py` | Checks whether the LCOR and APM association depends on CD8 infiltration. |
| `07_apm_positive_control/06_method_control_esr1_greb1.py` | Method check using the known ESR1 and GREB1 relationship. |
| `07_apm_positive_control/07_replication_metastatic_cohort.py` | Tests whether the LCOR and APM sign reversal replicates in an independent metastatic cohort. |
| `07_apm_positive_control/08_lcor_esr1_relationship.py` | Examines the relationship between LCOR and ESR1. |
| `07_apm_positive_control/09_replication_primary_cohort.py` | Second independent replication of the APM control and the ESR1 method check. |

## 08_umap_gene_trios

| Script | What it does |
|---|---|
| `08_umap_gene_trios/01_umap_trio_lcor_apm_esr1.py` | UMAP feature plots of LCOR, APM score and ESR1 by subtype. |
| `08_umap_gene_trios/02_umap_trio_lcor_apm_esr1_myeloid.py` | The same trio in myeloid cells. |
| `08_umap_gene_trios/03_umap_trio_lcor_cxcl16_cxcr6.py` | UMAP feature plots of LCOR, CXCL16 and CXCR6 by subtype. |

## 10_tcell_program

| Script | What it does |
|---|---|
| `10_tcell_program/01_build_signatures.py` | Builds the three gene signatures for the LCOR program and T-cell phenotype analysis. |
| `10_tcell_program/02_tcell_phenotype_test_primary.py` | Tests whether the tumor LCOR program associates with an LCOR-induced CD8 phenotype (primary tumors). |
| `10_tcell_program/03_tcell_phenotype_test_metastatic.py` | The same test in metastases. |
| `10_tcell_program/04_lcor_oe_vs_cxcl16_cxcr6_primary.py` | Correlates the LCOR_OE signature with CXCL16 and CXCR6 (primary tumors). |
| `10_tcell_program/05_lcor_oe_vs_cxcl16_cxcr6_metastatic.py` | Replicates that correlation in metastases. |
| `10_tcell_program/06_scatterplots.py` | Regenerates all LCOR correlations as per-patient scatterplots in a reference style. |
| `10_tcell_program/07_composition_plot.py` | Plots CD8 cluster composition per patient, ordered by malignant LCOR_OE score. |
| `10_tcell_program/08_celltypist_annotation.py` | Replaces Leiden reclustering with CellTypist annotation. |
| `10_tcell_program/09_celltypist_annotation_all_t_cells.py` | Applies CellTypist annotation to all T cells. |
| `10_tcell_program/10_cd8_vs_cd4_lineage.py` | Separates CD8 and CD4 T cells using CellTypist labels rather than marker thresholds. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
