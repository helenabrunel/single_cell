# Single-cell analysis of breast cancer

Single-cell transcriptomic analyses of breast cancer, centered on the relationship between stemness and immune evasion. Scripts are organized by scientific objective.

## Repository structure

| Folder | Objective |
|---|---|
| [`scripts/01_data_preparation`](scripts/01_data_preparation/README.md) | Turn raw inputs into filtered, normalized and integrated objects, and move data between Python and R. |
| [`scripts/02_integration_benchmark`](scripts/02_integration_benchmark/README.md) | Choose a batch-integration method for malignant cells and benchmark the candidates. |
| [`scripts/03_potency_scoring`](scripts/03_potency_scoring/README.md) | Estimate cellular potency with CytoTRACE2 and LandSCENT and compare the two. |
| [`scripts/04_stemness_cell_cycle`](scripts/04_stemness_cell_cycle/README.md) | Find which stemness signatures capture developmental potency independently of the cell cycle. |
| [`scripts/05_stemness_immune_axis`](scripts/05_stemness_immune_axis/README.md) | Define and characterize the stemness-by-immune axis and the SCIE and NSI subpopulations. |
| [`scripts/06_tim3_hypoxia`](scripts/06_tim3_hypoxia/README.md) | Test the relationship between TIM3 (HAVCR2) and hypoxia, controlling for sequencing depth. |
| [`scripts/07_ros_tme`](scripts/07_ros_tme/README.md) | Study reactive oxygen species metabolism across the tumor microenvironment. |
| [`scripts/08_gene_expression_by_tissue`](scripts/08_gene_expression_by_tissue/README.md) | Compare expression of genes of interest across lineages, tissues and subtypes. |
| [`scripts/09_lcor_cxcl16`](scripts/09_lcor_cxcl16/README.md) | Investigate the LCOR, CXCL16 and CXCR6 relationship across cell types and patients. |
| [`scripts/10_lcor_tcell_scores`](scripts/10_lcor_tcell_scores/README.md) | Relate tumor LCOR programs to T-cell phenotypes, using patient-level scores on non-integrated data. |
| [`scripts/11_gamma_delta_tcells`](scripts/11_gamma_delta_tcells/README.md) | Identify gamma-delta T cells, with TCR data and from expression alone. |
| [`scripts/12_patient_malignant_programs`](scripts/12_patient_malignant_programs/README.md) | Find recurrent malignant gene programs by processing each patient separately. |
| [`scripts/13_presentation`](scripts/13_presentation/README.md) | Build slides and figures for presentations. |
| `landscent_src/` | LandSCENT functions, including a sparse-matrix version of CompSRana, and their checks. |
| `signatures/` | Curated gene signatures. |
| `docs/` | Methods, results notes, the [decision log](docs/DECISIONS.md) and the script index. |

Each folder under `scripts/` has a README that explains every script. Within a folder, number prefixes give the order to run them in. Scripts sharing a number do the same analysis for a different context: `_primary` for primary tumors and `_metastatic` for metastases.

## Running a script

Data is not stored in this repository. Each script must be run from the folder that holds its data, listed in [docs/SCRIPT_INDEX.md](docs/SCRIPT_INDEX.md):

```bash
cd <folder from the index>
python <repository path>/scripts/<objective>/<script>.py
```

Some scripts contain absolute paths to the original machine (`/home/hbrunel/single_cell`). Update those paths to run them elsewhere.

## Requirements

Python packages are listed in `requirements.txt`.

R packages: AnnotationDbi, celda, cowplot, data.table, DelayedArray, dendextend, doParallel, dplyr, dynamicTreeCut, foreach, future, ggplot2, glmGamPoi, hdf5r, igraph, jsonlite, knitr, Matrix, msigdbr, NMF, org.Hs.eg.db, patchwork, readxl, rhdf5, scales, Seurat, SeuratObject, tidyr, UCell, wordspace.
