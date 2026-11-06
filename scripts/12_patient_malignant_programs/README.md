# Patient malignant programs

Find recurrent malignant gene programs by processing each patient separately.

| Script | What it does |
|---|---|
| `00_h5seurat_to_rds.R` | Builds the Seurat input for the per-patient workflow from an h5seurat file. |
| `01_split_by_patient_qc_metastatic.R` | Splits malignant cells by patient and runs QC and filtering (metastases). |
| `01_split_by_patient_qc_primary.R` | Splits malignant cells by patient and runs QC and filtering (primary tumors). |
| `02_sctransform_clustering_metastatic.R` | Runs SCTransform, PCA, UMAP and clustering per patient (metastases). |
| `02_sctransform_clustering_primary.R` | Runs SCTransform, PCA, UMAP and clustering per patient (primary tumors). |
| `03_cnmf_metastatic.R` | Identifies per-patient gene programs with consensus NMF (metastases). |
| `03_cnmf_primary.R` | Identifies per-patient gene programs with consensus NMF (primary tumors). |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
