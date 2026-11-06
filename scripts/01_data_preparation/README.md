# Data preparation

Turn raw inputs into filtered, normalized and integrated objects, and move data between Python and R.

| Script | What it does |
|---|---|
| `00_h5seurat_to_rds.R` | Builds a Seurat object directly from an h5seurat file, avoiding SeuratDisk incompatibilities. |
| `01_filter_primary_tumors.py` | Keeps primary tumor samples only, removing normal tissue and excluded groups before QC. |
| `01_load_10x_samples.py` | Loads raw 10x samples and concatenates them into one AnnData object. |
| `01_prepare_metastatic_cohort.py` | Prepares the metastatic cohort and builds its integrated subtype-by-compartment objects. |
| `02_add_clinical_metadata.py` | Attaches tissue type, ER/PR/HER2 status and derived subtype to the raw and processed objects. |
| `02_convert_residual_ensembl_ids.py` | Converts residual Ensembl-formatted gene names to HGNC symbols with mygene. |
| `02_qc_filter_tumor_only.py` | Maps Ensembl IDs to gene symbols, then runs QC, filtering and normalization on tumor-only data. |
| `03_qc_filter_mixed_tissue.py` | Computes QC metrics, filters cells and genes, and normalizes a cohort with normal and tumor tissue. |
| `03_split_and_integrate_compartments.py` | Builds the integrated objects for each subtype and compartment (epithelial, immune, stromal). |
| `04_cohort_overview.py` | Plots cohort composition and cells per patient or cell line, before and after QC. |
| `04_reintegrate_filtered_er_epithelium.py` | Re-integrates the ER epithelial object after excluding 11 samples, with unchanged parameters. |
| `05_subtype_overview.py` | Plots cohort composition by tissue type and, within tumors, by clinical subtype. |
| `06_convert_h5ad_to_seurat.R` | Converts a raw h5ad into a Seurat v5 object without loss, reading it directly with rhdf5. |
| `07_dump_seurat_for_verification.R` | Step 1 of the equivalence check: dumps the Seurat object to plain files. |
| `08_verify_seurat_equivalence.py` | Step 2 of the equivalence check: compares that dump against the original h5ad. |

## seurat_transfer

| Script | What it does |
|---|---|
| `seurat_transfer/01_export_reference.py` | Exports the Python-side reference used to validate a transfer to R/Seurat without converting objects. |
| `seurat_transfer/02_import_counts.R` | Rebuilds raw count matrices in R from the original sources and validates them against the reference. |
| `seurat_transfer/03_build_seurat.R` | Creates the initial Seurat v5 objects from the validated count matrices. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
