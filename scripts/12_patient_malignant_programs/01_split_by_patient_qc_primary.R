# 05_split_raw_by_patient_malignant_QC.R
# Per-patient processing of Regner (GSE243526) malignant cells, mirroring
# ../Klughammer/KlughammeR/01_split_raw_by_patient_malignant_QC.R step by step:
# QC plots, cell filtering, normalisation, cell-cycle scoring, scaling with
# CC.Difference regression, PCA + elbow plot, one RDS per patient.
#
# Dataset-specific choices:
#   - Input: gse243526_raw_seurat.rds (lossless conversion of gse243526_raw.h5ad,
#     see 04_convert_raw_to_seurat.R / 04b_verify_seurat_equivalence.py)
#   - Malignant cells: inferCNV == "high" in the authors' per-cell annotation
#     (Table S2, Full_Cohort_barcode_meta, from github.com/RegnerM2015/
#     scBreast_scRNA_scATAC_2024/miscellaneous), as defined in the paper.
#     Cells are matched by biopsy code + 10x barcode.
#   - Patient ID: meta.data$patient. Patient_14 has two samples (two foci of a
#     multifocal lobular tumour, 43E7BL / 43E7CL); they are pooled (no
#     integration), the sample is kept in orig.ident and shown in the QC plots.
#   - Subtype: receptor string built from the extended clinical table (Table S1,
#     gse243526_clinical_extended.tsv), then the same mapping as Klughammer:
#       any HER2+ -> HER2 | ER+ & HER2- -> ER | ER-/PR-/HER2- -> TNBC | else -> excluded
#     HER2 IHC 2+ (equivocal) with FISH "Not Amplified" is HER2- (ASCO/CAP).
#     Result: ER = 9 patients, TNBC = 1 (Patient_6), HER2 = 0.
#   - Patient_5 (ER-/PR+/HER2-) does not fit the rule. Added as TNBC with an
#     asterisk (project decision, 2026-10-14): neuroendocrine/small cell carcinoma in
#     Table S1, but basal-like (85% of malignant cells Basal_SC in Table S2; in the
#     authors' basal cohort). Flagged in meta.data$subtype_note.
#   - Features are already gene symbols: no renaming needed.
# Run from: ~/single_cell/Regner/

library(Seurat)
library(dplyr)
library(patchwork)
library(ggplot2)
library(Matrix)
library(readxl)

# Output folders
base_path <- "05_malignant_per_patient/"

folder_map <- list(
  "HER2" = paste0(base_path, "clust_rev_HER2_data/"),
  "TNBC" = paste0(base_path, "clust_rev_TNBC_data/"),
  "ER"   = paste0(base_path, "clust_rev_ER_data/")
)
for (d in folder_map) dir.create(d, recursive = TRUE, showWarnings = FALSE)

MIN_CELLS <- 100  # patients with fewer malignant cells after QC are skipped (same as Klughammer 02)

seurat_obj <- readRDS("gse243526_raw_seurat.rds")

# ── 0. Malignant cells (authors' inferCNV), subtype ───────────────────────
cnv <- read_excel("gse243526_authors_TableS2_scRNA_barcode_metadata.xlsx",
                  sheet = "Full_Cohort_barcode_meta")
cnv_key <- paste0(cnv$sample, "|", sub("_[0-9]+$", "", cnv$barcode))
stopifnot(!anyDuplicated(cnv_key))

obj_key <- paste0(as.character(seurat_obj$biopsy_code), "|",
                  sub(".*_", "", colnames(seurat_obj)))
stopifnot(all(cnv_key %in% obj_key))  # every annotated cell exists in our data

seurat_obj$inferCNV <- cnv$inferCNV[match(obj_key, cnv_key)]  # NA = removed by authors' QC
seurat_obj$author_cell_type <- cnv$cell_type_label[match(obj_key, cnv_key)]

cat("Authors' inferCNV calls (NA = not in authors' filtered data):\n")
print(table(seurat_obj$inferCNV, useNA = "ifany"))

seurat_obj <- subset(seurat_obj, cells = colnames(seurat_obj)[which(seurat_obj$inferCNV == "high")])

clin <- read.delim("gse243526_clinical_extended.tsv", check.names = FALSE)
er   <- ifelse(grepl("^Positive", clin$ER_IHC), "ER+", ifelse(grepl("^Negative", clin$ER_IHC), "ER-", NA))
pr   <- ifelse(grepl("^Positive", clin$PR_IHC), "PR+", ifelse(grepl("^Negative", clin$PR_IHC), "PR-", NA))
her2 <- case_when(
  clin$Her2_IHC == "Positive" | clin$Her2_FISH == "Amplified"         ~ "HER2+",
  clin$Her2_IHC == "Negative"                                          ~ "HER2-",
  clin$Her2_IHC == "Equivocal" & clin$Her2_FISH == "Not Amplified"     ~ "HER2-",
  TRUE                                                                 ~ NA_character_
)
receptors <- ifelse(is.na(er) | is.na(pr) | is.na(her2), NA, paste(er, pr, her2, sep = "/"))
names(receptors) <- clin$Sample_ID
seurat_obj$receptors_biopsy <- unname(receptors[as.character(seurat_obj$biopsy_code)])

map_subtype <- function(r) {
  r <- as.character(r)
  case_when(
    grepl("HER2+", r, fixed = TRUE)                                   ~ "HER2",
    grepl("ER+", r, fixed = TRUE) & grepl("HER2-", r, fixed = TRUE)   ~ "ER",
    r == "ER-/PR-/HER2-"                                              ~ "TNBC",
    TRUE                                                              ~ NA_character_
  )
}
seurat_obj$subtype_receptors <- map_subtype(seurat_obj$receptors_biopsy)

# Patient_5: TNBC* (see header)
p5 <- seurat_obj$patient == "Patient_5"
seurat_obj$subtype_receptors[p5] <- "TNBC"
seurat_obj$subtype_note <- NA_character_
seurat_obj$subtype_note[p5] <- "TNBC* (ER-/PR+/HER2-, neuroendocrine/small cell carcinoma; basal-like)"

cat("Malignant cells per patient and subtype:\n")
print(table(droplevels(seurat_obj$patient), seurat_obj$subtype_receptors, useNA = "ifany"))

# Get unique patients
patients <- sort(unique(as.character(seurat_obj@meta.data$patient)))

summary_rows <- list()

for (patient in patients) {

  cat("\n Processing patient:", patient, "\n")

  # ── 1. Subset patient (malignant cells only) ──────────────────────────
  pd1 <- subset(seurat_obj, cells = colnames(seurat_obj)[seurat_obj$patient == patient])
  n_raw <- ncol(pd1)

  # ── 2. Get subtype and assign folder ─────────────────────────────────
  patient_subtype <- unique(pd1@meta.data$subtype_receptors)

  if (length(patient_subtype) > 1) {
    warning(paste("Patient", patient, "has multiple subtypes:",
                  paste(patient_subtype, collapse = ", "), "— skipping."))
    next
  }

  if (is.na(patient_subtype) || !patient_subtype %in% names(folder_map)) {
    warning(paste("Patient", patient, "has unknown subtype:", patient_subtype, "— skipping."))
    next
  }

  folder <- folder_map[[patient_subtype]]

  # Resume: skip patients already saved by a previous (interrupted) run
  if (file.exists(paste0(folder, patient, "_cc_rgrsd.rds"))) {
    cat(" Already done — skipping.\n")
    summary_rows[[patient]] <- data.frame(patient, subtype = patient_subtype,
                                          n_raw, n_filt = NA, saved = TRUE)
    next
  }

  # ── 3. QC metrics ────────────────────────────────────────────────────
  pd1[["percent.mito"]] <- PercentageFeatureSet(pd1, pattern = "^MT-")
  pd1[["percent.ribo"]] <- PercentageFeatureSet(pd1, pattern = "^RP[SL]")

  # ── 4. QC plots (split by sample) ────────────────────────────────────
  vln <- VlnPlot(pd1,
                 features = c("nFeature_RNA", "nCount_RNA", "percent.mito", "percent.ribo"),
                 group.by = "orig.ident", ncol = 4, pt.size = 0.1) +
    plot_annotation(title = paste("QC —", patient, "(", patient_subtype, ")"))

  ggsave(filename = paste0(folder, patient, "_QC_violin.png"),
         plot = vln, width = 16, height = 6)

  scatter1 <- FeatureScatter(pd1, feature1 = "nCount_RNA", feature2 = "percent.mito",
                             group.by = "orig.ident")
  scatter2 <- FeatureScatter(pd1, feature1 = "nCount_RNA", feature2 = "nFeature_RNA",
                             group.by = "orig.ident")

  scatter_plot <- scatter1 + scatter2 +
    plot_annotation(title = paste("QC scatter —", patient))

  ggsave(filename = paste0(folder, patient, "_QC_scatter.png"),
         plot = scatter_plot, width = 14, height = 6)

  # ── 5. Filter cells ───────────────────────────────────────────────────
  pd1 <- subset(
    x      = pd1,
    subset = nCount_RNA   < 75000 &
      nCount_RNA   > 200   &
      percent.mito < 25
  )
  n_filt <- ncol(pd1)

  if (n_filt < MIN_CELLS) {
    warning(paste("Patient", patient, "has only", n_filt, "cells after QC — skipping."))
    summary_rows[[patient]] <- data.frame(patient, subtype = patient_subtype,
                                          n_raw, n_filt, saved = FALSE)
    next
  }

  # ── 6. Normalize & variable features ─────────────────────────────────
  pd1 <- NormalizeData(pd1)
  pd1 <- FindVariableFeatures(pd1, selection.method = "vst", nfeatures = 3000)

  # ── 7. Cell cycle scoring ─────────────────────────────────────────────
  s.genes   <- cc.genes.updated.2019$s.genes
  g2m.genes <- cc.genes.updated.2019$g2m.genes

  pd1 <- CellCycleScoring(pd1,
                          s.features   = s.genes,
                          g2m.features = g2m.genes,
                          set.ident    = TRUE)

  # ── 8. Scale & PCA (regressing CC difference) ────────────────────────
  pd1$CC.Difference <- pd1$S.Score - pd1$G2M.Score

  pd1 <- ScaleData(pd1,
                   vars.to.regress = "CC.Difference",
                   features        = rownames(pd1))

  pd1 <- RunPCA(pd1, features = VariableFeatures(pd1),
                npcs = min(50, ncol(pd1) - 1))

  # ── 9. Elbow plot ─────────────────────────────────────────────────────
  elbow <- ElbowPlot(pd1, ndims = min(50, ncol(pd1) - 1)) +
    ggtitle(paste("Elbow —", patient))

  ggsave(filename = paste0(folder, patient, "_elbow.png"),
         plot = elbow, width = 10, height = 6)

  # ── 10. Save RDS ──────────────────────────────────────────────────────
  # Atomic save: an interrupted run never leaves a truncated .rds behind
  tmp_file <- paste0(folder, patient, "_cc_rgrsd.rds.tmp")
  saveRDS(pd1, file = tmp_file)
  file.rename(tmp_file, paste0(folder, patient, "_cc_rgrsd.rds"))

  cat(" ✓ Saved:", paste0(folder, patient, "_cc_rgrsd.rds"), "\n")

  summary_rows[[patient]] <- data.frame(patient, subtype = patient_subtype,
                                        n_raw, n_filt, saved = TRUE)

  # Clean up memory
  rm(pd1)
  gc()
}

summary_df <- bind_rows(summary_rows)
write.csv(summary_df, paste0(base_path, "summary_cells_per_patient.csv"), row.names = FALSE)
print(summary_df)

cat("\n All patients processed.\n")
