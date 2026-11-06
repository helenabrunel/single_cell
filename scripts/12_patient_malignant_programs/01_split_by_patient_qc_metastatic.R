# 01_split_raw_by_patient_malignant_QC.R
# Per-patient processing of Klughammer malignant cells (adapted from the Wu et al.
# per-patient script): QC plots, cell filtering, normalisation, cell-cycle scoring,
# scaling with CC.Difference regression, PCA + elbow plot, one RDS per patient.
#
# Dataset-specific choices:
#   - Malignant cells: meta.data$compartments == "Malignant"
#     (= author_cell_type MBC + MBC_stem-like + MBC_chondroid)
#   - Patient ID: meta.data$donor_id (one sample per donor; donors with 2 channels
#     are pooled, the channel is kept in orig.ident and shown in the QC plots)
#   - Subtype: from meta.data$receptors_biopsy, same mapping as 01_prepare_dataset.py
#       any HER2+ -> HER2 | ER+ & HER2- -> ER | ER-/PR-/HER2- -> TNBC | nan -> excluded
#   - Features are Ensembl IDs: renamed to gene symbols (feature_name) so that the
#     ^MT- / ^RP[SL] patterns and cc.genes.updated.2019 work.
# Run from: ~/single_cell/Klughammer/KlughammeR/

library(Seurat)
library(dplyr)
library(patchwork)
library(ggplot2)
library(Matrix)

# Output folders
base_path <- "01_malignant_per_patient/"

folder_map <- list(
  "HER2" = paste0(base_path, "clust_rev_HER2_data/"),
  "TNBC" = paste0(base_path, "clust_rev_TNBC_data/"),
  "ER"   = paste0(base_path, "clust_rev_ER_data/")
)
for (d in folder_map) dir.create(d, recursive = TRUE, showWarnings = FALSE)

MIN_CELLS <- 100 # patients with fewer malignant cells after QC are skipped

seurat_obj <- readRDS("klughammer_seurat.rds")

# ── 0. Malignant cells, gene symbols, subtype ─────────────────────────────
seurat_obj <- subset(seurat_obj, subset = compartments == "Malignant")

counts <- LayerData(seurat_obj, assay = "RNA", layer = "counts")
mf     <- seurat_obj[["RNA"]][[]]
rownames(counts) <- make.unique(as.character(mf[rownames(counts), "feature_name"]))

md <- seurat_obj@meta.data
seurat_obj <- CreateSeuratObject(counts = counts, meta.data = md, project = "Klughammer")
rm(counts, md, mf); invisible(gc())

map_subtype <- function(r) {
  r <- as.character(r)
  case_when(
    grepl("HER2+", r, fixed = TRUE)                                   ~ "HER2",
    grepl("ER+", r, fixed = TRUE) & grepl("HER2-", r, fixed = TRUE)   ~ "ER",
    r == "ER-/PR-/HER2-"                                              ~ "TNBC",
    TRUE                                                              ~ NA_character_
  )
}
seurat_obj$subtype <- map_subtype(seurat_obj$receptors_biopsy)

cat("Malignant cells per patient and subtype:\n")
print(table(droplevels(seurat_obj$donor_id), seurat_obj$subtype, useNA = "ifany"))

# Get unique patients
patients <- sort(unique(as.character(seurat_obj@meta.data$donor_id)))

summary_rows <- list()

for (patient in patients) {

  cat("\n Processing patient:", patient, "\n")

  # ── 1. Subset patient (malignant cells only) ──────────────────────────
  pd1 <- subset(seurat_obj, cells = colnames(seurat_obj)[seurat_obj$donor_id == patient])
  n_raw <- ncol(pd1)

  # ── 2. Get subtype and assign folder ─────────────────────────────────
  patient_subtype <- unique(pd1@meta.data$subtype)

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

  # ── 4. QC plots (split by sequencing channel) ────────────────────────
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
