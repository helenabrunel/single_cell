# 02_SCTransform_clustering_per_patient.R
# Per-patient SCTransform (v2, regressing CC.Difference) + PCA + UMAP + clustering
# on the malignant-cell objects produced by 01_split_raw_by_patient_malignant_QC.R.
# Parameters kept identical to the original per-patient script:
#   SCTransform(vars.to.regress = "CC.Difference", vst.flavor = "v2")
#   RunPCA(npcs = 50) | RunUMAP(dims = 1:50) | FindNeighbors(dims = 1:50)
#   FindClusters(resolution = 0.7)
#
# Patients with fewer than MIN_CELLS malignant cells after QC are not clustered
# (too few cells for 50 PCs / resolution 0.7): their RDS files are deleted and they
# are logged as "too_few_cells". Cell numbers come from summary_cells_per_patient.csv
# (written by script 01); patients with n_filt = NA there (saved by an earlier,
# interrupted run of 01) are checked by the worker after reading the RDS.
#
# Memory-safe execution:
#   - Each patient runs in its own R process (Rscript <this script> <file>), so memory
#     is fully released between patients and an out-of-memory kill only loses that
#     patient; the driver logs the failure and continues with the next one.
#   - Resumable: patients whose _sctransform.rds already exists are skipped.
#   - Output is written to a temporary file and renamed at the end, so an interrupted
#     run never leaves a truncated .rds that would be skipped on resume.
#
# Run after script 01 has finished, from: ~/single_cell/Klughammer/KlughammeR/
#   Rscript 02_SCTransform_clustering_per_patient.R          # all patients
#   Rscript 02_SCTransform_clustering_per_patient.R <file>   # one patient (worker)

args <- commandArgs(trailingOnly = TRUE)

MIN_CELLS <- 100      # minimum malignant cells (after QC) to run SCTransform + clustering
RC_TOO_FEW_CELLS <- 3 # worker exit code when the patient is below MIN_CELLS

# ── Worker: process one patient ─────────────────────────────────────────────
if (length(args) == 1) {

  suppressMessages({
    library(dplyr)
    library(patchwork)
    library(glmGamPoi)
    library(ggplot2)
    library(Matrix)
    library(Seurat)
  })
  options(future.globals.maxSize = 16 * 1024^3)

  file     <- args[1]
  patient  <- sub("_cc_rgrsd\\.rds$", "", basename(file))
  out_file <- file.path(dirname(file), paste0(patient, "_sctransform.rds"))
  t0 <- Sys.time()

  # ── 1. Read RDS ─────────────────────────────────────────────────────────
  pd1 <- readRDS(file)
  cat("Patient:", patient, "| cells:", ncol(pd1), "\n")

  if (ncol(pd1) < MIN_CELLS) {
    cat("Only", ncol(pd1), "cells (<", MIN_CELLS, ") — not clustered.\n")
    quit(status = RC_TOO_FEW_CELLS)
  }

  # 50 PCs as in the original; capped only if a patient has <= 50 cells
  n_dims <- min(50, ncol(pd1) - 1)

  # ── 2. SCTransform + clustering pipeline ────────────────────────────────
  pd1 <- SCTransform(pd1,
                     vars.to.regress = "CC.Difference",
                     vst.flavor      = "v2",
                     verbose         = FALSE) %>%
    RunPCA(npcs = n_dims, verbose = FALSE) %>%
    RunUMAP(reduction = "pca", dims = 1:n_dims, verbose = FALSE) %>%
    FindNeighbors(reduction = "pca", dims = 1:n_dims, verbose = FALSE) %>%
    FindClusters(resolution = 0.7, verbose = FALSE)

  cat("Clusters:", nlevels(pd1$seurat_clusters), "\n")
  print(table(pd1$seurat_clusters))

  # ── 3. Save (atomic) ────────────────────────────────────────────────────
  tmp_file <- paste0(out_file, ".tmp")
  saveRDS(pd1, file = tmp_file)
  file.rename(tmp_file, out_file)

  cat("✓ Saved:", out_file, "| time:", format(Sys.time() - t0), "\n")
  quit(status = 0)
}

# ── Driver: loop over subtypes and patients ─────────────────────────────────
base_path <- "01_malignant_per_patient/"

input_dirs <- list(
  "ER"   = paste0(base_path, "clust_rev_ER_data/"),
  "HER2" = paste0(base_path, "clust_rev_HER2_data/"),
  "TNBC" = paste0(base_path, "clust_rev_TNBC_data/")
)

log_dir <- paste0(base_path, "logs_02_sctransform/")
dir.create(log_dir, showWarnings = FALSE)

this_script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
status <- list()

# Delete a patient's RDS files (input and any previous SCTransform output)
drop_patient <- function(input_dir, patient) {
  files <- paste0(input_dir, patient, c("_cc_rgrsd.rds", "_sctransform.rds"))
  files <- files[file.exists(files)]
  if (length(files) > 0) {
    file.remove(files)
    cat("    Deleted:", paste(basename(files), collapse = ", "), "\n")
  }
}

# ── Patients below MIN_CELLS after QC (from script 01 summary) ──────────────
summary_file <- paste0(base_path, "summary_cells_per_patient.csv")
if (!file.exists(summary_file)) {
  stop("Not found: ", summary_file, " — run script 01 to completion first.")
}
summary_df <- read.csv(summary_file)
too_few    <- summary_df$patient[!is.na(summary_df$n_filt) & summary_df$n_filt < MIN_CELLS]

cat("Patients with <", MIN_CELLS, "cells after QC (excluded):\n")
print(summary_df[summary_df$patient %in% too_few, ], row.names = FALSE)
if (any(is.na(summary_df$n_filt))) {
  cat("n_filt = NA (checked by the worker):",
      paste(summary_df$patient[is.na(summary_df$n_filt)], collapse = ", "), "\n")
}

for (subtype in names(input_dirs)) {

  input_dir <- input_dirs[[subtype]]

  # Get all _cc_rgrsd.rds files for this subtype
  rds_files <- list.files(input_dir, pattern = "_cc_rgrsd\\.rds$", full.names = TRUE)

  if (length(rds_files) == 0) {
    warning(paste("No files found in:", input_dir))
    next
  }

  cat("\n── Subtype:", subtype, "| Patients found:", length(rds_files), "\n")

  for (file in rds_files) {

    patient  <- sub("_cc_rgrsd\\.rds$", "", basename(file))
    out_file <- paste0(input_dir, patient, "_sctransform.rds")

    if (patient %in% too_few) {
      cat("  – Excluded:", patient, "(<", MIN_CELLS, "cells after QC)\n")
      drop_patient(input_dir, patient)
      status[[patient]] <- data.frame(patient, subtype, result = "too_few_cells")
      next
    }

    if (file.exists(out_file)) {
      cat("  Skipping", patient, "(already done)\n")
      status[[patient]] <- data.frame(patient, subtype, result = "already_done")
      next
    }

    cat("  Processing patient:", patient, "...", format(Sys.time(), "%H:%M"), "\n")
    log_file <- paste0(log_dir, patient, ".log")
    rc <- system2("Rscript", c(this_script, shQuote(file)),
                  stdout = log_file, stderr = log_file)

    if (rc == 0 && file.exists(out_file)) {
      cat("  ✓ Saved:", out_file, "\n")
      status[[patient]] <- data.frame(patient, subtype, result = "ok")
    } else if (rc == RC_TOO_FEW_CELLS) {
      cat("  – Excluded:", patient, "(<", MIN_CELLS, "cells) — see", log_file, "\n")
      drop_patient(input_dir, patient)
      status[[patient]] <- data.frame(patient, subtype, result = "too_few_cells")
    } else {
      # rc 137 = killed (usually out of memory)
      cat("  ✗ FAILED:", patient, "(exit code", rc, ") — see", log_file, "\n")
      status[[patient]] <- data.frame(patient, subtype, result = paste0("failed_rc", rc))
    }
  }
}

status_df <- do.call(rbind, status)
write.csv(status_df, paste0(base_path, "status_02_sctransform.csv"), row.names = FALSE)
print(status_df)

cat("\n All patients processed.\n")
