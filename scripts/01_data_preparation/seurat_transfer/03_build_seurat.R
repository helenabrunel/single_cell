# 03_build_seurat.R
# =================
# Creates the INITIAL Seurat (v5) object for Klughammer and Regner from the
# validated count matrices of 02_import_counts.R.
#
# Deliberately minimal — NO cell/gene filtering, NO QC metrics, NO
# normalisation. Downstream steps (malignant selection, QC, per-patient split,
# SCTransform, NMF) are left to the analyst.
#
#   counts layer    : raw integer UMIs, verified identical to the .rds input
#   meta.data       : all original cell metadata (author's nCount_RNA /
#                     nFeature_RNA kept as *_author, since Seurat recomputes them)
#   orig.ident      : sample (Klughammer: sampleid; Regner: sample_name)
#   RNA feature meta: ensembl_id / original symbol for every rowname
#   @misc$provenance: source files, md5, versions
#
# Output: seurat_transfer/<dataset>/<dataset>_seurat_raw.rds
#
# Run from: ~/single_cell/
#   Rscript seurat_transfer/03_build_seurat.R [klughammer|regner|both]

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
which_ds <- if (length(args)) args[1] else "both"
OUT <- "seurat_transfer"

build <- function(name, sample_col) {
  cat("\n================", name, "================\n")
  d <- readRDS(file.path(OUT, name, paste0(name, "_counts.rds")))
  meta <- d$meta
  stopifnot(identical(rownames(meta), colnames(d$counts)),
            identical(rownames(d$counts), d$features$rowname))

  # Keep author-provided QC columns under a different name so CreateSeuratObject's
  # own nCount_RNA / nFeature_RNA (from these exact counts) are not overwritten.
  for (col in c("nCount_RNA", "nFeature_RNA", "orig.ident")) {
    if (col %in% colnames(meta)) {
      colnames(meta)[colnames(meta) == col] <- paste0(col, "_author")
    }
  }

  obj <- CreateSeuratObject(counts = d$counts, assay = "RNA", project = name,
                            meta.data = meta, min.cells = 0, min.features = 0)
  obj$orig.ident <- factor(meta[[sample_col]])
  Idents(obj) <- "orig.ident"

  rownames(d$features) <- d$features$rowname
  for (col in setdiff(colnames(d$features), "rowname")) {
    obj[["RNA"]][[col]] <- d$features[rownames(obj), col]
  }
  obj@misc$provenance <- c(d$provenance, list(
    seurat_object_created = format(Sys.time()),
    Seurat = as.character(packageVersion("Seurat")),
    SeuratObject = as.character(packageVersion("SeuratObject")),
    note = "Raw counts only; no filtering/QC/normalisation applied."))

  # ── checks: the object must contain exactly the validated matrix ────────────
  cnt <- LayerData(obj, assay = "RNA", layer = "counts")
  stopifnot(
    identical(dim(obj), dim(d$counts)),
    identical(rownames(obj), rownames(d$counts)),
    identical(colnames(obj), colnames(d$counts)),
    identical(cnt@x, d$counts@x), identical(cnt@i, d$counts@i),
    identical(cnt@p, d$counts@p),
    all(obj$nCount_RNA == colSums(d$counts)),
    all(obj$nFeature_RNA == diff(d$counts@p)),
    all(colnames(meta) %in% colnames(obj[[]])))
  cat("  Counts layer identical to validated matrix: OK\n")
  cat(sprintf("  %d genes x %d cells | %d samples | Seurat %s\n",
              nrow(obj), ncol(obj), nlevels(obj$orig.ident),
              packageVersion("Seurat")))
  print(obj)

  f <- file.path(OUT, name, paste0(name, "_seurat_raw.rds"))
  saveRDS(obj, f, compress = FALSE)
  cat("  Saved:", f, "\n")
}

if (which_ds %in% c("klughammer", "both")) build("klughammer", "sampleid")
if (which_ds %in% c("regner", "both"))     build("regner", "sample_name")
