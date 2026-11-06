#!/usr/bin/env Rscript
# 10_landscent_scoring.R
# Run LandSCENT SR scoring on Visvader epithelial compartment (per subtype).
# Avoids full LandSCENT package install by sourcing only the two core functions
# (DoIntegPPI, CompSRana) and the built-in PPI network directly from GitHub.
#
# Outputs: landscent_scores_Epithelial_{ER,HER2,TNBC,Global}.csv
# Same naming convention as cytotrace2_scores_Epithelial_*.csv

suppressPackageStartupMessages({
  library(Matrix)
  library(igraph)
  library(wordspace)
  library(DelayedArray)
  library(parallel)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

# ── Download LandSCENT core files (once) ──────────────────────────────────────
LANDSCENT_DIR <- file.path("/home/hbrunel/single_cell", "landscent_src")
dir.create(LANDSCENT_DIR, showWarnings = FALSE)

BASE_URL <- "https://raw.githubusercontent.com/ChenWeiyan/LandSCENT/master"

download_if_missing <- function(url, dest) {
  if (!file.exists(dest)) {
    message("Downloading ", basename(dest), " ...")
    download.file(url, dest, quiet = TRUE)
  }
}

download_if_missing(
  paste0(BASE_URL, "/R/DoIntegPPI.R"),
  file.path(LANDSCENT_DIR, "DoIntegPPI.R")
)
download_if_missing(
  paste0(BASE_URL, "/R/CompSRana.R"),
  file.path(LANDSCENT_DIR, "CompSRana.R")
)

# PPI network (net13Jun12.m) — stored as RData in GitHub LFS / raw
PPI_PATH <- file.path(LANDSCENT_DIR, "net13Jun12.m.RData")
download_if_missing(
  paste0(BASE_URL, "/data/net13Jun12.m.RData"),
  PPI_PATH
)

# Source functions — suppress @import roxygen tags (they're comments, fine)
source(file.path(LANDSCENT_DIR, "DoIntegPPI.R"))
# The PPI network is >99% zeros (density 0.85%, avg degree ~72 over 8434 genes).
# CompSRana.R's per-cell entropy step builds a dense gene x gene matrix, which
# is ~350x slower and far more memory-hungry than necessary. CompSRana_sparse.R
# is a sparse-matrix reimplementation verified numerically identical (SR values
# match to ~1e-15, machine precision) against CompSRana.R on real data — see
# landscent_src/verify_sparse_landscent.R.
source(file.path(LANDSCENT_DIR, "CompSRana_sparse.R"))

# Load PPI network into environment
load(PPI_PATH)   # loads net13Jun12.m
message("PPI network: ", nrow(net13Jun12.m), " genes")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   <- "/home/hbrunel/single_cell/Visvader"
EXPORT_DIR <- "/home/hbrunel/single_cell/landscent_export"
CHUNK_DIR  <- file.path(DATA_DIR, "landscent_chunks")
dir.create(CHUNK_DIR, showWarnings = FALSE)
subtypes   <- c("ER", "HER2", "TNBC")
all_scores <- list()

# Cells are processed in small batches so peak memory is bounded by batch size,
# not by total subtype size (this script previously crashed 3x processing full
# subtypes at once — see landscent_src/verify_chunking.R). DoIntegPPI's log-
# transform normalises by lib_max = max library size *across the cells in that
# call*, so naive chunking silently changes results (~3% of the SR range) —
# lib_max must be computed once globally per subtype and applied identically
# to every chunk (verified to match an unchunked run to machine precision).
CHUNK_SIZE <- 5000

manual_log_transform <- function(exp_chunk, lib_max_global) {
  temp.m <- wordspace::normalize.cols(exp_chunk, method = "manhattan")
  temp.m@x <- log2((temp.m@x * lib_max_global) + 1.1)
  temp.dgT <- as(temp.m, "dgTMatrix")
  data.m <- as.matrix(temp.dgT)
  data.m[data.m == 0] <- log2(1.1)
  data.m
}

# ── Per-subtype scoring ───────────────────────────────────────────────────────
for (subtype in subtypes) {
  out_path <- file.path(DATA_DIR, paste0("landscent_scores_Epithelial_", subtype, ".csv"))

  if (file.exists(out_path)) {
    message("\n=== ", subtype, ": already scored — skipping ===")
    all_scores[[subtype]] <- read.csv(out_path)
    next
  }

  mtx_path   <- file.path(EXPORT_DIR, paste0("logcounts_", subtype, ".mtx"))
  genes_path <- file.path(EXPORT_DIR, paste0("genes_", subtype, ".txt"))
  cells_path <- file.path(EXPORT_DIR, paste0("cells_", subtype, ".csv"))

  if (!file.exists(mtx_path)) {
    stop("Export files not found for ", subtype,
         ".\nRun first: python3 Visvader/10a_export_for_landscent.py")
  }

  message("\n=== Processing ", subtype, " ===")

  # Read exported sparse matrix (genes × cells)
  message("  Reading sparse matrix ...")
  exp_mat  <- readMM(mtx_path)
  symbols  <- readLines(genes_path)
  cell_meta <- read.csv(cells_path, row.names = 1)

  rownames(exp_mat) <- symbols
  colnames(exp_mat) <- rownames(cell_meta)
  exp_mat <- as(exp_mat, "dgCMatrix")
  message("  Matrix: ", nrow(exp_mat), " genes × ", ncol(exp_mat), " cells")

  # Map gene symbols → Entrez IDs (PPI network uses Entrez)
  entrez_ids <- mapIds(org.Hs.eg.db, keys = symbols,
                       column = "ENTREZID", keytype = "SYMBOL",
                       multiVals = "first")
  keep <- !is.na(entrez_ids) & !duplicated(entrez_ids)
  exp_mat    <- exp_mat[keep, ]
  rownames(exp_mat) <- entrez_ids[keep]

  # Pre-subset to PPI genes only — avoids 28k×N dense conversion inside DoIntegPPI
  ppi_keep <- rownames(exp_mat) %in% rownames(net13Jun12.m)
  exp_mat  <- exp_mat[ppi_keep, ]
  message("  Genes in PPI network: ", nrow(exp_mat))

  # lib_max must be computed once across ALL cells in the subtype (not per
  # chunk) so every chunk's log-transform is normalised identically — see
  # comment near CHUNK_SIZE above.
  lib_max_global <- max(Matrix::colSums(exp_mat))
  n_cells <- ncol(exp_mat)
  chunk_starts <- seq(1, n_cells, by = CHUNK_SIZE)
  n_chunks <- length(chunk_starts)
  message("  ", n_cells, " cells in ", n_chunks, " chunks of up to ", CHUNK_SIZE,
          " (lib_max=", lib_max_global, ")")

  chunk_files <- character(n_chunks)
  for (ci in seq_along(chunk_starts)) {
    start_idx  <- chunk_starts[ci]
    end_idx    <- min(start_idx + CHUNK_SIZE - 1, n_cells)
    chunk_file <- file.path(CHUNK_DIR, paste0(subtype, "_chunk", sprintf("%04d", ci), ".csv"))
    chunk_files[ci] <- chunk_file

    if (file.exists(chunk_file)) {
      message("  chunk ", ci, "/", n_chunks, ": already done — skipping")
      next
    }

    t0 <- Sys.time()
    exp_chunk   <- exp_mat[, start_idx:end_idx]
    data_chunk  <- manual_log_transform(exp_chunk, lib_max_global)
    integ_chunk <- DoIntegPPI(exp.m = data_chunk, ppiA.m = net13Jun12.m, log_trans = FALSE)
    integ_chunk <- CompSRana_sparse(integ_chunk, local = FALSE, mc.cores = 1)

    sr_vec        <- integ_chunk$SR
    cell_barcodes <- colnames(integ_chunk$expMC)
    chunk_meta    <- cell_meta[start_idx:end_idx, , drop = FALSE]

    chunk_df <- data.frame(
      cell_id = rownames(chunk_meta),
      sample  = chunk_meta$sample,
      subtype = chunk_meta$subtype,
      SR      = sr_vec[match(rownames(chunk_meta), cell_barcodes)],
      row.names = NULL
    )
    write.csv(chunk_df, chunk_file, row.names = FALSE)

    rm(exp_chunk, data_chunk, integ_chunk); gc()
    message("  chunk ", ci, "/", n_chunks, ": ", end_idx - start_idx + 1, " cells in ",
            round(difftime(Sys.time(), t0, units = "secs"), 1), "s")
  }

  df <- do.call(rbind, lapply(chunk_files, read.csv))
  message("  SR range: ", round(min(df$SR, na.rm=TRUE), 4),
          " – ", round(max(df$SR, na.rm=TRUE), 4))
  message("  Writing ", basename(out_path))
  write.csv(df, out_path, row.names = FALSE)

  all_scores[[subtype]] <- df
  file.remove(chunk_files)
  rm(exp_mat); gc()
}

# ── Global CSV ────────────────────────────────────────────────────────────────
global_df  <- do.call(rbind, all_scores)
global_out <- file.path(DATA_DIR, "landscent_scores_Epithelial_Global.csv")
write.csv(global_df, global_out, row.names = FALSE)
message("\nGlobal scores written to ", basename(global_out))
message("SR summary (all subtypes):")
print(summary(global_df$SR))
message("\nDone.")
