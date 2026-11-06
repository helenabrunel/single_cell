# 02_import_counts.R
# ==================
# Rebuilds the raw UMI count matrices of Klughammer and Regner in R directly
# from the ORIGINAL sources (no object conversion), validates them against the
# Python reference written by 01_export_reference.py, and saves
#   seurat_transfer/<dataset>/<dataset>_counts.rds
# = list(counts = dgCMatrix genes x cells, meta = data.frame, features = data.frame,
#        provenance = list)
#
# Seurat is NOT needed here, so the transfer is independent of the Seurat /
# sctransform version used later (03_build_seurat.R).
#
# Any mismatch stops the script (stopifnot) — nothing is saved unless every
# check passes with exact integer equality.
#
# Run from: ~/single_cell/
#   Rscript seurat_transfer/02_import_counts.R [klughammer|regner|both]

suppressPackageStartupMessages({
  library(Matrix)
  library(rhdf5)
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
which_ds <- if (length(args)) args[1] else "both"
OUT <- "seurat_transfer"

# ── helpers ───────────────────────────────────────────────────────────────────
read_ref <- function(name) {
  d <- file.path(OUT, name)
  list(
    obs     = fread(file.path(d, "obs.csv.gz"), data.table = FALSE),
    var     = fread(file.path(d, "var.csv.gz"), data.table = FALSE),
    cells   = fread(file.path(d, "cell_checks.csv.gz"), data.table = FALSE),
    genes   = fread(file.path(d, "gene_checks.csv.gz"), data.table = FALSE),
    entries = fread(file.path(d, "entry_checks.csv.gz"), data.table = FALSE),
    summary = fromJSON(file.path(d, "summary.json"))
  )
}

# Same feature naming Seurat would apply (Read10X uses make.unique on symbols;
# CreateSeuratObject silently turns "_" into "-"), done explicitly here so the
# rownames never change behind our back. Mapping kept in `features`.
seurat_rownames <- function(symbol) make.unique(gsub("_", "-", symbol))

validate <- function(M, ref, label) {
  cat("\n── Validating", label, "──\n")
  s <- ref$summary
  chk <- function(desc, ok) {
    cat(sprintf("  %-48s %s\n", desc, if (isTRUE(ok)) "OK" else "FAIL"))
    if (!isTRUE(ok)) stop(label, ": check failed -> ", desc)
  }
  chk("dimensions (genes x cells)", identical(dim(M), c(s$n_genes, s$n_cells)))
  chk("cell barcodes identical & same order", identical(colnames(M), ref$cells$cell))
  chk("metadata rows identical & same order", identical(colnames(M), ref$obs$cell))
  chk("nnz", length(M@x) == s$nnz)
  chk("grand total UMIs", sum(M@x) == s$grand_total)
  chk("all counts non-negative integers", all(M@x >= 0 & M@x == round(M@x)))
  chk("per-cell total UMIs (all cells)", all(colSums(M) == ref$cells$total))
  chk("per-cell n detected (all cells)", all(diff(M@p) == ref$cells$n_genes))
  chk("per-gene total UMIs (all genes)", all(rowSums(M) == ref$genes$total))
  chk("per-gene n cells (all genes)",
      all(tabulate(M@i + 1L, nbins = nrow(M)) == ref$genes$n_cells))
  e <- ref$entries
  chk(sprintf("%d random entries cell x gene", nrow(e)),
      all(M[cbind(e$gene_idx, e$cell_idx)] == e$value))
  chk("dgCMatrix internally valid", { validObject(M); TRUE })
  invisible(TRUE)
}

finish <- function(M, ref, features, provenance, name) {
  meta <- ref$obs
  rownames(meta) <- meta$cell
  meta$cell <- NULL
  out <- list(counts = M, meta = meta, features = features,
              provenance = c(provenance, list(
                created = format(Sys.time()), R = R.version.string,
                Matrix = as.character(packageVersion("Matrix")))))
  f <- file.path(OUT, name, paste0(name, "_counts.rds"))
  saveRDS(out, f, compress = FALSE)
  cat("  Saved:", f, "\n")
}

# ── Klughammer: raw/X CSR arrays read straight from the CELLxGENE h5ad ────────
# CSR (cells x genes) has exactly the memory layout of CSC (genes x cells),
# so data/indices/indptr map 1:1 onto dgCMatrix x/i/p. No transposition, no copy
# through another tool.
if (which_ds %in% c("klughammer", "both")) {
  cat("\n================ Klughammer ================\n")
  src <- "Klughammer/data/dataset.h5ad"
  ref <- read_ref("klughammer")
  enc <- h5readAttributes(src, "raw/X")
  stopifnot(enc$`encoding-type` == "csr_matrix")
  shp <- as.numeric(enc$shape)                       # (n_cells, n_genes)

  x <- as.numeric(h5read(src, "raw/X/data"))
  i <- as.integer(h5read(src, "raw/X/indices", bit64conversion = "double"))
  p <- as.integer(h5read(src, "raw/X/indptr",  bit64conversion = "double"))
  cells <- as.character(h5read(src, paste0("obs/", h5readAttributes(src, "obs")$`_index`)))
  genes <- as.character(h5read(src, paste0("raw/var/", h5readAttributes(src, "raw/var")$`_index`)))
  h5closeAll()

  stopifnot(identical(genes, ref$var$ensembl_id))
  features <- data.frame(
    rowname    = seurat_rownames(ref$var$symbol),
    ensembl_id = ref$var$ensembl_id,
    symbol     = ref$var$symbol,
    biotype    = ref$var$feature_biotype,
    stringsAsFactors = FALSE)
  cat("  Symbols renamed by '_'->'-' / make.unique:",
      sum(features$rowname != features$symbol), "\n")

  M <- new("dgCMatrix", i = i, p = p, x = x,
           Dim = c(as.integer(shp[2]), as.integer(shp[1])),
           Dimnames = list(features$rowname, cells))
  rm(x, i, p); gc(verbose = FALSE)

  validate(M, ref, "klughammer")
  finish(M, ref, features,
         list(source = src, source_md5 = unname(tools::md5sum(src)),
              slot = "raw/X (integer UMI counts; .X is log-normalised)"),
         "klughammer")
  rm(M); gc(verbose = FALSE)
}

# ── Regner: original GEO 10x mtx files, one per sample ────────────────────────
# Barcodes prefixed "<sample_name>_" and samples ordered by GSM number, exactly
# as in Regner/01_load_samples.py, so cell names match the Python objects.
if (which_ds %in% c("regner", "both")) {
  cat("\n================ Regner ================\n")
  rna_dir <- "GSE243526_RNA"
  ref <- read_ref("regner")
  bfiles <- list.files(rna_dir, pattern = "^GSM\\d+_.+_RNA_barcodes\\.tsv\\.gz$")
  gsm    <- sub("^(GSM\\d+)_.*", "\\1", bfiles)
  sname  <- sub("^GSM\\d+_(.+)_RNA_barcodes\\.tsv\\.gz$", "\\1", bfiles)
  ord <- order(as.integer(sub("GSM", "", gsm)))
  gsm <- gsm[ord]; sname <- sname[ord]

  feat0 <- NULL; mats <- list(); md5 <- character()
  for (k in seq_along(gsm)) {
    pre <- file.path(rna_dir, paste0(gsm[k], "_", sname[k], "_RNA"))
    f_mtx <- paste0(pre, "_matrix.mtx.gz")
    feat <- fread(paste0(pre, "_features.tsv.gz"), header = FALSE, data.table = FALSE)
    bc   <- fread(paste0(pre, "_barcodes.tsv.gz"), header = FALSE, data.table = FALSE)$V1
    if (is.null(feat0)) feat0 <- feat
    stopifnot(identical(feat, feat0))                 # same gene reference
    m <- as(readMM(f_mtx), "CsparseMatrix")            # genes x cells
    stopifnot(nrow(m) == nrow(feat), ncol(m) == length(bc))
    colnames(m) <- paste0(sname[k], "_", bc)
    mats[[k]] <- m
    md5[paste0(gsm[k], "_", sname[k])] <- unname(tools::md5sum(f_mtx))
    cat(sprintf("  %s %-22s %6d cells\n", gsm[k], sname[k], ncol(m)))
  }
  M <- do.call(cbind, mats); rm(mats); gc(verbose = FALSE)
  if (!is(M, "dgCMatrix")) M <- as(M, "dgCMatrix")

  stopifnot(identical(feat0$V1, ref$var$ensembl_id),
            identical(feat0$V2, ref$var$symbol))
  features <- data.frame(
    rowname    = seurat_rownames(feat0$V2),
    ensembl_id = feat0$V1,
    symbol     = feat0$V2,
    feature_type = feat0$V3,
    stringsAsFactors = FALSE)
  cat("  Symbols renamed by '_'->'-' / make.unique:",
      sum(features$rowname != features$symbol), "\n")
  rownames(M) <- features$rowname

  validate(M, ref, "regner")
  finish(M, ref, features,
         list(source = rna_dir, source_md5 = as.list(md5),
              slot = "GEO 10x filtered matrices (integer UMI counts), unfiltered by Python QC; see meta$passed_python_qc"),
         "regner")
}

cat("\nAll checks passed.\n")
