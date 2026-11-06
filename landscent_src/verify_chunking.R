# Checks the chunked computation in CompSRana_sparse.R.
suppressPackageStartupMessages({
  library(Matrix); library(igraph); library(wordspace); library(DelayedArray)
  library(parallel); library(org.Hs.eg.db); library(AnnotationDbi)
})

LANDSCENT_DIR <- "/home/hbrunel/single_cell/landscent_src"
source(file.path(LANDSCENT_DIR, "DoIntegPPI.R"))
source(file.path(LANDSCENT_DIR, "CompSRana_sparse.R"))
load(file.path(LANDSCENT_DIR, "net13Jun12.m.RData"))

EXPORT_DIR <- "/home/hbrunel/single_cell/landscent_export"
exp_mat  <- readMM(file.path(EXPORT_DIR, "logcounts_HER2.mtx"))
symbols  <- readLines(file.path(EXPORT_DIR, "genes_HER2.txt"))
cell_meta <- read.csv(file.path(EXPORT_DIR, "cells_HER2.csv"), row.names = 1)
rownames(exp_mat) <- symbols
colnames(exp_mat) <- rownames(cell_meta)
exp_mat <- as(exp_mat, "dgCMatrix")

entrez_ids <- mapIds(org.Hs.eg.db, keys = symbols, column = "ENTREZID", keytype = "SYMBOL", multiVals = "first")
keep <- !is.na(entrez_ids) & !duplicated(entrez_ids)
exp_mat <- exp_mat[keep, ]; rownames(exp_mat) <- entrez_ids[keep]
ppi_keep <- rownames(exp_mat) %in% rownames(net13Jun12.m)
exp_mat <- exp_mat[ppi_keep, ]

n_test <- 900
exp_test <- exp_mat[, 1:n_test]

# ── Reference: unchunked, all 900 cells in one DoIntegPPI+CompSRana_sparse call ──
integ_all <- DoIntegPPI(exp.m = exp_test, ppiA.m = net13Jun12.m, log_trans = TRUE)
integ_all <- CompSRana_sparse(integ_all, local = FALSE, mc.cores = 1)
sr_unchunked <- integ_all$SR
names(sr_unchunked) <- colnames(integ_all$expMC)

# ── Chunked, using a GLOBAL lib_max applied manually per chunk ────────────────
manual_log_transform <- function(exp_chunk, lib_max_global) {
  temp.m <- wordspace::normalize.cols(exp_chunk, method = "manhattan")
  temp.m@x <- log2((temp.m@x * lib_max_global) + 1.1)
  temp.dgT <- as(temp.m, "dgTMatrix")
  data.m <- as.matrix(temp.dgT)
  data.m[data.m == 0] <- log2(1.1)
  data.m
}

lib_max_global <- max(Matrix::colSums(exp_test))
cat("global lib_max:", lib_max_global, "\n")

chunk_size <- 300
sr_chunked <- c()
for (start in seq(1, n_test, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n_test)
  exp_chunk <- exp_test[, start:end]
  data_chunk <- manual_log_transform(exp_chunk, lib_max_global)
  integ_c <- DoIntegPPI(exp.m = data_chunk, ppiA.m = net13Jun12.m, log_trans = FALSE)
  integ_c <- CompSRana_sparse(integ_c, local = FALSE, mc.cores = 1)
  sr_c <- integ_c$SR
  names(sr_c) <- colnames(integ_c$expMC)
  sr_chunked <- c(sr_chunked, sr_c)
}
sr_chunked <- sr_chunked[names(sr_unchunked)]

cat("n cells compared:", length(sr_unchunked), "\n")
cat("identical (all.equal, tol=1e-8):", isTRUE(all.equal(sr_unchunked, sr_chunked, tolerance = 1e-8)), "\n")
cat("max abs diff:", max(abs(sr_unchunked - sr_chunked)), "\n")
cat("SR range:", range(sr_unchunked), "\n")
cat("DONE\n")
