# Checks that CompSRana_sparse.R matches the dense CompSRana.R reference.
suppressPackageStartupMessages({
  library(Matrix)
  library(igraph)
  library(wordspace)
  library(DelayedArray)
  library(parallel)
})

LANDSCENT_DIR <- "/home/hbrunel/single_cell/landscent_src"
source(file.path(LANDSCENT_DIR, "DoIntegPPI.R"))
source(file.path(LANDSCENT_DIR, "CompSRana.R"))         # dense reference
source(file.path(LANDSCENT_DIR, "CompSRana_sparse.R"))  # sparse candidate

set.seed(1)

## ---------------------------------------------------------------------
## Test 1: small synthetic network + synthetic expression matrix
## ---------------------------------------------------------------------
cat("=== TEST 1: synthetic data ===\n")
n_genes <- 150
n_cells <- 40

# random symmetric sparse-ish binary adjacency, then force connectivity
A <- matrix(0, n_genes, n_genes)
A[upper.tri(A)] <- rbinom(n_genes * (n_genes - 1) / 2, 1, 0.05)
A <- A + t(A)
A[A > 1] <- 1
# ensure a connected ring backbone so the whole thing is one component
for (i in seq_len(n_genes - 1)) A[i, i + 1] <- A[i + 1, i] <- 1
A[n_genes, 1] <- A[1, n_genes] <- 1
diag(A) <- 0
rownames(A) <- colnames(A) <- paste0("G", seq_len(n_genes))

exp.m <- matrix(rpois(n_genes * n_cells, lambda = 8) + 1, nrow = n_genes)
exp.m <- log2(exp.m + 1.1)  # ensure strictly positive (no exact zeros), mimics log_trans output
rownames(exp.m) <- rownames(A)
colnames(exp.m) <- paste0("C", seq_len(n_cells))

integ <- DoIntegPPI(exp.m = exp.m, ppiA.m = A, log_trans = FALSE)
cat("adjMC dims:", dim(integ$adjMC), "\n")

dense  <- CompSRana(integ, local = TRUE, mc.cores = 1)
sparse <- CompSRana_sparse(integ, local = TRUE, mc.cores = 1)

cat("maxSR   dense vs sparse:", dense$maxSR, "vs", sparse$maxSR,
    "| diff:", abs(dense$maxSR - sparse$maxSR), "\n")
cat("SR      all.equal:", isTRUE(all.equal(dense$SR, sparse$SR, tolerance = 1e-8)), "\n")
cat("inv     all.equal:", isTRUE(all.equal(dense$inv, sparse$inv, tolerance = 1e-8)), "\n")
cat("s       all.equal:", isTRUE(all.equal(dense$s, sparse$s, tolerance = 1e-8)), "\n")
cat("ns      all.equal:", isTRUE(all.equal(dense$ns, sparse$ns, tolerance = 1e-8)), "\n")
cat("max abs SR diff:", max(abs(dense$SR - sparse$SR)), "\n")
cat("max abs s  diff:", max(abs(dense$s - sparse$s)), "\n")

## ---------------------------------------------------------------------
## Test 2: real PPI network + real HER2 data (small cell subset)
## ---------------------------------------------------------------------
cat("\n=== TEST 2: real PPI network + real HER2 subset (300 cells) ===\n")
load(file.path(LANDSCENT_DIR, "net13Jun12.m.RData"))  # net13Jun12.m

EXPORT_DIR <- "/home/hbrunel/single_cell/landscent_export"
mtx_path   <- file.path(EXPORT_DIR, "logcounts_HER2.mtx")
genes_path <- file.path(EXPORT_DIR, "genes_HER2.txt")
cells_path <- file.path(EXPORT_DIR, "cells_HER2.csv")

exp_mat  <- readMM(mtx_path)
symbols  <- readLines(genes_path)
cell_meta <- read.csv(cells_path, row.names = 1)
rownames(exp_mat) <- symbols
colnames(exp_mat) <- rownames(cell_meta)
exp_mat <- as(exp_mat, "dgCMatrix")

suppressPackageStartupMessages({ library(org.Hs.eg.db); library(AnnotationDbi) })
entrez_ids <- mapIds(org.Hs.eg.db, keys = symbols, column = "ENTREZID", keytype = "SYMBOL", multiVals = "first")
keep <- !is.na(entrez_ids) & !duplicated(entrez_ids)
exp_mat <- exp_mat[keep, ]
rownames(exp_mat) <- entrez_ids[keep]
ppi_keep <- rownames(exp_mat) %in% rownames(net13Jun12.m)
exp_mat <- exp_mat[ppi_keep, ]

# subset to first 300 cells to keep the DENSE reference computation tractable
n_test_cells <- 300
exp_mat_small <- exp_mat[, 1:n_test_cells]

integ2 <- DoIntegPPI(exp.m = exp_mat_small, ppiA.m = net13Jun12.m, log_trans = TRUE)
cat("adjMC dims:", dim(integ2$adjMC), "| cells:", ncol(integ2$expMC), "\n")

t0 <- Sys.time()
dense2 <- CompSRana(integ2, local = TRUE, mc.cores = 1)
cat("dense  time:", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n")

t0 <- Sys.time()
sparse2 <- CompSRana_sparse(integ2, local = TRUE, mc.cores = 1)
cat("sparse time:", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n")

cat("maxSR   dense vs sparse:", dense2$maxSR, "vs", sparse2$maxSR,
    "| diff:", abs(dense2$maxSR - sparse2$maxSR), "\n")
cat("SR      all.equal:", isTRUE(all.equal(dense2$SR, sparse2$SR, tolerance = 1e-8)), "\n")
cat("inv     all.equal:", isTRUE(all.equal(dense2$inv, sparse2$inv, tolerance = 1e-8)), "\n")
cat("s       all.equal:", isTRUE(all.equal(dense2$s, sparse2$s, tolerance = 1e-8)), "\n")
cat("ns      all.equal:", isTRUE(all.equal(dense2$ns, sparse2$ns, tolerance = 1e-8)), "\n")
cat("max abs SR diff:", max(abs(dense2$SR - sparse2$SR)), "\n")
cat("max abs s  diff:", max(abs(dense2$s - sparse2$s)), "\n")
cat("SR range (dense):", range(dense2$SR), "\n")

cat("\nDONE\n")
