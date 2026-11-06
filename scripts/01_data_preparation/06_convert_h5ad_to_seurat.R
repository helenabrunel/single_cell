# 04_convert_raw_to_seurat.R
# ==========================
# Lossless conversion of gse243526_raw.h5ad -> Seurat v5 object (.rds).
#
# The h5ad is read directly with rhdf5 (no Python / no intermediate
# converter). AnnData X is CSR cells x genes, which is byte-for-byte the
# CSC genes x cells layout of a dgCMatrix, so indices/indptr/data are
# copied verbatim (no transpose, no recomputation).
#
# Only unavoidable change: Seurat forbids "_" in feature names, so the 24
# make_unique() names (e.g. RGS5_1) become RGS5-1. The original names are
# kept in meta.features$original_var_name and misc$var_name_map.
#
# Equivalence is checked independently by 04b_verify_seurat_equivalence.{R,py}.
#
# Run from: ~/single_cell/Regner/
#   Rscript 04_convert_raw_to_seurat.R

suppressPackageStartupMessages({
  library(rhdf5)
  library(Matrix)
  library(Seurat)
  library(SeuratObject)
})

INFILE  <- "gse243526_raw.h5ad"
OUTFILE <- "gse243526_raw_seurat.rds"

attr_of <- function(path, name) h5readAttributes(INFILE, path)[[name]]

read_vec <- function(path) {
  v <- h5read(INFILE, path, bit64conversion = "double", drop = TRUE)
  as.vector(v)
}

# Read one AnnData column (array / categorical / string-array) exactly
read_column <- function(group, col) {
  path <- paste0(group, "/", col)
  enc  <- attr_of(path, "encoding-type")
  if (identical(enc, "categorical")) {
    cats  <- as.character(read_vec(paste0(path, "/categories")))
    codes <- as.integer(read_vec(paste0(path, "/codes")))
    stopifnot(all(codes >= -1L & codes < length(cats)))
    codes[codes == -1L] <- NA_integer_
    f <- factor(cats[codes + 1L], levels = cats,
                ordered = isTRUE(as.logical(attr_of(path, "ordered"))))
    return(f)
  }
  if (identical(enc, "string-array")) return(as.character(read_vec(path)))
  stopifnot(identical(enc, "array"))
  cls <- H5LS$dclass[H5LS$group == paste0("/", group) & H5LS$name == col]
  v   <- read_vec(path)
  if (cls == "INTEGER") {
    stopifnot(all(abs(v) <= .Machine$integer.max))  # int32/int64 -> exact R integer
    v <- as.integer(v)
  } else if (cls == "ENUM") {                       # h5py bool
    v <- as.logical(v)
  } else stopifnot(cls == "FLOAT")
  v
}

read_df <- function(group) {
  idx   <- as.character(read_vec(paste0(group, "/", attr_of(group, "_index"))))
  order <- as.character(attr_of(group, "column-order"))
  df <- data.frame(row.names = idx)
  for (col in order) df[[col]] <- read_column(group, col)
  stopifnot(identical(colnames(df), order), identical(rownames(df), idx))
  df
}

h5closeAll()
H5LS <- h5ls(INFILE, datasetinfo = TRUE)

# groups that must be empty in this file (fail loudly instead of dropping data)
for (g in c("layers", "obsm", "obsp", "varm", "varp", "uns"))
  if (any(H5LS$group == paste0("/", g))) stop("Group ", g, " is not empty: handle it")

cat("Reading obs / var ...\n")
obs <- read_df("obs")
var <- read_df("var")
h5closeAll()

cat("Reading X (CSR cells x genes) ...\n")
shape <- as.integer(attr_of("X", "shape"))
stopifnot(identical(attr_of("X", "encoding-type"), "csr_matrix"),
          shape[1] == nrow(obs), shape[2] == nrow(var))
indptr  <- as.integer(read_vec("X/indptr"))
indices <- as.integer(read_vec("X/indices"))
xdata   <- as.double(read_vec("X/data"))
h5closeAll()

new_genes <- gsub("_", "-", rownames(var), fixed = TRUE)
stopifnot(!anyDuplicated(new_genes))

counts <- new("dgCMatrix",
              i = indices, p = indptr, x = xdata,
              Dim = c(shape[2], shape[1]),
              Dimnames = list(new_genes, rownames(obs)))
rm(indices, indptr, xdata); invisible(gc())
stopifnot(validObject(counts))
cat("counts:", nrow(counts), "genes x", ncol(counts), "cells, nnz =", length(counts@x), "\n")

cat("Building Seurat object ...\n")
assay <- CreateAssay5Object(counts = counts, min.cells = 0, min.features = 0)
rm(counts); invisible(gc())
obj <- CreateSeuratObject(assay, assay = "RNA", project = "GSE243526",
                          min.cells = 0, min.features = 0)

stopifnot(identical(colnames(obj), rownames(obs)),
          identical(rownames(obj), new_genes))

# cell metadata (all obs columns, same order, same types)
obj <- AddMetaData(obj, obs)
obj$orig.ident <- obj$sample_name
Idents(obj) <- "sample_name"

# feature metadata (all var columns + original name)
var$original_var_name <- rownames(var)
rownames(var) <- new_genes
obj[["RNA"]] <- AddMetaData(obj[["RNA"]], var)

Misc(obj, "source") <- list(
  file          = normalizePath(INFILE),
  file_size     = file.size(INFILE),
  md5           = unname(tools::md5sum(INFILE)),
  converted_on  = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  converter     = "04_convert_raw_to_seurat.R (rhdf5, direct CSR->dgCMatrix)",
  anndata_X     = "raw UMI counts (int64), stored in layer 'counts'",
  versions      = list(R = R.version.string,
                       Seurat = as.character(packageVersion("Seurat")),
                       SeuratObject = as.character(packageVersion("SeuratObject")),
                       rhdf5 = as.character(packageVersion("rhdf5")))
)
Misc(obj, "var_name_map") <- data.frame(h5ad = var$original_var_name,
                                        seurat = new_genes)[var$original_var_name != new_genes, ]
Misc(obj, "obs_column_order") <- colnames(obs)
Misc(obj, "var_column_order") <- setdiff(colnames(var), "original_var_name")
Misc(obj, "uns") <- list()   # uns is empty in the source

print(obj)
cat("Saving", OUTFILE, "...\n")
saveRDS(obj, OUTFILE)
cat("Done.\n")
