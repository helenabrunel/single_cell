# 00_h5seurat_to_rds.R
# Build a Seurat object directly from ../data/dataset.h5seurat (SeuratDisk is
# incompatible with SeuratObject >= 5), then save as klughammer_seurat.rds.
# Requires hdf5r built against conda env 'hdf5lib' (no system HDF5 available).
# Run from: ~/single_cell/Klughammer/KlughammeR/
suppressMessages({library(hdf5r); library(Matrix); library(Seurat)})
t0 <- Sys.time()
f <- H5File$new("../data/dataset.h5seurat", mode = "r")

cells    <- f[["cell.names"]][]
features <- f[["assays/RNA/features"]][]

read_sparse <- function(g) {
  dims <- h5attr(g, "dims")
  enc  <- h5attr(g, "encoding-type")
  p    <- g[["indptr"]][]
  cat(sprintf("  encoding=%s dims=%s indptr=%d\n", enc, paste(dims, collapse = "x"), length(p)))
  stopifnot(length(p) == dims[2] + 1)  # one pointer per column (cell) -> CSC genes x cells
  m <- sparseMatrix(i = g[["indices"]][], p = p, x = as.numeric(g[["data"]][]),
                    dims = dims, index1 = FALSE, repr = "C")
  dimnames(m) <- list(features, cells)
  m
}

read_col <- function(x) {
  if (inherits(x, "H5Group")) {
    lev <- x[["categories"]][]; codes <- x[["codes"]][]
    codes[codes < 0] <- NA
    factor(lev[codes + 1L], levels = lev)
  } else x[]
}

read_tree <- function(x) {
  if (inherits(x, "H5Group")) lapply(setNames(names(x), names(x)), function(n) read_tree(x[[n]]))
  else x$read()
}

cat("counts\n"); counts <- read_sparse(f[["assays/RNA/counts"]])
cat("data\n");   data   <- read_sparse(f[["assays/RNA/data"]])
cat("counts integer:", all(counts@x == round(counts@x)), " max:", max(counts@x),
    " | data max:", max(data@x), "\n")

m  <- f[["meta.data"]]
md <- as.data.frame(lapply(setNames(h5attr(m, "column-order"), h5attr(m, "column-order")),
                           function(n) read_col(m[[n]])), check.names = FALSE)
rownames(md) <- cells

mfg <- f[["assays/RNA/meta.features"]]
mf  <- as.data.frame(lapply(setNames(names(mfg), names(mfg)), function(n) read_col(mfg[[n]])),
                     check.names = FALSE)
rownames(mf) <- features

assay <- CreateAssay5Object(counts = counts, data = data)
obj   <- CreateSeuratObject(assay, meta.data = md, project = "Klughammer")
rm(counts, data); invisible(gc())
obj[["RNA"]] <- AddMetaData(obj[["RNA"]], mf)

for (r in c("pca", "umap")) {
  emb <- f[[paste0("reductions/", r, "/cell.embeddings")]][, ]
  if (nrow(emb) != length(cells)) emb <- t(emb)
  rownames(emb) <- cells
  colnames(emb) <- paste0(toupper(r), "_", seq_len(ncol(emb)))
  stdev <- if (r == "pca") f[["reductions/pca/stdev"]][] else numeric(0)
  obj[[r]] <- CreateDimReducObject(embeddings = emb, stdev = stdev,
                                   key = paste0(toupper(r), "_"), assay = "RNA",
                                   misc = read_tree(f[[paste0("reductions/", r, "/misc")]]))
}

# Unstructured annotations (colors, rank_genes_groups, dendrogram, ...)
obj@misc <- read_tree(f[["misc"]])
ai <- f[["active.ident"]]
Idents(obj) <- setNames(factor(ai[["levels"]][][ai[["values"]][]], levels = ai[["levels"]][]), cells)
f$close_all()

# CreateSeuratObject and obj[["RNA"]] <- ... recompute nCount_RNA / nFeature_RNA
# on the filtered gene set; restore the authors' original values last.
obj@meta.data[colnames(md)] <- md
stopifnot(identical(obj@meta.data[colnames(md)], md))

print(obj)
cat("Load:", format(Sys.time() - t0), "\n")
saveRDS(obj, "klughammer_seurat.rds")
cat("Saved klughammer_seurat.rds. Total:", format(Sys.time() - t0), "\n")
