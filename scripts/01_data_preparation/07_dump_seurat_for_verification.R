# 04b_dump_seurat.R
# =================
# Step 1/2 of the equivalence check. Reloads gse243526_raw_seurat.rds and dumps
# everything (via Seurat's public accessors only) to a plain HDF5 file, which
# 04b_verify_seurat_equivalence.py compares against the original h5ad read by
# anndata. Also runs Seurat-side sanity checks (colSums vs total_counts, ...).
#
#   Rscript 04b_dump_seurat.R

suppressPackageStartupMessages({ library(rhdf5); library(Matrix); library(SeuratObject) })

RDS  <- "gse243526_raw_seurat.rds"
DUMP <- "gse243526_raw_seurat_dump.h5"

obj <- readRDS(RDS)
print(obj)
stopifnot(validObject(obj))

m <- LayerData(obj, assay = "RNA", layer = "counts")
stopifnot(is(m, "dgCMatrix"), validObject(m))

# ---- Seurat-side semantic checks ------------------------------------------
md <- obj[[]]
nc <- Matrix::colSums(m)
nf <- diff(m@p)                              # non-zeros per cell
stopifnot(all(nc == md$total_counts),
          all(nf == md$n_genes_by_counts),
          all(md$nCount_RNA == md$total_counts),
          all(md$nFeature_RNA == md$n_genes_by_counts))
mt <- obj[["RNA"]][["mt"]][, 1]
stopifnot(all(Matrix::colSums(m[mt, ]) == md$total_counts_mt))
mf <- obj[["RNA"]][[]]
stopifnot(all(Matrix::rowSums(m) == mf$total_counts),
          all(Matrix::rowSums(m > 0) == mf$n_cells_by_counts))
cat("Seurat-side sanity checks: OK\n")
cat("Layers:", Layers(obj[["RNA"]]), "| Assays:", Assays(obj),
    "| Reductions:", length(Reductions(obj)), "| Graphs:", length(Graphs(obj)), "\n")

# ---- dump --------------------------------------------------------------------
if (file.exists(DUMP)) file.remove(DUMP)
h5createFile(DUMP)
big <- function(name, v, type) {
  h5createDataset(DUMP, name, dims = length(v), storage.mode = type,
                  chunk = min(length(v), 1e7), level = 0)
  h5write(v, DUMP, name)
}
big("counts_i", m@i, "integer")
big("counts_p", m@p, "integer")
big("counts_x", m@x, "double")
h5write(dim(m), DUMP, "counts_dim")
h5write(colnames(obj), DUMP, "cells")
h5write(rownames(obj), DUMP, "genes")
h5write(c(Assays(obj), Layers(obj[["RNA"]])), DUMP, "assays_layers")

dump_df <- function(df, grp) {
  h5createGroup(DUMP, grp)
  h5write(colnames(df), DUMP, paste0(grp, "/__columns__"))
  for (col in colnames(df)) {
    v <- df[[col]]; p <- paste0(grp, "/", col)
    h5createGroup(DUMP, p)
    if (is.factor(v)) {
      codes <- as.integer(v) - 1L; codes[is.na(codes)] <- -1L
      h5write("factor", DUMP, paste0(p, "/kind"))
      h5write(codes, DUMP, paste0(p, "/codes"))
      h5write(levels(v), DUMP, paste0(p, "/levels"))
      h5write(as.integer(is.ordered(v)), DUMP, paste0(p, "/ordered"))
    } else if (is.logical(v)) {
      stopifnot(!anyNA(v))
      h5write("logical", DUMP, paste0(p, "/kind")); h5write(as.integer(v), DUMP, paste0(p, "/values"))
    } else if (is.character(v)) {
      h5write("character", DUMP, paste0(p, "/kind")); h5write(v, DUMP, paste0(p, "/values"))
    } else {
      h5write(class(v)[1], DUMP, paste0(p, "/kind")); h5write(v, DUMP, paste0(p, "/values"))
    }
  }
}
dump_df(md, "obs")
dump_df(mf, "var")
h5closeAll()
cat("Dump written:", DUMP, "\n")
