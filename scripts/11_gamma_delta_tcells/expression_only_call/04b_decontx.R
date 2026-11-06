# 04b_decontx.R
# Ambient-RNA correction with decontX (celda), one sample at a time, using
# all cells of the sample and the author cell types as clusters (z).
# Only the decontaminated counts of T/NK cells are written back.
#
# Input:  data/decontx/in/<sampleid>.h5   (from 04a_export_for_decontx.py)
# Output: data/decontx/out/<sampleid>.h5  (T/NK cells: decontaminated counts
#         as genes × cells CSC = cells × genes CSR, + per-cell contamination)
#
# Run from: ~/single_cell/Klughammer/gdT/
#   Rscript 04b_decontx.R
suppressMessages({library(rhdf5); library(Matrix); library(celda)})

dir.create("data/decontx/out", showWarnings = FALSE, recursive = TRUE)
files <- list.files("data/decontx/in", pattern = "\\.h5$", full.names = TRUE)

for (f in files) {
  s   <- sub("\\.h5$", "", basename(f))
  out <- file.path("data/decontx/out", basename(f))
  if (file.exists(out)) { cat(s, "already done\n"); next }
  tnk <- as.logical(h5read(f, "tnk"))
  if (!any(tnk)) { cat(s, ": no T/NK cells, skipped\n"); next }

  shp <- h5read(f, "shape")                       # cells, genes
  x <- new("dgCMatrix", Dim = as.integer(c(shp[2], shp[1])),
           p = as.integer(h5read(f, "indptr")), i = as.integer(h5read(f, "indices")),
           x = as.numeric(h5read(f, "data")))
  z <- h5read(f, "z")
  t0 <- Sys.time()
  res <- decontX(x, z = z, seed = 1, verbose = FALSE)
  dc  <- res$decontXcounts[, tnk, drop = FALSE]
  dc  <- as(dc, "CsparseMatrix")

  h5createFile(out)
  h5write(dc@x, out, "data"); h5write(dc@i, out, "indices"); h5write(dc@p, out, "indptr")
  h5write(c(ncol(dc), nrow(dc)), out, "shape")   # cells, genes
  h5write(h5read(f, "cells")[tnk], out, "cells")
  h5write(res$contamination[tnk], out, "contamination")
  cat(sprintf("%s: %d cells (%d T/NK), median contamination T/NK %.3f, %.1f min\n", s,
              ncol(x), sum(tnk), median(res$contamination[tnk]),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  rm(x, res, dc); gc(verbose = FALSE)
}
