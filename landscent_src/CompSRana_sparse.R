# CompSRana_sparse.R
# Sparse-matrix reimplementation of CompSRana/CompSRanaPRL/CompMaxSR from
# LandSCENT. Mathematically identical to the original dense CompSRana.R —
# see landscent_src/CompSRana.R for the reference implementation this was
# derived from and verified against (verify_sparse_landscent.R).
#
# The PPI network (net13Jun12.m) is >99% zeros (density 0.85%, avg degree ~72
# over 8434 genes). The original CompSRanaPRL builds a dense gene x gene
# matrix per single cell; this version keeps the same nonzero pattern
# throughout and only ever touches the ~72-per-row nonzero entries.

suppressPackageStartupMessages(library(Matrix))

CompMaxSR_sparse <- function(Integration.l) {
  adj.m <- Integration.l$adjMC
  if (!is(adj.m, "sparseMatrix")) adj.m <- as(adj.m, "dgCMatrix")

  fa <- function(x, extra = NULL) as.vector(adj.m %*% x)
  ap.o <- igraph::arpack(fa, options = list(n = nrow(adj.m), nev = 1, which = "LM"), sym = TRUE)

  Integration.l$maxSR <- log(ap.o$values)
  Integration.l
}

# adj.sp must already be a dgCMatrix (symmetric structural pattern).
CompSRanaPRL_sparse <- function(idx, exp.m, adj.sp, local = FALSE, maxSR = NULL) {
  exp.v <- exp.m[, idx]

  sumexp.v <- as.vector(adj.sp %*% exp.v)
  invP.v   <- exp.v * sumexp.v
  nf       <- sum(invP.v)
  invP.v   <- invP.v / nf

  # p.m[j, i] = adj[j, i] * exp.v[i] / sumexp.v[j]  ==  D1 %*% adj.sp %*% D2
  D1  <- Diagonal(x = 1 / sumexp.v)
  D2  <- Diagonal(x = exp.v)
  p.m <- D1 %*% adj.sp %*% D2

  px   <- p.m@x
  keep <- px > 0
  ent  <- numeric(length(px))
  ent[keep] <- -px[keep] * log(px[keep])
  ent.m <- p.m
  ent.m@x <- ent
  S.v <- rowSums(ent.m)

  SR <- sum(invP.v * S.v)
  if (!is.null(maxSR)) SR <- SR / maxSR

  if (local) {
    deg.v <- tabulate(p.m@i + 1L, nbins = nrow(p.m))
    NS.v  <- rep(0, nrow(p.m))
    ok    <- deg.v > 1
    NS.v[ok] <- S.v[ok] / log(deg.v[ok])
  } else {
    NS.v <- NULL
  }

  list(sr = SR, inv = invP.v, s = S.v, ns = NS.v)
}

CompSRana_sparse <- function(Integration.l, local = FALSE, mc.cores = 1) {
  Integration.l <- CompMaxSR_sparse(Integration.l)
  maxSR <- Integration.l$maxSR

  adj.sp <- Integration.l$adjMC
  if (!is(adj.sp, "sparseMatrix")) adj.sp <- as(adj.sp, "dgCMatrix")

  idx.l <- as.list(seq_len(ncol(Integration.l$expMC)))
  out.l <- parallel::mclapply(idx.l, CompSRanaPRL_sparse,
                               exp.m = Integration.l$expMC,
                               adj.sp = adj.sp,
                               local = local,
                               maxSR = maxSR,
                               mc.cores = mc.cores)

  Integration.l$SR  <- sapply(out.l, function(v) v[[1]])
  Integration.l$inv <- sapply(out.l, function(v) v[[2]])
  Integration.l$s   <- sapply(out.l, function(v) v[[3]])
  Integration.l$ns  <- sapply(out.l, function(v) v[[4]])
  Integration.l
}
