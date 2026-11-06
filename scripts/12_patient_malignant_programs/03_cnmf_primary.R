# Identifies per-patient gene programs with consensus NMF (primary tumors).
library(Seurat)
library(glmGamPoi)
library(Matrix)
library(NMF)
library(ggplot2)
library(future)
library(dplyr)
library(doParallel)
library(foreach)
library(dynamicTreeCut)
library(dendextend)
library(UCell)
library(msigdbr)
library(readxl)
library(cowplot)

# ===========================================================================
# cNMF_per_patient.R
# Per-patient NMF programs (PPs) -> cross-patient meta-programs (MPs) ->
# cell scoring, Gavish anchoring and MSigDB enrichment, on the Regner
# (GSE243526) malignant-cell objects.
#
# Copied from ../Klughammer/KlughammeR/03_cNMF_per_patient.R (itself from
# the lab's cNMF_pipeline_server.R); the NMF/MP/enrichment code is unchanged.
# Only paths and the dataset notes below differ.
# Notes for this dataset:
#   * input: malignant_cells_per_patient/ (server) or 05_malignant_per_patient/
#     (local), clust_rev_<subtype>_data/
#     Patient_X_sctransform.rds (script 06).
#   * NMF is fitted on the SCT assay (NMF_ASSAY = "SCT"): genes ranked by SCT
#     residual variance, values from the SCT "data" layer. Set
#     NMF_ASSAY = "RNA" for the original RNA log1p-CP10k behaviour.
#   * min_cells = 100 (was 350), the same threshold as scripts 05/06.
#     Patient_7 (200 cells) gets ranks 3-6. Patient_15 (116 cells) only allows
#     rank 3 (30 cells/rank), i.e. 3 W columns < min_size = 4, so it is
#     expected to return no stable PPs.
#   * Cohorts: ER 8 patients, TNBC 2 (Patient_6 and Patient_5*, ER-/PR+
#     neuroendocrine, basal-like), HER2 0 (folder empty, cohort skipped).
#     TNBC MPs cannot reach min_pat_n_absolute = 3 and will all be flagged
#     low_confidence.
#   * Patient_14 = two samples (two foci) pooled without integration; a single
#     SCT model.
#   * Large patients (Patient_12 5.2k cells, Patient_13 3.1k, Patient_5 2.8k)
#     dominate the runtime.
#
# Run from the Regner folder (the one holding malignant_cells_per_patient/;
# or set PROJ_ROOT), e.g.:
#   singularity exec <images_dir>/scie_programs.sif \
#     Rscript cNMF_per_patient.R
# Needs NMF, UCell, msigdbr, fgsea, dynamicTreeCut, dendextend (in the
# image; not installed on the local WSL machine).
#
# Server fixes inherited from the original:
#   * worker count from SLURM_CPUS_PER_TASK instead of a hardcoded 12
#   * cluster always shut down, even on error (on.exit)
#   * graphics devices and sink() always closed, even on error
#   * fixed: load_msigdb_paths() referenced go/c6/cp, which are commented
#     out -- a guaranteed "object 'go' not found" crash
#   * fixed: build_pp_projected_ranks() was called with mat_norm where a
#     Seurat object is expected; the tryCatch hid it and every patient
#     silently fell back to HVG-only ranks
#   * msigdbr 10.x renamed category= to collection=; both now work
#   * per-cohort failures are isolated instead of killing the run
# ===========================================================================

PROJ_ROOT     <- "."   # the Regner folder

# Server copy: Regner/malignant_cells_per_patient/clust_rev_<subtype>_data/
# Local copy:  Regner/05_malignant_per_patient/clust_rev_<subtype>_data/
base_data_dir <- file.path(PROJ_ROOT, "malignant_cells_per_patient")
if (!dir.exists(base_data_dir))
  base_data_dir <- file.path(PROJ_ROOT, "05_malignant_per_patient")
if (!dir.exists(base_data_dir))
  stop("Input folder not found: neither malignant_cells_per_patient/ nor ",
       "05_malignant_per_patient/ exists in ", normalizePath(PROJ_ROOT),
       " -- run from the Regner folder or fix base_data_dir")
base_out_dir  <- file.path(PROJ_ROOT, "NMF_per_patient")

# --- which per-patient objects to run NMF on ---------------------------------
#   "sctransform"  Patient_X_sctransform.rds  (script 06, default)
#   "cc_rgrsd"     Patient_X_cc_rgrsd.rds     (script 05)
# Same RNA counts in both, so identical programs.
INPUT_STAGE <- "sctransform"       # "sctransform" or "cc_rgrsd"

INPUT_SUBDIR  <- ""

# --- expression matrix that NMF is fitted on ---------------------------------
#   "SCT"  SCT assay from script 06: genes ranked by SCT residual variance,
#          NMF on the SCT "data" layer (log1p of the depth-corrected counts).
#          Non-negative, so valid for NMF. The Pearson residuals (scale.data,
#          where CC.Difference was regressed) are signed and cannot be used.
#   "RNA"  original behaviour: RNA counts, log1p CP10k, Seurat vst HVGs.
# Cell scoring (UCell) and the genome-wide projection use the RNA assay in
# both cases.
NMF_ASSAY <- "SCT"                 # "SCT" or "RNA"
if (NMF_ASSAY == "SCT" && INPUT_STAGE != "sctransform")
  stop("NMF_ASSAY = \"SCT\" needs INPUT_STAGE = \"sctransform\"")
INPUT_PATTERN <- paste0("_", INPUT_STAGE, "\\.rds$")
ID_PATTERN    <- paste0("^(.*)_", INPUT_STAGE, "\\.rds$")

# Gavish reference metaprograms. MSigDB C4 is tried first; this file is only
# the fallback. Set to NA if you do not have it on the server.
gavish_xlsx_file  <- NA_character_
gavish_xlsx_sheet <- "Cancer_MPs"

# --- MSigDB collections for enrichment -------------------------------------
# "H" alone is 50 Hallmark sets, which is why the first run returned only
# 7-20 hits per patient. CP (Reactome/KEGG/WikiPathways/BioCarta/PID) is
# ~4000 sets and C6 ~190. GOBP is >7000 and slow; BH correction is applied
# per collection, so adding one does not dilute another.
MSIGDB_COLLECTIONS <- c("H", "C6", "CP")   # add "GOBP" if you want GO terms

# --- NMF gene universe and run count ---------------------------------------
# ORIGINAL: n_hvg <- max(min(2000, floor(n_cells / 5)), 500)
#   floor(n_cells/5) only exceeds 500 above 2500 cells, so 10 of your 13
#   patients ran NMF on exactly 500 genes -- a thin basis for pulling out
#   7-9 distinct programs.
#
# RUNTIME WARNING: NMF cost scales roughly linearly in genes AND in nrun.
# Going 500 -> 2000 genes is ~4x on its own. CID4290A took 16.8h at 500
# genes x nrun=30; at 2000 x 30 it would be ~67h. Dropping nrun to 10
# brings that back to roughly 22h. Change these two together, not alone.
# The HVG count now SCALES with cell number instead of sitting on a flat
# floor. A 362-cell patient and a 2141-cell patient should not get the same
# gene universe.
#
#   n_hvg = clamp(n_cells * HVG_GENES_PER_CELL, HVG_MIN, HVG_TARGET)
#
HVG_TARGET         <- 2000   # upper bound, for the big patients
HVG_MIN            <- 800    # lower bound, for the small ones (was 500)
HVG_GENES_PER_CELL <- 1.0    # genes per cell between those bounds

# Low cell counts are the real hazard, and it is not over-fitting -- at rank
# 10 a 362-cell x 1500-gene matrix still has ~29 observations per parameter,
# better than the 500-gene version. The hazard is that with 362 cells, a gene
# detected in 1% of them is present in THREE cells, and such a gene can enter
# the HVG list on noise and drive a program built from a handful of cells.
#
# So filter by detection before ranking variability. A gene must appear in at
# least this fraction of cells AND in at least this many cells outright --
# the absolute floor is what protects the small patients, where 2% is only
# 7 cells. Set HVG_MIN_DETECT_FRAC <- 0 to disable and get the old behaviour.
HVG_MIN_DETECT_FRAC  <- 0.02   # >= 2% of cells
HVG_MIN_DETECT_CELLS <- 10     # and >= 10 cells, whichever is stricter

NRUN <- 10                   # NMF restarts per rank (was 30)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
set.seed(14)

# Worker count from the scheduler. The local version hardcoded 12, which
# oversubscribes a 6-CPU node and makes NMF slower, not faster. More workers
# than ranks is also wasted -- foreach parallelises across ranks only.
.slurm_cpus <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "")))
if (is.na(.slurm_cpus) || .slurm_cpus < 1) .slurm_cpus <- 4L
n_workers <- max(1L, min(.slurm_cpus, length(3:10)))

make_nmf_cluster <- function(n) {
  cl <- makeCluster(n)
  registerDoParallel(cl)
  cl
}
cl <- make_nmf_cluster(n_workers)

# A PSOCK worker that dies takes the whole cluster with it: every later
# %dopar% then fails instantly with "error writing to connection" / SIGPIPE,
# so one bad patient silently kills every patient after it. Check before each
# NMF and rebuild if needed -- a few seconds against hours of lost work.
cluster_alive <- function(cl) {
  isTRUE(tryCatch({ parallel::clusterEvalQ(cl, 1L); TRUE },
                  error = function(e) FALSE))
}
ensure_cluster <- function() {
  if (!cluster_alive(cl)) {
    log_msg(" worker cluster is dead -- restarting", n_workers, "workers")
    try(stopCluster(cl), silent = TRUE)
    cl <<- make_nmf_cluster(n_workers)
    log_msg(" cluster restarted:", if (cluster_alive(cl)) "OK" else "STILL DEAD")
  }
  invisible(NULL)
}
# Always release the workers, including on error -- otherwise a crashed run
# leaves orphan R processes pinned to the node.
on.exit({
  try(stopCluster(cl), silent = TRUE)
}, add = TRUE)

ranks <- 3:10
nrun <- NRUN
method <- "brunet"
min_size <- 4
persistence_frac <- 0.15

h_range <- c(seq(0.01, 0.1, by = 0.01),
             seq(0.12, 0.5, by = 0.02),
             seq(0.55, 0.9, by = 0.05))

# Step 1 — per-patient program gene selection
pp_top_n <- 50
within_pat_max_frac <- 0.5
cells_per_rank <- 30

# Step 2 — cross-patient meta-program parameters
mp_min_frac <- 0.1
mp_h_range <- seq(0.1, 0.9, by = 0.05)
mp_min_pat_frac <- 0.25
use_dynamic_tree_cut <- FALSE

# [CHG 1 / FIX J] Data-driven cutHeight parameters
#   cut_height_ceiling: maximum cutHeight as a fraction of the max merge height.
#     Prevents cutting above the tree root. 0.95 = allow up to 95% of max height.
#   cut_height_search_start: start searching from this fraction of the sorted
#     height range (from the TOP down). 0.4 = search in the upper 60% of heights.
#     The inter-cluster gap lives in the upper range (large sparse merges),
#     not the lower range (tight intra-cluster merges).
cut_height_ceiling <- 0.95
cut_height_search_start <- 0.4

# [FIX C] Cohort-size-adaptive parameters
min_pat_n_absolute <- 3
ubiq_frac_base <- 0.5
ubiq_frac_small_cohort <- 0.6
small_cohort_threshold <- 10

# Artifact gene tracking — KEPT in analysis, flagged post-hoc
artifact_prefixes <- c("RPL", "RPS", "MT-")
artifact_flag_frac <- 0.25
ubiq_frac <- 0.5

# Non-malignant signatures for doublet-like MP detection
nonmalignant_signatures <- list(
  T_cell       = c("CD3D","CD3E","CD3G","CD2","CD8A","CD8B","IL7R","TRAC"),
  B_cell       = c("CD79A","CD79B","MS4A1","CD19","IGHM","IGKC"),
  Myeloid      = c("LYZ","CD68","CD14","CSF1R","ITGAM","FCER1G"),
  Fibroblast   = c("COL1A1","COL1A2","COL3A1","DCN","LUM","PDGFRB"),
  Endothelial  = c("PECAM1","VWF","CLDN5","CDH5","FLT1","KDR")
)
doublet_sim_threshold <- 0.25

# Step 3 — usage matrix / cell scoring
score_method <- "UCell"

# Step 4 — anchoring & enrichment
msigdb_species <- "Homo sapiens"
enrich_top_n <- 50

cohorts <- list(
  list(folder = "clust_rev_ER_data",   adaptive_ranks = TRUE, min_cells = 100),
  list(folder = "clust_rev_HER2_data", adaptive_ranks = TRUE, min_cells = 100),
  list(folder = "clust_rev_TNBC_data", adaptive_ranks = TRUE, min_cells = 100)
)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_file <- file.path(base_out_dir, "pipeline_run.log")
dir.create(base_out_dir, recursive = TRUE, showWarnings = FALSE)
log_msg <- function(...) {
  # paste0 ran all arguments together ("Loaded40Gavish malignant MPs").
  msg <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
                paste(..., collapse = " "))
  cat(msg, "\n")
  cat(msg, "\n", file = log_file, append = TRUE)
}
# A png() left open by an error poisons every later plot in the run, and a
# sink() left open swallows all further output -- including the error that
# caused it. These close their device/connection no matter what.
safe_png <- function(path, expr, width = 1000, height = 900) {
  tryCatch({
    png(path, width = width, height = height)
    on.exit(try(dev.off(), silent = TRUE), add = TRUE)
    force(expr)
  }, error = function(e) log_msg("  plot failed (", basename(path), "): ",
                                 conditionMessage(e)))
}
safe_sink <- function(path, expr) {
  tryCatch({
    sink(path)
    on.exit({ if (sink.number() > 0) sink() }, add = TRUE)
    force(expr)
  }, error = function(e) {
    if (sink.number() > 0) sink()
    log_msg("  writing ", basename(path), " failed: ", conditionMessage(e))
  })
}

log_msg("=== Pipeline start (server) ===")
log_msg("workers: ", n_workers, " (SLURM_CPUS_PER_TASK=",
        Sys.getenv("SLURM_CPUS_PER_TASK", "unset"), ")")
log_msg("data in : ", base_data_dir)
log_msg("out to  : ", base_out_dir)
safe_sink(file.path(base_out_dir, "sessionInfo.txt"), print(sessionInfo()))
log_msg("HVG settings: target=", HVG_TARGET, " min=", HVG_MIN,
        " genes_per_cell=", HVG_GENES_PER_CELL,
        " detect>=", HVG_MIN_DETECT_FRAC, "/", HVG_MIN_DETECT_CELLS, "cells")
log_msg("MSigDB collections:", paste(MSIGDB_COLLECTIONS, collapse=", "))
log_msg("Parameters: ranks=", paste(ranks, collapse=","), " nrun=", nrun,
        " method=", method, " persistence_frac=", persistence_frac,
        " min_pat_n_absolute=", min_pat_n_absolute,
        " cut_height_ceiling=", cut_height_ceiling,
        " cut_height_search_start=", cut_height_search_start)

# ── ONE-TIME CACHE INVALIDATION ────────────────────────────────────────────
# Set TRUE for exactly one run after changing how w_pp_projected is computed
# (e.g. the partial-correlation fix). Flip back to FALSE afterward — leaving
# it TRUE would strip and rebuild the projection on every single run.
invalidate_stale_projections <- FALSE

if (invalidate_stale_projections) {
  stale_files <- list.files(base_out_dir, pattern = "_PP_result\\.rds$",
                            recursive = TRUE, full.names = TRUE)
  log_msg("Invalidating stale w_pp_projected in", length(stale_files), "cached PP results...")
  for (f in stale_files) {
    res <- readRDS(f)
    res$w_pp_projected <- NULL
    saveRDS(res, f)
  }
  log_msg("Done — projected ranks will be rebuilt from cached Seurat objects on this run.")
}

# ── ADAPTIVE RANK HELPER ──────────────────────────────────────────────────────
adaptive_ranks_for <- function(n_cells, ranks, cells_per_rank) {
  max_k <- floor(n_cells / cells_per_rank)
  max_k <- max(max_k, min(ranks))
  ranks[ranks <= max_k]
}

# ── ARTIFACT GENE TRACKING (flag, don't strip) ────────────────────────────────
is_artifact_gene <- function(genes) {
  keep <- rep(FALSE, length(genes))
  for (pfx in artifact_prefixes) {
    keep <- keep | grepl(paste0("^", pfx), genes)
  }
  keep
}

flag_artifact_mps <- function(mp_genes) {
  sapply(mp_genes, function(genes) {
    frac <- sum(is_artifact_gene(genes)) / length(genes)
    if (frac > artifact_flag_frac) "ribo_mito" else ""
  })
}

# ── STEP 1 FUNCTIONS — Per-patient robust programs ────────────────────────────

normalize_for_nmf <- function(mat) {
  lib_sizes <- Matrix::colSums(mat)
  mat_norm <- Matrix::t(Matrix::t(mat) / lib_sizes) * 1e4
  as.matrix(log1p(mat_norm))
}

# [FIX R — v11] Sparse, full-transcriptome normalization for PP gene-universe
#   widening. Unlike normalize_for_nmf() (which densifies — fine for a
#   ~500-2000 gene HVG matrix, not fine for the full ~20-30k gene panel),
#   this keeps the matrix sparse. log1p(0) == 0, so sparsity is preserved
#   through the log transform (only @x, the nonzero slot, is touched).
normalize_sparse_full <- function(mat) {
  lib_sizes <- Matrix::colSums(mat)
  mat_norm <- Matrix::t(Matrix::t(mat) / lib_sizes) * 1e4
  mat_norm@x <- log1p(mat_norm@x)
  mat_norm
}

# [FIX R — v11] Fast Pearson correlation of every row of a sparse matrix
#   against a single numeric vector, without densifying. Used to extend a
#   PP's gene ranking beyond the HVG set it was fit on: correlate every gene
#   in the full transcriptome against a per-cell PP score built from the
#   HVG-restricted top genes, giving fgsea a genome-wide preranked vector
#   instead of a ~500-2000-gene one.
#   mat: sparse dgCMatrix, genes x cells
#   y:   numeric vector, length ncol(mat) (a per-cell score)
#   Returns: named numeric vector of correlations, one per gene (row of mat)
sparse_row_cor <- function(mat, y) {
  n <- ncol(mat)
  y <- as.numeric(y)
  mean_y <- mean(y)
  sd_y <- sd(y)
  if (is.na(sd_y) || sd_y == 0) return(setNames(rep(NA_real_, nrow(mat)), rownames(mat)))
  
  row_means <- Matrix::rowMeans(mat)
  row_sq <- mat
  row_sq@x <- row_sq@x^2
  row_ss <- Matrix::rowSums(row_sq)
  row_sd <- sqrt(pmax(row_ss / n - row_means^2, 0))
  
  cov_xy <- as.numeric((mat %*% y) / n) - row_means * mean_y
  cor_xy <- cov_xy / (row_sd * sd_y)
  cor_xy[row_sd == 0] <- NA_real_
  setNames(cor_xy, rownames(mat))
}

# [FIX U — v13] Partial Pearson correlation of every row of a sparse matrix
#   against y, controlling for confound z (here: per-cell sequencing depth).
#   Removes the compositional artifact of library-size normalization: a cell
#   scoring high on one program mechanically pushes down the normalized
#   value of many unrelated genes (fixed-sum constraint), which shows up as
#   spurious negative correlation across huge swaths of the transcriptome —
#   independent of whether the score itself (y) is depth-robust. Partialling
#   out total counts removes this analytically via the standard
#   partial-correlation formula, no re-normalization or densifying needed.
sparse_partial_row_cor <- function(mat, y, z) {
  r_xy <- sparse_row_cor(mat, y)
  r_xz <- sparse_row_cor(mat, z)
  r_yz <- suppressWarnings(stats::cor(y, z))
  denom <- sqrt(pmax(1 - r_xz^2, 1e-8) * max(1 - r_yz^2, 1e-8))
  pcor <- (r_xy - r_xz * r_yz) / denom
  pcor[is.na(r_xy) | is.na(r_xz)] <- NA_real_
  pcor
}

# K-diagnostic computed from an already-fitted NMF object (no 2nd pass).
kdiag_from_fit <- function(fit, mat_norm, k) {
  err <- rss(fit, target = mat_norm) / sum(mat_norm^2)
  cons <- consensus(fit)
  offdiag <- cons[lower.tri(cons)]
  c(k = k, error = err, consensus_disp = sd(offdiag))
}

get_pp_genes_from_members <- function(w_all, stable_cls, top_n = 50,
                                      within_pat_max_frac = 0.5) {
  n_pp <- length(stable_cls)
  
  candidates_per_cl <- lapply(stable_cls, function(members) {
    w_mean <- rowMeans(w_all[, members, drop = FALSE])
    w_norm <- w_mean / sum(w_mean)
    w_sort <- sort(w_norm, decreasing = TRUE)
    w_sort <- w_sort[w_sort > 0]
    head(w_sort, min(2 * top_n, length(w_sort)))
  })
  
  all_candidate_genes <- unique(unlist(lapply(candidates_per_cl, names)))
  gene_pp_count <- setNames(integer(length(all_candidate_genes)), all_candidate_genes)
  for (cand in candidates_per_cl) {
    for (g in names(cand)) {
      if (g %in% all_candidate_genes) gene_pp_count[g] <- gene_pp_count[g] + 1L
    }
  }
  within_pat_freq <- gene_pp_count / n_pp
  # With a single PP every candidate has freq = 1, so down-weighting would
  # zero out all of them and leave an empty PP (-> NaN Jaccard, hclust crash).
  if (n_pp < 2) within_pat_freq[] <- 0

  ubiq_here <- names(within_pat_freq)[within_pat_freq > within_pat_max_frac]
  if (length(ubiq_here) > 0) {
    log_msg("  Within-patient near-ubiquitous genes (freq >",
            round(within_pat_max_frac, 2), ") down-weighted:", length(ubiq_here))
  }
  
  lapply(seq_along(stable_cls), function(i) {
    cand <- candidates_per_cl[[i]]
    genes <- names(cand)
    freq <- within_pat_freq[genes]
    freq[is.na(freq)] = 0
    adj_w <- cand * (1 - freq)
    adj_w <- sort(adj_w, decreasing = TRUE)
    adj_w <- adj_w[adj_w > 0]
    selected <- names(adj_w)[seq_len(min(top_n, length(adj_w)))]
    unique(selected[!is.na(selected)])
  })
}

find_stable_clusters <- function(hc, h_range, min_size, persistence_frac) {
  
  all_clusterings <- lapply(h_range, function(h) cutree(hc, h = h))
  
  cluster_history <- list()
  for (hi in seq_along(h_range)) {
    cl <- all_clusterings[[hi]]
    sizes <- table(cl)
    valid <- as.integer(names(sizes[sizes >= min_size]))
    for (v in valid) {
      members <- sort(which(cl == v))
      key <- paste(members, collapse = "-")
      if (is.null(cluster_history[[key]])) {
        cluster_history[[key]] <- list(members = members, count = 0L)
      }
      cluster_history[[key]]$count <- cluster_history[[key]]$count + 1L
    }
  }
  
  if (length(cluster_history) == 0) return(list())
  
  persistence <- sapply(cluster_history, function(x) x$count)
  min_persistence <- max(2L, ceiling(persistence_frac * length(h_range)))
  stable_keys <- names(persistence)[persistence >= min_persistence]
  
  if (length(stable_keys) == 0) return(list())
  
  stable_members <- lapply(stable_keys, function(k) cluster_history[[k]]$members)
  names(stable_members) <- stable_keys
  persistence_stable <- persistence[stable_keys]
  
  ord <- order(persistence_stable, decreasing = TRUE)
  stable_members <- stable_members[ord]
  persistence_stable <- persistence_stable[ord]
  
  # DISJOINT enforcement
  assigned <- integer(max(unlist(stable_members)))
  keep <- rep(TRUE, length(stable_members))
  for (i in seq_along(stable_members)) {
    members <- stable_members[[i]]
    free <- members[is.na(assigned[members]) | assigned[members] == 0]
    if (length(free) < min_size) {
      keep[i] <- FALSE
      next
    }
    stable_members[[i]] <- sort(free)
    assigned[free] <- i
  }
  stable_members <- stable_members[keep]
  lapply(stable_members, function(m) m)
}

find_patient_programs <- function(mat_norm, patient_name,
                                  ranks = 3:10, nrun = 30, method = "brunet",
                                  min_size = 4,
                                  h_range = c(seq(0.01, 0.1, by = 0.01),
                                              seq(0.12, 0.5, by = 0.02),
                                              seq(0.55, 0.9, by = 0.05)),
                                  top_n = 50, within_pat_max_frac = 0.5,
                                  persistence_frac = 0.15, seed = 14) {
  log_msg("-- Patient:", patient_name, "--")
  log_msg(" Matrix:", nrow(mat_norm), "genes x", ncol(mat_norm), "cells")
  
  # [FIX D] Save mat_norm to a temp file so each worker loads its own copy
  mat_norm_tmp <- tempfile(fileext = ".rds")
  saveRDS(mat_norm, mat_norm_tmp)
  
  # Parallel NMF across ranks — computes W, H, AND K-diagnostic in one pass
  # [FIX E] .pbackend = "seq": run NMF sequentially WITHIN each worker.
  #   The parallelism is already handled by foreach %dopar% across ranks.
  #   Using NULL fails because workers don't inherit the parent's registered
  #   foreach backend ("no registered backend" error).
  w_list <- foreach(k = ranks, .packages = c("NMF"),
                    .export = c("method","nrun","seed","kdiag_from_fit","mat_norm_tmp"),
                    .inorder = TRUE) %dopar% {
                      mat_norm_local <- readRDS(mat_norm_tmp)
                      set.seed(seed + k)
                      res <- nmf(mat_norm_local, rank = k, method = method, nrun = nrun,
                                 .pbackend = "seq", .options = "P")
                      W <- as.matrix(basis(res))   # force base matrix (avoids S4 dispatch issues)
                      H <- as.matrix(coef(res))
                      Wn <- sweep(W, 2, sqrt(colSums(W^2) + 1e-10), "/")
                      diag <- kdiag_from_fit(res, mat_norm_local, k)
                      rm(mat_norm_local, res)
                      list(W = Wn, H = H, k = k, kdiag = diag)
                    }
  
  unlink(mat_norm_tmp)
  
  kdiag <- do.call(rbind, lapply(w_list, function(x) x$kdiag))
  log_msg(" K-diagnostic:\n", paste(capture.output(print(kdiag)), collapse="\n"))
  
  w_all <- do.call(cbind, lapply(w_list, function(x) x$W))
  h_all <- lapply(w_list, function(x) x$H)
  names(h_all) <- paste0("k", ranks)
  
  cos_mat <- t(w_all) %*% w_all
  cos_mat <- pmin(pmax(cos_mat, 0), 1)
  dist_mat <- as.dist(1 - cos_mat)
  hc <- hclust(dist_mat, method = "average")
  
  stable_cls <- find_stable_clusters(hc, h_range, min_size, persistence_frac)
  best_nPP <- length(stable_cls)
  
  log_msg(" Stable per-patient programs (PPs):", best_nPP)
  if (best_nPP == 0) {
    log_msg(" No stable programs found — try lowering min_size or persistence_frac.")
    return(NULL)
  }
  
  pp_genes <- get_pp_genes_from_members(w_all, stable_cls,
                                        top_n = top_n,
                                        within_pat_max_frac = within_pat_max_frac)
  names(pp_genes) <- paste0(patient_name, "_PP", seq_along(pp_genes))
  
  w_pp <- do.call(cbind, lapply(stable_cls, function(m)
    rowMeans(w_all[, m, drop = FALSE])))
  colnames(w_pp) <- names(pp_genes)
  
  list(
    cos_mat = cos_mat,
    tree = hc,
    stable_cls = stable_cls,
    pp_genes = pp_genes,
    w_pp = w_pp,
    h_all = h_all,
    kdiag = kdiag,
    nPP = best_nPP
  )
}

write_genes_csv <- function(genes_list, path) {
  lens <- sapply(genes_list, length)
  max_len <- if (length(lens) == 0 || max(lens) == 0) 1L else max(lens)
  df <- do.call(cbind, lapply(names(genes_list), function(nm) {
    g <- genes_list[[nm]]
    if (length(g) == 0) g <- ""
    length(g) <- max_len
    g[is.na(g)] <- ""
    data.frame(g, stringsAsFactors = FALSE, check.names = FALSE)
  }))
  colnames(df) <- names(genes_list)
  write.csv(df, path, row.names = FALSE, na = "")
}

# [FIX S — v12] Standalone writer for the projected top-genes diagnostic CSV,
#   factored out of save_patient_results so ensure_projected_ranks() (the
#   cached-patient backfill path) can also produce it, not just first runs.
write_pp_projected_csv <- function(res, patient_name, pat_out_dir) {
  if (is.null(res$w_pp_projected) || length(res$w_pp_projected) == 0) return(invisible(NULL))
  top_k <- 30
  proj_rows <- lapply(names(res$w_pp_projected), function(pp) {
    v <- res$w_pp_projected[[pp]]
    if (is.null(v)) return(NULL)
    v <- sort(v[!is.na(v)], decreasing = TRUE)
    top_pos <- utils::head(v, top_k)
    top_neg <- utils::tail(v, top_k)
    data.frame(
      PP = pp,
      rank = c(seq_along(top_pos), -rev(seq_along(top_neg))),
      gene = c(names(top_pos), names(top_neg)),
      correlation = round(c(as.numeric(top_pos), as.numeric(top_neg)), 4),
      stringsAsFactors = FALSE
    )
  })
  proj_df <- do.call(rbind, proj_rows[!vapply(proj_rows, is.null, logical(1))])
  if (!is.null(proj_df) && nrow(proj_df) > 0) {
    write.csv(proj_df,
              file.path(pat_out_dir, paste0(patient_name, "_PP_projected_top_genes.csv")),
              row.names = FALSE)
  }
  invisible(proj_df)
}

save_patient_results <- function(res, patient_name, pat_out_dir, seu, hvg_genes) {
  
  saveRDS(res, file.path(pat_out_dir, paste0(patient_name, "_PP_result.rds")))
  write_genes_csv(res$pp_genes,
                  file.path(pat_out_dir, paste0(patient_name, "_PP_top_genes.csv")))
  
  # [CHG 2] Save HVG gene list for per-patient enrichment universe
  writeLines(hvg_genes, file.path(pat_out_dir, paste0(patient_name, "_hvg_genes.txt")))
  
  # [FIX R — v11] Diagnostic: top +/- correlated genes per PP from the
  # genome-wide projection, BEFORE any within-patient frequency down-
  # weighting. This is the list to eyeball if you want to know "what is
  # this PP actually about" independent of the pathway enrichment tables —
  # it isn't restricted to HVGs and isn't penalized for recurring across
  # this patient's other PPs.
  write_pp_projected_csv(res, patient_name, pat_out_dir)
  
  # K-stability diagnostic plot
  kdf <- as.data.frame(res$kdiag)
  p_err <- ggplot(kdf, aes(k, error)) +
    geom_line() + geom_point() +
    theme_minimal() + ggtitle(paste0(patient_name, " — NMF error vs K"))
  p_disp <- ggplot(kdf, aes(k, consensus_disp)) +
    geom_line() + geom_point() +
    theme_minimal() + ggtitle(paste0(patient_name, " — consensus dispersion vs K"))
  ggsave(file.path(pat_out_dir, paste0(patient_name, "_k_stability.png")),
         plot_grid(p_err, p_disp, ncol = 2), width = 10, height = 4)
  
  # Summary text
  sink(file.path(pat_out_dir, paste0(patient_name, "_PP_summary.txt")))
  cat("Patient:", patient_name, "\n")
  cat("Number of PPs:", res$nPP, "\n")
  cat("Method: persistence-based disjoint cluster detection (persistence_frac=",
      persistence_frac, ")\n\n")
  cat("K-stability diagnostic:\n"); print(res$kdiag); cat("\n")
  cat("Genes per PP:\n"); print(sapply(res$pp_genes, length)); cat("\n")
  for (pp in names(res$pp_genes)) {
    cat("---", pp, "---\n")
    cat(res$pp_genes[[pp]], sep = ", "); cat("\n\n")
  }
  sink()
  
  # Cosine heatmap
  png(file.path(pat_out_dir, paste0(patient_name, "_cosine_heatmap.png")),
      width = 1000, height = 900)
  heatmap(res$cos_mat, symm = TRUE,
          main = paste0(patient_name, " - W column cosine similarity"))
  dev.off()
  
  # Dendrogram via dendextend
  # [FIX H] Removed labels=FALSE from plot() — base R plot.window doesn't
  # accept 'labels', causing "invalid graphical parameter" warnings.
  # To suppress leaf labels, use labels_colors + hang instead.
  dend <- as.dendrogram(res$tree)
  png(file.path(pat_out_dir, paste0(patient_name, "_tree.png")),
      width = 1200, height = 700)
  par(mar = c(4,4,4,8))
  plot(dend, main = paste0(patient_name, " — ", res$nPP,
                           " stable PPs (persistence, disjoint)"),
       horiz = FALSE, leaflab = "none")
  if (length(res$stable_cls) > 0) {
    leaf_cols <- rep("grey50", nleaves(dend))
    pp_colors <- rainbow(length(res$stable_cls))
    for (i in seq_along(res$stable_cls)) {
      leaf_cols[res$stable_cls[[i]]] <- pp_colors[i]
    }
    labels_colors(dend) <- leaf_cols[order.dendrogram(dend)]
  }
  dev.off()
  
  saveRDS(seu, file.path(pat_out_dir, paste0(patient_name, "_seurat.rds")))
  invisible(seu)
}

# [FIX L] Helper: extract HVGs from a saved Seurat object and save to file.
#   Used when process_patient() returns early for already-processed patients.
#   The Seurat object has VariableFeatures stored from the original run.
ensure_hvg_file <- function(patient_name, pat_out_dir) {
  hvg_file <- file.path(pat_out_dir, paste0(patient_name, "_hvg_genes.txt"))
  if (file.exists(hvg_file)) return(readLines(hvg_file))
  
  seu_file <- file.path(pat_out_dir, paste0(patient_name, "_seurat.rds"))
  if (!file.exists(seu_file)) return(NULL)
  
  log_msg("  Extracting HVGs from cached Seurat for", patient_name)
  seu <- tryCatch(readRDS(seu_file), error = function(e) NULL)
  if (is.null(seu)) return(NULL)
  
  hvg <- tryCatch(VariableFeatures(seu), error = function(e) character(0))
  rm(seu); gc()
  
  if (length(hvg) < 100) return(NULL)
  writeLines(hvg, hvg_file)
  log_msg("  Saved HVG file for", patient_name, ":", length(hvg), "genes")
  hvg
}

# [FIX S — v12] Backfill projected ranks for already-processed (cached)
#   patients. process_patient()'s early-return path used to just return the
#   cached res as-is; if that res predates the projection feature (or
#   projection failed on a prior run), it has no w_pp_projected and fgsea
#   silently falls back to the narrow HVG-only NMF weights forever, even on
#   reruns. This rebuilds the projection from the cached Seurat object +
#   cached HVG list — no NMF re-run needed — and writes the updated res back
#   to disk so this only has to happen once per patient.
ensure_projected_ranks <- function(res, patient_name, pat_out_dir) {
  if (!is.null(res$w_pp_projected) && length(res$w_pp_projected) > 0) {
    return(res)  # already has it, nothing to do
  }
  
  seu_file <- file.path(pat_out_dir, paste0(patient_name, "_seurat.rds"))
  if (!file.exists(seu_file)) {
    log_msg("  WARNING: no cached Seurat object for", patient_name,
            "— cannot backfill projected ranks, fgsea will use NMF-weight fallback")
    return(res)
  }
  
  hvg <- ensure_hvg_file(patient_name, pat_out_dir)
  if (is.null(hvg) || length(hvg) < 100) {
    log_msg("  WARNING: no usable HVG list for", patient_name,
            "— cannot backfill projected ranks")
    return(res)
  }
  
  log_msg("  Backfilling projected ranks for", patient_name, "from cached Seurat object")
  seu <- tryCatch(readRDS(seu_file), error = function(e) NULL)
  if (is.null(seu)) {
    log_msg("  WARNING: failed to load cached Seurat for", patient_name)
    return(res)
  }
  
  res$w_pp_projected <- tryCatch({
    DefaultAssay(seu) <- "RNA"
    mat_full <- GetAssayData(seu, assay = "RNA", layer = "counts")
    mat_full_norm <- normalize_sparse_full(mat_full)
    out <- build_pp_projected_ranks(seu, mat_full_norm, res$pp_genes)
    rm(mat_full, mat_full_norm)
    out
  }, error = function(e) {
    log_msg("  WARNING: projected-rank backfill failed for", patient_name,
            "(", conditionMessage(e), ")")
    NULL
  })
  
  rm(seu); gc()
  
  if (!is.null(res$w_pp_projected) && length(res$w_pp_projected) > 0) {
    saveRDS(res, file.path(pat_out_dir, paste0(patient_name, "_PP_result.rds")))
    log_msg("  Projected ranks backfilled and cached for", patient_name)
  }
  res
}

process_patient <- function(rds_path, patient_name, pat_out_dir,
                            ranks, nrun, method, min_size, h_range,
                            top_n = 50, within_pat_max_frac = 0.5,
                            persistence_frac = 0.15, seed = 14,
                            adaptive_ranks = FALSE, min_cells = 60,
                            cells_per_rank = 30) {
  
  done_flag <- file.path(pat_out_dir, paste0(patient_name, "_PP_result.rds"))
  if (file.exists(done_flag)) {
    log_msg(" Already processed — loading pp_genes for", patient_name)
    # [FIX L] Ensure HVG file exists for cached patients (extract from Seurat)
    ensure_hvg_file(patient_name, pat_out_dir)
    res_cached <- readRDS(done_flag)
    # Cached results from before the single-PP fix can hold empty PPs.
    # Rebuild their gene lists from w_pp (= per-PP mean W) without re-running NMF.
    if (!is.null(res_cached$w_pp) &&
        any(lengths(res_cached$pp_genes) == 0)) {
      log_msg("  Rebuilding empty PP gene lists from cached W for", patient_name)
      pp_names <- names(res_cached$pp_genes)
      res_cached$pp_genes <- setNames(
        get_pp_genes_from_members(res_cached$w_pp,
                                  as.list(seq_len(ncol(res_cached$w_pp))),
                                  top_n = top_n,
                                  within_pat_max_frac = within_pat_max_frac),
        pp_names)
      res_cached$w_pp_projected <- NULL   # force backfill with the new genes
      saveRDS(res_cached, done_flag)
      write_genes_csv(res_cached$pp_genes,
                      file.path(pat_out_dir, paste0(patient_name, "_PP_top_genes.csv")))
    }
    # [FIX S] Backfill projected ranks if this cached result predates them
    res_cached <- ensure_projected_ranks(res_cached, patient_name, pat_out_dir)
    return(res_cached)
  }
  
  dir.create(pat_out_dir, recursive = TRUE, showWarnings = FALSE)
  
  seu <- tryCatch(readRDS(rds_path),
                  error = function(e) { log_msg(" ERROR loading:", conditionMessage(e)); NULL })
  if (is.null(seu)) return(invisible(NULL))
  
  n_cells <- ncol(seu)
  log_msg(" Cells:", n_cells)
  
  if (n_cells < min_cells) {
    log_msg(" Skipping", patient_name, "— only", n_cells, "cells (min_cells =", min_cells, ")")
    rm(seu); gc()
    return(invisible(NULL))
  }
  
  ranks_use <- if (adaptive_ranks) {
    r <- adaptive_ranks_for(n_cells, ranks, cells_per_rank)
    if (!identical(r, ranks))
      log_msg(" Adaptive ranks:", paste(r, collapse = "-"), "(", n_cells, "cells,",
              cells_per_rank, "cells/rank )")
    r
  } else ranks
  
  DefaultAssay(seu) <- "RNA"
  mat_full <- GetAssayData(seu, assay = "RNA", layer = "counts")  # [FIX R] keep full gene set for projection
  mat <- mat_full
  
  # Adaptive HVG count — ribo/mito genes NOT stripped (kept for biological check)
  # Scales with cell number; see HVG_* in CONFIG.
  n_hvg <- round(n_cells * HVG_GENES_PER_CELL)
  n_hvg <- max(HVG_MIN, min(HVG_TARGET, n_hvg))
  n_hvg <- min(n_hvg, nrow(mat))          # cannot ask for more than exist

  # Detection filter first: drop genes seen in too few cells before ranking
  # variability, so a 3-cell gene cannot become an HVG in a small patient.
  if (HVG_MIN_DETECT_FRAC > 0) {
    min_cells_det <- max(HVG_MIN_DETECT_CELLS,
                         ceiling(HVG_MIN_DETECT_FRAC * n_cells))
    det_counts    <- Matrix::rowSums(mat_full > 0)
    genes_detected <- rownames(mat_full)[det_counts >= min_cells_det]
    log_msg(" detection filter: >=", min_cells_det, "cells ->",
            length(genes_detected), "of", nrow(mat_full), "genes kept")
  } else {
    genes_detected <- rownames(mat_full)
  }

  if (NMF_ASSAY == "SCT") {
    # All genes in SCT residual-variance order (VariableFeatures only holds
    # the top 3000, too few to survive the detection filter for HVG_TARGET).
    sct_attr <- SCTResults(seu, assay = "SCT", slot = "feature.attributes")
    request <- nrow(sct_attr)
    genes.use <- rownames(sct_attr)[order(sct_attr$residual_variance,
                                          decreasing = TRUE)]
  } else {
    # Ask for extra so that enough survive the filter, then take the top n_hvg
    # in Seurat's own variability order.
    request <- min(nrow(mat), max(n_hvg * 3, n_hvg + 500))
    seu <- FindVariableFeatures(seu, assay = "RNA", nfeatures = request)
    genes.use <- VariableFeatures(seu)
  }
  genes.use <- genes.use[genes.use %in% genes_detected]
  genes.use <- genes.use[genes.use %in% rownames(mat)]
  genes.use <- utils::head(genes.use, n_hvg)

  log_msg(" HVGs:", length(genes.use), "used (target", n_hvg,
          "for", n_cells, "cells; requested", request, "before filtering)")
  if (length(genes.use) < n_hvg)
    log_msg(" NOTE: fewer HVGs than target -- detection filter is binding here")

  mat <- mat[genes.use, ]
  
  mat_norm <- if (NMF_ASSAY == "SCT") {
    as.matrix(GetAssayData(seu, assay = "SCT", layer = "data")[genes.use, ])
  } else normalize_for_nmf(mat)
  log_msg(" NMF input:", NMF_ASSAY, "assay")
  
  ensure_cluster()
  t_nmf <- Sys.time()

  run_nmf <- function() {
    find_patient_programs(mat_norm, patient_name,
                          ranks = ranks_use, nrun = nrun, method = method,
                          min_size = min_size, h_range = h_range,
                          top_n = top_n, within_pat_max_frac = within_pat_max_frac,
                          persistence_frac = persistence_frac, seed = seed)
  }

  res <- tryCatch(run_nmf(), error = function(e) {
    msg <- conditionMessage(e)
    log_msg(" ERROR in NMF:", msg)
    # A connection/SIGPIPE error means the CLUSTER died, not that this
    # patient's data is bad. Rebuild and give this patient one more go.
    if (grepl("connection|SIGPIPE|socket|cluster", msg, ignore.case = TRUE)) {
      log_msg(" that looks like a dead worker -- rebuilding cluster, retrying",
              patient_name)
      try(stopCluster(cl), silent = TRUE)
      cl <<- make_nmf_cluster(n_workers)
      return(tryCatch(run_nmf(), error = function(e2) {
        log_msg(" retry also failed:", conditionMessage(e2)); NULL
      }))
    }
    NULL
  })

  log_msg(sprintf(" NMF wall time for %s: %.1f min", patient_name,
                  as.numeric(difftime(Sys.time(), t_nmf, units = "mins"))))
  
  if (is.null(res)) {
    rm(seu, mat, mat_norm, mat_full); gc()
    return(invisible(NULL))
  }
  
  # [FIX R — v11] Build genome-wide projected rank vectors for fgsea, using
  # the full (non-HVG-restricted) transcriptome as the correlation target.
  # Kept in a tryCatch — if this fails for any reason, fgsea just falls back
  # to the old HVG-only w_pp ranking for this patient (see enrich_pp_fgsea).
  res$w_pp_projected <- tryCatch({
    mat_full_norm <- normalize_sparse_full(mat_full)
    # [SERVER FIX] was build_pp_projected_ranks(mat_norm, ...) -- the first
    # argument must be the Seurat object (the function calls colnames(seu)
    # and AddModuleScore_UCell(seu, ...)). Passing the matrix threw inside
    # the tryCatch, so every patient silently fell back to HVG-only ranks.
    build_pp_projected_ranks(seu, mat_full_norm, res$pp_genes)
  }, error = function(e) {
    log_msg(" WARNING: projected-rank build failed for", patient_name,
            "(", conditionMessage(e), ") — fgsea will fall back to NMF weights")
    NULL
  })
  
  rm(mat, mat_norm, mat_full); gc()
  
  # [CHG 2] Pass HVG genes to save_patient_results for per-patient enrichment
  tryCatch(save_patient_results(res, patient_name, pat_out_dir, seu,
                                hvg_genes = genes.use),
           error = function(e) log_msg(" ERROR saving:", conditionMessage(e)))
  
  rm(seu); gc()
  log_msg("Done:", patient_name)
  invisible(res)
}

# ── STEP 2 FUNCTIONS — Cross-patient Meta-Programs ───────────────────────────

jaccard <- function(a, b) {
  u <- length(union(a, b))
  if (u == 0) 0 else length(intersect(a, b)) / u
}
overlap_coef <- function(a, b) {
  m <- min(length(a), length(b))
  if (m == 0) 0 else length(intersect(a, b)) / m
}

build_overlap_matrix <- function(all_pp_genes, metric = c("jaccard", "overlap")) {
  metric <- match.arg(metric)
  n <- length(all_pp_genes)
  mat <- matrix(0, n, n, dimnames = list(names(all_pp_genes), names(all_pp_genes)))
  fn <- if (metric == "jaccard") jaccard else overlap_coef
  for (i in seq_len(n)) {
    for (j in i:n) {
      v <- fn(all_pp_genes[[i]], all_pp_genes[[j]])
      mat[i,j] = v; mat[j,i] = v
    }
  }
  mat
}

# [FIX C] build_mp_genes accepts min_pat_n_absolute and cohort_n_patients
build_mp_genes <- function(clust_res, all_pp_genes, top_n = 50,
                           min_pat_frac = 0.25, ubiq_genes = character(0),
                           min_pat_n_absolute = 3, cohort_n_patients = NA,
                           small_cohort_threshold = 10) {
  if (length(clust_res$valid_cls) == 0) {
    log_msg("  WARNING: no valid clusters passed mp_min_size — no MPs built")
    return(list())
  }
  
  # [FIX N] is_small_cohort dead code removed — v9 no longer flags MPs based
  # on cohort size, only on per-MP patient support (n_patients).
  
  res <- lapply(seq_along(clust_res$valid_cls), function(idx) {
    cl <- clust_res$valid_cls[idx]
    pp_in_cl <- names(clust_res$clusters)[clust_res$clusters == cl]
    if (length(pp_in_cl) == 0) {
      log_msg("  WARNING: cluster", cl, "has no PPs — returning empty MP")
      return(list(genes = character(0), low_confidence = FALSE, n_patients = 0))
    }
    pat_of_pp <- sub("_PP\\d+$", "", pp_in_cl)
    n_patients <- length(unique(pat_of_pp))
    min_pat_n <- max(min_pat_n_absolute, ceiling(min_pat_frac * n_patients))
    
    pat_gene_sets <- lapply(unique(pat_of_pp), function(pat) {
      pps <- pp_in_cl[pat_of_pp == pat]
      unique(unlist(all_pp_genes[pps]))
    })
    all_genes_union <- unlist(pat_gene_sets)
    if (length(all_genes_union) == 0) {
      log_msg("  WARNING: cluster", cl, "has no genes across its PPs — returning empty MP")
      return(list(genes = character(0), low_confidence = FALSE, n_patients = n_patients))
    }
    pat_counts <- table(all_genes_union)
    stable <- names(pat_counts)[pat_counts >= min_pat_n]
    stable <- stable[!stable %in% ubiq_genes]
    
    fallback_used <- FALSE
    if (length(stable) < 10) {
      fallback_used <- TRUE
      all_genes_flat <- unlist(all_pp_genes[pp_in_cl])
      all_genes_flat <- all_genes_flat[!all_genes_flat %in% ubiq_genes]
      if (length(all_genes_flat) == 0) {
        log_msg("  WARNING: cluster", cl, "produced no genes after ubiq filter — returning empty MP")
        return(list(genes = character(0), low_confidence = FALSE, n_patients = n_patients))
      }
      freq <- sort(table(all_genes_flat), decreasing = TRUE)
      stable <- names(freq)[seq_len(min(top_n, length(freq)))]
    }
    if (fallback_used)
      log_msg("  WARNING: MP", cl, "fell back to raw frequency (<10 stable genes) — lower confidence")
    
    stable <- unique(stable[!is.na(stable) & nchar(stable) > 0])
    all_genes_flat <- unlist(all_pp_genes[pp_in_cl])
    all_genes_flat <- all_genes_flat[!all_genes_flat %in% ubiq_genes]
    if (length(all_genes_flat) == 0 || length(stable) == 0) {
      log_msg("  WARNING: cluster", cl, "produced no genes after filtering — returning empty MP")
      return(list(genes = character(0), low_confidence = FALSE, n_patients = n_patients))
    }
    gene_freq <- sort(table(all_genes_flat[all_genes_flat %in% stable]), decreasing = TRUE)
    if (length(gene_freq) == 0)
      return(list(genes = character(0), low_confidence = FALSE, n_patients = n_patients))
    
    mp_genes <- names(gene_freq)[seq_len(min(top_n, length(gene_freq)))]
    # [FIX N] low_confidence should reflect per-MP support (patients contributing
    # PPs), NOT the size of the whole cohort. v8 used
    # "is_small_cohort || (n_patients < min_pat_n_absolute)" which flagged EVERY
    # MP in cohorts <10 patients as low_confidence, drowning MP3 (1 patient) in
    # false alarms alongside MP1/2/4 (4-5 patients). v9 flags only when the MP
    # itself is supported by fewer than min_pat_n_absolute (3) patients.
    low_conf <- n_patients < min_pat_n_absolute
    list(genes = mp_genes, low_confidence = low_conf, n_patients = n_patients)
  })
  names(res) <- paste0("clust", clust_res$valid_cls)
  res
}

score_sweep <- function(clusters, min_size) {
  sizes <- table(clusters)
  valid_sizes <- sizes[sizes >= min_size]
  n_valid <- length(valid_sizes)
  if (n_valid == 0) return(0)
  med_size <- median(as.integer(valid_sizes))
  coverage <- sum(valid_sizes) / length(clusters)
  n_valid * log1p(med_size) * coverage
}

flag_doublet_mps <- function(mp_genes, nonmalignant_signatures, threshold) {
  flags <- character(length(mp_genes))
  for (i in seq_along(mp_genes)) {
    best <- 0; best_type <- NA
    for (nm in names(nonmalignant_signatures)) {
      j <- jaccard(mp_genes[[i]], nonmalignant_signatures[[nm]])
      if (j > best) { best <- j; best_type <- nm }
    }
    if (best > threshold) flags[i] <- best_type
  }
  flags
}

# [CHG 1 / FIX J] Data-driven cutHeight via gap analysis on dendrogram merge
#   heights. v7 used RELATIVE jumps (diff/height) which locked onto the gap
#   between the two smallest heights (tiny denominators inflate the ratio),
#   and searched the LOWER range (tight intra-cluster merges). This produced
#   cutHeight=0.54, below all merges -> 0 MPs and a crash.
#
#   v8 uses ABSOLUTE jumps (simple differences) and searches the UPPER portion
#   of the height range, where the inter-cluster gap lives (large sparse merges
#   between clusters vs. small dense merges within clusters). We sort heights
#   ascending, search from index search_start upward, find the largest absolute
#   gap, and cut just above it.
#
#   hc: an hclust object
#   ceiling_frac: maximum cutHeight as fraction of max height (prevents
#     cutting above the tree root)
#   search_start: start searching from this fraction of the sorted height
#     range (0.4 = search in the upper 60% of heights)
#   Returns: a numeric cutHeight value
data_driven_cut_height <- function(hc, ceiling_frac = 0.95, search_start = 0.4) {
  heights <- sort(hc$height, decreasing = FALSE)
  n <- length(heights)
  if (n < 3) return(max(heights) * ceiling_frac)
  
  max_h <- max(heights)
  ceiling_val <- max_h * ceiling_frac
  
  # Search in the upper portion of the height range (from search_start upward).
  # The inter-cluster gap is where large sparse merges transition to even
  # larger ones — this is in the upper range, not the lower.
  search_begin <- max(2, floor(n * search_start))
  search_heights <- heights[search_begin:n]
  
  # Absolute jumps between consecutive merge heights
  jumps <- diff(search_heights)
  
  if (length(jumps) == 0) return(ceiling_val)
  
  # The largest absolute jump marks the inter-cluster boundary
  best_idx <- which.max(jumps)
  # cut just above the gap (at the height where the jump lands)
  cut_h <- search_heights[best_idx + 1]
  
  # Apply ceiling to prevent cutting above the tree root
  cut_h <- min(cut_h, ceiling_val)
  
  # Sanity: must be positive and below max
  cut_h <- max(cut_h, heights[min(n, 2)])
  cut_h <- min(cut_h, max_h)
  
  log_msg("  Data-driven cutHeight:", round(cut_h, 4),
          "(largest absolute gap at merge", search_begin + best_idx - 1,
          "of", n, ", ceiling=", round(ceiling_val, 4), ")")
  cut_h
}

find_meta_programs <- function(all_pp_genes, cohort_name, cohort_out_dir,
                               mp_min_size = 3, h_range = seq(0.1, 0.9, by = 0.05),
                               min_pat_frac = 0.25, ubiq_frac = 0.5,
                               use_dynamic_tree_cut = TRUE,
                               min_pat_n_absolute = 3,
                               small_cohort_threshold = 10,
                               cut_height_ceiling = 0.95,
                               cut_height_search_start = 0.4) {
  
  log_msg("Cross-patient MP clustering:", cohort_name)
  log_msg(" Total per-patient programs:", length(all_pp_genes))

  empty_pp <- lengths(all_pp_genes) == 0
  if (any(empty_pp)) {
    log_msg(" WARNING: dropping", sum(empty_pp), "empty PP(s):",
            paste(names(all_pp_genes)[empty_pp], collapse = ", "))
    all_pp_genes <- all_pp_genes[!empty_pp]
  }

  if (length(all_pp_genes) < mp_min_size) {
    log_msg(" Not enough programs to cluster — skipping.")
    return(NULL)
  }
  
  n_programs <- length(all_pp_genes)
  all_genes_vec <- unlist(all_pp_genes)
  gene_freq_tab <- sort(table(all_genes_vec), decreasing = TRUE)
  gene_freq_frac <- as.numeric(gene_freq_tab) / n_programs
  
  # [FIX C] Adaptive ubiq_frac
  n_patients_cohort <- length(unique(sub("_PP\\d+$", "", names(all_pp_genes))))
  ubiq_frac_eff <- ubiq_frac + 0.1 * max(0, 1 - n_patients_cohort / 20)
  ubiq_frac_eff <- min(ubiq_frac_eff, 0.6)
  if (ubiq_frac_eff != ubiq_frac)
    log_msg(" Adaptive ubiq_frac:", round(ubiq_frac, 2), "->", round(ubiq_frac_eff, 2),
            "(", n_patients_cohort, "patients )")
  
  ubiq_genes <- names(gene_freq_tab)[gene_freq_frac > ubiq_frac_eff]
  if (length(ubiq_genes) > 0) {
    log_msg(" Ubiquitous genes (in >", round(ubiq_frac_eff * 100), "% of PPs) excluded from MP gene lists:",
            length(ubiq_genes))
  }
  
  dir.create(cohort_out_dir, recursive = TRUE, showWarnings = FALSE)
  freq_df <- data.frame(gene = names(gene_freq_tab),
                        n_programs = as.integer(gene_freq_tab),
                        frac_programs = round(gene_freq_frac, 3),
                        ubiquitous = names(gene_freq_tab) %in% ubiq_genes,
                        stringsAsFactors = FALSE)
  write.csv(freq_df, file.path(cohort_out_dir, paste0(cohort_name, "_gene_frequency.csv")),
            row.names = FALSE)
  
  log_msg(" Building Jaccard + Overlap matrices...")
  jac_mat <- build_overlap_matrix(all_pp_genes, metric = "jaccard")
  ovl_mat <- build_overlap_matrix(all_pp_genes, metric = "overlap")
  
  # [CHG 1] cluster_and_cut now computes a data-driven cutHeight per metric
  #   and passes it to cutreeDynamic. This replaces the v6 default (~99% of
  #   height range) which over-merged PPs.
  cluster_and_cut <- function(sim_mat, metric_name) {
    dist_mat <- as.dist(1 - sim_mat)
    hc <- hclust(dist_mat, method = "average")
    
    if (use_dynamic_tree_cut) {
      # [CHG 1 / FIX J] Compute data-driven cutHeight for this metric
      cut_h <- data_driven_cut_height(hc, ceiling_frac = cut_height_ceiling,
                                      search_start = cut_height_search_start)
      
      clusters_raw <- cutreeDynamic(hc, distM = as.matrix(dist_mat),
                                    minClusterSize = mp_min_size,
                                    method = "hybrid", deepSplit = 2,
                                    cutHeight = cut_h)
      # [FIX B] Preserve names with setNames (as.integer strips them)
      pp_names <- rownames(sim_mat)
      clusters <- setNames(as.integer(clusters_raw), pp_names)
      ctab <- table(clusters[clusters > 0])
      valid_cls <- as.integer(names(ctab[ctab >= mp_min_size]))
      best_h <- cut_h  # store for dendrogram plot
      log_msg("  ", metric_name, "MPs (dynamicTreeCut, cutHeight=", round(cut_h, 4), "):",
              length(valid_cls))
    } else {
      scores <- sapply(h_range, function(h) score_sweep(cutree(hc, h = h), mp_min_size))
      best_h <- h_range[which.max(scores)]
      clusters <- setNames(as.integer(cutree(hc, h = best_h)), names(cutree(hc, h = best_h)))
      ctab <- table(clusters)
      valid_cls <- as.integer(names(ctab[ctab >= mp_min_size]))
      log_msg("  ", metric_name, "MPs (score_sweep):", length(valid_cls), "at h =", best_h)
    }
    list(hc = hc, clusters = clusters, valid_cls = valid_cls, best_h = best_h)
  }
  
  jac_clust <- cluster_and_cut(jac_mat, "Jaccard")
  ovl_clust <- cluster_and_cut(ovl_mat, "Overlap")
  
  jac_mp_res <- build_mp_genes(jac_clust, all_pp_genes,
                               min_pat_frac = min_pat_frac, ubiq_genes = ubiq_genes,
                               min_pat_n_absolute = min_pat_n_absolute,
                               cohort_n_patients = n_patients_cohort,
                               small_cohort_threshold = small_cohort_threshold)
  ovl_mp_res <- build_mp_genes(ovl_clust, all_pp_genes,
                               min_pat_frac = min_pat_frac, ubiq_genes = ubiq_genes,
                               min_pat_n_absolute = min_pat_n_absolute,
                               cohort_n_patients = n_patients_cohort,
                               small_cohort_threshold = small_cohort_threshold)
  
  jac_mp_genes  <- lapply(jac_mp_res,  function(x) x$genes)
  jac_mp_conf   <- lapply(jac_mp_res,  function(x) x$low_confidence)
  jac_mp_npat   <- lapply(jac_mp_res,  function(x) x$n_patients)
  ovl_mp_genes  <- lapply(ovl_mp_res,  function(x) x$genes)
  
  keep_jac <- sapply(jac_mp_genes, function(g) length(g) > 0)
  jac_mp_genes <- setNames(jac_mp_genes[keep_jac], paste0("MP", seq_len(sum(keep_jac))))
  jac_mp_conf  <- unlist(jac_mp_conf[keep_jac])
  jac_mp_npat  <- unlist(jac_mp_npat[keep_jac])
  
  keep_ovl <- sapply(ovl_mp_genes, function(g) length(g) > 0)
  ovl_mp_genes <- setNames(ovl_mp_genes[keep_ovl], paste0("MP", seq_len(sum(keep_ovl))))
  
  # Early exit if no MPs survived
  if (length(jac_mp_genes) == 0) {
    log_msg("  WARNING: no MPs survived gene filtering for", cohort_name,
            "— saving empty result and skipping flagging/scoring")
    saveRDS(
      list(jaccard = list(sim_mat = jac_mat, clust = jac_clust, mp_genes = list(),
                          doublet_flags = character(0), artifact_flags = character(0),
                          low_confidence = logical(0), n_patients = integer(0)),
           overlap = list(sim_mat = ovl_mat, clust = ovl_clust, mp_genes = list())),
      file.path(cohort_out_dir, paste0(cohort_name, "_MP_result.rds"))
    )
    return(list(jaccard = list(sim_mat = jac_mat, clust = jac_clust, mp_genes = list(),
                               doublet_flags = character(0), artifact_flags = character(0),
                               low_confidence = logical(0), n_patients = integer(0)),
                overlap = list(sim_mat = ovl_mat, clust = ovl_clust, mp_genes = list())))
  }
  
  jac_doublet_flags <- flag_doublet_mps(jac_mp_genes, nonmalignant_signatures,
                                        doublet_sim_threshold)
  jac_artifact_flags <- flag_artifact_mps(jac_mp_genes)
  
  for (i in seq_along(jac_mp_genes)) {
    flags_i <- c()
    if (nchar(jac_doublet_flags[i]) > 0)
      flags_i <- c(flags_i, paste0("doublet:", jac_doublet_flags[i]))
    if (nchar(jac_artifact_flags[i]) > 0)
      flags_i <- c(flags_i, jac_artifact_flags[i])
    if (isTRUE(jac_mp_conf[i]))
      flags_i <- c(flags_i, "low_confidence")
    if (length(flags_i) > 0)
      log_msg("  FLAG: MP", i, "->", paste(flags_i, collapse = " | "),
              "(n_patients=", jac_mp_npat[i], ")")
  }
  
  dir.create(cohort_out_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(
    list(jaccard = list(sim_mat = jac_mat, clust = jac_clust, mp_genes = jac_mp_genes,
                        doublet_flags = jac_doublet_flags,
                        artifact_flags = jac_artifact_flags,
                        low_confidence = jac_mp_conf,
                        n_patients = jac_mp_npat),
         overlap = list(sim_mat = ovl_mat, clust = ovl_clust, mp_genes = ovl_mp_genes)),
    file.path(cohort_out_dir, paste0(cohort_name, "_MP_result.rds"))
  )
  
  write_genes_csv(jac_mp_genes, file.path(cohort_out_dir, paste0(cohort_name, "_MP_genes_Jaccard.csv")))
  write_genes_csv(ovl_mp_genes, file.path(cohort_out_dir, paste0(cohort_name, "_MP_genes_Overlap.csv")))
  
  # Heatmaps
  save_mp_heatmap <- function(sim_mat, clust_res, metric_name) {
    if (nrow(sim_mat) < 2) {
      log_msg("  <2 PPs — skipping ", metric_name, " MP heatmap")
      return(invisible(NULL))
    }
    ord <- order(clust_res$clusters)
    mat_ord <- sim_mat[ord, ord]
    n_cls <- max(clust_res$clusters)
    # [FIX M] cutreeDynamic assigns cluster 0 to noise PPs. rainbow(n_cls)[0]
    # silently drops that element, shortening cl_cols below ncol(x) and
    # crashing heatmap("ColSideColors must be a character vector of length
    # ncol(x)"). Build colors explicitly: cluster 0 -> grey80, 1..n_cls -> rainbow.
    cl_cols <- if (n_cls > 0) {
      palette <- c("grey80", rainbow(n_cls))
      palette[clust_res$clusters[ord] + 1]
    } else rep("grey50", nrow(mat_ord))
    png(file.path(cohort_out_dir, paste0(cohort_name, "_MP_heatmap_", metric_name, ".png")),
        width = 1400, height = 1200)
    heatmap(mat_ord, Rowv = NA, Colv = NA, symm = TRUE,
            RowSideColors = cl_cols, ColSideColors = cl_cols,
            main = paste0(cohort_name, " — ", metric_name, " overlap"),
            col = colorRampPalette(c("white","orange","red"))(100))
    dev.off()
  }
  save_mp_heatmap(jac_mat, jac_clust, "Jaccard")
  save_mp_heatmap(ovl_mat, ovl_clust, "Overlap")
  
  # Dendrograms — [FIX H] use leaflab="none" instead of labels=FALSE
  # [CHG 1] Draw abline at cutHeight for BOTH dynamicTreeCut and score_sweep
  #   (v6 only drew it for score_sweep because best_h was NA for dynamicTreeCut)
  save_mp_tree <- function(hc, best_h, metric_name) {
    dend <- as.dendrogram(hc)
    png(file.path(cohort_out_dir, paste0(cohort_name, "_MP_tree_", metric_name, ".png")),
        width = max(1200, length(all_pp_genes) * 10), height = 700)
    par(mar = c(4,4,4,8))
    plot(dend, main = paste0(cohort_name, " — ", metric_name,
                             ifelse(is.na(best_h), " (dynamicTreeCut)", paste0(" cutHeight=", round(best_h, 4)))),
         leaflab = "none")
    if (!is.na(best_h)) abline(h = best_h, col = "blue", lty = 2)
    dev.off()
  }
  save_mp_tree(jac_clust$hc, jac_clust$best_h, "Jaccard")
  save_mp_tree(ovl_clust$hc, ovl_clust$best_h, "Overlap")
  
  # Summary text
  sink(file.path(cohort_out_dir, paste0(cohort_name, "_MP_summary.txt")))
  cat("Cohort:", cohort_name, "\n")
  cat("Total per-patient programs:", length(all_pp_genes), "\n")
  cat("Cohort patients:", n_patients_cohort, "\n")
  cat("Adaptive ubiq_frac:", round(ubiq_frac_eff, 2), "\n")
  cat("min_pat_n_absolute:", min_pat_n_absolute, "\n")
  cat("Jaccard cutHeight:", round(jac_clust$best_h, 4), "\n")
  cat("Overlap cutHeight:", round(ovl_clust$best_h, 4), "\n\n")
  for (metric in c("Jaccard", "Overlap")) {
    mp_genes <- if (metric == "Jaccard") jac_mp_genes else ovl_mp_genes
    clust <- if (metric == "Jaccard") jac_clust else ovl_clust
    cat("--", metric, "--\n")
    cat("MPs found:", length(mp_genes), "\n")
    cat("Genes per MP:\n"); print(sapply(mp_genes, length)); cat("\n")
    if (metric == "Jaccard") {
      cat("MP flags (doublet / ribo_mito / low_confidence / n_patients):\n")
      for (i in seq_along(mp_genes)) {
        dflag <- ifelse(nchar(jac_doublet_flags[i])>0, jac_doublet_flags[i], "clean")
        aflag <- ifelse(nchar(jac_artifact_flags[i])>0, jac_artifact_flags[i], "clean")
        lc    <- ifelse(isTRUE(jac_mp_conf[i]), "LOW_CONF", "ok")
        cat(" MP", i, ": doublet=", dflag, " ribo_mito=", aflag,
            " confidence=", lc, " n_patients=", jac_mp_npat[i], "\n")
      }
      cat("\n")
    }
    for (mp in names(mp_genes)) {
      cat("---", mp, "---\n")
      cat(mp_genes[[mp]], sep = ", "); cat("\n\n")
    }
  }
  sink()
  
  log_msg("MP results saved for", cohort_name)
  list(jaccard = list(sim_mat = jac_mat, clust = jac_clust, mp_genes = jac_mp_genes,
                      doublet_flags = jac_doublet_flags,
                      artifact_flags = jac_artifact_flags,
                      low_confidence = jac_mp_conf,
                      n_patients = jac_mp_npat),
       overlap = list(sim_mat = ovl_mat, clust = ovl_clust, mp_genes = ovl_mp_genes))
}

# ── STEP 3 — Cohort-wide usage matrix (memory-conscious: load-score-release) ─

score_cells_streaming <- function(seu_paths, mp_genes, cohort_name, cohort_out_dir,
                                  method = "UCell") {
  log_msg("Step 3: scoring cells against", length(mp_genes), "MPs (", method, ")")
  
  usage_all <- list()
  
  for (pat in names(seu_paths)) {
    log_msg("  Loading", pat, "...")
    seu <- tryCatch(readRDS(seu_paths[[pat]]),
                    error = function(e) { log_msg("   ERROR:", conditionMessage(e)); NULL })
    if (is.null(seu)) next
    
    signatures <- lapply(mp_genes, function(g) g[g %in% rownames(seu)])
    signatures <- Filter(function(g) length(g) >= 5, signatures)
    
    if (length(signatures) == 0) {
      log_msg("   WARNING: no MP signatures with >=5 genes in", pat, "— skipping")
      rm(seu); gc()
      next
    }
    
    if (method == "UCell") {
      seu <- AddModuleScore_UCell(seu, features = signatures, name = "_MP")
      mp_cols <- paste0(names(signatures), "_MP")
    } else {
      seu <- AddModuleScore(seu, features = signatures, assay = "RNA", name = "MP")
      mp_cols <- paste0("MP", seq_along(signatures))
    }
    
    df <- as.data.frame(seu@meta.data[, mp_cols, drop = FALSE])
    colnames(df) <- names(signatures)
    df$cell_id <- rownames(seu@meta.data)
    df$patient <- pat
    usage_all[[pat]] <- df
    
    log_msg("   scored", pat, ":", nrow(df), "cells")
    rm(seu); gc()
  }
  
  if (length(usage_all) == 0) {
    log_msg("  ERROR: no patients scored successfully")
    return(NULL)
  }
  
  usage <- do.call(rbind, usage_all)
  rownames(usage) <- usage$cell_id
  rm(usage_all); gc()
  
  mp_cols <- intersect(names(mp_genes), colnames(usage))
  usage$dominant_MP <- mp_cols[apply(usage[, mp_cols, drop = FALSE], 1, which.max)]
  
  write.csv(usage, file.path(cohort_out_dir, paste0(cohort_name, "_MP_usage_matrix.csv")),
            row.names = FALSE)
  saveRDS(usage, file.path(cohort_out_dir, paste0(cohort_name, "_MP_usage_matrix.rds")))
  
  # (a) Usage heatmap
  # base heatmap() needs >= 2 rows and 2 columns; small cohorts (e.g. TNBC,
  # 2 patients) can end up with a single MP.
  if (length(mp_cols) >= 2 && nrow(usage) >= 2) {
    ord <- order(usage$dominant_MP)
    png(file.path(cohort_out_dir, paste0(cohort_name, "_usage_heatmap.png")),
        width = 1200, height = 1400)
    heatmap(t(as.matrix(usage[ord, mp_cols])), Rowv = NA, Colv = NA,
            scale = "none", useRaster = TRUE,
            col = colorRampPalette(c("navy","yellow","red"))(100),
            main = paste0(cohort_name, " — MP usage (cells ordered by dominant MP)"),
            labCol = NA)
    dev.off()
  } else {
    log_msg("  Only ", length(mp_cols), " MP(s) — skipping usage heatmap")
  }
  
  # (b) Per-patient MP composition (stacked bar)
  # [FIX F] as.matrix() before prop.table — prop.table on a data.frame
  # triggers marginSums("x is not an array"). This was the REAL cause of
  # the marginSums crash, not S4 matrices.
  comp <- as.data.frame.matrix(table(usage$patient, usage$dominant_MP))
  comp_frac <- prop.table(as.matrix(comp), margin = 1)
  
  # [FIX G] Base R reshape instead of reshape::melt (reshape package not loaded)
  comp_long <- data.frame(
    patient = rep(rownames(comp_frac), times = ncol(comp_frac)),
    MP = rep(colnames(comp_frac), each = nrow(comp_frac)),
    fraction = as.vector(comp_frac),
    stringsAsFactors = FALSE
  )
  p_comp <- ggplot(comp_long, aes(patient, fraction, fill = MP)) +
    geom_bar(stat = "identity") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste0(cohort_name, " — MP composition per patient"))
  ggsave(file.path(cohort_out_dir, paste0(cohort_name, "_MP_composition.png")),
         p_comp, width = 10, height = 6)
  
  log_msg("Step 3 complete. Usage matrix:", nrow(usage), "cells x", length(mp_cols), "MPs")
  usage
}

# ── STEP 4 — Anchoring to published MPs + functional enrichment ──────────────

# [FIX A] GAVISH_3CA_MALIGNANT_METAPROGRAM_* gene sets are in MSigDB C4
load_gavish_mps <- function(xlsx_file, xlsx_sheet, species) {
  if (requireNamespace("msigdbr", quietly = TRUE)) {
    gav <- tryCatch({
      d <- msigdbr_safe(species, "C4")
      if (is.null(d)) NULL else dplyr::filter(d, grepl("GAVISH_3CA_MALIGNANT_METAPROGRAM", gs_name))
    }, error = function(e) NULL)
    if (!is.null(gav) && nrow(gav) > 0) {
      mp_list <- split(gav$gene_symbol, gav$gs_name)
      names(mp_list) <- sub("GAVISH_3CA_MALIGNANT_METAPROGRAM_(\\d+)_(.*)", "MP\\1_\\2",
                            names(mp_list))
      log_msg("  Loaded", length(mp_list), "Gavish malignant MPs from MSigDB C4 (3CA collection)")
      return(mp_list)
    }
    log_msg("  GAVISH sets not found in C4 — trying C2 as fallback...")
    gav2 <- tryCatch({
      d <- msigdbr_safe(species, "C2")
      if (is.null(d)) NULL else dplyr::filter(d, grepl("GAVISH_3CA_MALIGNANT_METAPROGRAM", gs_name))
    }, error = function(e) NULL)
    if (!is.null(gav2) && nrow(gav2) > 0) {
      mp_list <- split(gav2$gene_symbol, gav2$gs_name)
      names(mp_list) <- sub("GAVISH_3CA_MALIGNANT_METAPROGRAM_(\\d+)_(.*)", "MP\\1_\\2",
                            names(mp_list))
      log_msg("  Loaded", length(mp_list), "Gavish malignant MPs from MSigDB C2 (fallback)")
      return(mp_list)
    }
    log_msg("  GAVISH sets not found in C4 or C2 — check msigdbr version / DB release.")
  }
  
  if (is.null(xlsx_file) || is.na(xlsx_file) || !file.exists(xlsx_file)) {
    log_msg("  No xlsx provided either — anchoring skipped.")
    log_msg("  (Set gavish_xlsx_file to Gavish 2023 Supplementary Table S2 to enable.)")
    return(NULL)
  }
  
  sheets <- excel_sheets(xlsx_file)
  log_msg("  xlsx sheets found:", paste(sheets, collapse = ", "))
  if (is.null(xlsx_sheet)) {
    log_msg("  gavish_xlsx_sheet is NULL — set it to the malignant MP sheet.")
    return(NULL)
  }
  df <- read_excel(xlsx_file, sheet = xlsx_sheet)
  log_msg("  Loaded sheet '", xlsx_sheet, "':", ncol(df), "columns,", nrow(df), "rows")
  mp_list <- lapply(colnames(df), function(col) {
    genes <- df[[col]]
    genes <- genes[!is.na(genes) & nchar(genes) > 0]
    unique(genes)
  })
  names(mp_list) <- colnames(df)
  mp_list <- Filter(function(g) length(g) > 0, mp_list)
  log_msg("  Parsed", length(mp_list), "MPs from xlsx sheet '", xlsx_sheet, "'")
  mp_list
}

# [FIX O] Compute the FULL Jaccard matrix between two named gene-set lists
#   (query x reference). Unlike anchor_to_published() which returns only the
#   best match per query, this returns every pairwise Jaccard so downstream
#   analysis can inspect secondary matches, multi-MP mixtures, and cross-
#   patient similarity patterns to Gavish MPs.
#
#   query_sets:  named list of gene vectors (e.g. per-patient PPs or cohort MPs)
#   ref_sets:    named list of gene vectors (e.g. Gavish 3CA MPs)
#   Returns:     matrix rownames = names(query_sets), colnames = names(ref_sets)
full_jaccard_matrix <- function(query_sets, ref_sets) {
  m <- matrix(0, length(query_sets), length(ref_sets),
              dimnames = list(names(query_sets), names(ref_sets)))
  for (i in seq_along(query_sets)) {
    for (j in seq_along(ref_sets)) {
      m[i, j] <- jaccard(query_sets[[i]], ref_sets[[j]])
    }
  }
  m
}

# [FIX O] Save the full Jaccard matrix as CSV + a PNG heatmap.
save_full_jaccard_to_gavish <- function(query_sets, ref_sets, out_dir,
                                        label, title_suffix = "") {
  if (is.null(ref_sets) || length(ref_sets) == 0) {
    log_msg("  No Gavish reference — skipping full Jaccard matrix for", label)
    return(NULL)
  }
  if (length(query_sets) == 0) {
    log_msg("  Empty query set — skipping full Jaccard matrix for", label)
    return(NULL)
  }
  jm <- full_jaccard_matrix(query_sets, ref_sets)
  
  # CSV: row = query name (PP or MP), columns = Gavish MPs, values = Jaccard
  csv_path <- file.path(out_dir, paste0(label, "_full_jaccard_to_gavish.csv"))
  write.csv(round(jm, 4), csv_path, row.names = TRUE)
  log_msg("  Full Jaccard matrix saved:", basename(csv_path),
          "(", nrow(jm), "x", ncol(jm), ")")
  
  # Heatmap PNG — clamp very small values so the colour scale is readable
  png_path <- file.path(out_dir, paste0(label, "_full_jaccard_to_gavish.png"))
  png(png_path, width = max(1000, ncol(jm) * 22),
      height = max(500, nrow(jm) * 24))
  par(mar = c(11, 10, 4, 2))
  image(seq_len(nrow(jm)), seq_len(ncol(jm)), jm,
        axes = FALSE, xlab = "", ylab = "",
        col = colorRampPalette(c("white","yellow","orange","red"))(100),
        main = paste0("Jaccard: ", label, " vs Gavish 3CA MPs", title_suffix))
  axis(1, at = seq_len(nrow(jm)), labels = rownames(jm), las = 2, cex.axis = 0.8)
  axis(2, at = seq_len(ncol(jm)), labels = colnames(jm), las = 2, cex.axis = 0.6)
  # annotate cells with Jaccard values >= 0.05 for readability
  for (i in seq_len(nrow(jm))) {
    for (j in seq_len(ncol(jm))) {
      if (jm[i, j] >= 0.05) {
        text(i, j, sprintf("%.2f", jm[i, j]), cex = 0.6)
      }
    }
  }
  dev.off()
  invisible(jm)
}

anchor_to_published <- function(my_mp_genes, ref_mp_genes, cohort_out_dir, cohort_name) {
  if (is.null(ref_mp_genes)) return(NULL)
  log_msg("Step 4a: anchoring", length(my_mp_genes), "MPs to", length(ref_mp_genes), "published MPs")
  res <- lapply(seq_along(my_mp_genes), function(i) {
    best <- 0; best_name <- NA; second <- 0
    for (nm in names(ref_mp_genes)) {
      j <- jaccard(my_mp_genes[[i]], ref_mp_genes[[nm]])
      if (j > best) { second <- best; best <- j; best_name <- nm }
      else if (j > second) second <- j
    }
    data.frame(my_MP = names(my_mp_genes)[i],
               best_published = best_name,
               best_jaccard = round(best, 3),
               second_jaccard = round(second, 3),
               novel = best < 0.15,
               stringsAsFactors = FALSE)
  })
  anchor_df <- do.call(rbind, res)
  write.csv(anchor_df, file.path(cohort_out_dir, paste0(cohort_name, "_MP_anchoring.csv")),
            row.names = FALSE)
  log_msg("  Anchoring table saved. Novel MPs (best Jaccard < 0.15):",
          sum(anchor_df$novel))
  anchor_df
}

# [CHG 2 / FIX K] Load MSigDB pathways ONCE per cohort, reuse for both
#   per-patient and MP-level enrichment. Expanded libraries: H (Hallmark)
#   + C2 curated pathways (KEGG, Reactome, WikiPathways, BioCarta, PID)
#   + C5:GOBP (GO Biological Process) + C6 (oncogenic signatures).
#
#   [FIX K] v7 called msigdbr(subcategory="CP") which returns only 19 "SA_"
#   (Sigma-Aldrich) gene sets, NOT KEGG/Reactome/WikiPathways. The real
#   curated pathways are in subcategories CP:REACTOME, CP:WIKIPATHWAYS,
#   CP:KEGG_MEDICUS, CP:BIOCARTA, CP:PID, CP:KEGG_LEGACY. v8 loads all C2
#   and filters to these CP:* subcategories (4096 gene sets).
# msigdbr 10.x renamed category= to collection= and subcategory= to
# subcollection=. Try the new names, fall back to the old ones.
msigdbr_safe <- function(species, collection, subcollection = NULL) {
  out <- tryCatch({
    if (is.null(subcollection)) msigdbr::msigdbr(species = species, collection = collection)
    else msigdbr::msigdbr(species = species, collection = collection,
                          subcollection = subcollection)
  }, error = function(e) NULL)
  if (!is.null(out)) return(out)
  tryCatch({
    if (is.null(subcollection)) msigdbr::msigdbr(species = species, category = collection)
    else msigdbr::msigdbr(species = species, category = collection,
                          subcategory = subcollection)
  }, error = function(e) {
    log_msg("  WARNING: msigdbr failed for ", collection, ": ", conditionMessage(e))
    NULL
  })
}

# [SERVER FIX] The local version built h_p/go_p/c6_p/cp_p but go, c6 and cp
# were all commented out above, so this function always died with
# "object 'go' not found" before reaching any enrichment. Rewritten to load
# exactly the collections named in MSIGDB_COLLECTIONS and nothing else.
load_msigdb_paths <- function(species = "Homo sapiens",
                              collections = MSIGDB_COLLECTIONS) {
  log_msg("  Loading MSigDB pathways: ", paste(collections, collapse = ", "))
  make_paths <- function(df) split(df$gene_symbol, df$gs_name)

  cp_subcats <- c("CP:REACTOME", "CP:WIKIPATHWAYS", "CP:KEGG_MEDICUS",
                  "CP:BIOCARTA", "CP:PID", "CP:KEGG_LEGACY")

  paths <- list(); collection <- character(0)

  for (coll in collections) {
    df <- switch(coll,
      "H"    = msigdbr_safe(species, "H"),
      "C6"   = msigdbr_safe(species, "C6"),
      "GOBP" = msigdbr_safe(species, "C5", "BP"),
      "CP"   = {
        c2 <- msigdbr_safe(species, "C2")
        if (is.null(c2)) NULL else {
          sub_col <- if ("gs_subcollection" %in% names(c2)) "gs_subcollection" else "gs_subcat"
          if (sub_col %in% names(c2)) c2[c2[[sub_col]] %in% cp_subcats, ] else c2
        }
      },
      { log_msg("  WARNING: unknown collection '", coll, "' -- skipped"); NULL })

    if (is.null(df) || nrow(df) == 0) {
      log_msg("  WARNING: ", coll, " returned nothing -- skipped")
      next
    }
    p <- make_paths(df)
    paths <- c(paths, p)
    collection <- c(collection, rep(coll, length(p)))
    log_msg("    ", coll, ": ", length(p), " gene sets")
  }

  if (length(paths) == 0) {
    log_msg("  WARNING: no MSigDB pathways loaded at all -- enrichment will be skipped")
    return(structure(list(), collection = character(0)))
  }

  names(collection) <- names(paths)
  attr(paths, "collection") <- collection
  log_msg("  Total: ", length(paths), " pathways")
  paths
}

# [FIX P — v10, revised FIX R/S — v11] fgsea-preranked enrichment for
#   per-patient programs (PPs).
#
#   v10 ranked genes by the raw NMF weight (w_pp column), restricted to
#   whatever HVG set NMF was fit on (typically ~500-2000 genes). That's a
#   narrow universe for GSEA: many MSigDB pathways have most members absent
#   from the ranking entirely (never reaching minSize), and the ones that
#   pass have too little overlapping mass to clear a strict NES/padj bar.
#   Net effect: almost every PP came back empty.
#
#   [FIX R] v11 prefers a genome-wide rank vector per PP — built by scoring
#   each cell on the PP's HVG-restricted top genes, then correlating every
#   gene in the FULL transcriptome (not just HVGs) against that per-cell
#   score (see build_pp_projected_ranks()). This is the same logic as
#   projecting NMF/metaprogram scores back onto genes that Gavish et al. use
#   to extend metaprogram gene lists beyond a fixed HVG panel. Falls back to
#   the old w_pp-based (HVG-only, positive-loading) ranking if a projected
#   rank vector isn't supplied for a PP.
#
#   [CHG 3] Runs fgsea separately per MSigDB collection (H / GOBP / C6 / CP)
#   using the `collection` attribute set by load_msigdb_paths(), so BH
#   correction in one collection isn't diluted by the size of another
#   (GOBP alone is >7000 terms and was swamping the other three).
#
#   ranked_lists: named list, one numeric vector per PP, names = gene
#     symbols, values = the projected correlation (or NMF weight fallback).
#     Can be signed (projected) or non-negative (NMF weight fallback) —
#     scoreType is chosen automatically per PP based on whether negative
#     values are present.
#   w_pp: optional legacy fallback (genes x nPP NMF weight matrix) used only
#     for PPs missing from ranked_lists.
#   pp_names: character vector of PP names to process (used to drive w_pp
#     fallback iteration order)
#   paths: pre-loaded MSigDB pathway list (from load_msigdb_paths, tagged
#     with a "collection" attribute)
#   nes_min: reporting threshold on |NES| (loosened from 1.5 -> 1.0 by
#     default now that ranks are genome-wide rather than HVG-only; tune to
#     taste once you see the distribution of hits)
enrich_pp_fgsea <- function(ranked_lists = NULL, w_pp = NULL, pp_names,
                            out_dir, label,
                            paths = NULL,
                            min_size = 15, max_size = 500,
                            nes_min = 1.0, padj_max = 0.05,
                            top_leading = 20) {
  log_msg("  fgsea-preranked enrichment for", length(pp_names),
          "PP(s) (", label, ")")
  
  if (is.null(paths) || length(paths) == 0) {
    log_msg("  WARNING: no MSigDB paths supplied — skipping fgsea for", label)
    return(NULL)
  }
  collection <- attr(paths, "collection")
  if (is.null(collection)) {
    log_msg("  WARNING: paths have no 'collection' attribute — treating as one pool")
    collection <- setNames(rep("ALL", length(paths)), names(paths))
  }
  
  # Guard against fgsea's random tie-breaking; makes output reproducible
  set.seed(14)
  
  run_one_pp <- function(pp, ranks) {
    ranks <- ranks[!is.na(ranks)]
    ranks <- sort(ranks, decreasing = TRUE)
    if (length(ranks) < min_size + 5) {
      log_msg("    Skipping", pp, "— only", length(ranks), "ranked genes")
      return(NULL)
    }
    has_negative <- any(ranks < 0)
    score_type <- if (has_negative) "std" else "pos"
    
    collections_here <- unique(collection[names(paths)])
    per_coll <- lapply(collections_here, function(coll) {
      paths_coll <- paths[collection[names(paths)] == coll]
      res <- tryCatch(
        suppressWarnings(fgsea::fgseaMultilevel(
          pathways = paths_coll, stats = ranks,
          minSize = min_size, maxSize = max_size,
          scoreType = score_type
        )),
        error = function(e) {
          log_msg("    ERROR in fgsea for", pp, "/", coll, ":", conditionMessage(e))
          NULL
        }
      )
      if (is.null(res) || nrow(res) == 0) return(NULL)
      res$padj <- p.adjust(res$pval, method = "BH")  # correct within this collection only
      res$collection <- coll
      res
    })
    per_coll <- per_coll[!vapply(per_coll, is.null, logical(1))]
    if (length(per_coll) == 0) return(NULL)
    res <- do.call(rbind, per_coll)
    
    res <- res[res$padj < padj_max & abs(res$NES) >= nes_min, ]
    if (nrow(res) == 0) return(NULL)
    
    le_char <- vapply(res$leadingEdge, function(g) {
      if (length(g) == 0) return("")
      paste(utils::head(g, top_leading), collapse = "|")
    }, character(1))
    le_size <- lengths(res$leadingEdge)
    combined <- res$NES * (-log10(pmax(res$padj, 1e-300)))
    
    data.frame(
      gene_set        = pp,
      collection      = res$collection,
      pathway         = res$pathway,
      NES             = round(res$NES, 3),
      pval            = res$pval,
      padj            = res$padj,
      size            = res$size,
      leading_edge_n  = le_size,
      leading_edge    = le_char,
      combined_score  = round(combined, 3),
      rank_source     = if (has_negative) "projected_full_transcriptome" else "nmf_weight_hvg_only",
      stringsAsFactors = FALSE
    )
  }
  
  enrich_all <- list()
  for (pp in pp_names) {
    ranks <- NULL
    if (!is.null(ranked_lists) && !is.null(ranked_lists[[pp]])) {
      ranks <- ranked_lists[[pp]]
    } else if (!is.null(w_pp) && pp %in% colnames(w_pp)) {
      log_msg("    ", pp, ": no projected ranks — falling back to NMF weight (HVG-only)")
      ranks <- w_pp[, pp]
      ranks <- ranks[!is.na(ranks) & ranks > 0]
    }
    if (is.null(ranks) || length(ranks) < min_size + 5) {
      log_msg("    Skipping", pp, "— no usable rank vector")
      next
    }
    out <- run_one_pp(pp, ranks)
    if (!is.null(out)) enrich_all[[pp]] <- out
  }
  
  enrich_df <- do.call(rbind, enrich_all)
  if (!is.null(enrich_df) && nrow(enrich_df) > 0) {
    enrich_df <- enrich_df[order(enrich_df$gene_set,
                                 -enrich_df$combined_score), ]
    write.csv(enrich_df,
              file.path(out_dir, paste0(label, "_fgsea_enrichment.csv")),
              row.names = FALSE)
    log_msg("  fgsea table saved:", nrow(enrich_df),
            "significant pathway-PP pairs (|NES|>=", nes_min,
            ", padj<", padj_max, ", corrected per collection)")
  } else {
    log_msg("  No significant fgsea hits for", label,
            "at |NES|>=", nes_min, ", padj<", padj_max)
  }
  enrich_df
}

# [FIX R — v11, FIX T — v12] Build a genome-wide, signed rank vector per PP by:
#   1. Scoring each cell on the PP's existing HVG-restricted top genes using
#      UCell (rank-based within each cell — same method already used for
#      cohort-wide MP scoring in score_cells_streaming()).
#   2. Correlating every gene in the FULL transcriptome (mat_full_norm,
#      sparse, not HVG-restricted) against that per-cell score.
#   This extends the fgsea ranking universe from ~500-2000 HVGs to the whole
#   expressed transcriptome, and yields signed correlations (genes anti-
#   correlated with the program are informative too).
#
#   [FIX T] v11 scored cells with a plain colMeans() of log1p-normalized
#   expression. That's still correlated with each cell's total transcript
#   count even after library-size normalization (dropout/overdispersion at
#   low counts doesn't fully cancel it out) — so the "PP score" was partly
#   just "how many transcripts did this cell have," and nearly every gene in
#   the genome showed a weak same-direction correlation with it. Result:
#   40-60% of the entire MSigDB library came back "significant" per PP, a
#   classic GSEA failure mode from a confounded preranked statistic, not
#   real biology. UCell scores are built from each cell's own gene ranking
#   (like a per-cell Mann-Whitney rank statistic), which is largely
#   insensitive to total depth, and fixes this.
#
#   seu: the patient's Seurat object (RNA assay set as default, log-
#     normalized data already computed — Seurat's standard NormalizeData
#     output, which UCell reads from directly)
#   mat_full_norm: sparse, log1p-normalized, FULL matrix (genes x cells),
#     from normalize_sparse_full() on the un-subsetted count matrix — used
#     only for the correlation target (rownames/colnames), not for scoring
#   pp_genes: named list of character vectors, this patient's PP -> gene list
#     (i.e. res$pp_genes)
#   Returns: named list, one numeric vector per PP (names = gene symbols)
build_pp_projected_ranks <- function(seu, mat_full_norm, pp_genes,
                                     min_detect_frac = 0.01) {
  if (!identical(colnames(seu), colnames(mat_full_norm))) {
    log_msg("  WARNING: cell order mismatch between Seurat object and full matrix — skipping projection")
    return(list())
  }
  
  # [FIX U] Drop genes detected in almost no cells — near-zero-variance genes
  # produce numerically unstable, noise-driven correlations that inflate the
  # pathway hit count without adding real signal.
  n_cells <- ncol(mat_full_norm)
  detect_frac <- Matrix::rowSums(mat_full_norm > 0) / n_cells
  keep_genes <- detect_frac >= min_detect_frac
  mat_full_norm <- mat_full_norm[keep_genes, , drop = FALSE]
  log_msg("  Projection universe:", nrow(mat_full_norm), "genes detected in >=",
          round(min_detect_frac * 100, 1), "% of cells (of", length(keep_genes), "total)")
  
  # [FIX U] Confound to control for: total UMI counts per cell. Library-size
  # normalization makes every gene's normalized value partly a function of
  # this, which otherwise shows up as spurious correlation with any score.
  z <- tryCatch(seu$nCount_RNA, error = function(e) NULL)
  if (is.null(z)) z <- Matrix::colSums(GetAssayData(seu, assay = "RNA", layer = "counts"))
  z <- log1p(as.numeric(z))
  
  signatures <- lapply(pp_genes, function(genes) intersect(genes, rownames(seu)))
  signatures <- Filter(function(g) length(g) >= 3, signatures)
  if (length(signatures) == 0) return(list())
  
  scored <- tryCatch(
    UCell::AddModuleScore_UCell(seu, features = signatures, name = "_projscore"),
    error = function(e) {
      log_msg("  WARNING: UCell scoring failed for projection (", conditionMessage(e), ")")
      NULL
    }
  )
  if (is.null(scored)) return(list())
  
  out <- lapply(names(signatures), function(pp) {
    score_col <- paste0(pp, "_projscore")
    if (!score_col %in% colnames(scored@meta.data)) return(NULL)
    cell_score <- scored@meta.data[[score_col]]
    names(cell_score) <- colnames(scored)
    sparse_partial_row_cor(mat_full_norm, cell_score, z)
  })
  names(out) <- names(signatures)
  out[!vapply(out, is.null, logical(1))]
}

# [CHG 2] Generalized enrichment function — works for both PPs and MPs.
#   Replaces v6's enrich_mps(). Same hypergeometric test (phyper) + BH
#   correction, FDR < 0.05. Uses expressed-gene background [FIX I].
#
#   [FIX Q — v10] Now emits four extra columns to make output rankable:
#     n_overlap, overlap_ratio_gs, fold_enrichment, overlap_genes
#   Also writes a stringent secondary CSV alongside the lenient one, with
#   n_overlap >= 5 and fold_enrichment >= 3 filters — this typically cuts
#   300 hits down to ~30-60 readable ones.
#
#   gene_sets: named list of gene vectors (PPs or MPs)
#   out_dir: output directory for CSV
#   label: prefix for output filename and log messages (e.g. patient name or cohort_MP)
#   paths: pre-loaded MSigDB pathway list (from load_msigdb_paths)
#   expressed_genes: hypergeometric universe (HVG list for PPs, cohort union for MPs)
enrich_gene_sets <- function(gene_sets, out_dir, label,
                             species = "Homo sapiens", top_n = 50,
                             expressed_genes = NULL, paths = NULL,
                             stringent_min_overlap = 5,
                             stringent_min_fold = 3) {
  log_msg("  MSigDB enrichment for", length(gene_sets), "gene sets (", label, ")")
  
  # Load pathways if not pre-loaded
  if (is.null(paths)) {
    paths <- load_msigdb_paths(species)
  }
  
  # [FIX I] Use expressed-gene background if provided; otherwise fall back to
  # the union of all genes in the MSigDB pathways (less correct but functional).
  if (!is.null(expressed_genes) && length(expressed_genes) > 100) {
    universe <- expressed_genes
    log_msg("  Using expressed-gene background:", length(universe), "genes")
  } else {
    universe <- unique(unlist(paths))
    log_msg("  WARNING: no expressed-gene background provided — using MSigDB union (",
            length(universe), "genes). This may inflate significance.")
  }
  n_univ <- length(universe)
  
  enrich_all <- list()
  for (gs_name in names(gene_sets)) {
    gs_set <- gene_sets[[gs_name]]
    gs_set <- intersect(gs_set, universe)
    if (length(gs_set) < 3) next
    n_gs <- length(gs_set)
    
    # Compute p-value AND the metrics we need for ranking in one pass
    per_path <- lapply(names(paths), function(pw_name) {
      pw <- intersect(paths[[pw_name]], universe)
      if (length(pw) < 3) return(NULL)  # not eligible for testing given this universe
      inter <- intersect(gs_set, pw)
      k <- length(inter)
      # [FIX V — v14] Keep k=0 pathways in the correction pass (pval=1)
      # instead of dropping them before p.adjust. Dropping zero-overlap
      # pathways first shrinks the number of hypotheses BH corrects across,
      # understating the true multiple-testing burden and letting weak
      # single-gene overlaps look "significant" simply because the tested
      # set was artificially small.
      p <- if (k == 0) 1 else phyper(k - 1, length(pw), n_univ - length(pw), n_gs,
                                     lower.tail = FALSE)
      # Fold enrichment: observed / expected under the null
      expected <- (length(pw) * n_gs) / n_univ
      fe <- if (expected > 0) k / expected else NA_real_
      list(pathway = pw_name, pval = p,
           n_overlap = k, pw_size = length(pw),
           overlap_ratio_gs = k / n_gs,
           fold_enrichment = fe,
           overlap_genes = paste(inter, collapse = "|"))
    })
    per_path <- per_path[!vapply(per_path, is.null, logical(1))]
    if (length(per_path) == 0) next
    
    df_gs <- data.frame(
      gene_set        = gs_name,
      pathway         = vapply(per_path, `[[`, character(1), "pathway"),
      pval            = vapply(per_path, `[[`, numeric(1),   "pval"),
      n_overlap       = vapply(per_path, `[[`, integer(1),   "n_overlap"),
      pw_size         = vapply(per_path, `[[`, integer(1),   "pw_size"),
      overlap_ratio_gs = round(vapply(per_path, `[[`, numeric(1), "overlap_ratio_gs"), 3),
      fold_enrichment = round(vapply(per_path, `[[`, numeric(1), "fold_enrichment"), 2),
      overlap_genes   = vapply(per_path, `[[`, character(1), "overlap_genes"),
      stringsAsFactors = FALSE
    )
    # [FIX V] BH correction now runs over every pathway eligible for testing
    # (length(pw) >= 3 in this universe), including zero-overlap ones —
    # not just the ones that already looked promising.
    df_gs$padj <- p.adjust(df_gs$pval, method = "BH")
    df_gs$combined_score <- round(df_gs$fold_enrichment *
                                    (-log10(pmax(df_gs$padj, 1e-300))), 3)
    # [FIX V] Zero-overlap rows have served their purpose in the correction;
    # drop them now, and require >=2 overlapping genes for the lenient
    # table — a single shared gene isn't interpretable as "enrichment"
    # regardless of how significant the p-value looks for a small pathway.
    df_gs <- df_gs[df_gs$n_overlap >= 2 & df_gs$padj < 0.05, ]
    if (nrow(df_gs) == 0) next
    enrich_all[[gs_name]] <- df_gs
  }
  
  enrich_df <- do.call(rbind, enrich_all)
  if (!is.null(enrich_df) && nrow(enrich_df) > 0) {
    # Sort: gene_set (PP or MP), then combined_score descending — strongest first
    enrich_df <- enrich_df[order(enrich_df$gene_set,
                                 -enrich_df$combined_score), ]
    write.csv(enrich_df, file.path(out_dir, paste0(label, "_enrichment.csv")),
              row.names = FALSE)
    log_msg("  Enrichment table saved:", nrow(enrich_df),
            "significant pathway-gene_set pairs (lenient: FDR<0.05)")
    
    # Stringent secondary CSV — kills weak/small overlaps
    strict <- enrich_df[enrich_df$n_overlap  >= stringent_min_overlap &
                          enrich_df$fold_enrichment >= stringent_min_fold, ]
    if (nrow(strict) > 0) {
      write.csv(strict,
                file.path(out_dir, paste0(label, "_enrichment_stringent.csv")),
                row.names = FALSE)
      log_msg("  Stringent table saved:", nrow(strict),
              "hits (n_overlap>=", stringent_min_overlap,
              ", fold>=", stringent_min_fold, ")")
    }
  } else {
    log_msg("  No significant enrichments at FDR < 0.05 for", label)
  }
  enrich_df
}

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

gavish_mps <- load_gavish_mps(gavish_xlsx_file, gavish_xlsx_sheet, msigdb_species)

COHORT_STATUS <- list()

for (cohort in cohorts) {
 # One cohort failing should not discard the cohorts already finished.
 COHORT_STATUS[[cohort$folder]] <- tryCatch({
  cohort_data_dir <- file.path(base_data_dir, cohort$folder)
  cohort_out_dir  <- file.path(base_out_dir, cohort$folder)
  log_msg("Cohort:", cohort$folder)
  
  if (nzchar(INPUT_SUBDIR)) cohort_data_dir <- file.path(cohort_data_dir, INPUT_SUBDIR)
  if (!dir.exists(cohort_data_dir)) {
    log_msg(" Directory does not exist — skipping cohort: ", cohort_data_dir)
    # `next` aborts the tryCatch, so record the status before jumping
    COHORT_STATUS[[cohort$folder]] <- "skipped: input directory missing"
    next
  }
  rds_files <- list.files(cohort_data_dir, pattern = INPUT_PATTERN,
                          full.names = TRUE)
  if (length(rds_files) == 0) {
    log_msg(" No files matching ", INPUT_PATTERN, " in ", cohort_data_dir)
    present <- list.files(cohort_data_dir, pattern = "\\.rds$")
    if (length(present)) {
      log_msg("  .rds files that ARE there: ",
              paste(utils::head(present, 5), collapse = ", "),
              if (length(present) > 5) " ..." else "")
      log_msg("  Adjust INPUT_PATTERN / ID_PATTERN at the top of this script.")
    } else {
      log_msg("  Directory is empty — no patients of this subtype (expected for HER2), or scripts 05/06 have not run")
    }
    COHORT_STATUS[[cohort$folder]] <- "skipped: no input files"
    next
  }
  log_msg(" Found", length(rds_files), "patients.")
  
  # [CHG 2] Load MSigDB pathways ONCE per cohort — reused for all per-patient
  #   and MP-level enrichment calls in this cohort
  msigdb_paths <- load_msigdb_paths(msigdb_species)
  
  all_pp_genes <- list()
  seu_paths <- list()
  expressed_genes_cohort <- character(0)  # [FIX I] collect for enrichment background
  
  for (rds_path in rds_files) {
    fname <- basename(rds_path)
    patient_name <- sub(ID_PATTERN, "\\1", fname)
    if (identical(patient_name, fname)) {
      # pattern did not match; fall back to the filename without extension
      patient_name <- tools::file_path_sans_ext(fname)
    }
    pat_out_dir <- file.path(cohort_out_dir, patient_name)
    
    res <- process_patient(
      rds_path = rds_path, patient_name = patient_name, pat_out_dir = pat_out_dir,
      ranks = ranks, nrun = nrun, method = method, min_size = min_size,
      h_range = h_range, top_n = pp_top_n, within_pat_max_frac = within_pat_max_frac,
      persistence_frac = persistence_frac, seed = 14,
      adaptive_ranks = isTRUE(cohort$adaptive_ranks),
      min_cells = if (!is.null(cohort$min_cells)) cohort$min_cells else 60,
      cells_per_rank = cells_per_rank
    )
    
    if (!is.null(res)) {
      all_pp_genes <- c(all_pp_genes, res$pp_genes)
      seu_paths[[patient_name]] <- file.path(pat_out_dir, paste0(patient_name, "_seurat.rds"))
      # [FIX I] Collect expressed genes (from PP gene lists as a proxy for HVGs)
      expressed_genes_cohort <- union(expressed_genes_cohort, unlist(res$pp_genes))
      
      # [CHG 2 / FIX L] Per-patient GO enrichment on individual PPs
      #   Uses that patient's HVG list as the hypergeometric universe.
      #   [FIX L] ensure_hvg_file extracts HVGs from cached Seurat if the
      #   _hvg_genes.txt file doesn't exist yet (already-processed patients).
      hvg_file <- file.path(pat_out_dir, paste0(patient_name, "_hvg_genes.txt"))
      pat_universe <- if (file.exists(hvg_file)) {
        readLines(hvg_file)
      } else {
        # [FIX L] Try extracting from cached Seurat before falling back
        hvg <- ensure_hvg_file(patient_name, pat_out_dir)
        if (!is.null(hvg) && length(hvg) > 100) {
          hvg
        } else {
          log_msg("  WARNING: HVG file not found for", patient_name,
                  "— using PP gene union as universe")
          union(unlist(res$pp_genes), character(0))
        }
      }
      tryCatch(
        enrich_gene_sets(res$pp_genes, pat_out_dir, label = patient_name,
                         species = msigdb_species, top_n = enrich_top_n,
                         expressed_genes = pat_universe, paths = msigdb_paths),
        error = function(e) log_msg("  ERROR in per-patient enrichment for",
                                    patient_name, ":", conditionMessage(e))
      )
      
      # [FIX W — v14] PP-level fgsea DISABLED. Preranked GSEA assumes gene
      # labels are exchangeable under permutation, which only holds when the
      # ranking comes from something external to the tested expression
      # matrix (treatment vs control, genotype, etc.). Here the rank was a
      # per-cell score built FROM the same patient's expression data, then
      # correlated back against every gene in that same matrix — so real,
      # non-random gene-gene co-expression structure (cell cycle, stress,
      # hypoxia, any dominant axis of variation in that patient's cells)
      # drives huge numbers of pathways into "significance" regardless of
      # whether they define this specific PP. Partial correlation and
      # detection filtering fixed the *technical* confounds (depth,
      # compositionality) but not this — it's a property of testing a
      # self-derived phenotype, not a bug in how the correlation was
      # computed, and no amount of further cleaning removes it.
      #
      # PP pathway identity now comes from two methods that don't have this
      # problem: enrich_gene_sets() above (ORA on the fixed top-50 gene
      # list — a discrete test, not a self-derived continuous score) and
      # the Gavish Jaccard anchoring already computed per-cohort
      # (save_full_jaccard_to_gavish, anchor_to_published) — comparing
      # against published, expert-curated metaprogram gene lists.
      #
      # res$w_pp_projected is still built and still written to
      # <patient>_PP_projected_top_genes.csv as a descriptive list (what
      # correlates with this PP beyond its top 50) — just no formal
      # enrichment statistics are run on it anymore.
      #
      # enrich_pp_fgsea() itself is left defined above, unused, in case you
      # want it back for a different purpose (e.g. testing against a
      # genuinely external ranking, where the exchangeability assumption
      # would hold).
      
      rm(pat_universe); gc()
      
      rm(res); gc()
    }
  }
  
  log_msg("Total PPs collected for", cohort$folder, ":", length(all_pp_genes))
  
  mp_min_size <- max(3, ceiling(mp_min_frac * length(all_pp_genes)))
  log_msg(" MP minimum cluster size:", mp_min_size)
  
  mp_out_dir <- file.path(cohort_out_dir, "MetaPrograms")
  dir.create(mp_out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # [FIX O] Full Jaccard matrix of ALL per-patient PPs vs Gavish 3CA MPs
  #   (saved BEFORE MP clustering so PPs are preserved regardless of MP
  #   clustering outcome). One row per PP, one column per Gavish MP.
  save_full_jaccard_to_gavish(all_pp_genes, gavish_mps, mp_out_dir,
                              label = paste0(cohort$folder, "_PP"),
                              title_suffix = paste0(" (", cohort$folder,
                                                    " — ", length(all_pp_genes),
                                                    " PPs)"))
  
  mp_res <- tryCatch(
    find_meta_programs(all_pp_genes, cohort_name = cohort$folder,
                       cohort_out_dir = mp_out_dir, mp_min_size = mp_min_size,
                       h_range = mp_h_range, min_pat_frac = mp_min_pat_frac,
                       ubiq_frac = ubiq_frac_base, use_dynamic_tree_cut = use_dynamic_tree_cut,
                       min_pat_n_absolute = min_pat_n_absolute,
                       small_cohort_threshold = small_cohort_threshold,
                       cut_height_ceiling = cut_height_ceiling,
                       cut_height_search_start = cut_height_search_start),
    error = function(e) {
      log_msg(" ERROR in MP clustering:", conditionMessage(e))
      COHORT_STATUS[[cohort$folder]] <<- paste("FAILED: MP clustering:", conditionMessage(e))
      NULL
    }
  )
  
  rm(all_pp_genes); gc()
  
  if (is.null(mp_res)) {
    if (is.null(COHORT_STATUS[[cohort$folder]]))
      COHORT_STATUS[[cohort$folder]] <- "no MPs (too few PPs to cluster)"
    next
  }
  
  # ── Step 3: cohort-wide usage matrix (streaming) ───────────────────────────
  mp_genes_primary <- mp_res$jaccard$mp_genes
  clean_idx <- which(nchar(mp_res$jaccard$doublet_flags) == 0)
  mp_genes_clean <- mp_genes_primary[clean_idx]
  if (length(mp_genes_clean) < length(mp_genes_primary))
    log_msg(" Excluding", length(mp_genes_primary) - length(mp_genes_clean),
            "doublet-flagged MPs from cell scoring (ribo_mito-flagged MPs KEPT)")
  
  usage <- score_cells_streaming(seu_paths, mp_genes_clean, cohort$folder,
                                 mp_out_dir, method = score_method)
  
  rm(usage); gc()
  
  # ── Step 4a: anchoring to published 41 MPs ─────────────────────────────────
  anchor_df <- anchor_to_published(mp_genes_primary, gavish_mps, mp_out_dir, cohort$folder)
  
  # [FIX O] Full Jaccard matrix of MPs vs Gavish (all pairs, not just best match)
  save_full_jaccard_to_gavish(mp_genes_primary, gavish_mps, mp_out_dir,
                              label = paste0(cohort$folder, "_MP"),
                              title_suffix = paste0(" (", cohort$folder, ")"))
  
  # ── Step 4b: MSigDB functional enrichment (MP-level, KEPT) ─────────────────
  # [FIX I] Pass expressed-gene background to avoid inflated significance
  # [CHG 2] Use pre-loaded msigdb_paths and generalized enrich_gene_sets
  enrich_df <- enrich_gene_sets(mp_genes_primary, mp_out_dir,
                                label = paste0(cohort$folder, "_MP"),
                                species = msigdb_species, top_n = enrich_top_n,
                                expressed_genes = expressed_genes_cohort,
                                paths = msigdb_paths)
  
  # [CHG 2] Free MSigDB pathways after MP-level enrichment
  rm(msigdb_paths); gc()
  
  log_msg("=== Cohort", cohort$folder, "complete ===")
  
  rm(mp_res, mp_genes_primary, mp_genes_clean, anchor_df, enrich_df, seu_paths,
     expressed_genes_cohort)
  gc()
  log_msg(" Memory cleared between cohorts.")
  "ok"
 }, error = function(e) {
   log_msg(" COHORT FAILED: ", cohort$folder, " -- ", conditionMessage(e))
   paste("FAILED:", conditionMessage(e))
 })
}

log_msg("=== summary ===")
for (nm in names(COHORT_STATUS)) log_msg("  ", nm, ": ", COHORT_STATUS[[nm]])
status_df <- data.frame(cohort = names(COHORT_STATUS),
                        status = unlist(COHORT_STATUS), row.names = NULL)
write.csv(status_df, file.path(base_out_dir, "cohort_status.csv"), row.names = FALSE)

try(stopCluster(cl), silent = TRUE)
log_msg("=== Pipeline complete ===")

if (any(grepl("^FAILED", unlist(COHORT_STATUS)))) {
  quit(status = 1)   # non-zero exit so SLURM marks the job FAILED
}

