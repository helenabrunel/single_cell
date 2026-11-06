"""
part0_integration_comparison.py
================================
Compare batch-integration methods on malignant / epithelial cells.

Datasets
--------
  Visvader  : Epithelial Global  (compartment_epithelial.h5ad)
  Klughammer: Malignant Global   (klughammer_Malignant.h5ad)

Methods compared
----------------
  1. Unintegrated  — PCA only (no batch correction), baseline
  2. Harmony       — linear correction in PCA space (harmonypy)
  3. scVI          — non-linear VAE (scvi-tools)
  4. DRVI          — interpretable non-linear VAE (drvi-py)

For each dataset the script produces:
  figures/part0/<dataset>_umap_panel.png
      4 rows (methods) × 4 cols (patient, stemness, immune, CytoTRACE2)
  figures/part0/<dataset>_batch_mixing.png
      Fraction of k-NN from a different patient — per method boxplot
  figures/part0/<dataset>_correlation_preservation.png
      Per-patient Spearman(stemness, immune) distribution — per method
  figures/part0/<dataset>_metrics.csv
      Quantitative summary (median r, mean batch mixing)

Evaluation criteria
-------------------
  • Batch mixing    : higher fraction of inter-patient k-NN = better mixing
  • Signal survival : per-patient Spearman r(stemness × immune) close to
                      the unintegrated baseline = biological signal preserved

Notes on expression input
-------------------------
  Visvader   : raw counts available (layers["counts"]) — used for scVI / DRVI
  Klughammer : only log-normalised X available (no raw counts) —
               scVI uses gene_likelihood="normal"; DRVI uses is_count_data=False

Estimated runtime: ~60-90 min (dominated by scVI / DRVI training)
Set SKIP_HEAVY = True to run only Unintegrated + Harmony quickly.

Run from: ~/single_cell/
  python3 part0_integration_comparison.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings, gc
from scipy.stats import spearmanr
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

# ── Optional: skip scVI / DRVI for a quick check ─────────────────────────────
SKIP_HEAVY = False    # set True to run only Unintegrated + Harmony

# ── Paths and parameters ─────────────────────────────────────────────────────
OUTDIR   = "figures/part0"
SIGFILE  = "Visvader/signatures_confirmed.json"
N_HVG    = 2000
N_PCS    = 30
N_NEIGH  = 15
RANDOM   = 42
N_LATENT    = 20      # latent dims for scVI and DRVI
SUBSAMPLE   = 20_000  # cells for scVI/DRVI training; all cells get projected
BATCH_SIZE  = 1024    # large batch → fewer steps/epoch → tractable on CPU
MAX_EPOCHS  = 100     # cap; early stopping will usually stop earlier

os.makedirs(OUTDIR, exist_ok=True)

with open(SIGFILE) as f:
    ALL_SIGNATURES = json.load(f)

# ── Signatures ────────────────────────────────────────────────────────────────
STEM_SIGS = {
    "LIM_stem"    : "LIM_MAMMARY_STEM_CELL_UP",
    "stem_cons"   : "stem_consensus",
    "ESC_ASSOU"   : "ESC_ASSOU",
}
PRIMARY_STEM = "LIM_stem"   # used for UMAP colouring

IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
]

# Scanpy / Seurat standard cell-cycle gene lists
S_GENES = [
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP",
    "RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2",
    "RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
    "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
]
G2M_GENES = [
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
    "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
    "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
    "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1",
    "ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA",
]

# ── Dataset definitions ───────────────────────────────────────────────────────
DATASETS = {
    "Visvader_Epithelial": {
        "path"         : "Visvader/compartment_epithelial.h5ad",
        "batch_col"    : "sample",
        "subtype_col"  : "subtype",
        "count_layer"  : "counts",      # raw counts — preferred for scVI/DRVI
        "log_layer"    : "logcounts",
        "cc_in_obs"    : False,         # cell cycle must be computed
        "cytotrace_csv": "Visvader/cytotrace2_scores_Epithelial_Global.csv",
    },
    "Klughammer_Malignant": {
        "path"         : "Klughammer/klughammer_Malignant.h5ad",
        "batch_col"    : "sampleid",
        "subtype_col"  : "subtype",
        "count_layer"  : None,          # no raw counts; X is log-normalised
        "log_layer"    : None,          # use X directly
        "cc_in_obs"    : True,          # S.Score / G2M.Score already in obs
        "cytotrace_csv": "Klughammer/cytotrace2_scores_klughammer_Malignant_Global.csv",
    },
}

METHODS = (["Unintegrated", "Harmony"] if SKIP_HEAVY
           else ["Unintegrated", "Harmony", "scVI", "DRVI"])

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}
METHOD_COLORS = {
    "Unintegrated": "#888888",
    "Harmony"     : "#4393C3",
    "scVI"        : "#D6604D",
    "DRVI"        : "#762A83",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_log_x(adata, cfg):
    """Set adata.X to log-normalised counts for scoring / HVG."""
    if cfg["log_layer"] and cfg["log_layer"] in adata.layers:
        adata.X = adata.layers[cfg["log_layer"]].copy()
    # else X is already log-normalised (Klughammer)


def score_signatures(adata):
    """Score stemness, immunogenicity, and (if needed) cell cycle."""
    print("  Scoring signatures …")
    for short, sig_key in STEM_SIGS.items():
        genes = [g for g in ALL_SIGNATURES[sig_key] if g in adata.var_names]
        if len(genes) >= 5:
            sc.tl.score_genes(adata, genes, score_name=f"score_{short}",
                              use_raw=False)
            print(f"    {short}: {len(genes)}/{len(ALL_SIGNATURES[sig_key])} genes")

    imm_ok = [g for g in IMMUNOGENIC if g in adata.var_names]
    sc.tl.score_genes(adata, imm_ok, score_name="score_immune", use_raw=False)
    print(f"  immune: {len(imm_ok)}/{len(IMMUNOGENIC)} genes")


def score_cell_cycle(adata):
    """Compute Scanpy cell-cycle scores (stores S_score, G2M_score in obs)."""
    print("  Computing cell cycle scores …")
    s_ok   = [g for g in S_GENES   if g in adata.var_names]
    g2m_ok = [g for g in G2M_GENES if g in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_ok, g2m_genes=g2m_ok)
    # rename to match Klughammer convention
    adata.obs.rename(columns={"S_score": "S.Score", "G2M_score": "G2M.Score"},
                     inplace=True)


def load_cytotrace(adata, csv_path):
    """Merge CytoTRACE2_Score into obs (inner join on cell barcodes)."""
    ct = pd.read_csv(csv_path, index_col=0)[["CytoTRACE2_Score"]]
    adata.obs["CytoTRACE2"] = ct.reindex(adata.obs_names)["CytoTRACE2_Score"]
    n_missing = adata.obs["CytoTRACE2"].isna().sum()
    if n_missing > 0:
        print(f"    ⚠  CytoTRACE2: {n_missing} cells missing → filled with median")
        med = adata.obs["CytoTRACE2"].median()
        adata.obs["CytoTRACE2"].fillna(med, inplace=True)


def compute_hvg_pca(adata, batch_col, n_hvg=N_HVG, n_pcs=N_PCS):
    """HVGs (batch-aware) → scale → PCA. Stores X_pca in obsm."""
    print("  HVG + PCA …")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_hvg, flavor="seurat",
        batch_key=batch_col if adata.obs[batch_col].nunique() > 1 else None,
        subset=False,
    )
    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    if ad_hvg.layers.get("logcounts") is not None:
        ad_hvg.X = ad_hvg.layers["logcounts"].copy()
    sc.pp.scale(ad_hvg, max_value=10)
    sc.tl.pca(ad_hvg, svd_solver="arpack", n_comps=n_pcs, random_state=RANDOM)
    adata.obsm["X_pca_fresh"] = ad_hvg.obsm["X_pca"].copy()
    del ad_hvg; gc.collect()
    print(f"    HVGs selected: {adata.var['highly_variable'].sum()}")


# ── Integration functions ─────────────────────────────────────────────────────

def integrate_unintegrated(adata):
    """Just neighbours on raw PCA."""
    sc.pp.neighbors(adata, use_rep="X_pca_fresh", n_pcs=N_PCS,
                    n_neighbors=N_NEIGH, random_state=RANDOM)
    sc.tl.umap(adata, random_state=RANDOM)
    adata.obsm["X_umap_Unintegrated"]    = adata.obsm["X_umap"].copy()
    adata.obsp["conn_Unintegrated"]      = adata.obsp["connectivities"].copy()
    print("  ✓ Unintegrated")


def integrate_harmony(adata, batch_col):
    import harmonypy as hm
    ho  = hm.run_harmony(
        adata.obsm["X_pca_fresh"], adata.obs[[batch_col]],
        batch_col, max_iter_harmony=20, random_state=RANDOM, verbose=False,
    )
    Z = ho.Z_corr
    if Z.shape[0] == N_PCS and Z.shape[1] == adata.n_obs:
        Z = Z.T
    adata.obsm["X_harmony"] = Z
    sc.pp.neighbors(adata, use_rep="X_harmony", n_pcs=N_PCS,
                    n_neighbors=N_NEIGH, random_state=RANDOM)
    sc.tl.umap(adata, random_state=RANDOM)
    adata.obsm["X_umap_Harmony"] = adata.obsm["X_umap"].copy()
    adata.obsp["conn_Harmony"]   = adata.obsp["connectivities"].copy()
    print("  ✓ Harmony")


def integrate_scvi(adata, cfg):
    """
    scVI integration.
    Trains on a random SUBSAMPLE of cells for speed, then projects all cells
    into the latent space with model.get_latent_representation(adata_full).
    Uses raw counts when available (Visvader); log-normalised X with
    gene_likelihood='normal' when no raw counts (Klughammer).
    """
    from scvi.model import SCVI

    hvg_mask   = adata.var["highly_variable"].values
    ad_hvg     = adata[:, hvg_mask].copy()
    has_counts = cfg["count_layer"] is not None

    # Subsample for training
    n_train   = min(SUBSAMPLE, ad_hvg.n_obs)
    rng       = np.random.default_rng(RANDOM)
    train_idx = rng.choice(ad_hvg.n_obs, n_train, replace=False)
    ad_train  = ad_hvg[train_idx].copy()
    print(f"    scVI: training on {n_train:,} / {ad_hvg.n_obs:,} cells …")

    if has_counts:
        SCVI.setup_anndata(ad_train, layer=cfg["count_layer"],
                           batch_key=cfg["batch_col"])
        model = SCVI(ad_train, n_latent=N_LATENT, gene_likelihood="nb")
    else:
        SCVI.setup_anndata(ad_train, batch_key=cfg["batch_col"])
        model = SCVI(ad_train, n_latent=N_LATENT, gene_likelihood="normal")

    model.train(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                early_stopping=True, early_stopping_patience=10,
                check_val_every_n_epoch=5, progress_bar_refresh_rate=0)

    # Project all cells (not just training subset)
    if has_counts:
        SCVI.setup_anndata(ad_hvg, layer=cfg["count_layer"],
                           batch_key=cfg["batch_col"])
    else:
        SCVI.setup_anndata(ad_hvg, batch_key=cfg["batch_col"])
    latent = model.get_latent_representation(ad_hvg)
    adata.obsm["X_scVI"] = latent

    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=N_NEIGH,
                    random_state=RANDOM)
    sc.tl.umap(adata, random_state=RANDOM)
    adata.obsm["X_umap_scVI"] = adata.obsm["X_umap"].copy()
    adata.obsp["conn_scVI"]   = adata.obsp["connectivities"].copy()

    del ad_hvg, ad_train, model; gc.collect()
    print("  ✓ scVI")


def integrate_drvi(adata, cfg):
    """
    DRVI integration.
    Trains on a random SUBSAMPLE of cells for speed, then projects all cells.
    Raw counts when available; log-normalised with is_count_data=False otherwise.
    """
    from drvi.model import DRVI

    hvg_mask   = adata.var["highly_variable"].values
    ad_hvg     = adata[:, hvg_mask].copy()
    has_counts = cfg["count_layer"] is not None

    # Subsample for training
    n_train   = min(SUBSAMPLE, ad_hvg.n_obs)
    rng       = np.random.default_rng(RANDOM)
    train_idx = rng.choice(ad_hvg.n_obs, n_train, replace=False)
    ad_train  = ad_hvg[train_idx].copy()
    print(f"    DRVI: training on {n_train:,} / {ad_hvg.n_obs:,} cells …")

    if has_counts:
        DRVI.setup_anndata(ad_train, layer=cfg["count_layer"],
                           batch_key=cfg["batch_col"], is_count_data=True)
    else:
        DRVI.setup_anndata(ad_train, batch_key=cfg["batch_col"],
                           is_count_data=False)

    model = DRVI(ad_train, n_latent=N_LATENT * 2)  # DRVI uses more latent dims
    model.train(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                early_stopping=True, early_stopping_patience=10,
                check_val_every_n_epoch=5, progress_bar_refresh_rate=0)

    # Project all cells
    if has_counts:
        DRVI.setup_anndata(ad_hvg, layer=cfg["count_layer"],
                           batch_key=cfg["batch_col"], is_count_data=True)
    else:
        DRVI.setup_anndata(ad_hvg, batch_key=cfg["batch_col"],
                           is_count_data=False)
    latent = model.get_latent_representation(ad_hvg)
    adata.obsm["X_DRVI"] = latent

    sc.pp.neighbors(adata, use_rep="X_DRVI", n_neighbors=N_NEIGH,
                    random_state=RANDOM)
    sc.tl.umap(adata, random_state=RANDOM)
    adata.obsm["X_umap_DRVI"] = adata.obsm["X_umap"].copy()
    adata.obsp["conn_DRVI"]   = adata.obsp["connectivities"].copy()

    del ad_hvg, ad_train, model; gc.collect()
    print("  ✓ DRVI")


# ── Metrics ───────────────────────────────────────────────────────────────────

def batch_mixing_score(adata, batch_col, conn_key, n_sample=5000):
    """
    Mean fraction of k-NN from a *different* patient (higher = better mixing).
    Sampled over n_sample cells for speed.
    """
    conn    = adata.obsp[conn_key]
    batches = adata.obs[batch_col].values
    n_cells = conn.shape[0]
    rng     = np.random.default_rng(RANDOM)
    idx     = rng.choice(n_cells, min(n_sample, n_cells), replace=False)
    scores  = []
    for i in idx:
        row = np.asarray(conn[i].todense()).flatten()
        neigh = np.where(row > 0)[0]
        if len(neigh) == 0:
            continue
        diff = np.sum(batches[neigh] != batches[i])
        scores.append(diff / len(neigh))
    return np.array(scores)


def stemness_coherence(adata, stem_col, conn_key, n_sample=5000):
    """
    Pearson r between a cell's stemness score and the mean stemness score
    of its k-NN in the integrated graph.
    High value = high-stemness cells cluster together (biological preservation).
    Low value = stemness signal dispersed / lost in the embedding.

    Note: per-patient Spearman r(stem, immune) is invariant across integration
    methods (scores are computed from the expression matrix, not the embedding).
    This metric instead asks whether the embedding itself preserves the
    stemness gradient independent of patient of origin.
    """
    from scipy.stats import pearsonr
    conn     = adata.obsp[conn_key]
    stem     = adata.obs[stem_col].values.astype(float)
    n_cells  = conn.shape[0]
    rng      = np.random.default_rng(RANDOM)
    idx      = rng.choice(n_cells, min(n_sample, n_cells), replace=False)
    own, nbr = [], []
    for i in idx:
        row   = np.asarray(conn[i].todense()).flatten()
        neigh = np.where(row > 0)[0]
        if len(neigh) == 0:
            continue
        own.append(stem[i])
        nbr.append(stem[neigh].mean())
    if len(own) < 10:
        return np.nan
    r, _ = pearsonr(own, nbr)
    return r


def per_patient_corr(adata, stem_col, immune_col, batch_col, min_cells=50):
    """
    Per-patient Spearman r(stemness, immune) computed from expression scores.
    This is INVARIANT across integration methods (scores come from the
    expression matrix, not the embedding) and serves as the ground-truth
    anchor showing the signal exists within each patient's tumour.
    """
    rs = []
    for pat, grp in adata.obs.groupby(batch_col):
        if len(grp) < min_cells:
            continue
        x = grp[stem_col].values.astype(float)
        y = grp[immune_col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < min_cells:
            continue
        r, _ = spearmanr(x[mask], y[mask])
        rs.append(r)
    return np.array(rs)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _scatter(ax, coords, values, title, cmap="RdYlBu_r",
             vmin=None, vmax=None, s=0.3, alpha=0.4):
    """Generic UMAP scatter with colorbar."""
    vals = np.asarray(values)
    if vals.dtype.kind in ("U", "O", "S"):   # categorical
        cats  = pd.Categorical(vals)
        cmap_ = plt.cm.get_cmap("tab20", cats.nunique())
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=cats.codes, cmap=cmap_, s=s, alpha=alpha,
                   rasterized=True)
    else:
        fvals = vals.astype(float)
        vmin  = vmin if vmin is not None else np.nanpercentile(fvals, 1)
        vmax  = vmax if vmax is not None else np.nanpercentile(fvals, 99)
        sc_p  = ax.scatter(coords[:, 0], coords[:, 1],
                           c=fvals, cmap=cmap, vmin=vmin, vmax=vmax,
                           s=s, alpha=alpha, rasterized=True)
        plt.colorbar(sc_p, ax=ax, shrink=0.6, pad=0.01)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def plot_umap_panel(adata, batch_col, subtype_col, dataset_name, methods_done):
    """
    4-column panel per method:
      col 0 = patient, col 1 = subtype, col 2 = stemness, col 3 = immune
      + col 4 = CytoTRACE2 (if available)
    """
    has_ct  = "CytoTRACE2" in adata.obs.columns
    n_cols  = 5 if has_ct else 4
    n_rows  = len(methods_done)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Integration comparison — {dataset_name}",
                 fontsize=11, fontweight="bold")

    for r, method in enumerate(methods_done):
        umap_key = f"X_umap_{method}"
        if umap_key not in adata.obsm:
            for c in range(n_cols):
                axes[r, c].text(0.5, 0.5, "N/A",
                                ha="center", va="center", transform=axes[r, c].transAxes)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, 0].set_ylabel(method, fontsize=9, fontweight="bold")
            continue
        coords = adata.obsm[umap_key]

        # patient ID
        pat_cats = adata.obs[batch_col].astype("category")
        n_pats   = pat_cats.nunique()
        cmap_p   = plt.cm.get_cmap("tab20", n_pats)
        axes[r, 0].scatter(coords[:, 0], coords[:, 1],
                           c=pat_cats.cat.codes, cmap=cmap_p,
                           s=0.3, alpha=0.3, rasterized=True)
        axes[r, 0].set_title(f"Patient (n={n_pats})", fontsize=8)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(method, fontsize=9, fontweight="bold")

        # subtype
        for st, color in SUBTYPE_COLORS.items():
            m = adata.obs[subtype_col] == st
            if m.sum() > 0:
                axes[r, 1].scatter(coords[m, 0], coords[m, 1],
                                   c=color, s=0.3, alpha=0.3, rasterized=True)
        axes[r, 1].set_title("Subtype", fontsize=8)
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])

        # stemness
        stem_col = f"score_{PRIMARY_STEM}"
        if stem_col in adata.obs.columns:
            _scatter(axes[r, 2], coords, adata.obs[stem_col],
                     "Stemness (LIM_stem)")

        # immune
        if "score_immune" in adata.obs.columns:
            _scatter(axes[r, 3], coords, adata.obs["score_immune"],
                     "Immune (APM)", cmap="PuRd")

        # CytoTRACE2
        if has_ct:
            _scatter(axes[r, 4], coords, adata.obs["CytoTRACE2"],
                     "CytoTRACE2", cmap="YlOrRd_r")

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{dataset_name}_umap_panel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_batch_mixing(results, dataset_name, methods_done, n_patients):
    """Boxplot of batch-mixing scores per method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data   = [results[m]["batch_mixing"] for m in methods_done
               if results[m]["batch_mixing"] is not None]
    labels = [m for m in methods_done
               if results[m]["batch_mixing"] is not None]
    colors = [METHOD_COLORS.get(m, "#666666") for m in labels]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Fraction of k-NN from different patient", fontsize=10)
    ax.set_title(f"Batch mixing — {dataset_name}\n"
                 f"(higher = more inter-patient mixing)", fontsize=10)
    random_expectation = 1.0 - 1.0 / max(2, n_patients)
    ax.axhline(random_expectation, ls="--", color="grey", alpha=0.5,
               label=f"Random expectation (n={n_patients} patients)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{dataset_name}_batch_mixing.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_metrics_summary(results, dataset_name, methods_done, n_patients):
    """
    Two-panel summary figure:
      Left:  batch mixing (higher = better)
      Right: stemness coherence in embedding (higher = biological gradient preserved)
    """
    valid = [m for m in methods_done if results[m]["batch_mixing"] is not None]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Integration quality — {dataset_name}", fontsize=11,
                 fontweight="bold")

    colors = [METHOD_COLORS.get(m, "#666666") for m in valid]

    # Left: batch mixing (distributions)
    bm_data = [results[m]["batch_mixing"] for m in valid]
    bp = axes[0].boxplot(bm_data, patch_artist=True, widths=0.5,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_xticklabels(valid, fontsize=9)
    axes[0].set_ylabel("Fraction of k-NN from different patient", fontsize=9)
    axes[0].set_title("Batch mixing (higher = better)", fontsize=10)
    random_expectation = 1.0 - 1.0 / max(2, n_patients)
    axes[0].axhline(random_expectation, ls="--", color="grey", alpha=0.6,
                    label=f"Random (n={n_patients} pts)")
    axes[0].legend(fontsize=8)

    # Right: stemness coherence (bar)
    sc_vals = [results[m]["stemness_coherence"] if results[m]["stemness_coherence"]
               is not None else np.nan for m in valid]
    bars = axes[1].bar(valid, sc_vals, color=colors, alpha=0.75, edgecolor="black")
    axes[1].set_ylabel("Pearson r (own stemness vs k-NN mean stemness)", fontsize=9)
    axes[1].set_title("Stemness gradient coherence\n(higher = gradient preserved in embedding)",
                      fontsize=10)
    axes[1].set_ylim(0, 1)
    for bar, v in zip(bars, sc_vals):
        if not np.isnan(v):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{dataset_name}_metrics_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_corr_preservation(results, dataset_name, methods_done):
    """
    Boxplot of per-patient Spearman r(stemness, immune).
    Invariant across methods (expression-based scores), shown once as
    the per-patient ground truth signal that the integration should preserve.
    """
    # Use first method with data as representative (all are identical)
    for m in methods_done:
        if results[m]["per_patient_r"] is not None:
            r_arr = results[m]["per_patient_r"]
            break
    else:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([r_arr], patch_artist=True, widths=0.4,
                    medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#4393C3")
    bp["boxes"][0].set_alpha(0.7)
    ax.axhline(0, ls="--", color="grey", alpha=0.5)
    ax.set_xticklabels(["Per-patient\nSpearman r"], fontsize=10)
    ax.set_ylabel("Spearman r (LIM_stem × immune)", fontsize=10)
    ax.set_title(f"Per-patient stem × immune signal — {dataset_name}\n"
                 f"(n={len(r_arr)} patients, invariant across integration methods)",
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{dataset_name}_per_patient_r.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def save_metrics_csv(results, dataset_name, methods_done):
    rows = []
    for m in methods_done:
        r_arr  = results[m]["per_patient_r"]
        bm_arr = results[m]["batch_mixing"]
        sc     = results[m]["stemness_coherence"]
        rows.append({
            "method"                    : m,
            "mean_batch_mixing"         : np.mean(bm_arr)   if bm_arr is not None else np.nan,
            "stemness_coherence_r"      : sc                if sc     is not None else np.nan,
            "median_per_patient_r"      : np.median(r_arr)  if r_arr  is not None else np.nan,
            "iqr_per_patient_r"         : (np.percentile(r_arr, 75) -
                                           np.percentile(r_arr, 25)) if r_arr is not None else np.nan,
            "n_patients"                : len(r_arr)        if r_arr  is not None else 0,
        })
    df  = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, f"{dataset_name}_metrics.csv")
    df.to_csv(out, index=False)
    print(f"  ✓ {out}")
    print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f" DATASET: {dataset_name}")
    print(f"{'='*70}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n  Loading {cfg['path']} …")
    adata = sc.read_h5ad(cfg["path"])
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  Patients: {adata.obs[cfg['batch_col']].nunique()}")

    set_log_x(adata, cfg)

    # ── 2. Score ──────────────────────────────────────────────────────────────
    score_signatures(adata)
    if not cfg["cc_in_obs"]:
        score_cell_cycle(adata)
    load_cytotrace(adata, cfg["cytotrace_csv"])

    stem_col   = f"score_{PRIMARY_STEM}"
    immune_col = "score_immune"

    # ── 3. HVG + PCA (shared preprocessing) ──────────────────────────────────
    compute_hvg_pca(adata, cfg["batch_col"])

    # ── 4. Integration loop ───────────────────────────────────────────────────
    results   = {}
    n_patients = adata.obs[cfg["batch_col"]].nunique()

    # Per-patient Spearman r is invariant across methods (scores from expression).
    # Compute once and attach to all methods as ground-truth anchor.
    pat_r_anchor = per_patient_corr(adata, stem_col, immune_col,
                                    cfg["batch_col"])
    print(f"\n  Per-patient r anchor (n={len(pat_r_anchor)} patients): "
          f"median={np.median(pat_r_anchor):.3f}")

    for method in METHODS:
        print(f"\n  [{method}]")
        try:
            if method == "Unintegrated":
                integrate_unintegrated(adata)
            elif method == "Harmony":
                integrate_harmony(adata, cfg["batch_col"])
            elif method == "scVI":
                integrate_scvi(adata, cfg)
            elif method == "DRVI":
                integrate_drvi(adata, cfg)

            conn_key = f"conn_{method}"

            # Batch mixing: fraction of k-NN from a different patient
            bm = batch_mixing_score(adata, cfg["batch_col"], conn_key)

            # Stemness coherence: does the embedding cluster high-stemness cells
            # together regardless of patient? Varies by method.
            sc_score = stemness_coherence(adata, stem_col, conn_key)

            results[method] = {
                "batch_mixing"      : bm,
                "stemness_coherence": sc_score,
                "per_patient_r"     : pat_r_anchor,  # same for all methods
            }
            print(f"    batch mixing       (mean): {bm.mean():.3f}")
            print(f"    stemness coherence (Pearson r): {sc_score:.3f}")

        except Exception as exc:
            print(f"    ⚠  {method} failed: {exc}")
            import traceback; traceback.print_exc()
            results[method] = {
                "batch_mixing"      : None,
                "stemness_coherence": None,
                "per_patient_r"     : None,
            }

    methods_done = [m for m in METHODS if m in results]

    # ── 5. Figures ────────────────────────────────────────────────────────────
    print(f"\n  Saving figures for {dataset_name} …")
    plot_umap_panel(adata, cfg["batch_col"], cfg["subtype_col"],
                    dataset_name, methods_done)
    plot_metrics_summary(results, dataset_name, methods_done, n_patients)
    plot_batch_mixing(results, dataset_name, methods_done, n_patients)
    plot_corr_preservation(results, dataset_name, methods_done)
    save_metrics_csv(results, dataset_name, methods_done)

    del adata; gc.collect()

print(f"""
{'='*70}
 DONE
{'='*70}
 Figures: {OUTDIR}/
   <dataset>_umap_panel.png         — 4-col UMAP comparison (main visual)
   <dataset>_metrics_summary.png    — batch mixing + stemness coherence bars
   <dataset>_batch_mixing.png       — inter-patient k-NN mixing (boxplot)
   <dataset>_per_patient_r.png      — per-patient r(stem × immune) ground truth
   <dataset>_metrics.csv            — quantitative summary table

 Key metrics:
   batch_mixing          — fraction inter-patient k-NN (higher = better)
   stemness_coherence_r  — Pearson r(own_stem, kNN_mean_stem) in embedding
                           (higher = biological gradient preserved)
   median_per_patient_r  — ground truth per-patient signal (invariant)

 Next: part1_stemness_vs_cellcycle.py
{'='*70}
""")
