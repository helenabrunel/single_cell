"""
part0_malignant_integration.py
================================
Per-subtype integration of malignant / epithelial tumour cells.

For each dataset and each subtype (ER / HER2 / TNBC), cells are subset
and integrated independently so that batch correction operates within a
biologically homogeneous group.

Methods
-------
  Visvader  : Harmony  +  DRVI  (raw counts available → NB likelihood)
  Klughammer: DRVI only          (log-normalised only → normal likelihood)

UMAP panel per subtype: rows = methods,  5 columns:
  Patient  |  Subtype  |  Stemness (LIM_Stem)  |  Immune (consensus)  |  CytoTRACE2

Minimum patients to attempt integration: MIN_PATIENTS (default 3).
Klughammer HER2 (n=2) is automatically skipped.

Outputs  →  figures/part0/
  {Dataset}_{Subtype}_umap_panel.png
  {Dataset}_{Subtype}_drvi_model/     (cached DRVI models)

Run from: ~/single_cell/
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, gc, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

# ── configuration ──────────────────────────────────────────────────────────────
MIN_CELLS    = 500        # per-patient cell threshold (applied to full dataset)
MIN_PATIENTS = 3          # minimum patients to attempt integration per subtype
N_HVG        = 2000
N_PCS        = 30
N_NEIGH      = 15
RANDOM       = 42
N_LATENT     = 20
SUBSAMPLE    = 20_000
BATCH_SIZE   = 1024
MAX_EPOCHS   = 100
SUBTYPES     = ["ER", "HER2", "TNBC"]

OUTDIR = "figures/part0"
os.makedirs(OUTDIR, exist_ok=True)

# ── signatures ─────────────────────────────────────────────────────────────────
with open("Visvader/signatures_confirmed.json") as f:
    ALL_SIGS = json.load(f)
STEM_SIGS = list(ALL_SIGS.keys())

IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
]
PRIMARY_STEM = "LIM_MAMMARY_STEM_CELL_UP"

# ── colours ────────────────────────────────────────────────────────────────────
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
METHOD_COLORS  = {"Harmony": "#4393C3", "DRVI": "#762A83"}

# ── dataset definitions ────────────────────────────────────────────────────────
DATASETS = {
    "Visvader": {
        "path"         : "Visvader/compartment_epithelial.h5ad",
        "compartment_col"  : None,
        "compartment_value": None,
        "batch_col"    : "sample",
        "subtype_col"  : "subtype",
        "count_layer"  : "counts",
        "log_layer"    : "logcounts",
        "cytotrace_csv": "Visvader/cytotrace2_scores_Epithelial_Global.csv",
        "methods"      : ["Harmony", "DRVI"],
    },
    "Klughammer": {
        "path"         : "Klughammer/klughammer_prepared.h5ad",
        "compartment_col"  : "compartments",
        "compartment_value": "Malignant",
        "batch_col"    : "sampleid",
        "subtype_col"  : "subtype",
        "count_layer"  : None,
        "log_layer"    : None,
        "cytotrace_csv": "Klughammer/cytotrace2_scores_klughammer_Malignant_Global.csv",
        "methods"      : ["Harmony", "DRVI"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — data preparation
# ══════════════════════════════════════════════════════════════════════════════

def set_log_x(adata, cfg):
    if cfg["log_layer"] and cfg["log_layer"] in adata.layers:
        adata.X = adata.layers[cfg["log_layer"]].copy()


def filter_patients(adata, cfg, dataset_name):
    batch_col = cfg["batch_col"]
    counts    = adata.obs[batch_col].value_counts()
    keep      = counts[counts >= MIN_CELLS].index.tolist()
    removed   = len(counts) - len(keep)
    adata_f   = adata[adata.obs[batch_col].isin(keep)].copy()
    print(f"  QC: kept {len(keep)} patients / removed {removed} (<{MIN_CELLS} cells)")
    print(f"  Cells after filter: {adata_f.n_obs:,}")
    return adata_f


def score_signatures(adata):
    print("  Scoring stemness signatures ...")
    for sig_key in STEM_SIGS:
        genes = [g for g in ALL_SIGS[sig_key] if g in adata.var_names]
        if len(genes) >= 5:
            sc.tl.score_genes(adata, genes, score_name=f"score_{sig_key}", use_raw=False)
    imm_ok = [g for g in IMMUNOGENIC if g in adata.var_names]
    sc.tl.score_genes(adata, imm_ok, score_name="score_immune", use_raw=False)
    print(f"    immune: {len(imm_ok)}/{len(IMMUNOGENIC)} genes")


def load_cytotrace(adata, csv_path):
    ct  = pd.read_csv(csv_path, index_col=0)[["CytoTRACE2_Score"]]
    adata.obs["CytoTRACE2"] = ct.reindex(adata.obs_names)["CytoTRACE2_Score"]
    n_miss = adata.obs["CytoTRACE2"].isna().sum()
    if n_miss:
        med = adata.obs["CytoTRACE2"].median()
        adata.obs["CytoTRACE2"].fillna(med, inplace=True)
        print(f"    CytoTRACE2: {n_miss} missing values filled with median {med:.3f}")
    else:
        print(f"    CytoTRACE2: all {adata.n_obs:,} cells matched")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — integration
# ══════════════════════════════════════════════════════════════════════════════

def compute_hvg_pca(adata, batch_col):
    n_batches = adata.obs[batch_col].nunique()
    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, flavor="seurat",
        batch_key=batch_col if n_batches > 1 else None,
        subset=False,
    )
    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(ad_hvg, max_value=10)
    sc.tl.pca(ad_hvg, svd_solver="arpack", n_comps=N_PCS, random_state=RANDOM)
    adata.obsm["X_pca_fresh"] = ad_hvg.obsm["X_pca"].copy()
    del ad_hvg; gc.collect()
    print(f"    HVGs: {adata.var['highly_variable'].sum()}")


def integrate_harmony(adata, batch_col):
    import harmonypy as hm
    ho = hm.run_harmony(
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
    print("    ✓ Harmony")


def integrate_drvi(adata, cfg, model_dir):
    from drvi.model import DRVI
    has_counts   = cfg["count_layer"] is not None
    batch_col    = cfg["batch_col"]
    hvg_file     = model_dir + "_hvg_genes.txt"

    if os.path.exists(model_dir) and os.path.exists(hvg_file):
        # Reload cached model using the same HVG genes as training
        with open(hvg_file) as f:
            saved_genes = [l.strip() for l in f.readlines() if l.strip()]
        present = [g for g in saved_genes if g in adata.var_names]
        ad_hvg  = adata[:, present].copy()
        print(f"    Loading cached DRVI model ({len(present)} genes): {model_dir}")
        model = DRVI.load(model_dir, adata=ad_hvg)
    else:
        hvg_mask = adata.var["highly_variable"].values
        ad_hvg   = adata[:, hvg_mask].copy()
        os.makedirs(model_dir, exist_ok=True)
        with open(hvg_file, "w") as f:
            f.write("\n".join(ad_hvg.var_names.tolist()))

        n_train  = min(SUBSAMPLE, ad_hvg.n_obs)
        idx      = np.random.default_rng(RANDOM).choice(ad_hvg.n_obs, n_train, replace=False)
        ad_train = ad_hvg[idx].copy()
        print(f"    DRVI: training on {n_train:,} / {ad_hvg.n_obs:,} cells ...")
        if has_counts:
            DRVI.setup_anndata(ad_train, layer=cfg["count_layer"],
                               batch_key=batch_col, is_count_data=True)
        else:
            DRVI.setup_anndata(ad_train, batch_key=batch_col, is_count_data=False)
        model = DRVI(ad_train, n_latent=N_LATENT * 2)
        model.train(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                    early_stopping=True, early_stopping_patience=10,
                    check_val_every_n_epoch=5, progress_bar_refresh_rate=0)
        model.save(model_dir, overwrite=True)
        del ad_train; gc.collect()

    if has_counts:
        DRVI.setup_anndata(ad_hvg, layer=cfg["count_layer"],
                           batch_key=batch_col, is_count_data=True)
    else:
        DRVI.setup_anndata(ad_hvg, batch_key=batch_col, is_count_data=False)
    adata.obsm["X_DRVI"] = model.get_latent_representation(ad_hvg)
    sc.pp.neighbors(adata, use_rep="X_DRVI", n_neighbors=N_NEIGH, random_state=RANDOM)
    sc.tl.umap(adata, random_state=RANDOM)
    adata.obsm["X_umap_DRVI"] = adata.obsm["X_umap"].copy()
    del ad_hvg, model; gc.collect()
    print("    ✓ DRVI")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — plotting
# ══════════════════════════════════════════════════════════════════════════════

def _scatter(ax, xy, values, title, cmap="RdYlBu_r", s=0.8, alpha=0.5):
    vals = np.asarray(values)
    if vals.dtype.kind in ("U", "O", "S"):
        cats  = pd.Categorical(vals)
        n     = len(cats.categories)
        cmap_ = plt.cm.get_cmap("tab20", n)
        ax.scatter(xy[:, 0], xy[:, 1], c=cats.codes, cmap=cmap_,
                   s=s, alpha=alpha, rasterized=True)
        handles = [mpatches.Patch(color=cmap_(i / max(n - 1, 1)), label=str(c))
                   for i, c in enumerate(cats.categories)]
        ax.legend(handles=handles, fontsize=5, ncol=1,
                  loc="upper right", framealpha=0.5, markerscale=3)
    else:
        fv   = vals.astype(float)
        vmin = np.nanpercentile(fv, 2)
        vmax = np.nanpercentile(fv, 98)
        sc_p = ax.scatter(xy[:, 0], xy[:, 1], c=fv, cmap=cmap,
                          vmin=vmin, vmax=vmax, s=s, alpha=alpha, rasterized=True)
        plt.colorbar(sc_p, ax=ax, shrink=0.55, pad=0.01)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def plot_panel(adata, cfg, dataset_name, subtype, methods_done):
    """
    Grid: len(methods_done) rows  ×  4 columns
    Cols: Patient | Stemness (LIM_Stem) | Immune (consensus) | CytoTRACE2
    """
    batch_col  = cfg["batch_col"]
    stem_col   = f"score_{PRIMARY_STEM}"
    n_rows     = len(methods_done)
    col_titles = ["Patient", "Stemness\n(LIM_Stem)", "Immune\n(consensus)", "CytoTRACE2"]
    n_cols     = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"{dataset_name}  —  {subtype}  |  per-subtype integration\n"
        f"n = {adata.n_obs:,} cells  |  "
        f"{adata.obs[batch_col].nunique()} patients",
        fontsize=11, fontweight="bold")

    for r, method in enumerate(methods_done):
        umap_key = f"X_umap_{method}"
        axes[r, 0].set_ylabel(method, fontsize=10, fontweight="bold",
                               rotation=90, labelpad=8)
        if umap_key not in adata.obsm:
            for c in range(n_cols):
                axes[r, c].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[r, c].transAxes)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            continue

        xy = adata.obsm[umap_key]

        # Patient — coloured by patient, no legend
        cats  = pd.Categorical(adata.obs[batch_col].values)
        n_pat = len(cats.categories)
        cmap_ = plt.cm.get_cmap("tab20", n_pat)
        axes[r, 0].scatter(xy[:, 0], xy[:, 1],
                           c=cats.codes, cmap=cmap_,
                           s=0.8, alpha=0.5, rasterized=True)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        # Stemness
        if stem_col in adata.obs.columns:
            _scatter(axes[r, 1], xy, adata.obs[stem_col].values,
                     "Stemness (LIM_Stem)", cmap="RdYlBu_r")

        # Immune
        if "score_immune" in adata.obs.columns:
            _scatter(axes[r, 2], xy, adata.obs["score_immune"].values,
                     "Immune (consensus)", cmap="PuRd")

        # CytoTRACE2
        if "CytoTRACE2" in adata.obs.columns:
            _scatter(axes[r, 3], xy, adata.obs["CytoTRACE2"].values,
                     "CytoTRACE2", cmap="YlOrRd_r")

    # column headers (top row only)
    for c, ttl in enumerate(col_titles):
        axes[0, c].set_title(ttl, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{dataset_name}_{subtype}_umap_panel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  two phases: all Harmony first, then all DRVI
#
# Phase 1  (fast): Harmony for every dataset × subtype that supports it.
#                  Saves a temp h5ad per subtype so HVG/PCA/embeddings are
#                  preserved for Phase 2 without reloading the full dataset.
#                  Produces an immediate Harmony-only panel figure.
# Phase 2  (slow): Loads each temp h5ad, runs DRVI, regenerates the combined
#                  panel (Harmony + DRVI rows where applicable).
#                  Deletes the temp file once the figure is written.
# ══════════════════════════════════════════════════════════════════════════════

TMPDIR = os.path.join(OUTDIR, "_tmp_subtype")
os.makedirs(TMPDIR, exist_ok=True)

def _tmp_path(dataset_name, subtype):
    return os.path.join(TMPDIR, f"{dataset_name}_{subtype}.h5ad")


# ── PHASE 1: Harmony ──────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print(" PHASE 1 — Harmony (all datasets × subtypes)")
print("█" * 70)

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f" {dataset_name}  [Phase 1 — Harmony]")
    print(f"{'='*70}")

    print(f"\n  Loading {cfg['path']} ...")
    adata_full = sc.read_h5ad(cfg["path"])
    print(f"  {adata_full.n_obs:,} cells × {adata_full.n_vars:,} genes")

    if cfg["compartment_col"]:
        adata_full = adata_full[
            adata_full.obs[cfg["compartment_col"]] == cfg["compartment_value"]
        ].copy()
        print(f"  After compartment filter: {adata_full.n_obs:,} cells")

    set_log_x(adata_full, cfg)
    adata_full = filter_patients(adata_full, cfg, dataset_name)

    print("\n  Scoring ...")
    set_log_x(adata_full, cfg)
    score_signatures(adata_full)
    load_cytotrace(adata_full, cfg["cytotrace_csv"])

    batch_col   = cfg["batch_col"]
    subtype_col = cfg["subtype_col"]
    has_harmony = "Harmony" in cfg["methods"]

    for subtype in SUBTYPES:
        tmp = _tmp_path(dataset_name, subtype)
        print(f"\n--- {dataset_name}  ·  {subtype} ---")

        adata_st = adata_full[adata_full.obs[subtype_col] == subtype].copy()
        n_pats   = adata_st.obs[batch_col].nunique()
        print(f"  {adata_st.n_obs:,} cells | {n_pats} patients")

        if n_pats < MIN_PATIENTS:
            print(f"  ⚠  Skipping: only {n_pats} patients (min = {MIN_PATIENTS})")
            del adata_st; gc.collect()
            continue

        set_log_x(adata_st, cfg)
        compute_hvg_pca(adata_st, batch_col)

        if has_harmony:
            print("\n  [Harmony]")
            try:
                integrate_harmony(adata_st, batch_col)
                plot_panel(adata_st, cfg, dataset_name, subtype, ["Harmony"])
            except Exception as e:
                print(f"  ⚠  Harmony failed: {e}")

        print(f"  Saving temp: {tmp}")
        adata_st.write_h5ad(tmp)
        del adata_st; gc.collect()

    del adata_full; gc.collect()


# ── PHASE 2: DRVI ────────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print(" PHASE 2 — DRVI (all datasets × subtypes)")
print("█" * 70)

for dataset_name, cfg in DATASETS.items():
    if "DRVI" not in cfg["methods"]:
        continue

    print(f"\n{'='*70}")
    print(f" {dataset_name}  [Phase 2 — DRVI]")
    print(f"{'='*70}")

    batch_col = cfg["batch_col"]

    for subtype in SUBTYPES:
        tmp = _tmp_path(dataset_name, subtype)
        if not os.path.exists(tmp):
            print(f"\n--- {dataset_name}  ·  {subtype}: no temp file, skipped ---")
            continue

        print(f"\n--- {dataset_name}  ·  {subtype} ---")
        adata_st = sc.read_h5ad(tmp)
        print(f"  {adata_st.n_obs:,} cells | {adata_st.obs[batch_col].nunique()} patients")

        print("\n  [DRVI]")
        try:
            mdir = os.path.join(OUTDIR, f"{dataset_name}_{subtype}_drvi_model")
            integrate_drvi(adata_st, cfg, mdir)
        except Exception as e:
            print(f"  ⚠  DRVI failed: {e}")

        methods_done = [m for m in cfg["methods"]
                        if f"X_umap_{m}" in adata_st.obsm]
        if methods_done:
            plot_panel(adata_st, cfg, dataset_name, subtype, methods_done)

        os.remove(tmp)
        del adata_st; gc.collect()

# Clean up temp dir if empty
try:
    os.rmdir(TMPDIR)
except OSError:
    pass

print(f"\n{'='*70}")
print(" DONE")
print(f"{'='*70}")
