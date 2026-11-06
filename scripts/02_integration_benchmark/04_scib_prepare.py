"""
part0_scib_prepare.py
======================
Prepares per-subtype (ER / HER2 / TNBC) malignant/epithelial objects with the
embeddings needed for a scib-metrics benchmark, and persists them to disk
(unlike part0_malignant_integration.py, which deleted its per-subtype temp
files after making figures).

Embeddings produced per object:
  X_pca_fresh  — Unintegrated (HVG + scale + PCA, batch-aware HVG selection)
  X_harmony    — Harmony correction on top of X_pca_fresh
  X_DRVI       — DRVI latent space, REUSING the cached models trained by
                 part0_malignant_integration.py (figures/part0/{Dataset}_{Subtype}_drvi_model)
                 — no retraining, just projection.

scVI is NOT included here: no per-subtype scVI model was ever trained
(only pooled, whole-compartment models exist), and training 5 fresh models
would add hours of CPU time. This benchmarks exactly the methods that were
actually used for per-subtype integration (Harmony + DRVI) plus the
Unintegrated baseline.

bio_label: argmax of FINE_EPI marker-gene scores (Luminal_A, Luminal_B, HER2,
Basal_TNBC, Luminal_prog, Cycling_epi, EMT), computed on unintegrated
log-normalised expression. This is the bio-conservation reference label for
scib-metrics — deliberately NOT derived from any embedding, so it doesn't
circularly reward/penalize the method that produced it. Caveat (flagged to
the user): in malignant cells this label may still partly track patient-
specific tumour state rather than a universal cell program.

batch_key = patient (sample / sampleid) — the nuisance variable the
integration methods are correcting for.

Output: figures/part0/scib_inputs/{Dataset}_{Subtype}.h5ad
Run from: ~/single_cell/   (base conda env — needs scanpy/harmonypy/drvi)
"""

import scanpy as sc
import pandas as pd
import numpy as np
import json, os, gc, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

MIN_CELLS    = 500
MIN_PATIENTS = 3
N_HVG        = 2000
N_PCS        = 30
N_NEIGH      = 15
RANDOM       = 42
SUBTYPES     = ["ER", "HER2", "TNBC"]

OUTDIR    = "figures/part0/scib_inputs"
MODEL_DIR = "figures/part0"   # where the cached DRVI models from part0_malignant_integration.py live
os.makedirs(OUTDIR, exist_ok=True)

FINE_EPI = {
    "Luminal_A"   : ["ESR1","PGR","FOXA1","GATA3","XBP1"],
    "Luminal_B"   : ["ESR1","MKI67","TOP2A","CCNB1"],
    "HER2"        : ["ERBB2","GRB7","PGAP3"],
    "Basal_TNBC"  : ["KRT5","KRT14","KRT17","TP63","EGFR"],
    "Luminal_prog": ["KRT8","KRT18","EPCAM","CLDN3","CLDN4"],
    "Cycling_epi" : ["MKI67","TOP2A","PCNA","MCM2","CCNB1"],
    "EMT"         : ["VIM","CDH2","FN1","SNAI1","SNAI2","TWIST1","ZEB1"],
}

DATASETS = {
    "Visvader": {
        "path"            : "Visvader/compartment_epithelial.h5ad",
        "compartment_col"  : None,
        "compartment_value": None,
        "batch_col"       : "sample",
        "subtype_col"     : "subtype",
        "count_layer"     : "counts",
        "log_layer"       : "logcounts",
    },
    "Klughammer": {
        "path"            : "Klughammer/klughammer_prepared.h5ad",
        "compartment_col"  : "compartments",
        "compartment_value": "Malignant",
        "batch_col"       : "sampleid",
        "subtype_col"     : "subtype",
        "count_layer"     : None,
        "log_layer"       : None,
    },
}


def set_log_x(adata, cfg):
    if cfg["log_layer"] and cfg["log_layer"] in adata.layers:
        adata.X = adata.layers[cfg["log_layer"]].copy()


def filter_patients(adata, batch_col):
    counts  = adata.obs[batch_col].value_counts()
    keep    = counts[counts >= MIN_CELLS].index.tolist()
    removed = len(counts) - len(keep)
    adata_f = adata[adata.obs[batch_col].isin(keep)].copy()
    print(f"  QC: kept {len(keep)} patients / removed {removed} (<{MIN_CELLS} cells)")
    return adata_f


def compute_bio_label(adata):
    """Argmax of FINE_EPI marker scores on unintegrated expression -> categorical label."""
    cols = []
    for ct, genes in FINE_EPI.items():
        ok = [g for g in genes if g in adata.var_names]
        if len(ok) >= 2:
            sc.tl.score_genes(adata, ok, score_name=f"fine_{ct}", use_raw=False)
            cols.append(f"fine_{ct}")
    scores = adata.obs[cols].values
    best   = scores.argmax(axis=1)
    adata.obs["bio_label"] = pd.Categorical(
        [cols[i].replace("fine_", "") for i in best]
    )
    print("  bio_label distribution:")
    print(adata.obs["bio_label"].value_counts().to_string())


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
    print(f"    HVGs: {adata.var['highly_variable'].sum()}")
    del ad_hvg; gc.collect()


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
    print("    ✓ Harmony")


def project_drvi(adata, cfg, model_dir):
    """Load a cached DRVI model (trained by part0_malignant_integration.py) and
    project ALL cells of this subtype into its latent space. No retraining."""
    from drvi.model import DRVI
    hvg_file = model_dir + "_hvg_genes.txt"
    if not (os.path.exists(model_dir) and os.path.exists(hvg_file)):
        print(f"    ⚠  No cached DRVI model at {model_dir} — skipping DRVI for this object")
        return False

    with open(hvg_file) as f:
        saved_genes = [l.strip() for l in f if l.strip()]
    present = [g for g in saved_genes if g in adata.var_names]
    print(f"    Loading cached DRVI model ({len(present)}/{len(saved_genes)} genes): {model_dir}")
    ad_hvg = adata[:, present].copy()

    model = DRVI.load(model_dir, adata=ad_hvg)
    has_counts = cfg["count_layer"] is not None
    if has_counts:
        DRVI.setup_anndata(ad_hvg, layer=cfg["count_layer"],
                           batch_key=cfg["batch_col"], is_count_data=True)
    else:
        DRVI.setup_anndata(ad_hvg, batch_key=cfg["batch_col"], is_count_data=False)

    adata.obsm["X_DRVI"] = model.get_latent_representation(ad_hvg)
    del ad_hvg, model; gc.collect()
    print("    ✓ DRVI (projected from cached model)")
    return True


# ══════════════════════════════════════════════════════════════════════════
for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*70}\n {dataset_name}\n{'='*70}")

    print(f"  Loading {cfg['path']} ...")
    adata_full = sc.read_h5ad(cfg["path"])
    print(f"  {adata_full.n_obs:,} cells x {adata_full.n_vars:,} genes")

    if cfg["compartment_col"]:
        adata_full = adata_full[
            adata_full.obs[cfg["compartment_col"]] == cfg["compartment_value"]
        ].copy()
        print(f"  After compartment filter: {adata_full.n_obs:,} cells")

    set_log_x(adata_full, cfg)
    adata_full = filter_patients(adata_full, cfg["batch_col"])

    for subtype in SUBTYPES:
        outfile = os.path.join(OUTDIR, f"{dataset_name}_{subtype}.h5ad")
        print(f"\n--- {dataset_name} · {subtype} ---")

        if os.path.exists(outfile):
            print(f"  {outfile} exists — skipping")
            continue

        ad_st = adata_full[adata_full.obs[cfg["subtype_col"]] == subtype].copy()
        n_pats = ad_st.obs[cfg["batch_col"]].nunique()
        print(f"  {ad_st.n_obs:,} cells | {n_pats} patients")

        if n_pats < MIN_PATIENTS:
            print(f"  ⚠  Skipping: only {n_pats} patients (min = {MIN_PATIENTS})")
            del ad_st; gc.collect()
            continue

        set_log_x(ad_st, cfg)
        compute_bio_label(ad_st)
        compute_hvg_pca(ad_st, cfg["batch_col"])
        integrate_harmony(ad_st, cfg["batch_col"])

        model_dir = os.path.join(MODEL_DIR, f"{dataset_name}_{subtype}_drvi_model")
        project_drvi(ad_st, cfg, model_dir)

        # keep the object lean: drop expression matrix, keep only what scib-metrics needs
        keep_obs = [cfg["batch_col"], cfg["subtype_col"], "bio_label"]
        ad_slim = sc.AnnData(
            X=np.zeros((ad_st.n_obs, 1), dtype=np.float32),
            obs=ad_st.obs[keep_obs].copy(),
            obsm={k: v for k, v in ad_st.obsm.items()
                  if k in ("X_pca_fresh", "X_harmony", "X_DRVI")},
        )
        ad_slim.write_h5ad(outfile)
        print(f"  ✓ {outfile}  (embeddings: {list(ad_slim.obsm.keys())})")

        del ad_st, ad_slim; gc.collect()

    del adata_full; gc.collect()

print("\nDONE — outputs in figures/part0/scib_inputs/")
