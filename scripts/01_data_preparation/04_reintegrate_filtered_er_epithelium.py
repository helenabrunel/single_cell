"""
23_reintegrate_ER_filtered.py
=============================
Re-integrates the ER epithelial object after excluding 11 samples, with the same
pipeline and parameters as 03_split_integrate.py (Epithelial):
  renormalise from counts → 2000 HVGs (seurat, batch_key=sample) → scale →
  PCA (30) → Harmony (sample, theta=4) → neighbors (15) → UMAP → Leiden (0.4)
  → fine epithelial scores.

Excluded (user request; "ER-AF0319" in the request = ER-AH0319):
  ER-AH0319, ER-PM0360, ER-MH0029-7C, ER-MH0029-9C, ER-MH0056-T,
  ER-MH0032, ER-MH0064-T, ER-MH0167-T, ER-MH0173-T,
  mER-PM0178, mER-MH0068-T (metastatic)

Input:  epithelial_ER.h5ad
Output: epithelial_ER_filtered.h5ad
        figures/22_axis_signatures_ER_filtered/integration_ER_filtered.png
"""

import os, re, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.n_jobs = 4

DATA_DIR   = "/home/hbrunel/single_cell/Visvader"
INFILE     = os.path.join(DATA_DIR, "epithelial_ER.h5ad")
OUTFILE    = os.path.join(DATA_DIR, "epithelial_ER_filtered.h5ad")
FIG_DIR    = os.path.join(DATA_DIR, "figures", "22_axis_signatures_ER_filtered")
os.makedirs(FIG_DIR, exist_ok=True)
SAMPLE_COL = "sample"

EXCLUDE = ["ER-AH0319", "ER-PM0360", "ER-MH0029-7C", "ER-MH0029-9C", "ER-MH0056-T",
           "ER-MH0032", "ER-MH0064-T", "ER-MH0167-T", "ER-MH0173-T",
           "mER-PM0178", "mER-MH0068-T"]

# Same as 03_split_integrate.py, Epithelial
N_HVG, N_PCS, RES = 2000, 30, 0.4
# Harmony diversity penalty. Default (2) barely mixed patients here (median
# iLISI 1.23 of 9); theta=4 → 2.73, higher values add little (6 → 2.87, 8 → 2.94).
THETA = 4
FINE_EPI = {
    "Luminal_A"   : ["ESR1","PGR","FOXA1","GATA3","XBP1"],
    "Luminal_B"   : ["ESR1","MKI67","TOP2A","CCNB1"],
    "HER2"        : ["ERBB2","GRB7","PGAP3"],
    "Basal_TNBC"  : ["KRT5","KRT14","KRT17","TP63","EGFR"],
    "Luminal_prog": ["KRT8","KRT18","EPCAM","CLDN3","CLDN4"],
    "Cycling_epi" : ["MKI67","TOP2A","PCNA","MCM2","CCNB1"],
    "EMT"         : ["VIM","CDH2","FN1","SNAI1","SNAI2","TWIST1","ZEB1"],
}

ad = sc.read_h5ad(INFILE)
short = ad.obs[SAMPLE_COL].astype(str).str.replace(r"^GSM\d+_", "", regex=True)
missing = [s for s in EXCLUDE if s not in set(short)]
assert not missing, f"Samples not found: {missing}"
keep = ~short.isin(EXCLUDE)
ad = ad[keep.values].copy()
ad.obs[SAMPLE_COL] = ad.obs[SAMPLE_COL].astype(str).astype("category")
n_samples = ad.obs[SAMPLE_COL].nunique()
print(f"Kept {ad.n_obs:,} cells | {n_samples} samples")
print(ad.obs[SAMPLE_COL].value_counts().sort_index().to_string())

# Drop results of the previous integration
for k in ["X_pca", "X_pca_harmony", "X_umap"]:
    ad.obsm.pop(k, None)
for k in list(ad.uns):
    ad.uns.pop(k)
ad.obsp.clear() if hasattr(ad.obsp, "clear") else None
ad.obs = ad.obs.drop(columns=[c for c in ad.obs.columns
                              if c == "leiden" or c.startswith("score_") and
                              c.replace("score_", "") in FINE_EPI])

# Re-normalise from raw counts
ad.X = ad.layers["counts"].copy()
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
ad.layers["logcounts"] = ad.X.copy()

sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat",
                            batch_key=SAMPLE_COL, subset=False)
print(f"HVGs: {ad.var['highly_variable'].sum():,}")

ad_hvg = ad[:, ad.var["highly_variable"]].copy()
sc.pp.scale(ad_hvg, max_value=10)
sc.tl.pca(ad_hvg, svd_solver="arpack", n_comps=N_PCS)
ad.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
ad.uns["pca"] = ad_hvg.uns["pca"]
del ad_hvg; gc.collect()

ho = hm.run_harmony(ad.obsm["X_pca"], ad.obs[[SAMPLE_COL]], SAMPLE_COL,
                    theta=THETA, max_iter_harmony=20, random_state=42, verbose=False)
Z = np.asarray(ho.Z_corr)
if Z.shape[0] == N_PCS and Z.shape[1] == ad.n_obs:
    Z = Z.T
ad.obsm["X_pca_harmony"] = Z
print(f"Harmony done ({n_samples} samples, theta={THETA})")

sc.pp.neighbors(ad, use_rep="X_pca_harmony", n_pcs=N_PCS, n_neighbors=15)
sc.tl.umap(ad)
sc.tl.leiden(ad, resolution=RES, key_added="leiden")
print(f"UMAP | Leiden: {ad.obs['leiden'].nunique()} clusters")

ad.X = ad.layers["logcounts"].copy()
for ct, genes in FINE_EPI.items():
    ok = [g for g in genes if g in ad.var_names]
    if len(ok) >= 2:
        sc.tl.score_genes(ad, ok, score_name=f"score_{ct}", use_raw=False)

ad.write_h5ad(OUTFILE)
print(f"Saved {OUTFILE}")

# Quick integration check: sample + Leiden
coords = ad.obsm["X_umap"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
cats = sorted(ad.obs[SAMPLE_COL].unique())
cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
col = {p: cmap(i % cmap.N) for i, p in enumerate(cats)}
order = np.random.default_rng(0).permutation(ad.n_obs)
samp = ad.obs[SAMPLE_COL].astype(str).values
axes[0].scatter(coords[order, 0], coords[order, 1], c=[col[s] for s in samp[order]],
                s=2, alpha=0.7, linewidths=0, rasterized=True)
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=col[p], markersize=6,
                               label=re.sub(r"^GSM\d+_", "", p)) for p in cats],
               fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
axes[0].set_title("Sample", fontweight="bold")
leid = ad.obs["leiden"].astype(int).values
axes[1].scatter(coords[:, 0], coords[:, 1], c=leid, cmap="tab20", s=2, alpha=0.7,
                linewidths=0, rasterized=True)
for c in np.unique(leid):
    x, y = np.median(coords[leid == c], axis=0)
    axes[1].text(x, y, str(c), fontsize=9, fontweight="bold", ha="center", va="center")
axes[1].set_title(f"Leiden (res {RES})", fontweight="bold")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
fig.suptitle(f"ER epithelial — filtered ({n_samples} samples, n={ad.n_obs:,}), "
             f"Harmony theta={THETA}",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "integration_ER_filtered.png"), bbox_inches="tight", dpi=150)
print("Done.")
