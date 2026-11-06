"""
12_landscent_stemconsensus_umap.py
===================================
Extends 11_landscent_cytotrace_umap.py with a third potency/stemness axis:
stem_consensus (Visvader/signatures_confirmed.json), scored fresh on
unintegrated log-normalised expression.

Outputs (figures/V11/):
  umap3_{subtype}_comparison.png  — 3 UMAP panels (CytoTRACE2 | LandSCENT SR |
                                    stem_consensus) + 3 pairwise correlation
                                    scatters, per subtype
"""

import os, json, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 8})

DATA_DIR = "/home/hbrunel/single_cell/Visvader"
FIG_DIR  = os.path.join(DATA_DIR, "figures", "V11")
os.makedirs(FIG_DIR, exist_ok=True)

SUBTYPES = ["ER", "HER2", "TNBC"]

with open(os.path.join(DATA_DIR, "signatures_confirmed.json")) as f:
    SIGS = json.load(f)
STEM_GENES = SIGS["stem_consensus"]

CMAPS = {"CT2": "RdYlBu_r", "SR": "viridis", "STEM": "PRGn"}


def percentile_norm(values, lo=1, hi=99):
    v = pd.Series(values).dropna()
    return np.percentile(v, lo), np.percentile(v, hi)


def plot_panel(ax, coords, values, cmap, vmin, vmax, title, cbar_label, s=1.5, alpha=0.6):
    mask = ~np.isnan(values)
    sc_p = ax.scatter(coords[mask, 0], coords[mask, 1], c=values[mask], cmap=cmap,
                       vmin=vmin, vmax=vmax, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.tick_params(labelsize=6)
    cbar = plt.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                         ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7); cbar.ax.tick_params(labelsize=6)


def plot_corr(ax, x, y, xlab, ylab):
    valid = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[valid], y[valid], s=1, alpha=0.15, c="steelblue", rasterized=True)
    r = np.corrcoef(x[valid], y[valid])[0, 1]
    ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(f"r = {r:.3f}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)


for subtype in SUBTYPES:
    h5ad = os.path.join(DATA_DIR, f"epithelial_{subtype}.h5ad")
    if not os.path.exists(h5ad):
        print(f"Missing {h5ad}, skipping"); continue

    print(f"=== {subtype} ===")
    ad = sc.read_h5ad(h5ad)
    ad.X = ad.layers["logcounts"].copy()
    ok = [g for g in STEM_GENES if g in ad.var_names]
    sc.tl.score_genes(ad, ok, score_name="stem_consensus", use_raw=False)

    ct = pd.read_csv(os.path.join(DATA_DIR, f"cytotrace2_scores_Epithelial_{subtype}.csv"), index_col=0)["CytoTRACE2_Score"]
    sr_df = pd.read_csv(os.path.join(DATA_DIR, f"landscent_scores_Epithelial_{subtype}.csv"))
    sr = sr_df.set_index("cell_id")["SR"]

    ad.obs["CT2"] = ct.reindex(ad.obs_names).values
    ad.obs["SR"]  = sr.reindex(ad.obs_names).values

    coords = ad.obsm["X_umap"]
    ct_v   = ad.obs["CT2"].values.astype(float)
    sr_v   = ad.obs["SR"].values.astype(float)
    st_v   = ad.obs["stem_consensus"].values.astype(float)

    ct_vmin, ct_vmax = percentile_norm(ct_v)
    sr_vmin, sr_vmax = percentile_norm(sr_v)
    st_vmin, st_vmax = percentile_norm(st_v)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_panel(axes[0,0], coords, ct_v, CMAPS["CT2"], ct_vmin, ct_vmax,
               f"CytoTRACE2 — {subtype}", "CytoTRACE2 score")
    plot_panel(axes[0,1], coords, sr_v, CMAPS["SR"], sr_vmin, sr_vmax,
               f"LandSCENT SR — {subtype}", "Signaling entropy rate")
    plot_panel(axes[0,2], coords, st_v, CMAPS["STEM"], st_vmin, st_vmax,
               f"stem_consensus — {subtype}", "stem_consensus score")

    plot_corr(axes[1,0], ct_v, sr_v, "CytoTRACE2", "LandSCENT SR")
    plot_corr(axes[1,1], ct_v, st_v, "CytoTRACE2", "stem_consensus")
    plot_corr(axes[1,2], sr_v, st_v, "LandSCENT SR", "stem_consensus")

    fig.suptitle(f"CytoTRACE2 / LandSCENT SR / stem_consensus — {subtype} epithelial",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"umap3_{subtype}_comparison.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    del ad
    import gc; gc.collect()

print("Done.")
