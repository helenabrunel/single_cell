"""
11_landscent_cytotrace_umap.py
==============================
Visualise and compare CytoTRACE2 and LandSCENT (SR) on UMAP.

Requires: cytotrace2_scores_Epithelial_*.csv  (script 04)
          landscent_scores_Epithelial_*.csv    (script 10_landscent_scoring.R)

Outputs (figures/V11/):
  umap_{subtype}_comparison.png      — side-by-side UMAP (CytoTRACE2 | LandSCENT)
  umap_{subtype}_per_patient.png     — per-patient grid (one row per patient)
  umap_global_comparison.png         — all subtypes coloured by CytoTRACE2 | SR | subtype
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 8})

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/hbrunel/single_cell/Visvader"
FIG_DIR  = os.path.join(DATA_DIR, "figures", "V11")
os.makedirs(FIG_DIR, exist_ok=True)

SUBTYPES = ["ER", "HER2", "TNBC"]

SUBTYPE_COLORS = {"ER": "#E8825B", "HER2": "#6BAED6", "TNBC": "#74C476"}
CMAP_CYTOTRACE = "RdYlBu_r"   # high potency = warm
CMAP_SR        = "viridis"    # high entropy = bright


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scores(data_dir, subtype):
    """Return (cytotrace_series, sr_series) indexed by cell barcode."""
    ct_path = os.path.join(data_dir, f"cytotrace2_scores_Epithelial_{subtype}.csv")
    sr_path = os.path.join(data_dir, f"landscent_scores_Epithelial_{subtype}.csv")

    ct = pd.read_csv(ct_path, index_col=0)["CytoTRACE2_Score"]

    if not os.path.exists(sr_path):
        print(f"  WARNING: {sr_path} not found — run 10_landscent_scoring.R first")
        return ct, None

    sr_df = pd.read_csv(sr_path)
    sr = sr_df.set_index("cell_id")["SR"]
    return ct, sr


def percentile_norm(values, lo=1, hi=99):
    """Return vmin, vmax at given percentiles (ignoring NaN)."""
    v = values.dropna()
    return np.percentile(v, lo), np.percentile(v, hi)


def add_scores_to_obs(adata, ct, sr):
    """Join CytoTRACE2 and SR into adata.obs (align by cell barcode)."""
    adata.obs["CytoTRACE2"] = ct.reindex(adata.obs_names).values
    if sr is not None:
        adata.obs["SR"] = sr.reindex(adata.obs_names).values


def plot_umap_panel(ax, coords, values, cmap, vmin, vmax, title, cbar_label,
                    s=1.5, alpha=0.6):
    """Scatter UMAP coloured by `values` on `ax`."""
    mask = ~np.isnan(values)
    sc_plot = ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=values[mask], cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=s, alpha=alpha, linewidths=0, rasterized=True
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP1", fontsize=7)
    ax.set_ylabel("UMAP2", fontsize=7)
    ax.tick_params(labelsize=6)
    cbar = plt.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=ax, fraction=0.046, pad=0.04
    )
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    return sc_plot


# ── Per-subtype UMAP comparison ───────────────────────────────────────────────

for subtype in SUBTYPES:
    h5ad = os.path.join(DATA_DIR, f"epithelial_{subtype}.h5ad")
    if not os.path.exists(h5ad):
        print(f"Missing {h5ad}, skipping {subtype}")
        continue

    print(f"\n=== {subtype} ===")
    adata = sc.read_h5ad(h5ad)
    ct, sr = load_scores(DATA_DIR, subtype)
    add_scores_to_obs(adata, ct, sr)

    coords = adata.obsm["X_umap"]
    ct_vals = adata.obs["CytoTRACE2"].values.astype(float)
    ct_vmin, ct_vmax = percentile_norm(pd.Series(ct_vals))

    # ── Figure 1: side-by-side comparison ─────────────────────────────────────
    n_cols = 3 if sr is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    plot_umap_panel(axes[0], coords, ct_vals,
                    CMAP_CYTOTRACE, ct_vmin, ct_vmax,
                    f"CytoTRACE2 — {subtype}", "CytoTRACE2 score")

    if sr is not None:
        sr_vals = adata.obs["SR"].values.astype(float)
        sr_vmin, sr_vmax = percentile_norm(pd.Series(sr_vals))
        plot_umap_panel(axes[1], coords, sr_vals,
                        CMAP_SR, sr_vmin, sr_vmax,
                        f"LandSCENT SR — {subtype}", "Signaling entropy rate")

        # Correlation scatter in panel 3
        valid = ~(np.isnan(ct_vals) | np.isnan(sr_vals))
        axes[2].scatter(ct_vals[valid], sr_vals[valid],
                        s=1, alpha=0.2, c="steelblue", rasterized=True)
        r = np.corrcoef(ct_vals[valid], sr_vals[valid])[0, 1]
        axes[2].set_xlabel("CytoTRACE2 score", fontsize=8)
        axes[2].set_ylabel("LandSCENT SR", fontsize=8)
        axes[2].set_title(f"Correlation (r = {r:.3f})", fontsize=9, fontweight="bold")
        axes[2].tick_params(labelsize=6)
    else:
        axes[1].axis("off")

    fig.suptitle(f"Potency metrics comparison — {subtype} epithelial",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"umap_{subtype}_comparison.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Figure 2: per-patient grid ─────────────────────────────────────────────
    samples = sorted(adata.obs["sample"].unique())
    n_samples = len(samples)
    n_cols_pp = 2 if sr is None else 2   # CytoTRACE2 | SR per row
    n_rows_pp = n_samples

    # Target ~2.5" per panel to keep figure readable
    fig2 = plt.figure(figsize=(5 * n_cols_pp, 2.5 * n_rows_pp))
    gs = gridspec.GridSpec(n_rows_pp, n_cols_pp, figure=fig2,
                           hspace=0.4, wspace=0.3)

    for row_idx, sample in enumerate(samples):
        mask = adata.obs["sample"] == sample
        coords_s = coords[mask]
        ct_s = ct_vals[mask]

        ax_ct = fig2.add_subplot(gs[row_idx, 0])
        plot_umap_panel(ax_ct, coords_s, ct_s,
                        CMAP_CYTOTRACE, ct_vmin, ct_vmax,
                        f"{sample}\nCytoTRACE2", "CT2",
                        s=3, alpha=0.7)

        if sr is not None:
            sr_s = sr_vals[mask]
            ax_sr = fig2.add_subplot(gs[row_idx, 1])
            plot_umap_panel(ax_sr, coords_s, sr_s,
                            CMAP_SR, sr_vmin, sr_vmax,
                            f"{sample}\nLandSCENT SR", "SR",
                            s=3, alpha=0.7)

    fig2.suptitle(f"Per-patient potency — {subtype} epithelial",
                  fontsize=12, fontweight="bold", y=1.002)
    out2 = os.path.join(FIG_DIR, f"umap_{subtype}_per_patient.png")
    fig2.savefig(out2, bbox_inches="tight", dpi=120)
    plt.close(fig2)
    print(f"  Saved: {out2}")

    del adata
    import gc; gc.collect()


# ── Global overview (all subtypes on one UMAP) ────────────────────────────────
print("\n=== Global ===")
global_h5ad = os.path.join(DATA_DIR, "compartment_epithelial.h5ad")
if os.path.exists(global_h5ad):
    adata_g = sc.read_h5ad(global_h5ad)

    # Load global CSVs
    ct_g_path = os.path.join(DATA_DIR, "cytotrace2_scores_Epithelial_Global.csv")
    sr_g_path = os.path.join(DATA_DIR, "landscent_scores_Epithelial_Global.csv")
    ct_g = pd.read_csv(ct_g_path, index_col=0)["CytoTRACE2_Score"] \
        if os.path.exists(ct_g_path) else None
    sr_g = None
    if os.path.exists(sr_g_path):
        df_g = pd.read_csv(sr_g_path)
        sr_g = df_g.set_index("cell_id")["SR"]

    if ct_g is not None:
        adata_g.obs["CytoTRACE2"] = ct_g.reindex(adata_g.obs_names).values
    if sr_g is not None:
        adata_g.obs["SR"] = sr_g.reindex(adata_g.obs_names).values

    coords_g = adata_g.obsm["X_umap"]
    subtype_labels = adata_g.obs["subtype"].values

    n_panels = 2 + (1 if ct_g is not None else 0) + (1 if sr_g is not None else 0)
    fig_g, axes_g = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Panel 0: subtype coloured
    for st, col in SUBTYPE_COLORS.items():
        m = subtype_labels == st
        axes_g[0].scatter(coords_g[m, 0], coords_g[m, 1],
                          c=col, label=st, s=1, alpha=0.4,
                          linewidths=0, rasterized=True)
    axes_g[0].set_title("Subtypes", fontsize=9, fontweight="bold")
    axes_g[0].legend(markerscale=5, fontsize=7, framealpha=0.7)
    axes_g[0].set_xlabel("UMAP1", fontsize=7); axes_g[0].set_ylabel("UMAP2", fontsize=7)
    axes_g[0].tick_params(labelsize=6)

    panel = 1
    if ct_g is not None:
        ct_gv = adata_g.obs["CytoTRACE2"].values.astype(float)
        vn, vx = percentile_norm(pd.Series(ct_gv))
        plot_umap_panel(axes_g[panel], coords_g, ct_gv,
                        CMAP_CYTOTRACE, vn, vx,
                        "CytoTRACE2 — Global", "CytoTRACE2 score")
        panel += 1

    if sr_g is not None:
        sr_gv = adata_g.obs["SR"].values.astype(float)
        vn, vx = percentile_norm(pd.Series(sr_gv))
        plot_umap_panel(axes_g[panel], coords_g, sr_gv,
                        CMAP_SR, vn, vx,
                        "LandSCENT SR — Global", "Signaling entropy rate")
        panel += 1

    # Hide unused panels
    for i in range(panel, n_panels):
        axes_g[i].axis("off")

    fig_g.suptitle("Potency metrics — All epithelial (Global)",
                   fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_g = os.path.join(FIG_DIR, "umap_global_comparison.png")
    fig_g.savefig(out_g, bbox_inches="tight", dpi=150)
    plt.close(fig_g)
    print(f"  Saved: {out_g}")
else:
    print(f"  {global_h5ad} not found — skipping global")

print(f"\nAll figures saved to {FIG_DIR}")
