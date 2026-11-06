"""
14_lcor_cxcl16_hexbin.py
=========================
Hexbin correlation of LCOR vs CXCL16 expression in malignant (epithelial)
cells — full cohort + per subtype (ER, HER2, TNBC).

Same hexbin style as 05_hexbin_correlation.py (gridsize=90, RdYlBu_r,
p0.5-p99.5 + 20% padding), but on raw log-normalized expression of the two
genes (not z-scored signature scores) — axes are in logcounts units, and
x=0 / y=0 dashed lines mark "gene not detected in this cell".

Both genes are sparsely detected (~14-15% of malignant cells; see console
output for per-object dropout stats), typical of scRNA-seq. Pearson r is
reported alongside Spearman rho, which is less sensitive to the zero
inflation.

Input:  compartment_epithelial.h5ad
Output: figures/14_LCOR_CXCL16_malignant/
    hexbin_malignant_LCOR_vs_CXCL16.png
    correlation_table.csv

Run from: ~/single_cell/Visvader/
  python3 14_lcor_cxcl16_hexbin.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
H5AD_FILE   = "compartment_epithelial.h5ad"
OUTDIR      = "figures/14_LCOR_CXCL16_malignant"
GRIDSIZE    = 90
SUBTYPE_COL = "subtype"
GENE_X, GENE_Y = "LCOR", "CXCL16"
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hexbin_panel(ax, x, y, title, xlabel, ylabel, gridsize=90,
                  restrict_to_detected=False):
    """
    Hexbin, same footprint as 05_hexbin_correlation.py but with a log color
    scale (bins='log') — needed here because LCOR/CXCL16 are both sparsely
    detected (~14% of cells), so the double-zero bin holds ~70%+ of all
    cells and swamps a linear color scale.

    If restrict_to_detected, only cells with x>0 AND y>0 are plotted
    (co-expressors), to look at magnitude correlation without the
    dropout-dominated double-zero mass.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if restrict_to_detected:
        det = (x > 0) & (y > 0)
        x, y = x[det], y[det]
    if len(x) < 20:
        ax.set_visible(False); return None

    xmin, xmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ymin, ymax = np.percentile(y, 0.5), np.percentile(y, 99.5)
    xpad = 0.20 * (xmax - xmin); ypad = 0.20 * (ymax - ymin)
    xl = (xmin - xpad, xmax + xpad)
    yl = (ymin - ypad, ymax + ypad)
    if xl[0] == xl[1]:
        xl = (xl[0] - 0.5, xl[1] + 0.5)
    if yl[0] == yl[1]:
        yl = (yl[0] - 0.5, yl[1] + 0.5)

    ax.set_facecolor("white")
    ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="RdYlBu_r",
                   mincnt=1, linewidths=0.02, bins="log",
                   extent=[xl[0], xl[1], yl[0], yl[1]], zorder=2)
    if not restrict_to_detected:
        ax.axhline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
        ax.axvline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")

    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    pstr  = f"p={p:.1e}"   if p   >= 1e-300 else "p<1e-300"
    pstr_s= f"p={p_s:.1e}" if p_s >= 1e-300 else "p<1e-300"
    ax.text(0.97, 0.03,
            f"Pearson r={r:.3f} ({pstr})\nSpearman ρ={rho:.3f} ({pstr_s})\nn={len(x):,}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    return hb, r, p, rho, p_s


def make_hexbin_figure(obs, x_col, y_col, x_label, y_label, fname,
                        suptitle, restrict_to_detected=False):
    """All cells + per subtype columns."""
    x_all = obs[x_col].values.astype(float)
    y_all = obs[y_col].values.astype(float)
    mask  = ~(np.isnan(x_all) | np.isnan(y_all))
    if restrict_to_detected:
        mask &= (x_all > 0) & (y_all > 0)

    subtypes_here = [st for st in ["ER", "HER2", "TNBC"]
                     if SUBTYPE_COL in obs.columns
                     and st in obs[SUBTYPE_COL].values]

    n_cols = 1 + len(subtypes_here)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5*n_cols, 5.5))
    axes = np.array(axes).flatten()
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    corr_rows = []

    out = hexbin_panel(axes[0], x_all, y_all,
                        f"All (n={mask.sum():,})", x_label, y_label,
                        gridsize=GRIDSIZE, restrict_to_detected=restrict_to_detected)
    if out:
        hb, r, p, rho, p_s = out
        plt.colorbar(hb, ax=axes[0], shrink=0.8, label="log10(cell count)")
        corr_rows.append({"Group": "All", "pearson_r": r, "pearson_p": p,
                           "spearman_rho": rho, "spearman_p": p_s,
                           "n": int(mask.sum())})

    for i, subtype in enumerate(subtypes_here):
        ax      = axes[i+1]
        st_mask = (obs[SUBTYPE_COL] == subtype).values
        n_st    = (st_mask & mask).sum()
        out = hexbin_panel(ax, x_all[st_mask], y_all[st_mask],
                            f"{subtype} (n={n_st:,})",
                            x_label, "", gridsize=GRIDSIZE,
                            restrict_to_detected=restrict_to_detected)
        if out:
            hb, r, p, rho, p_s = out
            plt.colorbar(hb, ax=ax, shrink=0.8, label="log10(cell count)")
            corr_rows.append({"Group": subtype, "pearson_r": r, "pearson_p": p,
                               "spearman_rho": rho, "spearman_p": p_s,
                               "n": int(n_st)})
        ax.set_title(f"{subtype} (n={n_st:,})",
                     fontsize=9, fontweight="bold",
                     color=SUBTYPE_COLORS.get(subtype, "black"))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return corr_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print(f" 14 -- {GENE_X} vs {GENE_Y} -- malignant (epithelial) cells")
print("="*65)

adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}")

missing = [g for g in (GENE_X, GENE_Y) if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

# Raw counts -> dropout stats (sanity check, same spirit as prior TIM3 scripts)
idx = [adata.var_names.get_loc(g) for g in (GENE_X, GENE_Y)]
counts = adata.layers["counts"][:, idx]
counts = np.asarray(counts.todense()) if issparse(counts) else np.asarray(counts)
for i, g in enumerate((GENE_X, GENE_Y)):
    v = counts[:, i]
    print(f"  {g}: detected in {100*(v>0).mean():.1f}% of cells "
          f"(max count={int(v.max())}, mean|expr>0={v[v>0].mean():.2f})")

# Co-detection contingency (neither / X-only / Y-only / both) — overall + per subtype
x_pos, y_pos = counts[:, 0] > 0, counts[:, 1] > 0
subtype_arr  = adata.obs[SUBTYPE_COL].astype(str).values
contingency_rows = []
for grp, m in [("All", np.ones(adata.n_obs, dtype=bool))] + \
              [(st, subtype_arr == st) for st in ["ER", "HER2", "TNBC"] if (subtype_arr == st).any()]:
    n = m.sum()
    row = {
        "Group": grp, "n": int(n),
        "neither_pct":        round(100 * (~x_pos[m] & ~y_pos[m]).sum() / n, 1),
        f"{GENE_X}_only_pct": round(100 * ( x_pos[m] & ~y_pos[m]).sum() / n, 1),
        f"{GENE_Y}_only_pct": round(100 * (~x_pos[m] &  y_pos[m]).sum() / n, 1),
        "both_pct":           round(100 * ( x_pos[m] &  y_pos[m]).sum() / n, 1),
        "both_n":             int((x_pos[m] & y_pos[m]).sum()),
    }
    contingency_rows.append(row)
df_contingency = pd.DataFrame(contingency_rows)
df_contingency.to_csv(f"{OUTDIR}/codetection_contingency.csv", index=False)
print("\n  Co-detection breakdown (% of cells):")
print(df_contingency.to_string(index=False))

# Log-normalized expression for plotting
if "logcounts" in adata.layers:
    expr = adata.layers["logcounts"][:, idx]
else:
    expr = adata.X[:, idx]
expr = np.asarray(expr.todense()) if issparse(expr) else np.asarray(expr)

obs = adata.obs.copy()
obs[f"{GENE_X}_logcounts"] = expr[:, 0]
obs[f"{GENE_Y}_logcounts"] = expr[:, 1]

corr_rows = make_hexbin_figure(
    obs, f"{GENE_X}_logcounts", f"{GENE_Y}_logcounts",
    f"{GENE_X} (logcounts)", f"{GENE_Y} (logcounts)",
    f"hexbin_malignant_{GENE_X}_vs_{GENE_Y}",
    suptitle=f"Malignant (epithelial) cells — {GENE_X} vs {GENE_Y} (all cells, log color scale)",
)
df_corr = pd.DataFrame(corr_rows)
df_corr["subset"] = "all_cells"

corr_rows_codet = make_hexbin_figure(
    obs, f"{GENE_X}_logcounts", f"{GENE_Y}_logcounts",
    f"{GENE_X} (logcounts)", f"{GENE_Y} (logcounts)",
    f"hexbin_malignant_{GENE_X}_vs_{GENE_Y}_codetected_only",
    suptitle=f"Malignant (epithelial) cells — {GENE_X} vs {GENE_Y} "
             f"(magnitude among co-detected cells only)",
    restrict_to_detected=True,
)
df_corr_codet = pd.DataFrame(corr_rows_codet)
df_corr_codet["subset"] = "codetected_only"

df_corr_all = pd.concat([df_corr, df_corr_codet], ignore_index=True)
df_corr_all.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)
print("\n  Correlations (all cells + codetected-only):")
print(df_corr_all.to_string(index=False))
print(f"\n  Saved: {OUTDIR}/hexbin_malignant_{GENE_X}_vs_{GENE_Y}.png")
print(f"  Saved: {OUTDIR}/hexbin_malignant_{GENE_X}_vs_{GENE_Y}_codetected_only.png")
print(f"  Saved: {OUTDIR}/codetection_contingency.csv")
print(f"  Saved: {OUTDIR}/correlation_table.csv")
