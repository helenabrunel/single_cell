"""
03_depth_artifact_diagnostic_hexbin.py
======================================
Per-patient hexbin of LCOR vs CXCL16 (codetected cells only, "sin ceros"),
grouped by subtype -- one figure per subtype (ER / HER2 / TNBC), one panel
per patient within it.

WHY THIS SCRIPT EXISTS
-----------------------
14_lcor_cxcl16_hexbin.py showed a clear positive trend in the
codetected-only version, but visual inspection within a single subtype
suggested several distinct linear trends rather than one. This script
breaks that down to the patient level (same hexbin style, same
restrict_to_detected=True logic as 14) so each patient's own "line" can be
seen directly, instead of everyone pooled into one subtype panel.

Also produces a DEPTH-CORRECTED version of the same hexbins: for each
patient/gene, a constant per-patient Poisson rate vs total_counts gives a
depth-expected count, and the Pearson residual (raw - expected)/sqrt(expected)
is plotted instead of raw logcounts -- see README.txt for the method and
its caveats vs the pooled NB model in 15_lcor_cxcl16_depth_patient.py.

Input:  compartment_epithelial.h5ad
Output: figures/16_LCOR_CXCL16_by_patient/
    hexbin_by_patient_<SUBTYPE>.png                 (raw logcounts)
    hexbin_by_patient_<SUBTYPE>_depth_corrected.png  (depth-adjusted residuals)
    persample_correlation.csv   (r_raw and r_depth_corrected per patient)
    diagnostic_within_patient_lines.png
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/02_correlacion/scripts/03_depth_artifact_diagnostic_hexbin.py
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

H5AD_FILE   = "compartment_epithelial.h5ad"
OUTDIR      = "LCOR-CXCL16/02_correlacion/figures/03_depth_artifact_diagnostic_hexbin"
GRIDSIZE    = 40
SUBTYPE_COL = "subtype"
SAMPLE_COL  = "sample"
GENE_X, GENE_Y = "LCOR", "CXCL16"
MIN_N       = 20
N_COLS      = 4
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}


def hexbin_panel(ax, x, y, title, gridsize=40):
    if len(x) < MIN_N:
        ax.set_visible(False)
        return None

    xmin, xmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ymin, ymax = np.percentile(y, 0.5), np.percentile(y, 99.5)
    xpad = 0.20 * (xmax - xmin) or 0.5
    ypad = 0.20 * (ymax - ymin) or 0.5
    xl = (xmin - xpad, xmax + xpad)
    yl = (ymin - ypad, ymax + ypad)

    ax.set_facecolor("white")
    ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="RdYlBu_r",
                   mincnt=1, linewidths=0.02, bins="log",
                   extent=[xl[0], xl[1], yl[0], yl[1]], zorder=2)
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)

    r, p = stats.pearsonr(x, y)
    ax.text(0.97, 0.03, f"r={r:.2f}\nn={len(x)}",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85))
    return r, p, len(x)


print("="*65)
print(f" 03 -- {GENE_X} vs {GENE_Y} -- hexbin per patient, grouped by subtype")
print("="*65)

adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}")

idx = [adata.var_names.get_loc(g) for g in (GENE_X, GENE_Y)]
counts = adata.layers["counts"][:, idx]
counts = np.asarray(counts.todense()) if issparse(counts) else np.asarray(counts)
log = adata.layers["logcounts"][:, idx] if "logcounts" in adata.layers else adata.X[:, idx]
log = np.asarray(log.todense()) if issparse(log) else np.asarray(log)

obs = adata.obs.copy()
obs[f"{GENE_X}_raw"] = counts[:, 0]
obs[f"{GENE_Y}_raw"] = counts[:, 1]
obs[f"{GENE_X}_log"] = log[:, 0]
obs[f"{GENE_Y}_log"] = log[:, 1]
obs["codet"] = (obs[f"{GENE_X}_raw"] > 0) & (obs[f"{GENE_Y}_raw"] > 0)
del adata

# ── Depth-corrected residuals, computed PER PATIENT ────────────────────────────
# For each patient and each gene, fit the simplest depth model -- a constant
# detection RATE for that patient (rate = total raw UMIs / total counts) -- so
# the expected count for a cell is mu_i = total_counts_i * rate. This is the
# same offset-by-depth logic as 15_lcor_cxcl16_depth_patient.py (Poisson GLM
# with offset(log total_counts)), just fit within each patient separately
# instead of pooled, since we're already slicing the data patient by patient
# here. The Pearson residual (raw - mu)/sqrt(mu) is what's left of that gene's
# count after removing what sequencing depth alone would predict for that
# cell -- i.e. "more/less than expected given how deep this cell was
# sequenced", the natural depth-corrected quantity for a hexbin.
print("\n  Computing per-patient depth-adjusted residuals (Poisson offset)...")
for gene in (GENE_X, GENE_Y):
    resid = np.full(len(obs), np.nan)
    for s, g in obs.groupby(SAMPLE_COL):
        rate = g[f"{gene}_raw"].sum() / g["total_counts"].sum()
        mu = g["total_counts"].values * rate
        r = (g[f"{gene}_raw"].values - mu) / np.sqrt(np.maximum(mu, 1e-9))
        resid[obs.index.get_indexer(g.index)] = r
    obs[f"{gene}_resid"] = resid
print("  Done.")

def plot_grid(x_col, y_col, x_label, y_label, fname_suffix, subtitle_suffix):
    """Build the per-subtype, per-patient hexbin grid for a given pair of
    columns (raw logcounts, or depth-adjusted residuals), and return the
    per-patient correlation rows."""
    rows = []
    for subtype in ["ER", "HER2", "TNBC"]:
        st_obs = obs[(obs[SUBTYPE_COL] == subtype) & obs["codet"]]
        if st_obs.empty:
            continue
        sample_order = (st_obs.groupby(SAMPLE_COL).size()
                         .sort_values(ascending=False))
        samples = [s for s in sample_order.index if sample_order[s] >= MIN_N]
        if not samples:
            print(f"  {subtype}: no patient with >={MIN_N} codetected cells, skipping")
            continue

        n_cols = min(N_COLS, len(samples))
        n_rows = int(np.ceil(len(samples) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2*n_cols, 3.2*n_rows))
        axes = np.array(axes).reshape(-1)
        fig.suptitle(f"{subtype} -- {GENE_X} vs {GENE_Y}, codetected-only, per patient{subtitle_suffix}",
                     fontsize=11, fontweight="bold", color=SUBTYPE_COLORS[subtype])

        for i, s in enumerate(samples):
            g = st_obs[st_obs[SAMPLE_COL] == s]
            out = hexbin_panel(axes[i], g[x_col].values, g[y_col].values,
                                title=s.replace("GSM4909", "..."), gridsize=GRIDSIZE)
            if out:
                r, p, n = out
                rows.append({"sample": s, "subtype": subtype, "n_codet": n, "r": r, "p": p})
        for j in range(len(samples), len(axes)):
            axes[j].set_visible(False)

        for ax in axes[:len(samples)]:
            ax.set_xlabel(x_label, fontsize=7)
            ax.set_ylabel(y_label, fontsize=7)

        plt.tight_layout()
        fname = f"{OUTDIR}/hexbin_by_patient_{subtype}{fname_suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}  ({len(samples)} patients)")
    return rows


print("\n  Building RAW (logcounts) hexbins...")
corr_rows_raw = plot_grid(f"{GENE_X}_log", f"{GENE_Y}_log",
                           f"{GENE_X} (logcounts)", f"{GENE_Y} (logcounts)",
                           fname_suffix="", subtitle_suffix="")

print("\n  Building DEPTH-CORRECTED (Poisson residual) hexbins...")
corr_rows_dc = plot_grid(f"{GENE_X}_resid", f"{GENE_Y}_resid",
                          f"{GENE_X} (depth-adj. residual)", f"{GENE_Y} (depth-adj. residual)",
                          fname_suffix="_depth_corrected",
                          subtitle_suffix="\n(depth-corrected: Poisson residual vs per-patient rate)")

df_corr_raw = pd.DataFrame(corr_rows_raw).rename(columns={"r": "r_raw", "p": "p_raw"})
df_corr_dc  = pd.DataFrame(corr_rows_dc).rename(columns={"r": "r_depth_corrected", "p": "p_depth_corrected"})
df_corr = df_corr_raw.merge(df_corr_dc, on=["sample", "subtype", "n_codet"]).sort_values(["subtype", "r_raw"])
df_corr.to_csv(f"{OUTDIR}/persample_correlation.csv", index=False)
print("\n  Per-patient correlations (raw vs depth-corrected):")
print(df_corr.to_string(index=False))
print(f"\n  Saved: {OUTDIR}/persample_correlation.csv")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC -- why does more than one "line" appear WITHIN a single patient?
# Same discrete-UMI/depth artifact as 07f_TIM3_hypoxia_depth_adjusted.py:
# with only 1-4 raw UMIs per cell, each (LCOR_raw, CXCL16_raw) integer pair
# traces its own curve in normalized logcounts space as sequencing depth
# varies across cells -- not a distinct biological subpopulation.
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Diagnostic: raw-count / depth origin of the within-patient 'lines'...")
top_samples = (df_corr.sort_values("n_codet", ascending=False)["sample"].head(3).tolist())

fig, axes = plt.subplots(2, len(top_samples), figsize=(4.5*len(top_samples), 8.5))
for col, s in enumerate(top_samples):
    g = obs[(obs[SAMPLE_COL] == s) & obs["codet"]].copy()
    g["pair"] = g[f"{GENE_X}_raw"].astype(str) + "-" + g[f"{GENE_Y}_raw"].astype(str)
    top_pairs = g["pair"].value_counts().head(6).index

    ax = axes[0, col]
    for p in top_pairs:
        gp = g[g["pair"] == p]
        ax.scatter(gp[f"{GENE_X}_log"], gp[f"{GENE_Y}_log"], s=10, alpha=0.65, label=p)
    ax.set_title(f"{s}\ncolored by ({GENE_X}_raw, {GENE_Y}_raw)", fontsize=8.5)
    ax.legend(fontsize=6, title="raw pair", ncol=2)
    ax.set_xlabel(f"{GENE_X} (logcounts)", fontsize=8)
    ax.set_ylabel(f"{GENE_Y} (logcounts)", fontsize=8)

    ax = axes[1, col]
    sca = ax.scatter(g[f"{GENE_X}_log"], g[f"{GENE_Y}_log"], s=10,
                      c=np.log10(g["total_counts"]), cmap="viridis", alpha=0.75)
    plt.colorbar(sca, ax=ax, label="log10(total_counts)", shrink=0.85)
    ax.set_title(f"{s}\ncolored by sequencing depth", fontsize=8.5)
    ax.set_xlabel(f"{GENE_X} (logcounts)", fontsize=8)
    ax.set_ylabel(f"{GENE_Y} (logcounts)", fontsize=8)

fig.suptitle("Each 'line' within a patient = one fixed raw-UMI pair,\n"
             "stretched out by that cell's own sequencing depth (not a biological subgroup)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/diagnostic_within_patient_lines.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/diagnostic_within_patient_lines.png")

readme = [
    "03 -- LCOR x CXCL16 -- hexbin per patient, grouped by subtype",
    "",
    "WHAT THE 'MULTIPLE LINES WITHIN A SINGLE PATIENT' ARE",
    "-"*60,
    "In the higher-n patient panels (e.g. ER-MH0173-T, ER-MH0167-T) the",
    "codetected-only scatter shows several parallel diagonal streaks rather",
    "than one cloud. diagnostic_within_patient_lines.png shows why: with",
    "only 1-4 raw UMIs per cell for these genes, each cell's normalized",
    "logcounts value is ~log1p(raw_count * size_factor), where size_factor",
    "depends on that cell's own total_counts. So EVERY distinct integer",
    "pair (LCOR_raw, CXCL16_raw) -- e.g. (1,1), (1,2), (2,1), (1,3), (2,2) --",
    "traces its own deterministic curve as depth varies across cells.",
    "",
    "Top row: same cells colored by their raw UMI pair -- each color is",
    "exactly one line. Bottom row: same cells colored by sequencing depth --",
    "color grades smoothly along each line (high depth at the lower/left",
    "end, low depth at the upper/right end of each streak).",
    "",
    "This is the same discrete-count/depth artifact documented for HAVCR2",
    "in 07f_TIM3_hypoxia_depth_adjusted.py, here affecting two genes at",
    "once. It means the 'several lines' are NOT distinct biological",
    "subpopulations within a patient -- they are a mechanical consequence",
    "of normalizing very low, discrete raw counts. No further subgroup",
    "hunting on this hexbin (by patient or within a patient) is expected",
    "to reveal real structure; the depth-offset regression in",
    "15_lcor_cxcl16_depth_patient.py is the appropriate way to test the",
    "LCOR-CXCL16 relationship on this kind of sparse count data.",
    "",
    "DEPTH-CORRECTED HEXBINS (hexbin_by_patient_<subtype>_depth_corrected.png)",
    "-"*60,
    "For each patient and each gene, fit a constant per-patient detection",
    "rate = sum(raw UMIs)/sum(total_counts), so the DEPTH-EXPECTED count for",
    "cell i is mu_i = total_counts_i * rate. The Pearson residual",
    "(raw_i - mu_i)/sqrt(mu_i) is what's left of that gene's count after",
    "removing what sequencing depth alone predicts for that cell -- i.e.",
    "'more/less than expected given how deeply this cell was sequenced'.",
    "The depth-corrected hexbins plot LCOR residual vs CXCL16 residual",
    "(still restricted to codetected cells, for visual comparability with",
    "the raw version) and persample_correlation.csv reports both r_raw and",
    "r_depth_corrected side by side.",
    "",
    "CAVEATS on this residual: it only removes a simple constant-rate",
    "depth effect per patient (Poisson, no overdispersion/NB adjustment,",
    "unlike the pooled cluster-robust NB model in 02_depth_artifact_cell_level.py) -- treat it as",
    "a quick visual check, not a replacement for the 02_depth_artifact_cell_level.py regression",
    "numbers (RR=0.96 pooled / RR=1.06 within-patient) when reporting a",
    "headline result.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")
