"""
part1_plot_per_patient.py
--------------------------
Generate per-patient distribution figures for part1 metrics.
For each metric and signature, shows every patient as an individual dot
plus a box (median / IQR).

Outputs (figures/part1/):
  per_patient_r_ct2.png       — r(sig, CytoTRACE2) per patient
  per_patient_r_immune.png    — r(sig, immune) per patient
  per_patient_r_cc.png        — r(sig, S.Score) and r(sig, G2M) per patient
  per_patient_R2_cc.png       — R²(CC→sig) per patient
  per_patient_all.png         — 4-panel composite (all above)
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT     = "figures/part1"
MIN_PAT = 50    # min cells per patient to include

DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_processed.h5ad",
                   "batch_col": "sample",   "subtype_col": "subtype"},
    "Klughammer": {"file": "klughammer_malignant_processed.h5ad",
                   "batch_col": "sampleid", "subtype_col": "subtype"},
}

STEM_SIGS = [
    "ESC_ASSOU", "ESC_WONG", "BENPORATH_ES1", "EMP",
    "fMaSC_over_aMaSC", "LIM_MAMMARY_STEM_CELL_UP",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP", "LIM_MAMMARY_LUMINAL_MATURE_UP",
    "stem_consensus",
]
SIG_LABELS = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BENPORATH_ES1"                    : "BenPorath",
    "EMP"                              : "EMP",
    "fMaSC_over_aMaSC"                 : "fMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"         : "LIM_Stem",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
    "LIM_MAMMARY_LUMINAL_MATURE_UP"    : "LIM_LumMat",
    "stem_consensus"                   : "stem_cons",
}

DS_COLORS    = {"Visvader": "#2563EB", "Klughammer": "#7C3AED"}
DS_MARKERS   = {"Visvader": "o",       "Klughammer": "D"}
SUBTYPE_COLS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}


# ══════════════════════════════════════════════════════════════════════════════
# Compute per-patient r for all signatures
# ══════════════════════════════════════════════════════════════════════════════

def r2_cc(sig, s, g2m):
    m = np.isfinite(sig) & np.isfinite(s) & np.isfinite(g2m)
    if m.sum() < 20:
        return np.nan
    X = np.column_stack([s[m], g2m[m]])
    return float(LinearRegression().fit(X, sig[m]).score(X, sig[m]))

def r2_ct2(sig, ct2):
    m = np.isfinite(sig) & np.isfinite(ct2)
    if m.sum() < 20:
        return np.nan
    X = ct2[m].reshape(-1, 1)
    return float(LinearRegression().fit(X, sig[m]).score(X, sig[m]))

def safe_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return np.nan
    r, _ = spearmanr(x[m], y[m])
    return float(r)


all_pat_df = {}    # ds_name → DataFrame(patient, subtype, sig, metric, value)

for ds_name, cfg in DATASETS.items():
    print(f"Loading {ds_name} …")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs
    del adata
    batch_col   = cfg["batch_col"]
    subtype_col = cfg["subtype_col"]
    sigs = [s for s in STEM_SIGS if f"score_{s}" in obs.columns]
    rows = []
    for pat in obs[batch_col].unique():
        g = obs[obs[batch_col] == pat]
        if len(g) < MIN_PAT:
            continue
        subtype = str(g[subtype_col].iloc[0])
        s_s   = g["S.Score"].values.astype(float)
        g2m_s = g["G2M.Score"].values.astype(float)
        ct2_s = g["CytoTRACE2"].values.astype(float)
        imm_s = g["score_immune"].values.astype(float)
        for sig in sigs:
            sv = g[f"score_{sig}"].values.astype(float)
            rows.append({
                "patient" : pat,
                "subtype" : subtype,
                "sig"     : sig,
                "r_CT2"   : safe_r(sv, ct2_s),
                "r_S"     : safe_r(sv, s_s),
                "r_G2M"   : safe_r(sv, g2m_s),
                "r_immune": safe_r(sv, imm_s),
                "R2_CC"   : r2_cc(sv, s_s, g2m_s),
                "R2_CT2"  : r2_ct2(sv, ct2_s),
                "n_cells" : len(g),
            })
    all_pat_df[ds_name] = pd.DataFrame(rows)
    print(f"  {obs[batch_col].nunique()} patients, {len(sigs)} sigs → {len(rows)} rows")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helper
# ══════════════════════════════════════════════════════════════════════════════

def per_patient_strip(axes_row, metric, ylabel, hline=0,
                      title_suffix="", ylim=None):
    """
    One row of axes (one per dataset). For a given metric, plot:
    - box (Q1–Q3, median) per signature across patients
    - individual patient dots coloured by subtype
    """
    sig_order  = [s for s in STEM_SIGS if s in
                  all_pat_df[list(all_pat_df)[0]]["sig"].unique()]
    x_labels   = [SIG_LABELS[s] for s in sig_order]
    xs         = np.arange(len(sig_order))

    for ax, (ds_name, df) in zip(axes_row, all_pat_df.items()):
        ax.set_facecolor("#F9FAFB")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#D1D5DB")
        ax.tick_params(colors="#374151", labelsize=8)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)

        for xi, sig in enumerate(sig_order):
            vals = df[df["sig"] == sig][metric].dropna().values
            if len(vals) == 0:
                continue

            # box
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            bw = 0.35
            ax.fill_betweenx([q1, q3], xi - bw/2, xi + bw/2,
                             color="#BFDBFE", alpha=0.7, zorder=2)
            ax.plot([xi - bw/2, xi + bw/2], [med, med],
                    color="#1D4ED8", lw=2, zorder=3)
            ax.plot([xi, xi], [vals.min(), q1], color="#93C5FD",
                    lw=1, zorder=2)
            ax.plot([xi, xi], [q3, vals.max()], color="#93C5FD",
                    lw=1, zorder=2)

            # individual patient dots (jittered)
            sub_df = df[df["sig"] == sig][["subtype", metric]].dropna()
            rng    = np.random.default_rng(42)
            jitter = rng.uniform(-0.18, 0.18, len(sub_df))
            for (_, row), jit in zip(sub_df.iterrows(), jitter):
                col = SUBTYPE_COLS.get(row["subtype"], "#9CA3AF")
                ax.scatter(xi + jit, row[metric], color=col,
                           s=22, alpha=0.75, zorder=4,
                           edgecolors="white", linewidths=0.3)

        if hline is not None:
            ax.axhline(hline, color="#6B7280", ls="--", lw=0.9, alpha=0.6)

        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9, color="#374151")
        ax.set_xlim(-0.6, len(sig_order) - 0.4)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"{ds_name}  ({all_pat_df[ds_name]['patient'].nunique()} patients)",
                     fontsize=10, fontweight="bold", color="#111827")

    # subtype legend — placed outside the right axis to avoid overlap
    handles = [mpatches.Patch(color=c, label=st)
               for st, c in SUBTYPE_COLS.items()]
    handles += [mpatches.Patch(color="#BFDBFE", label="IQR box"),
                plt.Line2D([0], [0], color="#1D4ED8", lw=2, label="Median")]
    axes_row[-1].legend(handles=handles, fontsize=7.5,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        bbox_transform=axes_row[-1].transAxes,
                        framealpha=0.9, edgecolor="#D1D5DB")


# ══════════════════════════════════════════════════════════════════════════════
# Individual figures
# ══════════════════════════════════════════════════════════════════════════════

METRICS = [
    ("r_CT2",    "Spearman r (sig, CytoTRACE2)\nper patient",
     0,  "r(sig, CytoTRACE2)", (-0.6, 0.85)),
    ("r_immune", "Spearman r (sig, immunogenic consensus)\nper patient",
     0,  "r(sig, immune)",     (-0.55, 0.55)),
    ("R2_CC",    "R² (cell cycle → signature)\nper patient",
     None, "R²(CC→sig)",       (-0.02, 0.85)),
    ("R2_CT2",   "R² (CytoTRACE2 → signature)\nper patient",
     None, "R²(CT2→sig)",      (-0.02, 0.55)),
]

for metric, fig_title, hl, ylab, ylim in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(fig_title, fontsize=13, fontweight="bold",
                 color="#111827", y=1.02)
    per_patient_strip(axes, metric, ylab, hline=hl, ylim=ylim)
    plt.tight_layout()
    fname = f"per_patient_{metric}.png"
    fig.savefig(os.path.join(OUT, fname), dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ {fname}")

# r_S and r_G2M side by side (CC contamination)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
fig.patch.set_facecolor("white")
fig.suptitle("Cell-cycle contamination per patient\n"
             "r(sig, S.Score)  and  r(sig, G2M.Score)",
             fontsize=13, fontweight="bold", color="#111827", y=1.01)
per_patient_strip(axes[0], "r_S",   "r(sig, S.Score)",   hline=0)
per_patient_strip(axes[1], "r_G2M", "r(sig, G2M.Score)", hline=0)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "per_patient_r_cc.png"),
            dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("✓ per_patient_r_cc.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4-panel composite
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.patch.set_facecolor("white")
fig.suptitle("Per-patient metric distributions across all 9 stemness signatures\n"
             "Dots = individual patients coloured by subtype  ·  "
             "Box = IQR  ·  Line = median",
             fontsize=12, fontweight="bold", color="#111827", y=1.005)

panel_specs = [
    ("r_CT2",   "r(sig, CytoTRACE2)",        0,    None),
    ("r_immune","r(sig, immunogenic)",        0,    None),
    ("R2_CC",   "R²(CC → signature)",        None, (-0.02, 0.88)),
    ("R2_CT2",  "R²(CT2 → signature)",       None, (-0.02, 0.55)),
]

for row, (metric, ylab, hl, ylim) in enumerate(panel_specs):
    per_patient_strip(axes[row], metric, ylab, hline=hl, ylim=ylim)
    # row letter
    axes[row][0].text(-0.12, 1.05, chr(65 + row),
                      transform=axes[row][0].transAxes,
                      fontsize=14, fontweight="bold", color="#111827")

plt.tight_layout(h_pad=3.0)
fig.savefig(os.path.join(OUT, "per_patient_all.png"),
            dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("✓ per_patient_all.png")

print("\nDone. All figures saved to figures/part1/")
