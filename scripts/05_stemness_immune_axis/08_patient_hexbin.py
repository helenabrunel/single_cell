"""
Part 2 – Step 2  Per-patient density hexbin (stem × immune space).

One hexbin per patient, coloured by cell density.
Threshold lines at P_ANALYSIS are overlaid to show where phenotype
zones fall relative to the bulk distribution.

Outputs
-------
figures/part2/02_hexbin_Visvader.png
figures/part2/02_hexbin_Klughammer.png
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings, math

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
os.makedirs("figures/part2", exist_ok=True)
OUT = "figures/part2"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── configuration ──────────────────────────────────────────────────────────────
DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_ccfree.h5ad",  "batch": "sample",   "subtype_col": "subtype"},
    "Klughammer": {"file": "klughammer_malignant_ccfree.h5ad", "batch": "sampleid", "subtype_col": "subtype"},
}
P_ANALYSIS = 0.10    # symmetric threshold shown on hexbins
MIN_CELLS  = 200
SUBTYPES   = ["ER", "HER2", "TNBC"]

with open(os.path.join(OUT, "01_selected_signature.txt")) as f:
    selected_sig = f.read().strip()

RAW_COLS = {
    "ESC_ASSOU"  : "score_ESC_ASSOU",
    "ESC_WONG"   : "score_ESC_WONG",
    "BenPorath"  : "score_BENPORATH_ES1",
    "EMP"        : "score_EMP",
    "fMaSC"      : "score_fMaSC_over_aMaSC",
    "LIM_Stem"   : "score_LIM_MAMMARY_STEM_CELL_UP",
    "LIM_LumProg": "score_LIM_MAMMARY_LUMINAL_PROGENITOR_UP",
    "LIM_LumMat" : "score_LIM_MAMMARY_LUMINAL_MATURE_UP",
    "stem_cons"  : "score_stem_consensus",
}
STEM_COL   = RAW_COLS[selected_sig]
IMMUNE_COL = "score_immune"

# ── colours ────────────────────────────────────────────────────────────────────
BG   = "#FFFFFF"
DARK = "#1F2937"
GREY = "#6B7280"
SUBTYPE_COLORS = {"ER": "#2563EB", "HER2": "#059669", "TNBC": "#DC2626"}
DS_ACCENTS     = {"Visvader": "#2563EB", "Klughammer": "#7C3AED"}
C_SCIE = "#26C6DA"; C_NSI = "#EF5350"; C_SCI = "#4CAF50"

# ── one figure per dataset ─────────────────────────────────────────────────────
for ds_name, cfg in DATASETS.items():
    print(f"\n{ds_name} ...")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs.copy(); del adata

    # collect patients per subtype
    pat_by_st = {st: [] for st in SUBTYPES}
    for pat in sorted(obs[cfg["batch"]].unique()):
        g  = obs[obs[cfg["batch"]] == pat]
        st = str(g[cfg["subtype_col"]].iloc[0])
        if st in SUBTYPES and len(g) >= MIN_CELLS:
            pat_by_st[st].append(pat)

    n_per_st = max(len(v) for v in pat_by_st.values())
    n_rows   = len(SUBTYPES)
    n_cols   = n_per_st if n_per_st > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"{ds_name}  –  stem × immune density  |  "
        f"signature: {selected_sig}  |  "
        f"threshold lines: P = {int(P_ANALYSIS*100)}%",
        fontsize=11, fontweight="bold", color=DS_ACCENTS[ds_name], y=1.01)

    for row_i, st in enumerate(SUBTYPES):
        pats = pat_by_st[st]
        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            ax.set_facecolor("#F0F0F0")
            for sp in ax.spines.values():
                sp.set_color("#D1D5DB")

            if col_i >= len(pats):
                ax.axis("off")
                continue

            pat = pats[col_i]
            g   = obs[obs[cfg["batch"]] == pat]
            s   = g[STEM_COL].values.astype(float)
            im  = g[IMMUNE_COL].values.astype(float)
            ok  = np.isfinite(s) & np.isfinite(im)
            s, im = s[ok], im[ok]

            hb = ax.hexbin(s, im, gridsize=30, cmap="RdYlBu_r",
                           mincnt=1, linewidths=0.1, edgecolors="none")

            # symmetric per-patient threshold lines
            thr_s_hi = np.percentile(s,  (1 - P_ANALYSIS) * 100)
            thr_s_lo = np.percentile(s,        P_ANALYSIS  * 100)
            thr_i_lo = np.percentile(im,       P_ANALYSIS  * 100)
            thr_i_hi = np.percentile(im, (1 - P_ANALYSIS) * 100)

            ax.axvline(thr_s_hi, color=C_SCIE, lw=1.2, ls="--", alpha=0.9)
            ax.axvline(thr_s_lo, color=C_NSI,  lw=1.2, ls="--", alpha=0.9)
            ax.axhline(thr_i_lo, color=C_SCIE, lw=1.2, ls="--", alpha=0.9)
            ax.axhline(thr_i_hi, color=C_NSI,  lw=1.2, ls="--", alpha=0.9)

            ax.set_title(pat, fontsize=6, color=DARK, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(st, fontsize=9, fontweight="bold",
                              color=SUBTYPE_COLORS[st], labelpad=4)

    plt.tight_layout()
    out = os.path.join(OUT, f"02_hexbin_{ds_name}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {out}")
