"""
Part 2 – Step 3  Phenotype classification and distribution.

One symmetric threshold P applied to both axes per patient:

    stem_high  = top P%     (>= (100-P)th percentile)
    stem_low   = bottom P%  (<=      P th percentile)
    immune_high = top P%    (>= (100-P)th percentile)
    immune_low  = bottom P% (<=      P th percentile)

This defines 9 zones. Three corners are labelled phenotypes:
    SCIE  stem_high + immune_low    (bottom-right)
    NSI   stem_low  + immune_high   (top-left)
    SCI   stem_high + immune_high   (top-right)
    NC    all other zones (unclassified)

Fractions are independent — they do NOT sum to 100 %.

Outputs
-------
figures/part2/03_phenotype_fractions.csv
figures/part2/03_phenotypes_<dataset>.png   per-patient grouped bars
figures/part2/03_phenotypes_by_subtype.png  box plots per subtype
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
os.makedirs("figures/part2", exist_ok=True)
OUT = "figures/part2"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── configuration ──────────────────────────────────────────────────────────────
P_ANALYSIS = 0.10   # single symmetric percentile for both axes

DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_ccfree.h5ad",  "batch": "sample",   "subtype_col": "subtype"},
    "Klughammer": {"file": "klughammer_malignant_ccfree.h5ad", "batch": "sampleid", "subtype_col": "subtype"},
}
SUBTYPES   = ["ER", "HER2", "TNBC"]
MIN_CELLS  = 200
PHENOTYPES = ["SCIE", "NSI", "SCI"]

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
BORDER = "#E5E7EB"
C_SCIE = "#26C6DA"; C_NSI = "#EF5350"; C_SCI = "#4CAF50"
PH_COLORS = {"SCIE": C_SCIE, "NSI": C_NSI, "SCI": C_SCI}
ST_COLORS  = {"ER": "#2563EB", "HER2": "#059669", "TNBC": "#DC2626"}
DS_ACCENTS = {"Visvader": "#2563EB", "Klughammer": "#7C3AED"}

print(f"Signature  : {selected_sig}")
print(f"P_ANALYSIS : {P_ANALYSIS*100:.0f}%  "
      f"→ top/bottom {P_ANALYSIS*100:.0f}% per patient on each axis")

# ── classify per patient ───────────────────────────────────────────────────────
all_rows = []

for ds_name, cfg in DATASETS.items():
    print(f"\n{ds_name} ...")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs.copy(); del adata

    for pat in obs[cfg["batch"]].unique():
        g  = obs[obs[cfg["batch"]] == pat]
        st = str(g[cfg["subtype_col"]].iloc[0])
        if st not in SUBTYPES or len(g) < MIN_CELLS:
            continue

        s  = g[STEM_COL].values.astype(float)
        im = g[IMMUNE_COL].values.astype(float)
        ok = np.isfinite(s) & np.isfinite(im)
        if ok.sum() < MIN_CELLS:
            continue
        s, im = s[ok], im[ok]

        # symmetric per-patient thresholds
        thr_s_hi = np.percentile(s,  (1 - P_ANALYSIS) * 100)
        thr_s_lo = np.percentile(s,        P_ANALYSIS  * 100)
        thr_i_hi = np.percentile(im, (1 - P_ANALYSIS) * 100)
        thr_i_lo = np.percentile(im,       P_ANALYSIS  * 100)

        stem_hi = s  >= thr_s_hi
        stem_lo = s  <= thr_s_lo
        imm_hi  = im >= thr_i_hi
        imm_lo  = im <= thr_i_lo

        scie = (stem_hi & imm_lo).mean() * 100   # bottom-right
        nsi  = (stem_lo & imm_hi).mean() * 100   # top-left
        sci  = (stem_hi & imm_hi).mean() * 100   # top-right
        nc   = (~(stem_hi | stem_lo) & ~(imm_hi | imm_lo)).mean() * 100

        all_rows.append({
            "dataset": ds_name, "patient": pat, "subtype": st,
            "n_cells": ok.sum(),
            "SCIE": scie, "NSI": nsi, "SCI": sci, "NC": nc,
        })

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUT, "03_phenotype_fractions.csv"), index=False)
print(f"\n  Saved 03_phenotype_fractions.csv  ({len(df)} patients)")

# ── summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"Phenotype fractions  (mean % across patients)  –  P = {int(P_ANALYSIS*100)}%")
print("=" * 65)
for ds_name in DATASETS:
    sub = df[df["dataset"] == ds_name]
    print(f"\n{ds_name}:")
    print(f"  {'Subtype':8s}  {'SCIE':>7s}  {'NSI':>7s}  {'SCI':>7s}  {'NC':>7s}  n")
    for st in SUBTYPES:
        g = sub[sub["subtype"] == st]
        if len(g):
            print(f"  {st:8s}  {g['SCIE'].mean():6.2f}%  {g['NSI'].mean():6.2f}%  "
                  f"{g['SCI'].mean():6.2f}%  {g['NC'].mean():6.2f}%  n={len(g)}")

# ── Figure 1: per-patient grouped bars, one panel per subtype ─────────────────
bar_w   = 0.25
offsets = np.array([-bar_w, 0, bar_w])

for ds_name in DATASETS:
    sub = df[df["dataset"] == ds_name]
    fig, axes = plt.subplots(1, len(SUBTYPES),
                             figsize=(5 * len(SUBTYPES), 5), sharey=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"{ds_name}  –  Rare phenotype fractions per patient  (% of all cells)\n"
        f"Signature: {selected_sig}  |  P = {int(P_ANALYSIS*100)}%  "
        f"|  NC (centre cells) not shown",
        fontsize=11, fontweight="bold", color=DS_ACCENTS[ds_name], y=1.02)

    for ax, st in zip(axes, SUBTYPES):
        g  = sub[sub["subtype"] == st].sort_values("patient").reset_index(drop=True)
        xs = np.arange(len(g))
        for off, ph in zip(offsets, PHENOTYPES):
            ax.bar(xs + off, g[ph].values, width=bar_w * 0.9,
                   color=PH_COLORS[ph], alpha=0.85,
                   edgecolor="white", linewidth=0.3, zorder=3, label=ph)
        ax.set_xticks(xs)
        ax.set_xticklabels(g["patient"].values, rotation=90, fontsize=6)
        ax.set_title(st, fontsize=11, fontweight="bold", color=ST_COLORS[st], pad=5)
        ax.set_ylabel("% of cells", fontsize=9)
        ax.set_facecolor("#F9FAFB")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(BORDER)
        ax.grid(axis="y", color=BORDER, lw=0.7, zorder=0)

    handles = [plt.Rectangle((0,0),1,1, facecolor=PH_COLORS[ph], alpha=0.85, label=ph)
               for ph in PHENOTYPES]
    fig.legend(handles=handles, title="Phenotype", fontsize=9,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
               framealpha=0.9, edgecolor=BORDER)
    plt.tight_layout()
    out = os.path.join(OUT, f"03_phenotypes_{ds_name}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {out}")

# ── Figure 2: box plots – columns = subtypes, x-axis = phenotypes ─────────────
fig, axes = plt.subplots(2, len(SUBTYPES),
                         figsize=(5 * len(SUBTYPES), 9), sharey="row")
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"Rare phenotype fractions by subtype  –  {selected_sig}  |  P = {int(P_ANALYSIS*100)}%\n"
    "Each dot = one patient  |  NC (centre) not shown",
    fontsize=11, fontweight="bold", color=DARK, y=1.02)

for row_i, ds_name in enumerate(DATASETS):
    sub = df[df["dataset"] == ds_name]
    for col_i, st in enumerate(SUBTYPES):
        ax = axes[row_i, col_i]
        g  = sub[sub["subtype"] == st]
        for xi, ph in enumerate(PHENOTYPES):
            vals = g[ph].values
            col  = PH_COLORS[ph]
            ax.boxplot(vals, positions=[xi], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.30),
                       medianprops=dict(color=DARK, lw=2),
                       whiskerprops=dict(color=GREY), capprops=dict(color=GREY),
                       flierprops=dict(marker="o", color=GREY, alpha=0.4, ms=4))
            jit = np.random.default_rng(xi).uniform(-0.15, 0.15, len(vals))
            ax.scatter(xi + jit, vals, color=col, s=28, alpha=0.8,
                       zorder=4, edgecolors="white", lw=0.5)
        ax.set_xticks(range(len(PHENOTYPES)))
        ax.set_xticklabels(PHENOTYPES, fontsize=10, fontweight="bold")
        ax.set_facecolor("#F9FAFB")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(BORDER)
        ax.grid(axis="y", color=BORDER, lw=0.7)
        if row_i == 0:
            ax.set_title(st, fontsize=12, fontweight="bold", color=ST_COLORS[st], pad=6)
        if col_i == 0:
            ax.set_ylabel(f"{ds_name}\nfraction (%)", fontsize=9.5,
                          color=DS_ACCENTS[ds_name], fontweight="bold")

plt.tight_layout()
out2 = os.path.join(OUT, "03_phenotypes_by_subtype.png")
fig.savefig(out2, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  {out2}")
