"""
02c_subtype_overview.py
=========================
Summary figures: cohort composition by tissue type (tumor / normal /
cell line) and, within tumors, by clinical subtype (derived from ER/PR/
HER2 IHC in 01b_add_clinical_metadata.py).

Key context: "Patient_1".."Patient_4" are normal breast tissue, not
tumor. Among the 11 tumor patients (12 biopsies incl. Patient_14 x2):
5 HR+/HER2-, 5 HER2-equivocal (ambiguous IHC, not FISH-confirmed
HER2+), 1 TNBC, 0 confirmed HER2+.

Uses gse243526_sample_summary.csv (pre-filter counts) + gse243526_
processed.h5ad opened backed="r" (post-filter counts + clinical obs
columns added by 01b) — no need to load the expression matrix.

Output:
  gse243526_subtype_summary.csv
  figures/02c_subtype_overview/subtype_overview.png

Run from: ~/single_cell/Regner/
  python3 02c_subtype_overview.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings

warnings.filterwarnings("ignore")

RAW_SUMMARY = "gse243526_sample_summary.csv"
PROCESSED   = "gse243526_processed.h5ad"
OUTDIR      = "figures/02c_subtype_overview"
OUTFILE     = "gse243526_subtype_summary.csv"
os.makedirs(OUTDIR, exist_ok=True)

# Consistent with Visvader/Klughammer conventions (ER=green, HER2=purple,
# TNBC=red) — HR+/HER2- plays the "ER-bucket" role here.
SUBTYPE_COLORS = {
    "HR+/HER2-":      "#4DAF4A",
    "HER2-equivocal": "#984EA3",
    "TNBC":           "#E41A1C",
    "normal":         "#4C72B0",
    "cell_line":      "#898781",
}
TISSUE_LABELS = {
    "primary breast tumor":   "Tumor",
    "normal breast tissue":   "Normal",
    "breast cancer cell line":"Cell line",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD")
print("="*65)

raw = pd.read_csv(RAW_SUMMARY)

adata = sc.read_h5ad(PROCESSED, backed="r")
obs = adata.obs[["sample_name", "tissue", "subtype"]].drop_duplicates()
post_counts = adata.obs.groupby("sample_name", observed=True).size()
adata.file.close()

raw = raw.merge(obs, on="sample_name", how="left")
raw["n_cells_filtered"] = raw["sample_name"].map(post_counts).fillna(0).astype(int)
print(f"  Samples: {len(raw)}")
print(raw[["sample_name", "tissue", "subtype", "n_cells", "n_cells_filtered"]]
      .to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGGREGATE TO PATIENT LEVEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. AGGREGATE")
print("="*65)

cohort = raw.groupby("patient", observed=True).agg(
    tissue         = ("tissue", "first"),
    subtype        = ("subtype", "first"),
    n_biopsies     = ("sample_name", "nunique"),
    n_cells_raw    = ("n_cells", "sum"),
    n_cells_filtered = ("n_cells_filtered", "sum"),
).reset_index()

tissue_order  = ["primary breast tumor", "normal breast tissue", "breast cancer cell line"]
subtype_order = ["HR+/HER2-", "HER2-equivocal", "TNBC", "normal", "cell_line"]
cohort["tissue"]  = pd.Categorical(cohort["tissue"], categories=tissue_order, ordered=True)
cohort["subtype"] = pd.Categorical(cohort["subtype"], categories=subtype_order, ordered=True)
cohort = cohort.sort_values(["tissue", "subtype", "n_cells_filtered"],
                            ascending=[True, True, False]).reset_index(drop=True)
cohort.to_csv(OUTFILE, index=False)

tumor_n  = (cohort["tissue"] == "primary breast tumor").sum()
normal_n = (cohort["tissue"] == "normal breast tissue").sum()
line_n   = (cohort["tissue"] == "breast cancer cell line").sum()
subtype_counts = cohort.loc[cohort["tissue"] == "primary breast tumor",
                            "subtype"].value_counts().reindex(
                            ["HR+/HER2-", "HER2-equivocal", "TNBC"]).fillna(0).astype(int)

print(cohort.to_string(index=False))
print(f"\n  Tumor patients : {tumor_n}")
print(f"  Normal patients: {normal_n}")
print(f"  Cell lines     : {line_n}")
print(f"  Tumor subtypes : {dict(subtype_counts)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 3. FIGURE")
print("="*65)

fig = plt.figure(figsize=(11, 12))
gs  = fig.add_gridspec(3, 4, height_ratios=[1, 1.3, 6], hspace=0.35, wspace=0.5)

# --- Row 1: tissue-level stat tiles -------------------------------------------
tiles = [(str(tumor_n), "Tumor patients"), (str(normal_n), "Normal patients"),
         (str(line_n), "Cell lines")]
for i, (value, label) in enumerate(tiles):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.text(0.5, 0.55, value, ha="center", va="center",
            fontsize=30, fontweight="bold", color="#0b0b0b")
    ax.text(0.5, 0.12, label, ha="center", va="center",
            fontsize=10.5, color="#52514e")

# --- Row 2: tumor subtype bar --------------------------------------------------
ax = fig.add_subplot(gs[1, :])
labels = ["HR+/HER2-", "HER2-equivocal", "TNBC"]
values = [subtype_counts.get(l, 0) for l in labels]
colors = [SUBTYPE_COLORS[l] for l in labels]
bars = ax.bar(labels, values, color=colors, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.05, str(v),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(values) + 1.2)
ax.set_ylabel("N patients")
ax.set_title("Tumor patients by clinical subtype (ER/PR/HER2 IHC)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.99, 0.95, "HER2-equivocal = ambiguous IHC, not FISH-confirmed HER2+",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        color="#898781", style="italic")

# --- Row 3: cells per patient, colored by tissue/subtype -----------------------
ax = fig.add_subplot(gs[2, :])
y = np.arange(len(cohort))
bar_colors = [SUBTYPE_COLORS[s] for s in cohort["subtype"]]
removed = cohort["n_cells_raw"] - cohort["n_cells_filtered"]

ax.barh(y, cohort["n_cells_filtered"], color=bar_colors)
ax.barh(y, removed, left=cohort["n_cells_filtered"], color="#e1e0d9")

ylabels = [f"{p}{' (' + TISSUE_LABELS.get(t, t) + ')' if s in ('normal','cell_line') else ''}"
           for p, t, s in zip(cohort["patient"], cohort["tissue"], cohort["subtype"])]
ax.set_yticks(y)
ax.set_yticklabels(ylabels, fontsize=9)
ax.invert_yaxis()

for yi, (raw_n, kept_n) in enumerate(zip(cohort["n_cells_raw"], cohort["n_cells_filtered"])):
    ax.text(raw_n + max(cohort["n_cells_raw"]) * 0.01, yi,
            f"{kept_n:,}", va="center", fontsize=8, color="#0b0b0b")

# Section separators (tissue blocks)
block_sizes = cohort.groupby("tissue", observed=True).size()
cursor = 0
for size in block_sizes:
    if size == 0:
        continue
    cursor += size
    if cursor < len(cohort):
        ax.axhline(cursor - 0.5, color="#898781", lw=0.8, ls="--")

ax.set_xlabel("N cells (colored segment = post-QC, gray = removed in QC)")
ax.set_title("Cells per patient / cell line, by subtype", fontsize=11)
ax.legend(handles=[mpatches.Patch(color=c, label=l)
                   for l, c in SUBTYPE_COLORS.items()],
          fontsize=8.5, loc="lower right", frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e1e0d9", lw=0.6, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("Regner cohort (GSE243526) — tissue & subtype overview",
             fontsize=14, fontweight="bold", y=0.995)
plt.savefig(f"{OUTDIR}/subtype_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/subtype_overview.png")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  {OUTFILE}
  {OUTDIR}/subtype_overview.png
══════════════════════════════════════════════════════════════
""")
