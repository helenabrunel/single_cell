"""
02b_cohort_overview.py
=======================
Summary figures: cohort composition (patients vs. cell lines, number of
biopsies) and cells per patient/cell line, before and after QC filtering.

Uses gse243526_sample_summary.csv (pre-filter, from 01_load_samples.py) and
gse243526_processed.h5ad opened backed="r" (obs only — avoids loading the
full expression matrix just to count cells per sample).

Output:
  gse243526_cohort_summary.csv
  figures/02b_cohort_overview/cohort_overview.png

Run from: ~/single_cell/Regner/
  python3 02b_cohort_overview.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings

warnings.filterwarnings("ignore")

RAW_SUMMARY   = "gse243526_sample_summary.csv"
PROCESSED     = "gse243526_processed.h5ad"
OUTDIR        = "figures/02b_cohort_overview"
OUTFILE       = "gse243526_cohort_summary.csv"
os.makedirs(OUTDIR, exist_ok=True)

KEPT_COLOR    = "#4C72B0"
REMOVED_COLOR = "#C3C2B7"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD")
print("="*65)

raw = pd.read_csv(RAW_SUMMARY)
print(f"  Pre-filter summary: {len(raw)} samples")

adata = sc.read_h5ad(PROCESSED, backed="r")
post_counts = adata.obs.groupby("sample_name", observed=True).size()
adata.file.close()
print(f"  Post-filter obs loaded (backed): {post_counts.sum():,} cells")


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGGREGATE TO PATIENT / CELL-LINE LEVEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. AGGREGATE")
print("="*65)

raw["n_cells_filtered"] = raw["sample_name"].map(post_counts).fillna(0).astype(int)

cohort = raw.groupby("patient", observed=True).agg(
    is_cell_line   = ("is_cell_line", "first"),
    n_biopsies     = ("sample_name", "nunique"),
    n_cells_raw    = ("n_cells", "sum"),
    n_cells_filtered = ("n_cells_filtered", "sum"),
).reset_index()
cohort["pct_retained"] = (100 * cohort["n_cells_filtered"] /
                          cohort["n_cells_raw"]).round(1)
cohort = cohort.sort_values(
    ["is_cell_line", "n_cells_filtered"], ascending=[True, False]
).reset_index(drop=True)
cohort.to_csv(OUTFILE, index=False)

n_patients   = (~cohort["is_cell_line"]).sum()
n_cell_lines = cohort["is_cell_line"].sum()
n_biopsies   = int(raw["sample_name"].nunique())
multi_biopsy = cohort.loc[cohort["n_biopsies"] > 1, "patient"].tolist()

print(cohort.to_string(index=False))
print(f"\n  Patients   : {n_patients}")
print(f"  Cell lines : {n_cell_lines}")
print(f"  Biopsies/libraries: {n_biopsies}")
print(f"  Multi-biopsy patients: {multi_biopsy}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 3. FIGURE")
print("="*65)

fig = plt.figure(figsize=(10, 11))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 6], hspace=0.15, wspace=0.4)

# --- Stat tiles ---------------------------------------------------------------
tiles = [
    (str(n_patients),   "Patients"),
    (str(n_cell_lines),  "Cell lines"),
    (str(n_biopsies),   "Biopsies / libraries"),
]
note = f"({', '.join(multi_biopsy)}: 2 biopsies each)" if multi_biopsy else ""
for i, (value, label) in enumerate(tiles):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.text(0.5, 0.6, value, ha="center", va="center",
            fontsize=34, fontweight="bold", color="#0b0b0b")
    ax.text(0.5, 0.18, label, ha="center", va="center",
            fontsize=11, color="#52514e")
    if label.startswith("Biopsies") and note:
        ax.text(0.5, 0.0, note, ha="center", va="center",
                fontsize=8, color="#898781", style="italic")

# --- Cells per patient / cell line (stacked kept vs removed) ------------------
ax = fig.add_subplot(gs[1, :])

y = np.arange(len(cohort))
removed = cohort["n_cells_raw"] - cohort["n_cells_filtered"]
ax.barh(y, cohort["n_cells_filtered"], color=KEPT_COLOR, label="Kept (post-QC)")
ax.barh(y, removed, left=cohort["n_cells_filtered"],
        color=REMOVED_COLOR, label="Removed in QC")

labels = [f"{p}{'  (cell line)' if cl else ''}"
          for p, cl in zip(cohort["patient"], cohort["is_cell_line"])]
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()

for yi, (raw_n, kept_n) in enumerate(zip(cohort["n_cells_raw"],
                                          cohort["n_cells_filtered"])):
    ax.text(raw_n + max(cohort["n_cells_raw"]) * 0.01, yi,
            f"{kept_n:,}", va="center", fontsize=8, color="#0b0b0b")

# Separator between patients and cell lines
if n_cell_lines > 0 and n_patients > 0:
    ax.axhline(n_patients - 0.5, color="#898781", lw=0.8, ls="--")

ax.set_xlabel("N cells")
ax.set_title("Cells per patient / cell line (raw → post-QC)", fontsize=12)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e1e0d9", lw=0.6, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("Regner cohort (GSE243526) — overview", fontsize=14,
             fontweight="bold", y=0.985)
plt.savefig(f"{OUTDIR}/cohort_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {OUTDIR}/cohort_overview.png")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  {OUTFILE}
  {OUTDIR}/cohort_overview.png
══════════════════════════════════════════════════════════════
""")
