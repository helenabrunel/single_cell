"""
03_malignant_structure_no_harmony.py
====================================
One consolidated overview panel of the malignant (epithelial) compartment,
on the NON-Harmony embedding, to re-orient before further analysis: is
there obvious patient/batch structure, and where do LCOR / CXCL16 / CXCR6
actually sit on it.

WHY THIS SCRIPT EXISTS
-----------------------
Scripts 14-19 and 22 dug into specific statistical questions (depth
artifacts, per-patient correlation, LIANA, pseudobulk) piecemeal. This
script instead builds ONE panel, side by side, on a single embedding, to
get oriented visually before deciding the next analysis:
  - Does the malignant compartment cluster by PATIENT (expected -- tumor
    cells are patient-specific / clonal, unlike immune cells) or by
    something else?
  - Does it cluster by SEQUENCING DEPTH (would flag a technical artifact
    dominating the embedding)?
  - Where do LCOR, CXCL16, and CXCR6 sit on that structure?

EMBEDDING: reuses the NO-Harmony embedding already computed by
20e_recompute_umap_no_harmony.py (figures/20c_LCOR_malignant_only/
umap_no_harmony.npy -- HVG(2000, seurat) -> PCA(30) -> neighbors(30,15) ->
UMAP, no batch correction), NOT recomputed here. Row order = adata.obs_names
of compartment_epithelial.h5ad, verified by n_obs match before use.
No-Harmony (rather than the pipeline's Harmony-integrated embedding) is
used deliberately: Harmony is designed to REMOVE patient-level structure,
which is exactly one of the two things this panel needs to inspect.

CXCR6 IS INCLUDED DESPITE BEING A T-CELL MARKER: it should be near-absent
in malignant cells (confirmed in 01_celltype_specificity_overview.py's compartmental-specificity
check). Showing it here is a visual sanity check on that, not a search for
signal in the wrong compartment.

Input:  compartment_epithelial.h5ad
        figures/20c_LCOR_malignant_only/umap_no_harmony.npy
Output: figures/23_malignant_no_harmony_overview/
    fig_overview_panel.png   (2x3: subtype, patient, depth | LCOR, CXCL16, CXCR6)
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/01_exploratorio/scripts/03_malignant_structure_no_harmony.py
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 10})

OUTDIR = "LCOR-CXCL16/01_exploratorio/figures/03_malignant_structure_no_harmony"
UMAP_NPY = "LCOR-CXCL16/01_exploratorio/figures/03_malignant_structure_no_harmony/umap_no_harmony.npy"
GENES = ["LCOR", "CXCL16", "CXCR6"]
SUBTYPES = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(" 03 -- malignant compartment, no-Harmony overview panel")
print("="*65)

print("\n  Loading compartment_epithelial.h5ad...")
adata = sc.read_h5ad("compartment_epithelial.h5ad")
print(f"  Cells: {adata.n_obs:,}   Patients: {adata.obs['sample'].nunique()}")

print(f"  Loading precomputed no-Harmony embedding: {UMAP_NPY}")
umap = np.load(UMAP_NPY)
if umap.shape[0] != adata.n_obs:
    raise SystemExit(f"  Row count mismatch: umap has {umap.shape[0]} rows, "
                      f"adata has {adata.n_obs} -- re-run 20e first.")
print(f"  Embedding shape: {umap.shape} -- matches adata.n_obs, OK")

missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

gene_expr = {}
for g in GENES:
    gidx = adata.var_names.get_loc(g)
    v = adata.layers["logcounts"][:, gidx]
    v = np.asarray(v.todense()).ravel() if issparse(v) else np.asarray(v).ravel()
    gene_expr[g] = v

subtype = adata.obs["subtype"].astype(str).values
sample  = adata.obs["sample"].astype(str).values
depth   = adata.obs["total_counts"].values.astype(float)
log10_depth = np.log10(depth)
n_patients = adata.obs["sample"].nunique()

plot_order = np.random.RandomState(0).permutation(adata.n_obs)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL -- 2 rows x 3 cols: subtype, patient, depth | LCOR, CXCL16, CXCR6
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building overview panel...")
fig, axes = plt.subplots(2, 3, figsize=(17, 11))

# (0,0) subtype
ax = axes[0, 0]
subtype_po = subtype[plot_order]
for st in SUBTYPES:
    m = subtype_po == st
    if not m.any():
        continue
    ax.scatter(umap[plot_order][m, 0], umap[plot_order][m, 1], s=1.3,
               color=SUBTYPE_COLORS[st], alpha=0.6, label=f"{st} (n={m.sum():,})", rasterized=True)
ax.set_title("BC subtype", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, markerscale=8, loc="best")

# (0,1) patient -- qualitative palette, no legend (34 categories -- legend unreadable)
ax = axes[0, 1]
sample_codes = pd.Categorical(sample).codes
cmap_patient = plt.get_cmap("gist_ncar", n_patients)
ax.scatter(umap[plot_order, 0], umap[plot_order, 1], s=1.3,
           c=sample_codes[plot_order], cmap=cmap_patient, alpha=0.65, rasterized=True)
ax.set_title(f"Patient (n={n_patients}, no legend -- see caveat if\nclusters align 1:1 with color patches)",
             fontsize=11, fontweight="bold")

# (0,2) sequencing depth
ax = axes[0, 2]
order = np.argsort(log10_depth[plot_order])
idx_ord = plot_order[order]
sca = ax.scatter(umap[idx_ord, 0], umap[idx_ord, 1], s=1.3, c=log10_depth[idx_ord],
                  cmap="viridis", alpha=0.7, rasterized=True)
cb = plt.colorbar(sca, ax=ax, shrink=0.8)
cb.set_label("log10(total_counts)", fontsize=9)
ax.set_title("Sequencing depth per cell", fontsize=12, fontweight="bold")

# (1, 0:3) gene expression
for i, gene in enumerate(GENES):
    ax = axes[1, i]
    expr = gene_expr[gene]
    order = np.argsort(expr[plot_order])
    idx_ord = plot_order[order]
    sca = ax.scatter(umap[idx_ord, 0], umap[idx_ord, 1], s=1.3, c=expr[idx_ord],
                      cmap="Purples" if gene == "LCOR" else ("Oranges" if gene == "CXCL16" else "Blues"),
                      alpha=0.75, rasterized=True)
    cb = plt.colorbar(sca, ax=ax, shrink=0.8)
    cb.set_label("logcounts", fontsize=9)
    ax.set_title(f"{gene}  (detected in {100*(expr>0).mean():.1f}% of cells)",
                 fontsize=12, fontweight="bold")

for ax in axes.flatten():
    ax.set_xlabel("UMAP1", fontsize=9); ax.set_ylabel("UMAP2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f"Malignant (epithelial) compartment -- NO Harmony -- {adata.n_obs:,} cells, "
             f"{n_patients} patients\nstructure (subtype/patient/depth) vs. LCOR / CXCL16 / CXCR6 expression",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUTDIR}/fig_overview_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_overview_panel.png")

# ══════════════════════════════════════════════════════════════════════════════
# Quick numeric read on patient clustering (silhouette-free, cheap proxy):
# fraction of each patient's cells falling in its own single densest UMAP
# hexbin cell, vs a null of the same size drawn uniformly at random -- a
# quick, honest "do patients form their own separate blobs" signal without
# a full clustering analysis.
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Quick patient-separation proxy (2D histogram concentration)...")
from scipy.stats import binned_statistic_2d
xedges = np.linspace(umap[:, 0].min(), umap[:, 0].max(), 60)
yedges = np.linspace(umap[:, 1].min(), umap[:, 1].max(), 60)
xi = np.digitize(umap[:, 0], xedges)
yi = np.digitize(umap[:, 1], yedges)
bin_id = xi.astype(np.int64) * (len(yedges) + 2) + yi.astype(np.int64)

rows = []
for s in pd.unique(sample):
    m = sample == s
    n = m.sum()
    if n < 20:
        continue
    bins_s = bin_id[m]
    top_bin_frac = pd.Series(bins_s).value_counts(normalize=True).iloc[0]
    rows.append({"sample": s, "n_cells": int(n), "top_bin_frac": top_bin_frac})
df_sep = pd.DataFrame(rows).sort_values("top_bin_frac", ascending=False)
df_sep.to_csv(f"{OUTDIR}/patient_separation_proxy.csv", index=False)
print(df_sep.to_string(index=False))
print(f"\n  Median top_bin_frac across patients: {df_sep['top_bin_frac'].median():.3f}")
print("  (High values = a large share of that patient's cells sit in one small")
print("   UMAP neighborhood -> patient-specific clustering, expected for tumor cells.")
print("   Low/spread-out values = that patient's cells are scattered across the")
print("   embedding, mixing with other patients/subtypes.)")
print(f"  Saved: {OUTDIR}/patient_separation_proxy.csv")

# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
median_frac = df_sep["top_bin_frac"].median()
readme = [
    "03 -- malignant compartment, no-Harmony overview panel",
    "="*70,
    "",
    "PURPOSE",
    "-"*70,
    "A single re-orientation panel before further LCOR/CXCL16/CXCR6 analysis:",
    "does the malignant compartment cluster by patient (expected, tumor cells",
    "are patient-specific) or by depth (would flag a technical artifact), and",
    "where do the 3 genes of interest sit relative to that structure.",
    "",
    "EMBEDDING",
    "-"*70,
    "Reused from 20e_recompute_umap_no_harmony.py (NOT recomputed here):",
    "HVG(2000, seurat) -> PCA(30) -> neighbors(30 PCs, 15 neighbors) -> UMAP,",
    "no Harmony batch correction. Deliberately no-Harmony because Harmony is",
    "designed to remove exactly the patient-level structure this panel checks.",
    "",
    "PANEL LAYOUT (fig_overview_panel.png)",
    "-"*70,
    "  Row 1: BC subtype | Patient (34 patients, qualitative cmap, no legend) | "
    "sequencing depth (log10 total_counts)",
    "  Row 2: LCOR | CXCL16 | CXCR6 (all logcounts, own color scale per gene)",
    "  CXCR6 is a T-cell marker, included as a compartmental-specificity sanity",
    "  check (should be near-absent here, per 01_celltype_specificity_overview.py) -- not a search for",
    "  signal in the wrong compartment.",
    "",
    "PATIENT-SEPARATION PROXY",
    "-"*70,
    "  For each patient (n_cells>=20): the fraction of that patient's cells",
    "  falling into its single densest 60x60 UMAP grid cell (patient_separation_",
    "  proxy.csv). This is a cheap, honest way to quantify 'do patients form",
    "  their own separate blobs' without a full clustering/silhouette analysis.",
    f"  Median top_bin_frac across patients: {median_frac:.3f}",
    "  High values -> that patient's malignant cells occupy a tight, largely",
    "  patient-exclusive region of the embedding (expected for tumor cells --",
    "  each patient's malignant clone is genetically/transcriptionally its own).",
    "  Low values -> that patient's cells are spread across the embedding,",
    "  overlapping with other patients (could reflect a less distinct tumor",
    "  cell state, or just fewer cells for that patient to form a tight blob).",
    "",
    "HOW TO READ THIS FOR THE LCOR-CXCL16 QUESTION",
    "-"*70,
    "If the embedding is dominated by patient identity (which is normal/expected",
    "for malignant cells, not a defect to fix), that is the underlying reason",
    "14_lcor_cxcl16_hexbin.py, 02_depth_artifact_cell_level.py, 03_depth_artifact_diagnostic_hexbin.py found within-patient depth-offset regression essential:",
    "any pooled cell-level correlation is heavily shaped by which patients",
    "happen to sit where, not necessarily a shared biological gradient across",
    "all tumors. It also explains why 22's per-patient pseudobulk approach is",
    "the more interpretable summary once this structure is visually confirmed.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Output: {OUTDIR}/
============================================================
""")
