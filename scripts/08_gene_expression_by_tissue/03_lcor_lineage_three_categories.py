"""
20b_lcor_epithelial_lineage_3cat.py
========================================
Simplified re-take on 20_lcor_epithelial_lineage_normal.py: instead of the
10-signature winner-takes-all label (7 precomputed cancer molecular-subtype
signatures + 3 normal mammary-lineage signatures), this script classifies
epithelial cells using ONLY the 3 normal mammary lineage signatures --
Basal, Luminal_progenitor, Mature_luminal -- discarding the precomputed
cancer molecular-subtype scheme entirely. Cells that don't score positively
on any of the 3 lineage signatures are labeled "Others" (kept, not dropped,
so the object/figures still account for 100% of epithelial cells).

Steps 1-3 (recovering Normal-tissue epithelial cells from
GSE161529_cellwhisperer.h5ad, combining with compartment_epithelial.h5ad,
and the joint PCA/UMAP embedding) are unchanged from 20_lcor_epithelial_lineage_normal.py
-- see that script's docstring for full rationale/caveats (no Harmony
integration; supplementary visualization only, dotplot doesn't depend on it).

OUTPUTS
-------
figures/20b_LCOR_epithelial_lineage_3cat/
    epithelial_lineage_3cat.h5ad  -- combined tumor+normal epithelial object,
                                      obs["epi_lineage"] in {Basal,
                                      Luminal_progenitor, Mature_luminal, Others}
    fig_umap_epi_lineage.png      -- panel A equivalent (cell population)
    fig_umap_LCOR.png             -- panel B equivalent (LCOR expression)
    fig_dotplot_LCOR.png          -- panel C equivalent (LCOR x lineage x subtype)
    dotplot_LCOR_summary.csv
    README.txt

Run from: ~/single_cell/Visvader/
  python3 20b_lcor_epithelial_lineage_3cat.py
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

OUTDIR = "figures/20b_LCOR_epithelial_lineage_3cat"
os.makedirs(OUTDIR, exist_ok=True)
GENE = "LCOR"
SUBTYPES = ["ER", "HER2", "TNBC", "Normal"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C", "Normal": "#377EB8"}

# Compartment marker genes -- copied verbatim from 03_split_integrate.py
COMPARTMENT_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT5", "KRT14"],
    "Immune":     ["PTPRC", "CD3D", "CD3E", "CD8A", "CD8B", "MS4A1", "CD14",
                   "LYZ", "NKG7", "GNLY", "FCGR3A", "CD68", "MZB1", "TPSAB1"],
    "Stromal":    ["DCN", "FAP", "COL1A1", "COL1A2", "ACTA2", "PECAM1",
                   "VWF", "CDH5", "CLDN5", "PDGFRB", "RGS5"],
}
THRESHOLD, IMMUNE_EXCL, PTPRC_EXCL = 0.1, 0.2, 0.5
MIN_GENES, MIN_COUNTS, MAX_MT, MIN_CELLS = 200, 500, 30, 3

# Epithelial lineage markers -- ONLY the normal mammary lineage scheme
# (the cancer molecular-subtype scheme from 03_split_integrate.py's FINE_EPI
# is deliberately discarded here, per request).
LINEAGE_MARKERS = {
    "Basal":              ["KRT5", "KRT14", "TP63", "ACTA2", "MYLK", "OXTR", "KRT17"],
    "Luminal_progenitor": ["KIT", "ELF5", "ALDH1A3", "KRT8", "KRT18", "SLPI"],
    "Mature_luminal":     ["ESR1", "PGR", "FOXA1", "GATA3", "ANKRD30A", "AR"],
}
LINEAGE_ORDER = list(LINEAGE_MARKERS.keys()) + ["Others"]
LINEAGE_COLORS = {
    "Basal": "#E41A1C", "Luminal_progenitor": "#377EB8",
    "Mature_luminal": "#4DAF4A", "Others": "#BDBDBD",
}


def winner_takes_all(obs, marker_dict, label, other_label="Others"):
    score_cols = [f"score_{c}" for c in marker_dict if f"score_{c}" in obs.columns]
    names = [c.replace("score_", "") for c in score_cols]
    sm = obs[score_cols].values
    best_idx, best = sm.argmax(axis=1), sm.max(axis=1)
    result = np.array([names[i] if s > 0 else other_label for i, s in zip(best_idx, best)])
    print(f"  {label} distribution:\n{pd.Series(result).value_counts().to_string()}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 -- recover Normal-tissue Epithelial cells from the raw combined object
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print(" 20b -- LCOR by 3-category epithelial lineage x subtype")
print("="*65)
print("\nSTEP 1 -- Normal-tissue epithelial cells from GSE161529_cellwhisperer.h5ad")

raw = sc.read_h5ad("GSE161529_cellwhisperer.h5ad")
normal = raw[(raw.obs["subtype"] == "Normal") & (raw.obs["tissue"] == "Total")].copy()
del raw
print(f"  Normal, tissue=Total: {normal.n_obs:,} cells, {normal.obs['sample'].nunique()} samples")

mapping = pd.read_csv("ensembl_to_symbol_mapping.csv", index_col=0)
ens2sym = dict(zip(mapping["ensembl_id"], mapping.index))
normal.var_names = [ens2sym.get(e, e) for e in normal.var_names]
normal.var_names_make_unique()

normal.var["mt"] = normal.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(normal, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
max_genes = int(np.percentile(normal.obs["n_genes_by_counts"], 99))
qc_mask = ((normal.obs["n_genes_by_counts"] >= MIN_GENES) &
           (normal.obs["n_genes_by_counts"] <= max_genes) &
           (normal.obs["total_counts"] >= MIN_COUNTS) &
           (normal.obs["pct_counts_mt"] <= MAX_MT))
print(f"  QC (MIN_GENES={MIN_GENES}, MAX_GENES=p99={max_genes}, MIN_COUNTS={MIN_COUNTS}, "
      f"MAX_MT={MAX_MT}): {qc_mask.sum():,}/{normal.n_obs:,} cells pass")
normal = normal[qc_mask].copy()
sc.pp.filter_genes(normal, min_cells=MIN_CELLS)
print(f"  After gene filter (min_cells={MIN_CELLS}): {normal.n_obs:,} cells x {normal.n_vars:,} genes")

normal.layers["counts"] = normal.X.copy()
sc.pp.normalize_total(normal, target_sum=1e4)
sc.pp.log1p(normal)
normal.layers["logcounts"] = normal.X.copy()

for comp, genes in COMPARTMENT_MARKERS.items():
    ok = [g for g in genes if g in normal.var_names]
    if len(ok) >= 2:
        sc.tl.score_genes(normal, ok, score_name=f"score_{comp}", use_raw=False)

score_cols = [f"score_{c}" for c in COMPARTMENT_MARKERS if f"score_{c}" in normal.obs.columns]
sm = normal.obs[score_cols].values
best_idx, best = sm.argmax(axis=1), sm.max(axis=1)
comp_raw = [score_cols[i].replace("score_", "") if s >= THRESHOLD else "Unassigned"
            for i, s in zip(best_idx, best)]
normal.obs["compartment_raw"] = comp_raw

if "PTPRC" in normal.var_names:
    x = normal[:, "PTPRC"].X
    ptprc = np.asarray(x.todense()).flatten() if issparse(x) else np.asarray(x).flatten()
else:
    ptprc = np.zeros(normal.n_obs)
imm_scores = normal.obs.get("score_Immune", pd.Series(0.0, index=normal.obs_names)).values
comp_clean = [
    "Immune" if (raw_c == "Epithelial" and (pt > PTPRC_EXCL or imm > IMMUNE_EXCL)) else raw_c
    for raw_c, imm, pt in zip(comp_raw, imm_scores, ptprc)
]
normal.obs["compartment"] = comp_clean
print(f"  Compartment distribution (Normal, tissue=Total):\n"
      f"{pd.Series(comp_clean).value_counts().to_string()}")

normal_epi = normal[normal.obs["compartment"] == "Epithelial"].copy()
normal_epi.obs["subtype"] = "Normal"
print(f"  -> Normal-tissue Epithelial: {normal_epi.n_obs:,} cells, "
      f"{normal_epi.obs['sample'].nunique()} samples")
del normal


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 -- combine with tumor-epithelial cells (compartment_epithelial.h5ad)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2 -- combining with tumor-epithelial cells (compartment_epithelial.h5ad)")
tumor_epi = sc.read_h5ad("compartment_epithelial.h5ad")
print(f"  Tumor epithelial: {tumor_epi.n_obs:,} cells x {tumor_epi.n_vars:,} genes")

shared_genes = sorted(set(tumor_epi.var_names) & set(normal_epi.var_names))
print(f"  Shared genes: {len(shared_genes):,}  |  {GENE} present: {GENE in shared_genes}")
if GENE not in shared_genes:
    raise SystemExit(f"{GENE} not found in the shared gene set -- aborting.")


def minimal(a, genes):
    b = a[:, genes].copy()
    b.obs = b.obs[["sample", "subtype"]].copy()
    b.obs["subtype"] = b.obs["subtype"].astype(str)
    return b

tumor_sub = minimal(tumor_epi, shared_genes)
normal_sub = minimal(normal_epi, shared_genes)
del tumor_epi, normal_epi

combined = ad.concat([tumor_sub, normal_sub], join="inner", label="origin",
                      keys=["tumor", "normal"], index_unique="-")
combined.obs_names_make_unique()
print(f"  Combined: {combined.n_obs:,} cells x {combined.n_vars:,} genes")
print(combined.obs["subtype"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 -- joint embedding (PCA -> neighbors -> UMAP; NO Harmony -- see docstring)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3 -- joint UMAP (no batch integration; supplementary visualization only)")
combined.X = combined.layers["logcounts"].copy()
sc.pp.highly_variable_genes(combined, n_top_genes=2000, flavor="seurat")
sc.pp.pca(combined, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(combined, n_pcs=30, n_neighbors=15)
sc.tl.umap(combined)
print("  UMAP done.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 -- epithelial lineage scoring (3 normal-lineage signatures ONLY,
#           winner-takes-all, "Others" for cells with no positive score)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4 -- epithelial lineage scoring (3 signatures, winner-takes-all)")
for ct, genes in LINEAGE_MARKERS.items():
    ok = [g for g in genes if g in combined.var_names]
    print(f"  {ct}: {len(ok)}/{len(genes)} markers present")
    if len(ok) >= 2:
        sc.tl.score_genes(combined, ok, score_name=f"score_{ct}", use_raw=False)

combined.obs["epi_lineage"] = winner_takes_all(combined.obs, LINEAGE_MARKERS, "epi_lineage")
print("\n  epi_lineage x subtype:")
print(pd.crosstab(combined.obs["epi_lineage"], combined.obs["subtype"])
      .reindex(LINEAGE_ORDER).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OBJECT -- combined tumor+normal epithelial AnnData, 3-category labels only
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving filtered object (3-category epi_lineage only)...")
gidx = combined.var_names.get_loc(GENE)
lcor_log = combined.layers["logcounts"][:, gidx]
lcor_log = np.asarray(lcor_log.todense()).ravel() if issparse(lcor_log) else np.asarray(lcor_log).ravel()
combined.obs[f"{GENE}_logcounts"] = lcor_log
counts = combined.layers["counts"][:, gidx]
counts = np.asarray(counts.todense()).ravel() if issparse(counts) else np.asarray(counts).ravel()
combined.obs[f"{GENE}_counts"] = counts

h5ad_path = f"{OUTDIR}/epithelial_lineage_3cat.h5ad"
combined.write_h5ad(h5ad_path)
print(f"  Saved: {h5ad_path}  ({combined.n_obs:,} cells x {combined.n_vars:,} genes)")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A -- UMAP by epi_lineage
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding panel A (UMAP x epi_lineage)...")
umap = combined.obsm["X_umap"]

fig, ax = plt.subplots(figsize=(7.0, 6.5))
plot_order = np.random.RandomState(0).permutation(combined.n_obs)
lineage_arr = combined.obs["epi_lineage"].values[plot_order]
for ct in LINEAGE_ORDER:
    m = lineage_arr == ct
    if not m.any():
        continue
    ax.scatter(umap[plot_order][m, 0], umap[plot_order][m, 1], s=1.2,
               color=LINEAGE_COLORS[ct], alpha=0.6,
               label=f"{ct} (n={m.sum():,})", rasterized=True)
ax.set_title("Epithelial cell population (3-category lineage)\n(Tumor + recovered Normal, joint UMAP)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, markerscale=6, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_umap_epi_lineage.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_umap_epi_lineage.png")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B -- UMAP by LCOR expression
# ══════════════════════════════════════════════════════════════════════════════
print("Building panel B (UMAP x LCOR)...")
fig, ax = plt.subplots(figsize=(6.5, 6.5))
order = np.argsort(lcor_log[plot_order])
idx_ord = plot_order[order]
sca = ax.scatter(umap[idx_ord, 0], umap[idx_ord, 1], s=1.2, c=lcor_log[idx_ord],
                  cmap="Purples", alpha=0.7, rasterized=True)
plt.colorbar(sca, ax=ax, shrink=0.8, label="logcounts")
ax.set_title(f"{GENE} expression\n(detected in {100*(lcor_log>0).mean():.1f}% of cells)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_umap_LCOR.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_umap_LCOR.png")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C -- dotplot: LCOR x epi_lineage x subtype
# ══════════════════════════════════════════════════════════════════════════════
print("Building panel C (dotplot)...")
rows = []
for st in SUBTYPES:
    d = combined.obs[combined.obs["subtype"] == st]
    for lin in LINEAGE_ORDER:
        dg = d[d["epi_lineage"] == lin]
        n = len(dg)
        pct = 100 * (dg[f"{GENE}_counts"] > 0).mean() if n > 0 else np.nan
        meanlog = dg[f"{GENE}_logcounts"].mean() if n > 0 else np.nan
        rows.append({"subtype": st, "epi_lineage": lin, "n_cells": n,
                     "pct_detected": pct, "mean_logcounts": meanlog})
dot_df = pd.DataFrame(rows)
dot_df.to_csv(f"{OUTDIR}/dotplot_LCOR_summary.csv", index=False)
print(f"  Saved: {OUTDIR}/dotplot_LCOR_summary.csv")

vmax = np.nanmax(dot_df["mean_logcounts"].values)
norm = Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.plasma


def size_for(pct):
    return 20 + 6.5 * pct

fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(LINEAGE_ORDER) + 1.8))
for xi, st in enumerate(SUBTYPES):
    for yi, lin in enumerate(LINEAGE_ORDER):
        row = dot_df[(dot_df.subtype == st) & (dot_df.epi_lineage == lin)]
        if row.empty or row["n_cells"].values[0] == 0:
            continue
        pct = row["pct_detected"].values[0]
        meanlog = row["mean_logcounts"].values[0]
        ax.scatter(xi, yi, s=size_for(pct), c=[cmap(norm(meanlog))],
                   edgecolor="black", linewidth=0.4, zorder=3)

ax.set_xticks(range(len(SUBTYPES)))
ax.set_xticklabels(SUBTYPES, fontsize=9)
for xi, st in enumerate(SUBTYPES):
    ax.get_xticklabels()[xi].set_color(SUBTYPE_COLORS[st])
    ax.get_xticklabels()[xi].set_fontweight("bold")
ax.set_yticks(range(len(LINEAGE_ORDER)))
ax.set_yticklabels(LINEAGE_ORDER, fontsize=8)
ax.set_xlim(-0.6, len(SUBTYPES) - 0.4)
ax.set_ylim(-0.6, len(LINEAGE_ORDER) - 0.4)
ax.invert_yaxis()
ax.set_axisbelow(True)
ax.grid(axis="both", color="#EAEAEA", linewidth=0.7, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
ax.set_title(f"{GENE} expression by epithelial lineage x subtype", fontsize=10.5, fontweight="bold")

cbar_ax = fig.add_axes([0.95, 0.30, 0.02, 0.40])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label(f"{GENE} mean logcounts", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)

legend_ax = fig.add_axes([0.97, 0.74, 0.12, 0.18])
legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis("off")
legend_ax.set_title("% detected", fontsize=7, pad=3)
for i, pv in enumerate([10, 50, 90]):
    y = 0.78 - i * 0.42
    legend_ax.scatter([0.25], [y], s=size_for(pv), c="#999999",
                       edgecolors="black", linewidths=0.4, clip_on=False)
    legend_ax.text(0.55, y, f"{pv}%", fontsize=6.5, va="center")

plt.subplots_adjust(right=0.88)
fig.savefig(f"{OUTDIR}/fig_dotplot_LCOR.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_dotplot_LCOR.png")


# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
readme = [
    "20b -- LCOR by 3-category epithelial lineage x subtype",
    "="*70,
    "",
    "Simplified re-take on 20_lcor_epithelial_lineage_normal.py: classifies",
    "epithelial cells (tumor + recovered Normal) using ONLY the 3 normal",
    "mammary lineage signatures -- Basal, Luminal_progenitor, Mature_luminal.",
    "The precomputed cancer molecular-subtype scheme (Luminal_A, Luminal_B,",
    "HER2_sig, Basal_TNBC, Luminal_prog, Cycling_epi, EMT -- from",
    "03_split_integrate.py's FINE_EPI) is discarded entirely, not just hidden.",
    "Cells with no positive score on any of the 3 lineage signatures are",
    "labeled 'Others' (kept, not dropped).",
    "",
    "NORMAL RECOVERY",
    "-"*70,
    f"  Subtype=='Normal', tissue=='Total' (matches how tumor samples were",
    f"  filtered), after QC + compartment scoring: {normal_sub.n_obs:,} Epithelial cells.",
    "  Same QC thresholds/marker genes as 02_qc_filter_normalize.py /",
    "  03_split_integrate.py (MIN_GENES=200, MIN_COUNTS=500, MAX_MT=30%,",
    "  MAX_GENES=p99, MIN_CELLS=3; compartment scoring threshold=0.1, PTPRC",
    "  exclusion >0.5, immune-score exclusion >0.2) -- applied to isolate the",
    "  Epithelial compartment within Normal samples specifically.",
    "",
    "EPITHELIAL LINEAGE (dotplot rows / UMAP color in panel A)",
    "-"*70,
    "  Basal:              KRT5, KRT14, TP63, ACTA2, MYLK, OXTR, KRT17",
    "  Luminal_progenitor: KIT, ELF5, ALDH1A3, KRT8, KRT18, SLPI",
    "  Mature_luminal:     ESR1, PGR, FOXA1, GATA3, ANKRD30A, AR",
    "  Others: every score <= 0 (winner-takes-all across the 3 above only).",
    "",
    "EMBEDDING CAVEAT",
    "-"*70,
    "  Joint UMAP (tumor-epithelial + normal-epithelial, on their shared",
    "  gene set) uses plain PCA -> neighbors -> UMAP, WITHOUT Harmony batch",
    "  integration (unlike the rest of the project). Supplementary",
    "  visualization only -- the LCOR dotplot does not depend on it.",
    "",
    "OUTPUTS",
    "-"*70,
    "  epithelial_lineage_3cat.h5ad -- combined tumor+normal epithelial object,",
    "    obs['epi_lineage'] in {Basal, Luminal_progenitor, Mature_luminal, Others}",
    "  fig_umap_epi_lineage.png, fig_umap_LCOR.png, fig_dotplot_LCOR.png",
    "  dotplot_LCOR_summary.csv (subtype, epi_lineage, n_cells, pct_detected,",
    "  mean_logcounts)",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Output: {OUTDIR}/
============================================================
""")
