"""
03_split_integrate.py
=====================
Produces 16 integrated objects in a 4 × 4 design:

  Subtype  ×  Compartment
  ───────────────────────────────────────────────
  Global   ×  All / Epithelial / Immune / Stromal
  ER       ×  All / Epithelial / Immune / Stromal
  HER2     ×  All / Epithelial / Immune / Stromal
  TNBC     ×  All / Epithelial / Immune / Stromal

Output files:
  # All cells (no compartment split)
  all_global.h5ad
  all_ER.h5ad  /  all_HER2.h5ad  /  all_TNBC.h5ad

  # Epithelial
  compartment_epithelial.h5ad
  epithelial_ER.h5ad  /  epithelial_HER2.h5ad  /  epithelial_TNBC.h5ad

  # Immune
  compartment_immune.h5ad
  immune_ER.h5ad  /  immune_HER2.h5ad  /  immune_TNBC.h5ad

  # Stromal
  compartment_stromal.h5ad
  stromal_ER.h5ad  /  stromal_HER2.h5ad  /  stromal_TNBC.h5ad

Compartment assignment:
  - Winner-takes-all score (threshold=0.1)
  - PTPRC hard exclusion: Epithelial with PTPRC > 0.5 → Immune
  - Immune score exclusion: Epithelial with immune score > 0.2 → Immune

Input:  GSE161529_tumor_processed.h5ad
Output: 16 .h5ad files + figures/03_split/

Run from: ~/single_cell/
  python3 03_split_integrate.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings, gc
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.n_jobs = 4

try:
    import harmonypy as hm
except ImportError:
    os.system("pip install harmonypy -q")
    import harmonypy as hm

FILE        = "GSE161529_tumor_processed.h5ad"
OUTDIR      = "figures/03_split"
SAMPLE_COL  = "sample"
SUBTYPE_COL = "subtype"
SUBTYPES    = ["ER", "HER2", "TNBC"]
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}
COMP_COLORS = {
    "Epithelial": "#E41A1C",
    "Immune":     "#377EB8",
    "Stromal":    "#4DAF4A",
    "Unassigned": "#CCCCCC",
}

# ── Compartment marker genes ──────────────────────────────────────────────────
COMPARTMENT_MARKERS = {
    "Epithelial": ["EPCAM","KRT8","KRT18","KRT19","CDH1","KRT5","KRT14"],
    "Immune":     ["PTPRC","CD3D","CD3E","CD8A","CD8B","MS4A1","CD14",
                   "LYZ","NKG7","GNLY","FCGR3A","CD68","MZB1","TPSAB1"],
    "Stromal":    ["DCN","FAP","COL1A1","COL1A2","ACTA2","PECAM1",
                   "VWF","CDH5","CLDN5","PDGFRB","RGS5"],
}
THRESHOLD   = 0.1
IMMUNE_EXCL = 0.2
PTPRC_EXCL  = 0.5

# ── Fine cell type markers ────────────────────────────────────────────────────
FINE_EPI = {
    "Luminal_A"   : ["ESR1","PGR","FOXA1","GATA3","XBP1"],
    "Luminal_B"   : ["ESR1","MKI67","TOP2A","CCNB1"],
    "HER2"        : ["ERBB2","GRB7","PGAP3"],
    "Basal_TNBC"  : ["KRT5","KRT14","KRT17","TP63","EGFR"],
    "Luminal_prog": ["KRT8","KRT18","EPCAM","CLDN3","CLDN4"],
    "Cycling_epi" : ["MKI67","TOP2A","PCNA","MCM2","CCNB1"],
    "EMT"         : ["VIM","CDH2","FN1","SNAI1","SNAI2","TWIST1","ZEB1"],
}
FINE_IMM = {
    "CD8_T"      : ["CD8A","CD8B","GZMK","GZMB","PRF1"],
    "CD4_T"      : ["CD4","IL7R","TCF7","LEF1"],
    "Treg"       : ["FOXP3","IL2RA","CTLA4","IKZF2"],
    "NK"         : ["GNLY","NKG7","KLRD1","NCAM1"],
    "Macro_M1"   : ["CD68","TNF","IL1B","CXCL10"],
    "Macro_M2"   : ["CD163","MRC1","TGFB1","IL10","CCL18"],
    "B_cell"     : ["MS4A1","CD19","CD79A"],
    "Plasmablast": ["MZB1","JCHAIN","IGHG1"],
    "Dendritic"  : ["CLEC9A","CD1C","FCER1A"],
    "Mast"       : ["TPSAB1","CPA3","KIT"],
}
FINE_STR = {
    "CAF_myo"    : ["ACTA2","TAGLN","MYL9","FAP"],
    "CAF_matrix" : ["DCN","COL1A1","COL1A2","LUM","MFAP5"],
    "Endothelial": ["PECAM1","VWF","CDH5","CLDN5"],
    "Pericyte"   : ["RGS5","PDGFRB","MCAM"],
}
FINE_ALL = {**FINE_EPI, **FINE_IMM, **FINE_STR}

FINE_MARKERS = {
    "All":        FINE_ALL,
    "Epithelial": FINE_EPI,
    "Immune":     FINE_IMM,
    "Stromal":    FINE_STR,
}

# ── Integration parameters per compartment ────────────────────────────────────
PARAMS = {
    "All":        {"n_hvg": 3000, "n_pcs": 40, "res": 0.5},
    "Epithelial": {"n_hvg": 2000, "n_pcs": 30, "res": 0.4},
    "Immune":     {"n_hvg": 2000, "n_pcs": 25, "res": 0.6},
    "Stromal":    {"n_hvg": 1500, "n_pcs": 20, "res": 0.4},
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def process_and_save(ad, comp_name, subtype_label,
                     n_hvg, n_pcs, resolution, outfile, fig_title):
    """
    Full pipeline: renormalize → HVGs → PCA → Harmony → UMAP → Leiden
    → fine scores → save h5ad + UMAP figure.
    """
    n_cells   = ad.n_obs
    n_samples = ad.obs[SAMPLE_COL].nunique()
    print(f"\n  [{subtype_label} / {comp_name}] "
          f"{n_cells:,} cells | {n_samples} samples")

    if n_cells < 50:
        print(f"  Too few cells — skipping"); return

    # Re-normalize from raw counts
    if "counts" in ad.layers:
        ad.X = ad.layers["counts"].copy()
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        ad.layers["logcounts"] = ad.X.copy()
    else:
        ad.X = ad.layers["logcounts"].copy()

    # HVGs
    sc.pp.highly_variable_genes(
        ad, n_top_genes=n_hvg, flavor="seurat",
        batch_key=SAMPLE_COL if n_samples > 1 else None,
        subset=False
    )
    print(f"  HVGs: {ad.var['highly_variable'].sum():,}")

    # Scale + PCA on HVG submatrix
    ad_hvg   = ad[:, ad.var["highly_variable"]].copy()
    ad_hvg.X = ad_hvg.layers["logcounts"].copy()
    sc.pp.scale(ad_hvg, max_value=10)
    sc.tl.pca(ad_hvg, svd_solver="arpack", n_comps=n_pcs)
    ad.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    ad.uns["pca"]    = ad_hvg.uns["pca"]
    del ad_hvg; gc.collect()

    # Harmony
    if n_samples > 1:
        ho  = hm.run_harmony(
            ad.obsm["X_pca"], ad.obs[[SAMPLE_COL]],
            SAMPLE_COL, max_iter_harmony=20,
            random_state=42, verbose=False
        )
        Z = ho.Z_corr
        if Z.shape[0] == n_pcs and Z.shape[1] == n_cells:
            Z = Z.T
        ad.obsm["X_pca_harmony"] = Z
        use_rep = "X_pca_harmony"
        print(f"  ✓ Harmony ({n_samples} samples)")
    else:
        use_rep = "X_pca"
        print(f"  Single sample — skipping Harmony")

    # Neighbors + UMAP + Leiden
    sc.pp.neighbors(ad, use_rep=use_rep, n_pcs=n_pcs, n_neighbors=15)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=resolution, key_added="leiden")
    print(f"  ✓ UMAP | Leiden: {ad.obs['leiden'].nunique()} clusters")

    # Fine cell type scores
    ad.X = ad.layers["logcounts"].copy()
    fine = FINE_MARKERS.get(comp_name, FINE_ALL)
    for ct, genes in fine.items():
        ok = [g for g in genes if g in ad.var_names]
        if len(ok) >= 2:
            sc.tl.score_genes(ad, ok, score_name=f"score_{ct}",
                              use_raw=False)

    # Save h5ad
    ad.write_h5ad(outfile)
    print(f"  ✓ {outfile}")

    # UMAP figure
    coords    = ad.obsm["X_umap"]
    sc_cols   = [c for c in ad.obs.columns if c.startswith("score_")
                 and not any(k in c for k in
                             ["Epithelial","Immune","Stromal"])]
    n_panels  = 3 + len(sc_cols)
    nc = 4; nr = (n_panels + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axes = np.array(axes).flatten()
    fig.suptitle(fig_title, fontsize=11, fontweight="bold")

    # Leiden
    cats = ad.obs["leiden"].astype("category")
    cmap = plt.cm.get_cmap("tab20", cats.nunique())
    axes[0].scatter(coords[:,0], coords[:,1], c=cats.cat.codes,
                    s=0.4, alpha=0.4, cmap=cmap, rasterized=True)
    handles = [mpatches.Patch(color=cmap(j/cats.nunique()), label=str(c))
               for j, c in enumerate(cats.cat.categories)]
    axes[0].legend(handles=handles, fontsize=6, ncol=2, loc="upper right")
    axes[0].set_title("Leiden"); axes[0].set_xticks([]); axes[0].set_yticks([])

    # Subtype (only meaningful for global objects)
    for st, color in SUBTYPE_COLORS.items():
        m = ad.obs[SUBTYPE_COL] == st
        if m.sum() > 0:
            axes[1].scatter(coords[m,0], coords[m,1], c=color,
                            s=0.3, alpha=0.4, rasterized=True)
    axes[1].legend(handles=[mpatches.Patch(color=c, label=l)
                             for l, c in SUBTYPE_COLORS.items()
                             if l in ad.obs[SUBTYPE_COL].values],
                   fontsize=8)
    axes[1].set_title("Subtype")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Sample
    cats2 = ad.obs[SAMPLE_COL].astype("category")
    cmap2 = plt.cm.get_cmap("tab20", cats2.nunique())
    axes[2].scatter(coords[:,0], coords[:,1], c=cats2.cat.codes,
                    s=0.3, alpha=0.3, cmap=cmap2, rasterized=True)
    axes[2].set_title(f"Sample ({cats2.nunique()})")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    # Score panels
    for i, col in enumerate(sc_cols[:nc*nr-3]):
        ax = axes[3+i]
        sc_p = ax.scatter(coords[:,0], coords[:,1],
                          c=ad.obs[col], s=0.4, alpha=0.3,
                          cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                          rasterized=True)
        plt.colorbar(sc_p, ax=ax, shrink=0.6)
        ax.set_title(col.replace("score_",""), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    figpath = outfile.replace(".h5ad", "_umap.png")
    figpath = os.path.join(OUTDIR, os.path.basename(figpath))
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {figpath}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD + COMPARTMENT ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD + COMPARTMENT ASSIGNMENT")
print("="*65)

adata = sc.read_h5ad(FILE)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")
print(f"  Subtypes: {dict(adata.obs[SUBTYPE_COL].value_counts())}")

if "logcounts" in adata.layers:
    adata.X = adata.layers["logcounts"].copy()

# Score compartments
for comp, genes in COMPARTMENT_MARKERS.items():
    ok = [g for g in genes if g in adata.var_names]
    print(f"  {comp}: {len(ok)}/{len(genes)} marker genes")
    if len(ok) >= 2:
        sc.tl.score_genes(adata, ok, score_name=f"score_{comp}",
                          use_raw=False)

# Winner-takes-all
score_cols   = [f"score_{c}" for c in COMPARTMENT_MARKERS
                if f"score_{c}" in adata.obs.columns]
score_matrix = adata.obs[score_cols].values
best_scores  = score_matrix.max(axis=1)
best_idx     = score_matrix.argmax(axis=1)
compartment  = [
    score_cols[i].replace("score_","") if s >= THRESHOLD else "Unassigned"
    for i, s in zip(best_idx, best_scores)
]
adata.obs["compartment_raw"] = pd.Categorical(compartment)

# PTPRC hard exclusion
if "PTPRC" in adata.var_names:
    x     = adata[:,"PTPRC"].X
    ptprc = np.asarray(x.todense()).flatten() if issparse(x) \
            else np.asarray(x).flatten()
    adata.obs["PTPRC_expr"] = ptprc
else:
    adata.obs["PTPRC_expr"] = 0.0

imm_scores = adata.obs.get("score_Immune",
                            pd.Series(0.0, index=adata.obs_names))
comp_clean = []
n_rescued  = 0
for raw, imm, ptprc in zip(adata.obs["compartment_raw"],
                            imm_scores,
                            adata.obs["PTPRC_expr"]):
    if raw == "Epithelial" and (ptprc > PTPRC_EXCL or imm > IMMUNE_EXCL):
        comp_clean.append("Immune"); n_rescued += 1
    else:
        comp_clean.append(raw)

adata.obs["compartment"] = pd.Categorical(comp_clean)
print(f"\n  Reassigned Epithelial→Immune: {n_rescued:,}")
print(f"\n  Compartment distribution:")
print(adata.obs["compartment"].value_counts().to_string())

# Compartment assignment figure
if "X_umap" in adata.obsm:
    coords = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Compartment assignment — ER / HER2 / TNBC",
                 fontsize=12, fontweight="bold")
    for ax, (col, title) in zip(axes, [
        ("compartment_raw", "Raw (score only)"),
        ("compartment",     "Filtered (+PTPRC)"),
        ("PTPRC_expr",      "PTPRC expression"),
    ]):
        if col == "PTPRC_expr":
            vals = adata.obs[col].values.astype(float)
            sc_p = ax.scatter(coords[:,0], coords[:,1], c=vals,
                              s=0.3, alpha=0.3, cmap="RdYlBu_r",
                              vmin=0, vmax=2, rasterized=True)
            plt.colorbar(sc_p, ax=ax, label="logcounts")
        else:
            for comp, color in COMP_COLORS.items():
                m = adata.obs[col] == comp
                if m.sum() > 0:
                    ax.scatter(coords[m,0], coords[m,1], c=color,
                               s=0.3, alpha=0.4, label=comp,
                               rasterized=True)
            ax.legend(fontsize=7)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/compartment_assignment.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUTDIR}/compartment_assignment.png")

adata.write_h5ad("GSE161529_tumor_compartments.h5ad")
print(f"  ✓ GSE161529_tumor_compartments.h5ad")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 4 × 4 INTEGRATION
#    Rows: Global / ER / HER2 / TNBC
#    Cols: All / Epithelial / Immune / Stromal
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. 4 × 4 INTEGRATION")
print("="*65)

# Define all 16 jobs
JOBS = []
for subtype_label, subtype_mask_fn in [
    ("Global", lambda ad: np.ones(ad.n_obs, dtype=bool)),
    ("ER",     lambda ad: (ad.obs[SUBTYPE_COL] == "ER").values),
    ("HER2",   lambda ad: (ad.obs[SUBTYPE_COL] == "HER2").values),
    ("TNBC",   lambda ad: (ad.obs[SUBTYPE_COL] == "TNBC").values),
]:
    for comp_label, comp_mask_fn in [
        ("All",        lambda ad: np.ones(ad.n_obs, dtype=bool)),
        ("Epithelial", lambda ad: (ad.obs["compartment"] == "Epithelial").values),
        ("Immune",     lambda ad: (ad.obs["compartment"] == "Immune").values),
        ("Stromal",    lambda ad: (ad.obs["compartment"] == "Stromal").values),
    ]:
        # Output filename
        if comp_label == "All":
            outfile = f"all_{subtype_label.lower()}.h5ad"
        elif subtype_label == "Global":
            outfile = f"compartment_{comp_label.lower()}.h5ad"
        else:
            outfile = f"{comp_label.lower()}_{subtype_label}.h5ad"

        JOBS.append({
            "subtype_label": subtype_label,
            "subtype_mask":  subtype_mask_fn,
            "comp_label":    comp_label,
            "comp_mask":     comp_mask_fn,
            "outfile":       outfile,
            "params":        PARAMS[comp_label],
        })

print(f"  Total jobs: {len(JOBS)}")
for j in JOBS:
    print(f"    {j['subtype_label']:8s} × {j['comp_label']:12s} → {j['outfile']}")

print("\n  Starting processing...")

for job in JOBS:
    subtype_label = job["subtype_label"]
    comp_label    = job["comp_label"]
    outfile       = job["outfile"]
    p             = job["params"]

    # Skip if already done
    if os.path.exists(outfile):
        print(f"\n  [{subtype_label}/{comp_label}] "
              f"{outfile} exists — skipping")
        continue

    # Select cells
    mask = job["subtype_mask"](adata) & job["comp_mask"](adata)
    if mask.sum() < 50:
        print(f"\n  [{subtype_label}/{comp_label}] "
              f"Too few cells ({mask.sum()}) — skipping")
        continue

    ad = adata[mask].copy()

    process_and_save(
        ad,
        comp_name=comp_label,
        subtype_label=subtype_label,
        n_hvg=p["n_hvg"],
        n_pcs=p["n_pcs"],
        resolution=p["res"],
        outfile=outfile,
        fig_title=f"UMAP — {subtype_label} / {comp_label}"
    )

    del ad; gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  16 objects (4 subtypes × 4 compartments):

  All cells:
    all_global.h5ad
    all_ER.h5ad / all_HER2.h5ad / all_TNBC.h5ad

  Epithelial:
    compartment_epithelial.h5ad
    epithelial_ER.h5ad / epithelial_HER2.h5ad / epithelial_TNBC.h5ad

  Immune:
    compartment_immune.h5ad
    immune_ER.h5ad / immune_HER2.h5ad / immune_TNBC.h5ad

  Stromal:
    compartment_stromal.h5ad
    stromal_ER.h5ad / stromal_HER2.h5ad / stromal_TNBC.h5ad

  Figures: {OUTDIR}/
    compartment_assignment.png
    *_umap.png (16 figures)

  Next: python3 cytotrace2_scoring.py
══════════════════════════════════════════════════════════════
""")
