"""
K01_prepare.py
==============
Prepare the Klughammer metastatic breast cancer dataset and produce
16 integrated objects in a 4 × 4 design:

  Subtype     ×  Compartment
  ─────────────────────────────────────────────────────
  Global      ×  All / Malignant / Lymphoid / Myeloid / Stromal
  ER          ×  All / Malignant / Lymphoid / Myeloid / Stromal
  HER2        ×  All / Malignant / Lymphoid / Myeloid / Stromal
  TNBC        ×  All / Malignant / Lymphoid / Myeloid / Stromal

Steps:
  1. Convert Ensembl IDs → Gene Symbols (feature_name column)
  2. Map receptors_biopsy → subtype (ER / HER2 / TNBC)
     - HER2+ (any) → HER2
     - ER+/HER2-   → ER
     - ER-/PR-/HER2- → TNBC
  3. Filter to ER / HER2 / TNBC only
  4. 4×4 Harmony integration:
     - HVGs + PCA + Harmony (batch = sampleid)
     - UMAP + Leiden per object
  5. Save 20 objects + overview figures

Notes:
  - X already log-normalized (no re-normalization)
  - No raw counts — CytoTRACE2 handled in K02
  - Cell cycle already in obs (S.Score, G2M.Score, Phase)
  - Compartments already annotated (Malignant/Lymphoid/Myeloid/Stromal)

Input:  data/dataset.h5ad
Output: klughammer_*.h5ad (20 objects) + figures/K01_prepare/

Run from: ~/single_cell/Klughammer/
  python3 K01_prepare.py
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

FILE       = "data/dataset.h5ad"
OUTDIR     = "figures/K01_prepare"
SAMPLE_COL = "sampleid"
SUBTYPE_COL= "subtype"
SUBTYPES   = ["ER", "HER2", "TNBC"]
COMPARTMENTS = ["Malignant", "Lymphoid", "Myeloid", "Stromal"]
os.makedirs(OUTDIR, exist_ok=True)

SUBTYPE_COLORS = {
    "ER"  : "#4DAF4A",
    "HER2": "#984EA3",
    "TNBC": "#E41A1C",
}
COMP_COLORS = {
    "Malignant": "#E41A1C",
    "Lymphoid" : "#377EB8",
    "Myeloid"  : "#FF7F00",
    "Stromal"  : "#4DAF4A",
}

# Integration parameters per compartment
PARAMS = {
    "All"      : {"n_hvg": 3000, "n_pcs": 40, "res": 0.5},
    "Malignant": {"n_hvg": 2000, "n_pcs": 30, "res": 0.4},
    "Lymphoid" : {"n_hvg": 2000, "n_pcs": 25, "res": 0.6},
    "Myeloid"  : {"n_hvg": 2000, "n_pcs": 25, "res": 0.5},
    "Stromal"  : {"n_hvg": 1500, "n_pcs": 20, "res": 0.4},
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def process_and_save(ad, comp_name, subtype_label,
                     n_hvg, n_pcs, resolution, outfile):
    """HVGs → PCA → Harmony → UMAP → Leiden → save."""
    n_cells   = ad.n_obs
    n_samples = ad.obs[SAMPLE_COL].nunique()
    print(f"\n  [{subtype_label} / {comp_name}] "
          f"{n_cells:,} cells | {n_samples} samples")

    if n_cells < 50:
        print("  Too few cells — skipping"); return

    # HVGs
    sc.pp.highly_variable_genes(
        ad, n_top_genes=n_hvg, flavor="seurat",
        batch_key=SAMPLE_COL if n_samples > 1 else None,
        subset=False
    )
    print(f"  HVGs: {ad.var['highly_variable'].sum():,}")

    # Scale + PCA on HVG submatrix
    ad_hvg = ad[:, ad.var["highly_variable"]].copy()
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

    # UMAP + Leiden
    sc.pp.neighbors(ad, use_rep=use_rep, n_pcs=n_pcs, n_neighbors=15)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=resolution, key_added="leiden")
    print(f"  ✓ UMAP | Leiden: {ad.obs['leiden'].nunique()} clusters")

    # UMAP figure
    coords = ad.obsm["X_umap"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Klughammer — {subtype_label} / {comp_name}",
                 fontsize=12, fontweight="bold")

    # Leiden
    cats = ad.obs["leiden"].astype("category")
    cmap = plt.cm.get_cmap("tab20", cats.nunique())
    axes[0].scatter(coords[:,0], coords[:,1], c=cats.cat.codes,
                    s=0.4, alpha=0.4, cmap=cmap, rasterized=True)
    handles = [mpatches.Patch(color=cmap(j/cats.nunique()), label=str(c))
               for j,c in enumerate(cats.cat.categories)]
    axes[0].legend(handles=handles, fontsize=6, ncol=2, loc="upper right")
    axes[0].set_title("Leiden")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Subtype (only meaningful for global)
    for st, color in SUBTYPE_COLORS.items():
        m = ad.obs[SUBTYPE_COL] == st
        if m.sum() > 0:
            axes[1].scatter(coords[m,0], coords[m,1], c=color,
                            s=0.3, alpha=0.4, rasterized=True)
    axes[1].legend(handles=[mpatches.Patch(color=c, label=l)
                             for l,c in SUBTYPE_COLORS.items()
                             if l in ad.obs[SUBTYPE_COL].values],
                   fontsize=8)
    axes[1].set_title("Subtype")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Biopsy site
    sites  = ad.obs["site_biopsy"].astype("category")
    cmap_s = plt.cm.get_cmap("tab10", sites.nunique())
    axes[2].scatter(coords[:,0], coords[:,1], c=sites.cat.codes,
                    s=0.3, alpha=0.3, cmap=cmap_s, rasterized=True)
    handles_s = [mpatches.Patch(color=cmap_s(j/sites.nunique()),
                                label=str(c))
                 for j,c in enumerate(sites.cat.categories)]
    axes[2].legend(handles=handles_s, fontsize=7)
    axes[2].set_title("Biopsy site")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    figpath = os.path.join(
        OUTDIR,
        f"umap_{subtype_label.lower()}_{comp_name.lower()}.png"
    )
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()

    # Save object
    ad.write_h5ad(outfile)
    print(f"  ✓ {outfile}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD")
print("="*65)
adata = sc.read_h5ad(FILE)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GENE SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. ENSEMBL → GENE SYMBOLS")
print("="*65)

if "feature_name" in adata.var.columns:
    symbols = adata.var["feature_name"].tolist()
    seen = {}
    unique_syms = []
    for i, s in enumerate(symbols):
        s = str(s).strip() if s and str(s) != "nan" \
            else adata.var_names[i]
        if s in seen:
            seen[s] += 1
            unique_syms.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            unique_syms.append(s)
    adata.var["ensembl_id"] = adata.var_names.tolist()
    adata.var["symbol"]     = unique_syms
    adata.var_names         = unique_syms
    adata.var_names_make_unique()
    adata.var[["ensembl_id","symbol"]].to_csv(
        "ensembl_to_symbol_mapping.csv")
    print(f"  ✓ Symbols updated: {list(adata.var_names[:5])}")
    print(f"  ✓ Mapping saved: ensembl_to_symbol_mapping.csv")
else:
    print("  ⚠️  feature_name not found — keeping Ensembl IDs")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAP receptors_biopsy → subtype
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 3. SUBTYPE MAPPING")
print("="*65)

def map_subtype(r):
    if pd.isna(r) or str(r) in ("nan",""):
        return None
    r = str(r)
    if "HER2+" in r:
        return "HER2"
    elif "ER+" in r and "HER2-" in r:
        return "ER"
    elif r == "ER-/PR-/HER2-":
        return "TNBC"
    return None

adata.obs[SUBTYPE_COL] = pd.Categorical(
    adata.obs["receptors_biopsy"].map(map_subtype)
)
print("  Subtype distribution:")
print(adata.obs[SUBTYPE_COL].value_counts(dropna=False).to_string())

# Filter to mapped subtypes only
n_before = adata.n_obs
adata    = adata[adata.obs[SUBTYPE_COL].notna()].copy()
print(f"\n  Kept: {adata.n_obs:,}/{n_before:,} cells")
print(f"\n  Subtype × compartment:")
print(pd.crosstab(adata.obs[SUBTYPE_COL],
                  adata.obs["compartments"]).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4. OVERVIEW UMAP
# ══════════════════════════════════════════════════════════════════════════════
if "X_umap" in adata.obsm:
    coords = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Klughammer — overview (original UMAP)",
                 fontsize=12, fontweight="bold")
    for ax, (col, cmap_dict, title) in zip(axes, [
        ("compartments", COMP_COLORS,    "Compartments"),
        (SUBTYPE_COL,    SUBTYPE_COLORS, "Subtype"),
        ("site_biopsy",  None,           "Biopsy site"),
    ]):
        if cmap_dict:
            for label, color in cmap_dict.items():
                m = adata.obs[col] == label
                if m.sum() > 0:
                    ax.scatter(coords[m,0], coords[m,1], c=color,
                               s=0.3, alpha=0.4, rasterized=True)
            ax.legend(handles=[mpatches.Patch(color=c, label=l)
                                for l,c in cmap_dict.items()
                                if l in adata.obs[col].values],
                      fontsize=8)
        else:
            cats  = adata.obs[col].astype("category")
            cmap_ = plt.cm.get_cmap("tab10", cats.nunique())
            ax.scatter(coords[:,0], coords[:,1], c=cats.cat.codes,
                       s=0.3, alpha=0.3, cmap=cmap_, rasterized=True)
            ax.legend(handles=[mpatches.Patch(
                color=cmap_(j/cats.nunique()), label=str(c))
                for j,c in enumerate(cats.cat.categories)], fontsize=7)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/umap_overview_original.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ {OUTDIR}/umap_overview_original.png")

# Save full prepared object
adata.write_h5ad("klughammer_prepared.h5ad")
print(f"  ✓ klughammer_prepared.h5ad ({adata.n_obs:,} cells)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 4 × 4 INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 5. 4 × 4 INTEGRATION")
print("="*65)

# Define all 20 jobs (4 subtypes × 5 compartments including All)
JOBS = []
for subtype_label, subtype_fn in [
    ("Global", lambda ad, s="Global": np.ones(ad.n_obs, dtype=bool)),
    ("ER",     lambda ad, s="ER":     (ad.obs[SUBTYPE_COL]==s).values),
    ("HER2",   lambda ad, s="HER2":   (ad.obs[SUBTYPE_COL]==s).values),
    ("TNBC",   lambda ad, s="TNBC":   (ad.obs[SUBTYPE_COL]==s).values),
]:
    for comp_label, comp_fn in [
        ("All",      lambda ad, c="All":      np.ones(ad.n_obs, dtype=bool)),
        ("Malignant",lambda ad, c="Malignant":(ad.obs["compartments"]==c).values),
        ("Lymphoid", lambda ad, c="Lymphoid": (ad.obs["compartments"]==c).values),
        ("Myeloid",  lambda ad, c="Myeloid":  (ad.obs["compartments"]==c).values),
        ("Stromal",  lambda ad, c="Stromal":  (ad.obs["compartments"]==c).values),
    ]:
        if comp_label == "All":
            outfile = f"klughammer_all_{subtype_label.lower()}.h5ad"
        elif subtype_label == "Global":
            outfile = f"klughammer_{comp_label}.h5ad"
        else:
            outfile = f"klughammer_{comp_label}_{subtype_label}.h5ad"

        JOBS.append({
            "subtype_label": subtype_label,
            "subtype_fn"   : subtype_fn,
            "comp_label"   : comp_label,
            "comp_fn"      : comp_fn,
            "outfile"      : outfile,
            "params"       : PARAMS[comp_label],
        })

print(f"  Total jobs: {len(JOBS)}")
for j in JOBS:
    print(f"    {j['subtype_label']:8s} × {j['comp_label']:12s}"
          f" → {j['outfile']}")

print("\n  Starting integration...")

for job in JOBS:
    sl  = job["subtype_label"]
    cl  = job["comp_label"]
    out = job["outfile"]
    p   = job["params"]

    if os.path.exists(out):
        print(f"\n  [{sl}/{cl}] {out} exists — skipping")
        continue

    mask = job["subtype_fn"](adata) & job["comp_fn"](adata)
    if mask.sum() < 50:
        print(f"\n  [{sl}/{cl}] Too few cells ({mask.sum()}) — skipping")
        continue

    ad = adata[mask].copy()
    process_and_save(ad, cl, sl,
                     n_hvg=p["n_hvg"], n_pcs=p["n_pcs"],
                     resolution=p["res"], outfile=out)
    del ad; gc.collect()


print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  Full object:
    klughammer_prepared.h5ad

  All cells:
    klughammer_all_global.h5ad
    klughammer_all_ER.h5ad / _HER2.h5ad / _TNBC.h5ad

  Malignant:
    klughammer_Malignant.h5ad
    klughammer_Malignant_ER.h5ad / _HER2.h5ad / _TNBC.h5ad

  Lymphoid:
    klughammer_Lymphoid.h5ad
    klughammer_Lymphoid_ER.h5ad / _HER2.h5ad / _TNBC.h5ad

  Myeloid:
    klughammer_Myeloid.h5ad
    klughammer_Myeloid_ER.h5ad / _HER2.h5ad / _TNBC.h5ad

  Stromal:
    klughammer_Stromal.h5ad
    klughammer_Stromal_ER.h5ad / _HER2.h5ad / _TNBC.h5ad

  Subtype mapping:
    ER   : {(adata.obs[SUBTYPE_COL]=='ER').sum():,}
    HER2 : {(adata.obs[SUBTYPE_COL]=='HER2').sum():,}
    TNBC : {(adata.obs[SUBTYPE_COL]=='TNBC').sum():,}

  Figures: {OUTDIR}/

  Next: python3 K02_cytotrace2.py
══════════════════════════════════════════════════════════════
""")
