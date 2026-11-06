"""
ROS_TME_analysis.py
===================
Analysis of GO:0072593 (reactive oxygen species metabolic process)
across all 3 TME compartments (Epithelial, Immune, Stromal)
stratified by breast cancer subtype.

GO:0072593:
  "The chemical reactions and pathways involving reactive oxygen species —
   molecules formed by incomplete one-electron reduction of oxygen.
   They contribute to microbicidal activity of phagocytes, signal
   transduction, gene expression, and oxidative damage to biopolymers."

Gene families:
  NOX_complex  NADPH oxidases (ROS production)
  SOD          Superoxide dismutases (ROS scavenging)
  Catalase     CAT
  GPX          Glutathione peroxidases
  PRDX         Peroxiredoxins
  Thioredoxin  TXN/TXNRD system
  Glutathione  GSR/GST/GPX system
  NOS          Nitric oxide synthases — scored only if expressed (>1% cells)
  Other_ROS    NRF2, HMOX1, MPO, XDH, metallothioneins...

Fixes vs previous version:
  - score_and_znorm() skips families with <1% expressing cells (fixes NOS)
  - Violin: stat bars positioned relative to data range (no overflow)
  - Violin: y-axis clipped to p1-p99 + headroom
  - Boxplot: proper grouped layout with colored subtype labels
  - Family violins: NOS excluded if not expressed; y-axis clipped per panel

Inputs (auto-detects scored or compartment objects):
  [scored_|compartment_]epithelial.h5ad
  [scored_|compartment_]immune.h5ad
  [scored_|compartment_]stromal.h5ad

Outputs: figures/ROS_TME/

Run from: ~/single_cell/GO0072593/
  python3 ROS_TME_analysis.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

# ── Paths ──────────────────────────────────────────────────────────────────
FILES = {
    "Epithelial": "compartment_epithelial.h5ad",
    "Immune":     "compartment_immune.h5ad",
    "Stromal":    "compartment_stromal.h5ad",
}
OUTDIR      = "figures/06_ROS"
SUBTYPE_COL = "subtype"
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
}

# ══════════════════════════════════════════════════════════════════════════════
# GO:0072593 GENE FAMILIES
# ══════════════════════════════════════════════════════════════════════════════
GO_0072593_FAMILIES = {
    "NOX_complex": [
        "NOX1","NOX2","NOX3","NOX4","NOX5",
        "CYBA","CYBB",
        "NCF1","NCF2","NCF4",
        "RAC1","RAC2",
        "DUOX1","DUOX2","DUOXA1","DUOXA2",
    ],
    "SOD": ["SOD1","SOD2","SOD3"],
    "Catalase": ["CAT"],
    "GPX": ["GPX1","GPX2","GPX3","GPX4","GPX5","GPX6","GPX7","GPX8"],
    "PRDX": ["PRDX1","PRDX2","PRDX3","PRDX4","PRDX5","PRDX6"],
    "Thioredoxin": [
        "TXN","TXN2","TXNRD1","TXNRD2","TXNRD3",
        "GLRX","GLRX2","GLRX3",
    ],
    "Glutathione": [
        "GSR","GSTM1","GSTM2","GSTM3","GSTM4","GSTM5",
        "GSTP1","GSTO1","GSTO2","GSTT1","GSTT2",
        "GGT1","GGT5",
    ],
    "NOS": ["NOS1","NOS2","NOS3"],
    "Other_ROS": [
        "MPO","XDH","HMOX1","HMOX2","NQO1","SRXN1",
        "PARK7","SIRT3","NFE2L2","KEAP1",
        "SELENOP","SELENBP1","MT1A","MT1B","MT2A",
    ],
}

GO_0072593 = list(dict.fromkeys(
    [g for genes in GO_0072593_FAMILIES.values() for g in genes]
))

print(f"GO:0072593 — {len(GO_0072593)} genes across "
      f"{len(GO_0072593_FAMILIES)} families")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def to_dense(x):
    return np.asarray(x.todense()).flatten() if issparse(x) \
           else np.asarray(x).flatten()


def score_and_znorm(adata, gene_list, col, min_pct_cells=0.01):
    """
    Score a gene signature and add z-normalized version.
    Skips scoring if fewer than min_pct_cells (default 1%) of cells
    express any gene in the list — avoids meaningless negative scores
    for gene sets absent in a compartment (e.g. NOS in epithelial).
    Returns (success, genes_present).
    """
    present = [g for g in gene_list if g in adata.var_names]
    if len(present) < 3:
        return False, present

    # Expression check: what fraction of cells express ≥1 gene?
    x = adata[:, present].X
    n_expr = (x > 0).sum(axis=1)
    if issparse(x):
        n_expr = np.asarray(n_expr).flatten()
    pct = (n_expr > 0).mean()

    if pct < min_pct_cells:
        print(f"    ⚠️  {col}: only {pct:.1%} cells express genes — skipping")
        return False, present

    sc.tl.score_genes(adata, gene_list=present,
                      score_name=col, use_raw=False)
    raw = adata.obs[col].values.astype(float)
    # Z-score: for heatmaps and cross-compartment boxplot
    adata.obs[f"{col}_z"] = (raw - raw.mean()) / (raw.std() + 1e-8)
    # Percentile rank 0-100 within compartment: for violins
    # Always positive and intuitive: "cell is at Xth percentile of ROS activity"
    from scipy.stats import rankdata
    adata.obs[f"{col}_pct"] = rankdata(raw) / len(raw) * 100
    return True, present


def get_mean_expr(adata, genes, groupby):
    """Mean log-expression of genes per group."""
    genes_ok = [g for g in genes if g in adata.var_names]
    if not genes_ok:
        return pd.DataFrame()
    rows = []
    for grp in adata.obs[groupby].unique():
        mask = adata.obs[groupby] == grp
        x    = adata[mask, genes_ok].X
        mean = np.asarray(x.mean(axis=0)).flatten() if issparse(x) \
               else np.asarray(x).mean(axis=0)
        rows.append(pd.Series(mean, index=genes_ok, name=grp))
    return pd.DataFrame(rows)


def violin_by_subtype(ax, data_dict, subtypes, title, comp_name,
                      stat_pairs=None):
    """
    Draw a violin plot with proper layout.
    data_dict: {subtype: array}
    stat_pairs: list of (a, b) tuples for Mann-Whitney annotation
    """
    present  = [st for st in subtypes if st in data_dict and
                len(data_dict[st]) > 0]
    data_arr = [data_dict[st] for st in present]
    if not data_arr:
        return

    parts = ax.violinplot(data_arr, showmedians=True, showextrema=False,
                          positions=range(len(present)))
    for pc, st in zip(parts["bodies"], present):
        pc.set_facecolor(SUBTYPE_COLORS.get(st, "#AAAAAA"))
        pc.set_alpha(0.75)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.5)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
    ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.4,
               label="50th percentile")
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=COMP_COLORS.get(comp_name, "black"))

    # Y limits based on data (clip outliers)
    all_vals  = np.concatenate(data_arr)
    y_p1      = np.percentile(all_vals, 1)
    y_p99     = np.percentile(all_vals, 99)
    y_range   = y_p99 - y_p1

    # Statistical annotations — positioned relative to data range
    bar_start = y_p99 + y_range * 0.10
    bar_step  = y_range * 0.13
    n_drawn   = 0

    if stat_pairs:
        for a, b in stat_pairs:
            if a not in present or b not in present:
                continue
            va = data_dict[a]
            vb = data_dict[b]
            _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            stars = ("***" if p < 0.001 else "**" if p < 0.01
                     else "*" if p < 0.05 else "ns")
            xa, xb = present.index(a), present.index(b)
            yline  = bar_start + n_drawn * bar_step
            tick   = bar_step * 0.12
            ax.plot([xa, xb], [yline, yline], color="black", lw=1)
            ax.plot([xa, xa], [yline - tick, yline], color="black", lw=1)
            ax.plot([xb, xb], [yline - tick, yline], color="black", lw=1)
            ax.text((xa + xb) / 2, yline + bar_step * 0.05, stars,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            n_drawn += 1

    y_top = bar_start + n_drawn * bar_step + y_range * 0.15
    ax.set_ylim(y_p1 - y_range * 0.08, y_top)
    ax.set_xlim(-0.6, len(present) - 0.4)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 1. LOAD AND SCORE")
print("="*65)

adatas      = {}
genes_found = {}
scored_fams = {}   # which families were successfully scored per compartment

for comp_name, fpath in FILES.items():
    print(f"\n  {comp_name}: {fpath}")
    ad = sc.read_h5ad(fpath)
    if "logcounts" in ad.layers:
        ad.X = ad.layers["logcounts"].copy()

    # Full GO signature
    ok, present = score_and_znorm(ad, GO_0072593, "score_ROS")
    genes_found[comp_name] = present
    pct = 100 * len(present) / len(GO_0072593)
    print(f"    Full signature: {len(present)}/{len(GO_0072593)} genes "
          f"({pct:.0f}%)")

    # Per-family scores
    fams_ok = []
    print(f"    Families:")
    for fam, genes in GO_0072593_FAMILIES.items():
        ok_fam, _ = score_and_znorm(ad, genes, f"score_ROS_{fam}")
        if ok_fam:
            fams_ok.append(fam)
            print(f"      ✓ {fam}")
    scored_fams[comp_name] = fams_ok

    adatas[comp_name] = ad
    print(f"    Cells: {ad.n_obs:,}")

subtypes_order = sorted(
    set().union(*[set(ad.obs[SUBTYPE_COL].unique())
                  for ad in adatas.values()])
)
print(f"\n  Subtypes: {subtypes_order}")

STAT_PAIRS = [("TNBC","ER"), ("TNBC","HER2"), ("HER2","ER")]


# ══════════════════════════════════════════════════════════════════════════════
# 2. UMAP PER COMPARTMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 2. UMAPs")
print("="*65)

# Overview: ROS score + subtype for all 3 compartments
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("GO:0072593 ROS metabolic process — UMAP per compartment",
             fontsize=13, fontweight="bold")

for col_idx, (comp_name, ad) in enumerate(adatas.items()):
    coords   = ad.obsm["X_umap"]
    subtypes = ad.obs[SUBTYPE_COL].astype(str)
    vals     = ad.obs["score_ROS"].values

    ax = axes[0][col_idx]
    sc_p = ax.scatter(coords[:,0], coords[:,1], c=vals,
                      s=0.4, alpha=0.3, cmap="YlOrRd",
                      vmin=np.percentile(vals, 2),
                      vmax=np.percentile(vals, 98),
                      rasterized=True)
    plt.colorbar(sc_p, ax=ax, label="ROS score", shrink=0.8)
    ax.set_title(f"{comp_name} — ROS score", fontsize=10,
                 fontweight="bold", color=COMP_COLORS[comp_name])
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1][col_idx]
    for subtype, color in SUBTYPE_COLORS.items():
        mask = subtypes == subtype
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=0.3, alpha=0.4, rasterized=True)
    handles = [mpatches.Patch(color=c, label=l)
               for l, c in SUBTYPE_COLORS.items()
               if l in subtypes.values]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_title(f"{comp_name} — Tumor subtype", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUTDIR}/umap_ROS_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ umap_ROS_overview.png")

# Per-compartment family UMAPs
for comp_name, ad in adatas.items():
    coords    = ad.obsm["X_umap"]
    fam_cols  = [f"score_ROS_{f}" for f in scored_fams[comp_name]
                 if f"score_ROS_{f}" in ad.obs.columns]
    all_cols  = ["score_ROS"] + fam_cols

    nc = 4; nr = (len(all_cols) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axes = np.array(axes).flatten()
    fig.suptitle(f"GO:0072593 family scores — {comp_name}",
                 fontsize=12, fontweight="bold")

    for i, col in enumerate(all_cols):
        ax   = axes[i]
        vals = ad.obs[col].values
        sc_p = ax.scatter(coords[:,0], coords[:,1], c=vals,
                          s=0.4, alpha=0.3, cmap="YlOrRd",
                          vmin=np.percentile(vals, 2),
                          vmax=np.percentile(vals, 98),
                          rasterized=True)
        plt.colorbar(sc_p, ax=ax, shrink=0.7)
        label = col.replace("score_ROS_","").replace("score_ROS","ALL")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(all_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/umap_ROS_families_{comp_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ umap_ROS_families_{comp_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. VIOLIN: ROS SCORE BY SUBTYPE — all compartments
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 3. VIOLIN — ROS BY SUBTYPE")
print("="*65)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    "GO:0072593 ROS score by tumor subtype — all compartments\n"
    "(percentile rank within compartment; dashed line = 50th percentile)",
    fontsize=12, fontweight="bold", y=1.02
)

for ax, (comp_name, ad) in zip(axes, adatas.items()):
    data_dict = {
        st: ad.obs.loc[ad.obs[SUBTYPE_COL] == st, "score_ROS_pct"].dropna().values
        for st in subtypes_order
        if st in ad.obs[SUBTYPE_COL].values
    }
    violin_by_subtype(ax, data_dict, subtypes_order,
                      comp_name, comp_name, stat_pairs=STAT_PAIRS)
    ax.set_ylabel("ROS score (percentile rank)", fontsize=10)
    ax.set_ylim(0, None)  # percentile rank is always 0-100

plt.tight_layout()
plt.savefig(f"{OUTDIR}/violin_ROS_by_subtype.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ violin_ROS_by_subtype.png")

# Per-family violins — Epithelial (one panel per scored family)
ad_epi    = adatas["Epithelial"]
fam_cols  = [(f, f"score_ROS_{f}_z") for f in scored_fams["Epithelial"]
             if f"score_ROS_{f}_z" in ad_epi.obs.columns]

if fam_cols:
    nc = 4; nr = (len(fam_cols) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axes = np.array(axes).flatten()
    fig.suptitle(
        "GO:0072593 family scores by subtype — Epithelial\n"
        "(percentile rank within compartment; families with <1% expressing cells excluded)",
        fontsize=12, fontweight="bold", y=1.01
    )
    epi_subtypes = [st for st in subtypes_order
                    if st in ad_epi.obs[SUBTYPE_COL].values]

    for i, (fam_name, col) in enumerate(fam_cols):
        ax = axes[i]
        col_pct = col.replace("_z", "_pct")
        data_dict = {
            st: ad_epi.obs.loc[ad_epi.obs[SUBTYPE_COL] == st, col_pct]
                           .dropna().values
            for st in epi_subtypes
        }
        violin_by_subtype(ax, data_dict, epi_subtypes,
                          fam_name, "Epithelial")
        ax.set_ylabel("Percentile rank" if i % nc == 0 else "")
        ax.set_ylim(0, None)

    for ax in axes[len(fam_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/violin_ROS_families_Epithelial.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ violin_ROS_families_Epithelial.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HEATMAP: MEAN EXPRESSION PER SUBTYPE × COMPARTMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 4. HEATMAPS")
print("="*65)

all_genes_present = sorted(set().union(
    *[set(genes_found[c]) for c in genes_found]
))

# Per-compartment heatmap
for comp_name, ad in adatas.items():
    df = get_mean_expr(ad, all_genes_present, SUBTYPE_COL)
    if df.empty:
        continue
    genes_here    = [g for g in all_genes_present if g in df.columns]
    subtypes_here = [st for st in subtypes_order if st in df.index]
    sub_df        = df.loc[subtypes_here, genes_here]
    sub_z         = sub_df.apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0
    )

    fig, ax = plt.subplots(figsize=(max(14, len(genes_here)*0.3), 5))
    sns.heatmap(sub_z, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "z-score"})

    for label_obj in ax.get_yticklabels():
        txt = label_obj.get_text()
        label_obj.set_color(SUBTYPE_COLORS.get(txt, "black"))
        label_obj.set_fontweight("bold")

    ax.set_title(f"GO:0072593 gene expression by subtype — {comp_name}",
                 fontsize=11, fontweight="bold",
                 color=COMP_COLORS[comp_name])
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/heatmap_ROS_genes_{comp_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ heatmap_ROS_genes_{comp_name}.png")

# Combined heatmap: all compartments × subtypes
rows, row_labels = [], []
for comp_name, ad in adatas.items():
    df = get_mean_expr(ad, all_genes_present, SUBTYPE_COL)
    if df.empty:
        continue
    genes_here = [g for g in all_genes_present if g in df.columns]
    for subtype in subtypes_order:
        if subtype in df.index:
            rows.append(df.loc[subtype, genes_here])
            row_labels.append(f"{comp_name} — {subtype}")

if rows:
    combined   = pd.DataFrame(rows, index=row_labels)
    combined_z = combined.apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1
    )
    fig, ax = plt.subplots(
        figsize=(max(16, len(combined_z.columns)*0.28),
                 max(10, len(row_labels)*0.45))
    )
    sns.heatmap(combined_z, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Mean expression (z-score)"})
    ax.set_title(
        "GO:0072593 — Mean expression per subtype × compartment\n"
        "(row z-score normalized)",
        fontsize=12, fontweight="bold"
    )
    for label_obj in ax.get_yticklabels():
        txt = label_obj.get_text()
        for comp, col in COMP_COLORS.items():
            if comp in txt:
                label_obj.set_color(col)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/heatmap_ROS_combined.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ heatmap_ROS_combined.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCATTER: ROS vs IMMUNE EVASION (epithelial)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 5. ROS vs IMMUNE EVASION — EPITHELIAL")
print("="*65)

if "score_immunoevasion_z" in ad_epi.obs.columns:
    st_epi = sorted(ad_epi.obs[SUBTYPE_COL].unique())
    fig, axes = plt.subplots(1, len(st_epi), figsize=(4.5*len(st_epi), 5))
    axes = np.array(axes).flatten()
    fig.suptitle(
        "GO:0072593 ROS score vs Immune Evasion — Epithelial, by subtype",
        fontsize=12, fontweight="bold"
    )
    for ax, subtype in zip(axes, st_epi):
        mask = ad_epi.obs[SUBTYPE_COL] == subtype
        sub  = ad_epi.obs[mask]
        if len(sub) < 20:
            ax.set_visible(False); continue
        x = sub["score_ROS_z"].values
        y = sub["score_immunoevasion_z"].values
        ax.scatter(x, y, c=SUBTYPE_COLORS.get(subtype,"#AAAAAA"),
                   s=0.5, alpha=0.3, rasterized=True)
        r, p = stats.pearsonr(x, y)
        pstr = f"p={p:.1e}" if p >= 1e-300 else "p<1e-300"
        ax.set_title(f"{subtype}\n(n={mask.sum():,})", fontsize=9,
                     fontweight="bold",
                     color=SUBTYPE_COLORS.get(subtype,"black"))
        ax.text(0.05, 0.95, f"r={r:.3f}\n{pstr}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.set_xlabel("ROS score (z)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Immune evasion (z)", fontsize=9)
        ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.4)
        ax.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/scatter_ROS_vs_immunoevasion.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ scatter_ROS_vs_immunoevasion.png")
else:
    print("  ⚠️  score_immunoevasion_z not found — run 04_cell_state_scoring.py first")


# ══════════════════════════════════════════════════════════════════════════════
# 6. BOXPLOT: CROSS-COMPARTMENT × SUBTYPE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" 6. BOXPLOT CROSS-COMPARTMENT")
print("="*65)

dfs = []
for comp_name, ad in adatas.items():
    obs = ad.obs[[SUBTYPE_COL, "score_ROS_z"]].copy()
    obs["compartment"] = comp_name
    dfs.append(obs)
df_combined = pd.concat(dfs, ignore_index=True)

comp_order   = ["Epithelial","Immune","Stromal"]
group_width  = 1.3
comp_offsets = {c: (i - 1) * 0.38 for i, c in enumerate(comp_order)}

fig, ax = plt.subplots(figsize=(16, 6))

xtick_pos, xtick_lab = [], []

for st_idx, subtype in enumerate(subtypes_order):
    group_center = st_idx * group_width
    xtick_pos.append(group_center)
    xtick_lab.append(subtype)

    for comp_name in comp_order:
        x    = group_center + comp_offsets[comp_name]
        data = df_combined.loc[
            (df_combined["compartment"] == comp_name) &
            (df_combined[SUBTYPE_COL]   == subtype),
            "score_ROS_z"
        ].dropna().values

        if len(data) == 0:
            continue

        bp = ax.boxplot(
            data, positions=[x], widths=0.28,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1), capprops=dict(lw=1),
            boxprops=dict(linewidth=0.8)
        )
        bp["boxes"][0].set_facecolor(COMP_COLORS[comp_name])
        bp["boxes"][0].set_alpha(0.75)

    # Light alternating background
    if st_idx % 2 == 0:
        ax.axvspan(group_center - group_width*0.46,
                   group_center + group_width*0.46,
                   alpha=0.04, color="gray", zorder=0)

# Vertical separators
for st_idx in range(len(subtypes_order) - 1):
    ax.axvline((st_idx + 0.5) * group_width + 0.05,
               color="#DDDDDD", lw=1, ls="--", zorder=0)

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lab, fontsize=11, fontweight="bold", rotation=0)
for label, subtype in zip(ax.get_xticklabels(), subtypes_order):
    label.set_color(SUBTYPE_COLORS.get(subtype, "black"))

ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=0)
ax.set_ylabel("ROS score (z-score)", fontsize=11)
ax.set_title(
    "GO:0072593 ROS metabolic process — all compartments × subtypes",
    fontsize=12, fontweight="bold", pad=12
)
ax.set_xlim(-0.75, (len(subtypes_order) - 1) * group_width + 0.75)

legend_handles = [mpatches.Patch(color=c, label=l, alpha=0.75)
                  for l, c in COMP_COLORS.items()]
ax.legend(handles=legend_handles, fontsize=10, loc="upper right",
          framealpha=0.9, title="Compartment", title_fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/boxplot_ROS_cross_compartment.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ boxplot_ROS_cross_compartment.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
══════════════════════════════════════════════════════════════
 SUMMARY — GO:0072593 ROS METABOLIC PROCESS
══════════════════════════════════════════════════════════════
  Signature: {len(GO_0072593)} genes, {len(GO_0072593_FAMILIES)} families
  Genes found per compartment:
{chr(10).join(f"    {c}: {len(genes_found[c])}" for c in genes_found)}
  Families scored per compartment:
{chr(10).join(f"    {c}: {scored_fams[c]}" for c in scored_fams)}

  Figures in {OUTDIR}/:
    umap_ROS_overview.png              ROS score + subtype, all compartments
    umap_ROS_families_*.png            per-family UMAPs
    violin_ROS_by_subtype.png          main violin (fixed layout)
    violin_ROS_families_Epithelial.png per-family violins (NOS excluded if absent)
    heatmap_ROS_genes_*.png            per-compartment gene heatmaps
    heatmap_ROS_combined.png           all compartments × subtypes
    scatter_ROS_vs_immunoevasion.png   ROS vs immune evasion (epithelial)
    boxplot_ROS_cross_compartment.png  main boxplot (fixed layout)
══════════════════════════════════════════════════════════════
""")
