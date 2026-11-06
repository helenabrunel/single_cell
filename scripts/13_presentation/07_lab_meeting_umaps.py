"""
presentation_umaps.py
=====================
UMAP figures for the lab meeting presentation.

For each dataset × subtype, produces a 2×3 panel figure:
  Row 1: CC phase  |  stem_consensus score  |  EMP score
  Row 2: Immune score  |  fMaSC score  |  SCIE / NSI / Other label

SCIE = top-tertile stemness × bottom-tertile immune (per subtype, using fMaSC for Visvader)
NSI  = bottom-tertile stemness × top-tertile immune

Cells from other subtypes are shown in light gray for spatial context.

Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 presentation_umaps.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import json, os, warnings
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
SUBTYPES       = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
PHASE_COLORS   = {"G1": "#CCCCCC", "S": "#F4A261", "G2M": "#E63946"}
MIN_CELLS      = 500

# Signature to use for SCIE/NSI labelling — one per dataset
# (best SCIE fold enrichment from subpop_metrics)
SCIE_SIG = {
    "Visvader":   "fMaSC_over_aMaSC",
    "Klughammer": "LIM_MAMMARY_STEM_CELL_UP",
}

IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
]
S_GENES = [
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1",
    "GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51",
    "RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1",
    "CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
]
G2M_GENES = [
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","SMC4","CCNB2","CKAP2L",
    "CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP",
    "CDCA3","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5",
    "CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR",
    "CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA",
]

SIG_SHORT = {
    "ESC_ASSOU": "ESC_ASSOU", "ESC_WONG": "ESC_WONG",
    "BENPORATH_ES1": "BENPORATH", "EMP": "EMP",
    "fMaSC_over_aMaSC": "fMaSC",
    "LIM_MAMMARY_STEM_CELL_UP": "LIM_MaSC",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
    "LIM_MAMMARY_LUMINAL_MATURE_UP": "LIM_LumMat",
    "stem_consensus": "stem_consensus",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def score_sig(adata, name, genes):
    present = [g for g in genes if g in adata.var_names]
    if len(present) >= 5:
        sc.tl.score_genes(adata, present, score_name=f"sig_{name}", use_raw=False)

def clip_scores(arr, lo=1, hi=99):
    vmin, vmax = np.nanpercentile(arr, lo), np.nanpercentile(arr, hi)
    return np.clip(arr, vmin, vmax), vmin, vmax

def umap_continuous(ax, umap_all, scores_all, mask_focus, title,
                    cmap="RdYlBu_r", s_bg=1, s_fg=2):
    """
    Plot continuous score on UMAP. Cells outside mask_focus are gray.
    Returns the ScalarMappable for colorbar.
    """
    # Background (other subtypes)
    bg = ~mask_focus
    ax.scatter(umap_all[bg, 0], umap_all[bg, 1],
               c="#EEEEEE", s=s_bg, linewidths=0, rasterized=True, zorder=1)

    scores_fg, vmin, vmax = clip_scores(scores_all[mask_focus])
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(umap_all[mask_focus, 0], umap_all[mask_focus, 1],
               c=scores_fg, cmap=cmap, norm=norm,
               s=s_fg, linewidths=0, rasterized=True, zorder=2)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=7); ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")
    return ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

def umap_categorical(ax, umap_all, labels_all, mask_focus, title,
                     color_map, s_bg=1, s_fg=2):
    """
    Plot categorical labels on UMAP. 'Other' cells within focus are semi-transparent.
    """
    bg = ~mask_focus
    ax.scatter(umap_all[bg, 0], umap_all[bg, 1],
               c="#EEEEEE", s=s_bg, linewidths=0, rasterized=True, zorder=1)

    for label, color in color_map.items():
        mask = mask_focus & (labels_all == label)
        ax.scatter(umap_all[mask, 0], umap_all[mask, 1],
                   c=color, s=s_fg, linewidths=0, rasterized=True,
                   zorder=3 if label != "Other" else 2,
                   alpha=1.0 if label != "Other" else 0.15,
                   label=label)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=7); ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

def assign_scie_nsi(stem_scores, imm_scores):
    """Global tertile-based SCIE/NSI labels."""
    t33_s = np.nanpercentile(stem_scores, 33)
    t67_s = np.nanpercentile(stem_scores, 67)
    t33_i = np.nanpercentile(imm_scores,  33)
    t67_i = np.nanpercentile(imm_scores,  67)
    labels = np.full(len(stem_scores), "Other", dtype=object)
    labels[(stem_scores >= t67_s) & (imm_scores <= t33_i)] = "SCIE"
    labels[(stem_scores <= t33_s) & (imm_scores >= t67_i)] = "NSI"
    return labels

# ── Dataset loader ────────────────────────────────────────────────────────────
def load_dataset(cfg):
    ds    = cfg["name"]
    print(f"\nLoading {ds} …")
    adata = sc.read_h5ad(cfg["h5ad"])

    # CC scores
    if not cfg["cc_in_obs"]:
        sc.tl.score_genes_cell_cycle(
            adata,
            s_genes  =[g for g in S_GENES   if g in adata.var_names],
            g2m_genes=[g for g in G2M_GENES if g in adata.var_names],
        )
        cfg["phase_col"] = "phase"
    else:
        cfg["phase_col"] = cfg.get("phase_col_name", "Phase")

    # All signatures
    with open(cfg["sigfile"]) as f:
        sigs = json.load(f)
    for name, genes in sigs.items():
        score_sig(adata, name, genes)

    # Immune
    imm = [g for g in IMMUNOGENIC if g in adata.var_names]
    sc.tl.score_genes(adata, imm, score_name="sig_immune", use_raw=False)

    # Patient filter
    counts = adata.obs[cfg["patient_col"]].value_counts()
    keep   = counts[counts >= MIN_CELLS].index
    adata  = adata[adata.obs[cfg["patient_col"]].isin(keep)].copy()
    print(f"  {adata.n_obs:,} cells after filter")

    return adata, cfg

DATASETS = {
    "Visvader": {
        "name"     : "Visvader",
        "h5ad"     : "Visvader/compartment_epithelial.h5ad",
        "sigfile"  : "Visvader/signatures_confirmed.json",
        "patient_col": "sample",
        "subtype_col": "subtype",
        "cc_in_obs"  : False,
        "outdir"   : "Visvader/figures/presentation",
    },
    "Klughammer": {
        "name"     : "Klughammer",
        "h5ad"     : "Klughammer/klughammer_Malignant.h5ad",
        "sigfile"  : "Klughammer/signatures_confirmed.json",
        "patient_col": "donor_id",
        "subtype_col": "subtype",
        "cc_in_obs"  : True,
        "phase_col_name": "Phase",
        "outdir"   : "Klughammer/figures/presentation",
    },
}

# ── Per-subtype UMAP figure ───────────────────────────────────────────────────
def make_umap_figure(adata, cfg, subtype):
    ds_name    = cfg["name"]
    sub_col    = cfg["subtype_col"]
    phase_col  = cfg["phase_col"]
    scie_sig   = SCIE_SIG[ds_name]
    scie_short = SIG_SHORT.get(scie_sig, scie_sig)
    outdir     = cfg["outdir"]

    mask_focus = (adata.obs[sub_col] == subtype).values
    n_focus    = mask_focus.sum()
    if n_focus < 100:
        print(f"  SKIP {subtype}: only {n_focus} cells")
        return

    umap = adata.obsm["X_umap"]

    # Scores for focus cells (all cells for background coordinates)
    obs = adata.obs

    def get_score(col):
        return obs[col].values.astype(float)

    stem_con  = get_score("sig_stem_consensus")
    emp       = get_score("sig_EMP")
    immune    = get_score("sig_immune")
    scie_s    = get_score(f"sig_{scie_sig}")
    phases    = obs[phase_col].values

    # SCIE/NSI labels — computed only on focus cells
    scie_labels_all = np.full(len(adata), "Other", dtype=object)
    idx_focus = np.where(mask_focus)[0]
    lbl = assign_scie_nsi(scie_s[mask_focus], immune[mask_focus])
    for i, l in zip(idx_focus, lbl):
        scie_labels_all[i] = l

    # Phase as array
    phase_labels_all = phases

    s_bg = max(0.5, min(2.0, 50000 / len(adata)))
    s_fg = max(1.0, min(4.0, 80000 / n_focus))

    fig = plt.figure(figsize=(24, 11))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.30)

    # ── Row 1, col 0: Patient ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    patients     = adata.obs[cfg["patient_col"]].values
    patient_ids  = sorted(set(patients[mask_focus]))
    n_patients   = len(patient_ids)
    pat_cmap     = plt.cm.get_cmap("tab20", max(n_patients, 20))
    pat_color    = {p: pat_cmap(i) for i, p in enumerate(patient_ids)}

    ax.scatter(umap[~mask_focus, 0], umap[~mask_focus, 1],
               c="#EEEEEE", s=s_bg, linewidths=0, rasterized=True, zorder=1)
    for pid, color in pat_color.items():
        m = mask_focus & (patients == pid)
        ax.scatter(umap[m, 0], umap[m, 1], c=[color], s=s_fg,
                   linewidths=0, rasterized=True, zorder=2)
    ax.set_title(f"Patient  (n={n_patients})", fontsize=9, fontweight="bold",
                 color=SUBTYPE_COLORS.get(subtype, "black"))
    ax.set_xlabel("UMAP 1", fontsize=7); ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6); ax.set_facecolor("white")
    ax.text(0.02, 0.02, "each colour = one patient",
            transform=ax.transAxes, fontsize=6, style="italic", color="gray")

    # ── Row 1, col 1: CC phase ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    phase_color_map = {**PHASE_COLORS, "Other": "#EEEEEE"}
    bg = ~mask_focus
    ax.scatter(umap[bg, 0], umap[bg, 1],
               c="#EEEEEE", s=s_bg, linewidths=0, rasterized=True, zorder=1)
    for phase, color in PHASE_COLORS.items():
        m = mask_focus & (phase_labels_all == phase)
        ax.scatter(umap[m, 0], umap[m, 1], c=color, s=s_fg,
                   linewidths=0, rasterized=True, zorder=2, label=phase)
    ax.set_title("Cell cycle phase", fontsize=9, fontweight="bold",
                 color=SUBTYPE_COLORS.get(subtype, "black"))
    ax.set_xlabel("UMAP 1", fontsize=7); ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6); ax.set_facecolor("white")
    leg_elems = [Line2D([0],[0], marker="o", color="w",
                         markerfacecolor=c, markersize=7, label=p)
                 for p, c in PHASE_COLORS.items()]
    ax.legend(handles=leg_elems, fontsize=7, loc="lower right",
              framealpha=0.8, markerscale=1.2)

    # ── Row 1, col 2: stem_consensus ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    sm = umap_continuous(ax, umap, stem_con, mask_focus,
                         "stem_consensus score", cmap="RdYlBu_r",
                         s_bg=s_bg, s_fg=s_fg)
    plt.colorbar(sm, ax=ax, shrink=0.7, label="Score (clipped 1–99%)")

    # ── Row 1, col 3: EMP ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    sm = umap_continuous(ax, umap, emp, mask_focus,
                         "EMP score  (CC-clean)", cmap="RdYlBu_r",
                         s_bg=s_bg, s_fg=s_fg)
    plt.colorbar(sm, ax=ax, shrink=0.7, label="Score (clipped 1–99%)")

    # ── Row 2, col 0: CC phase (repeat for alignment) ────────────────────────
    # (kept empty — use for legend or leave blank)
    fig.add_subplot(gs[1, 0]).set_visible(False)

    # ── Row 2, col 1: Immune score ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    sm = umap_continuous(ax, umap, immune, mask_focus,
                         "Immune score (APM/IFN)", cmap="RdYlBu_r",
                         s_bg=s_bg, s_fg=s_fg)
    plt.colorbar(sm, ax=ax, shrink=0.7, label="Score (clipped 1–99%)")

    # ── Row 2, col 2: fMaSC / LIM_MaSC ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    sm = umap_continuous(ax, umap, scie_s, mask_focus,
                         f"{scie_short} score\n(best SCIE enrichment for {ds_name})",
                         cmap="RdYlBu_r", s_bg=s_bg, s_fg=s_fg)
    plt.colorbar(sm, ax=ax, shrink=0.7, label="Score (clipped 1–99%)")

    # ── Row 2, col 3: SCIE / NSI labels ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    scie_color_map = {
        "SCIE" : "#1A73E8",
        "NSI"  : "#D32F2F",
        "Other": "#CCCCCC",
    }
    umap_categorical(ax, umap, scie_labels_all, mask_focus,
                     f"SCIE / NSI label\n({scie_short} × Immune, global tertiles)",
                     scie_color_map, s_bg=s_bg, s_fg=s_fg)
    n_scie = (lbl == "SCIE").sum()
    n_nsi  = (lbl == "NSI").sum()
    pct_s  = 100 * n_scie / n_focus
    pct_n  = 100 * n_nsi  / n_focus
    leg_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1A73E8",
               markersize=7, label=f"SCIE  (n={n_scie:,}, {pct_s:.1f}%)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#D32F2F",
               markersize=7, label=f"NSI   (n={n_nsi:,}, {pct_n:.1f}%)"),
    ]
    ax.legend(handles=leg_elems, fontsize=7, loc="lower right", framealpha=0.8)

    fig.suptitle(
        f"{ds_name} — {subtype} subtype  (n={n_focus:,} cells)\n"
        f"Row 1: cell cycle contamination context  ·  "
        f"Row 2: immune landscape and SCIE/NSI classification",
        fontsize=11, fontweight="bold",
        color=SUBTYPE_COLORS.get(subtype, "black"),
    )

    path = os.path.join(outdir, f"figF_umap_{ds_name}_{subtype}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ds_name, cfg in DATASETS.items():
        os.makedirs(cfg["outdir"], exist_ok=True)
        adata, cfg = load_dataset(cfg)
        for subtype in SUBTYPES:
            print(f"  Plotting {ds_name} — {subtype} …")
            make_umap_figure(adata, cfg, subtype)
        del adata
    print("\nDone.")
