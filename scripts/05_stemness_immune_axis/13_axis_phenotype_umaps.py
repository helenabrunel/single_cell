"""
22d_axis_phenotype_umaps.py
===========================
Per subtype (Harmony-integrated epithelial objects), four UMAPs:
  1) SCIE / NSI / SCI / Low annotation  (adapted annotate_phenotype)
  2) Patient ID
  3) CytoTRACE2 potency score (CytoTRACE2 package scale and palette)
  4) Immune consensus score (immunogenic_consensus, 51 genes)

annotate_phenotype adaptation
-----------------------------
The original R function computes the quantile cut-offs within each patient.
Here every subtype is one Harmony-integrated dataset, so the cut-offs are
computed once over all cells of the subtype (GROUP_COL = None). Setting
GROUP_COL = "sample" reproduces the original per-patient behaviour.
  SI   = SCIE - NSI
  SCIE : SI >= q(SI, HI_Q)
  NSI  : SI <= q(SI, LO_Q)
  SCI  : SCIE >= q(SCIE, HI_Q) and NSI >= q(NSI, HI_Q)   (overwrites SCIE/NSI)
  Low  : rest

Inputs: scores_Epithelial_{subtype}.csv (22_axis_signatures_umap.py),
        cytotrace2_scores_Epithelial_{subtype}.csv, epithelial_{subtype}.h5ad
Outputs: figures/22_axis_signatures/phenotype/ (copied to axis_based signatures/UMAPs/)
"""

import os, re, sys, shutil, gc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 8})

DATA_DIR  = "/home/hbrunel/single_cell/Visvader"
SCORE_DIR = os.path.join(DATA_DIR, "figures", "22_axis_signatures")
FIG_DIR   = os.path.join(SCORE_DIR, "phenotype")
COPY_DIR  = "/home/hbrunel/single_cell/axis_based signatures/UMAPs"

SUBTYPES  = ["ER", "HER2", "TNBC"]
EPI_H5AD  = "epithelial_{}.h5ad"

# --er-filtered: ER only, re-integrated without 11 samples (23_reintegrate_ER_filtered.py)
ER_FILTERED = "--er-filtered" in sys.argv
if ER_FILTERED:
    SUBTYPES  = ["ER"]
    EPI_H5AD  = "epithelial_{}_filtered.h5ad"
    SCORE_DIR = os.path.join(DATA_DIR, "figures", "22_axis_signatures_ER_filtered")
    FIG_DIR   = os.path.join(SCORE_DIR, "phenotype")
    COPY_DIR  = "/home/hbrunel/single_cell/axis_based signatures/UMAPs_ER_filtered"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(COPY_DIR, exist_ok=True)
HI_Q, LO_Q = 0.85, 0.15
GROUP_COL = None  # None = whole integrated subtype; "sample" = per patient (original)

LABEL_COLORS = {"SCIE": "#00B0F0", "NSI": "#FBA094", "SCI": "#AFCB4B", "Low": "#D9D9D9"}
PLOT_ORDER   = ["Low", "NSI", "SCIE", "SCI"]

# Immune consensus = immunogenic_consensus (list provided by the user)
IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA","HLA-DRB1",
    "HLA-DRB3","HLA-DRB4","HLA-DRB5","HLA-F","TAP1","TAP2","TAPBP",
    "CIITA","PSME1","PSME2","IFI30","IFNA1","IFNA2","IFNA4","IFNA5","IFNA6","IFNA7",
    "IFNA8","IFNA10","IFNA13","IFNA14","IFNA16","IFNA17","IFNA21","KIR2DS1","KIR2DS3",
    "KIR2DS4","KIR2DS5","KLRC2","KLRC3","KLRC4","CD4","CD8A","CD8B","LTA","CD74",
]


def annotate_phenotype(df, stem_col, immune_col, group_col=None, hi_q=HI_Q, lo_q=LO_Q):
    annotation = pd.Series("Low", index=df.index)
    groups = [df.index] if group_col is None else df.groupby(group_col).groups.values()
    for cells in groups:
        m = df.loc[cells]
        si = m[stem_col] - m[immune_col]
        is_scie = si >= si.quantile(hi_q)
        is_nsi  = si <= si.quantile(lo_q)
        is_sci  = (m[stem_col] >= m[stem_col].quantile(hi_q)) & \
                  (m[immune_col] >= m[immune_col].quantile(hi_q))
        annotation[is_scie[is_scie].index] = "SCIE"
        annotation[is_nsi[is_nsi].index]   = "NSI"
        annotation[is_sci[is_sci].index]   = "SCI"  # overwrites SCIE/NSI
    return pd.Categorical(annotation, categories=list(LABEL_COLORS))


def short_patient(s):
    return re.sub(r"^GSM\d+_", "", s)


def style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticks([]); ax.set_yticks([])


def plot_annotation(ax, coords, labels, title):
    lab = np.asarray(labels)
    for l in PLOT_ORDER:
        m = lab == l
        ax.scatter(coords[m, 0], coords[m, 1], c=LABEL_COLORS[l], s=2,
                   alpha=0.5 if l == "Low" else 0.85, linewidths=0, rasterized=True)
    handles = [Line2D([], [], marker="o", ls="", color=LABEL_COLORS[l], markersize=6,
                      label=f"{l} ({(lab == l).sum():,}; {100 * (lab == l).mean():.1f}%)")
               for l in LABEL_COLORS]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    style(ax, title)


def plot_patient(ax, coords, patients, title):
    cats = sorted(pd.unique(patients))
    cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(patients))  # avoid one patient covering the others
    color_of = {p: cmap(i % cmap.N) for i, p in enumerate(cats)}
    ax.scatter(coords[order, 0], coords[order, 1],
               c=[color_of[p] for p in np.asarray(patients)[order]],
               s=2, alpha=0.7, linewidths=0, rasterized=True)
    handles = [Line2D([], [], marker="o", ls="", color=color_of[p], markersize=6,
                      label=short_patient(p)) for p in cats]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="center left",
              bbox_to_anchor=(1.0, 0.5), ncol=1 if len(cats) <= 12 else 2)
    style(ax, title)


# CytoTRACE2 potency score scale (same as cytotrace2_py plot_all_outputs; see 22f)
POTENCY_LABELS = ["Differentiated", "Unipotent", "Oligopotent",
                  "Multipotent", "Pluripotent", "Totipotent"]
POTENCY_COLORS = ["#5E4FA2", "#66C2A5", "#E6F598", "#FEE08B", "#F46D43", "#9E0142"]
_centres = (2 * np.arange(6) + 1) / 12
POTENCY_CMAP = LinearSegmentedColormap.from_list(
    "potency_cols",
    [(0, POTENCY_COLORS[0])] + list(zip(_centres, POTENCY_COLORS)) + [(1, POTENCY_COLORS[-1])])


def plot_potency_score(ax, coords, score, title):
    order = np.argsort(score)
    p = ax.scatter(coords[order, 0], coords[order, 1], c=score[order], cmap=POTENCY_CMAP,
                   vmin=0, vmax=1, s=2, alpha=0.8, linewidths=0, rasterized=True)
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title("Potency score", fontsize=8, loc="left", pad=8)
    cbar.set_ticks(_centres)
    cbar.set_ticklabels(POTENCY_LABELS, fontsize=7)
    for b in np.arange(1, 6) / 6:
        cbar.ax.axhline(b, color="white", lw=0.5)
    style(ax, title)


def plot_continuous(ax, coords, values, cmap, title, label):
    vmin, vmax = np.nanpercentile(values, [1, 99])
    order = np.argsort(values)
    p = ax.scatter(coords[order, 0], coords[order, 1], c=values[order], cmap=cmap,
                   vmin=vmin, vmax=vmax, s=2, alpha=0.8, linewidths=0, rasterized=True)
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=8)
    style(ax, title)


counts, outputs = [], []

for subtype in SUBTYPES:
    print(f"=== {subtype} ===")
    a = sc.read_h5ad(os.path.join(DATA_DIR, EPI_H5AD.format(subtype)))
    a.X = a.layers["logcounts"]
    imm = [g for g in IMMUNOGENIC if g in a.var_names]
    print(f"  Immune consensus genes: {len(imm)}/{len(IMMUNOGENIC)}")
    sc.tl.score_genes(a, imm, score_name="immune_consensus", use_raw=False, random_state=0)
    coords = np.asarray(a.obsm["X_umap"])

    df = pd.read_csv(os.path.join(SCORE_DIR, f"scores_Epithelial_{subtype}.csv"), index_col=0)
    assert (df.index == a.obs_names).all()
    df["immune_consensus"] = a.obs["immune_consensus"].values
    ct2 = pd.read_csv(os.path.join(DATA_DIR, f"cytotrace2_scores_Epithelial_{subtype}.csv"),
                      index_col=0)
    df["CytoTRACE2"] = ct2["CytoTRACE2_Score"].reindex(df.index).values
    df["SI"] = df["SCIE"] - df["NSI"]
    df["phenotype"] = annotate_phenotype(df, "SCIE", "NSI", group_col=GROUP_COL)
    df.to_csv(os.path.join(FIG_DIR, f"phenotype_{subtype}.csv"))
    del a; gc.collect()

    vc = df["phenotype"].value_counts()
    for l in LABEL_COLORS:
        counts.append({"subtype": subtype, "phenotype": l, "n_cells": int(vc.get(l, 0)),
                       "pct": round(100 * vc.get(l, 0) / len(df), 2)})
    print("  " + ", ".join(f"{l} {vc.get(l, 0):,}" for l in LABEL_COLORS))

    n = len(df)
    panels = {
        "phenotype": lambda ax: plot_annotation(ax, coords, df["phenotype"],
                                                f"{subtype} — SCIE / NSI / SCI"),
        "patient":   lambda ax: plot_patient(ax, coords, df["sample"].values,
                                             f"{subtype} — patient"),
        "CytoTRACE2": lambda ax: plot_potency_score(ax, coords, df["CytoTRACE2"].values,
                                                    f"{subtype} — CytoTRACE2"),
        "immune_consensus": lambda ax: plot_continuous(ax, coords,
                                                       df["immune_consensus"].values,
                                                       "viridis",
                                                       f"{subtype} — immune consensus",
                                                       "immune consensus score"),
    }

    for name, draw in panels.items():
        fig, ax = plt.subplots(figsize=(7, 6.5))
        draw(ax)
        out = os.path.join(FIG_DIR, f"umap_{subtype}_{name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
        outputs.append(out)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, draw in zip(axes.flat, panels.values()):
        draw(ax)
    fig.suptitle(f"{subtype} — epithelial cells, Harmony UMAP (n={n:,})",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"umap_{subtype}_4panels.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    outputs.append(out)

out = os.path.join(FIG_DIR, "phenotype_counts.csv")
pd.DataFrame(counts).to_csv(out, index=False)
outputs.append(out)

for f in outputs:
    shutil.copy(f, COPY_DIR)
print("Done.")
