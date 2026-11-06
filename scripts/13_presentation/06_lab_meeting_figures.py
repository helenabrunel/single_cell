"""
presentation_figures.py
========================
Three figures for the lab meeting presentation.

  A — CC contamination: scatter of G2M.Score vs signature score for
      stem_consensus (bad, R²≈0.45) vs ESC_ASSOU (better, R²≈0.25),
      coloured by cell cycle phase. One row per subtype.

  B — Stemness × Immune 2D landscape: hexbin using the top-ranked signature
      (ESC_ASSOU) with per-patient z-score normalisation, pooled per subtype
      × dataset. SCIE and NSI corners annotated.

  C — Per-patient consistency: 3×3 grid of individual patient hexbins
      (ER subtype, Visvader, top-9 patients by cell count) using ESC_ASSOU.
      Confirms the pattern holds within individual patients.

Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 presentation_figures.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import json, os, warnings
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Config ────────────────────────────────────────────────────────────────────
BEST_SIG = "ESC_ASSOU"       # top-ranked signature (rank_combined, Visvader)
BAD_SIG  = "stem_consensus"  # highest R²_CC — illustrates CC contamination

OUTDIR_V = "Visvader/figures/presentation"
OUTDIR_K = "Klughammer/figures/presentation"
MIN_CELLS = 500

SUBTYPES       = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
PHASE_COLORS   = {"G1": "#AAAAAA", "S": "#F4A261", "G2M": "#E63946"}

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
IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)

def zscore_per_patient(vals, patients):
    """Return per-patient z-scored copy of vals (pandas Series)."""
    out = vals.copy().astype(float)
    for pid in patients.unique():
        mask = patients == pid
        v = out[mask]
        sd = v.std()
        out[mask] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out

from scipy.stats import pearsonr

def r2_from_xy(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 30:
        return np.nan
    return float(LinearRegression()
                 .fit(x[valid].reshape(-1,1), y[valid])
                 .score(x[valid].reshape(-1,1), y[valid]))

def score_sig(adata, name, genes):
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 5:
        print(f"  SKIP {name}: only {len(present)} genes found")
        return
    sc.tl.score_genes(adata, present, score_name=f"sig_{name}", use_raw=False)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_visvader():
    print("Loading Visvader …")
    adata = sc.read_h5ad("Visvader/compartment_epithelial.h5ad")
    print(f"  {adata.n_obs:,} cells")

    print("  Computing CC scores …")
    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes  =[g for g in S_GENES   if g in adata.var_names],
        g2m_genes=[g for g in G2M_GENES if g in adata.var_names],
    )
    # scanpy uses 'phase' (lowercase)

    print("  Scoring signatures …")
    with open("Visvader/signatures_confirmed.json") as f:
        sigs = json.load(f)
    for name, genes in sigs.items():
        score_sig(adata, name, genes)
    imm = [g for g in IMMUNOGENIC if g in adata.var_names]
    sc.tl.score_genes(adata, imm, score_name="sig_immune", use_raw=False)

    ct2 = pd.read_csv("Visvader/cytotrace2_scores_Epithelial_Global.csv",
                      index_col=0)[["CytoTRACE2_Score"]].rename(
                      columns={"CytoTRACE2_Score": "CytoTRACE2"})
    adata.obs = adata.obs.join(ct2, how="left")

    counts = adata.obs["sample"].value_counts()
    keep   = counts[counts >= MIN_CELLS].index
    adata  = adata[adata.obs["sample"].isin(keep)].copy()
    print(f"  {adata.n_obs:,} cells after ≥{MIN_CELLS} filter")

    sig_cols = [c for c in adata.obs.columns if c.startswith("sig_")]
    cols = ["sample","subtype","S_score","G2M_score","phase","CytoTRACE2"] + sig_cols
    df = adata.obs[cols].copy()
    df["CC"] = (df["S_score"] + df["G2M_score"]) / 2
    del adata
    return df

def load_klughammer():
    print("Loading Klughammer …")
    adata = sc.read_h5ad("Klughammer/klughammer_Malignant.h5ad")
    print(f"  {adata.n_obs:,} cells")

    print("  Scoring signatures …")
    with open("Klughammer/signatures_confirmed.json") as f:
        sigs = json.load(f)
    for name, genes in sigs.items():
        score_sig(adata, name, genes)
    imm = [g for g in IMMUNOGENIC if g in adata.var_names]
    sc.tl.score_genes(adata, imm, score_name="sig_immune", use_raw=False)

    ct2 = pd.read_csv("Klughammer/cytotrace2_scores_klughammer_Malignant_Global.csv",
                      index_col=0)[["CytoTRACE2_Score"]].rename(
                      columns={"CytoTRACE2_Score": "CytoTRACE2"})
    adata.obs = adata.obs.join(ct2, how="left")

    counts = adata.obs["donor_id"].value_counts()
    keep   = counts[counts >= MIN_CELLS].index
    adata  = adata[adata.obs["donor_id"].isin(keep)].copy()
    print(f"  {adata.n_obs:,} cells after ≥{MIN_CELLS} filter")

    # Klughammer already has S.Score, G2M.Score, Phase
    sig_cols = [c for c in adata.obs.columns if c.startswith("sig_")]
    cols = ["donor_id","subtype","S.Score","G2M.Score","Phase","CytoTRACE2"] + sig_cols
    df = adata.obs[cols].copy()
    df = df.rename(columns={"donor_id":"sample","S.Score":"S_score",
                             "G2M.Score":"G2M_score","Phase":"phase"})
    df["CC"] = (df["S_score"] + df["G2M_score"]) / 2
    del adata
    return df

# Pre-computed R²_CC from ranking CSV (S+G2M bivariate, per subtype)
R2_CC_TABLE = {
    "stem_consensus": {"ER": 0.297, "HER2": 0.392, "TNBC": 0.661},
    "ESC_ASSOU":      {"ER": 0.068, "HER2": 0.143, "TNBC": 0.534},
    "EMP":            {"ER": 0.000, "HER2": 0.000, "TNBC": 0.068},
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": {"ER": 0.001, "HER2": 0.003, "TNBC": 0.012},
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — CC Contamination
# ══════════════════════════════════════════════════════════════════════════════
def fig_A_cc_contamination(df, dataset_name, outdir):
    """
    Two-panel figure using ER subtype only (clearest contrast):
      Left  — violin of per-patient z-scored stem_consensus by CC phase
               (R²_CC = 0.30 → high contamination)
      Right — violin of per-patient z-scored ESC_ASSOU by CC phase
               (R²_CC = 0.07 → much lower contamination)

    ER is used because it offers the clearest contrast between the two
    signatures; TNBC and HER2 show high CC contamination in both signatures
    and would mislead the audience.

    A bottom bar chart shows R²_CC for all 9 signatures to give the full
    picture and justify using ESC_ASSOU as the best rank_combined trade-off.
    """
    # Load full R2_CC table from CSV for bar chart
    ranking = pd.read_csv(
        os.path.join(dataset_name, "figures/part1",
                     f"ranking_with_immune_{dataset_name}.csv")
    )
    r2_er = (ranking[ranking["subtype"] == "ER"]
             .set_index("signature")["R2_CC"]
             .sort_values())
    sig_short = {
        "ESC_ASSOU": "ESC_ASSOU", "ESC_WONG": "ESC_WONG",
        "BENPORATH_ES1": "BENPORATH", "EMP": "EMP",
        "fMaSC_over_aMaSC": "fMaSC",
        "LIM_MAMMARY_STEM_CELL_UP": "LIM_MaSC",
        "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
        "LIM_MAMMARY_LUMINAL_MATURE_UP": "LIM_LumMat",
        "stem_consensus": "stem_consensus",
    }
    r2_er.index = [sig_short.get(s, s) for s in r2_er.index]

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.4, 1],
                             hspace=0.45, wspace=0.35)

    # ── Top row: violin by phase for ER subtype ───────────────────────────
    er = df[df["subtype"] == "ER"].copy()

    for col, (sig_name, r2_val) in enumerate([
        (BAD_SIG,  R2_CC_TABLE[BAD_SIG]["ER"]),
        (BEST_SIG, R2_CC_TABLE[BEST_SIG]["ER"]),
    ]):
        ax = fig.add_subplot(gs[0, col])
        sig_col = f"sig_{sig_name}"
        z = zscore_per_patient(er[sig_col], er["sample"])

        phases_ordered = ["G1", "S", "G2M"]
        data_by_phase  = [z[er["phase"] == p].values for p in phases_ordered]
        colors         = [PHASE_COLORS[p] for p in phases_ordered]

        parts = ax.violinplot(data_by_phase, positions=[0, 1, 2],
                              showmedians=True, showextrema=False)
        for pc, col_c in zip(parts["bodies"], colors):
            pc.set_facecolor(col_c)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(phases_ordered, fontsize=10)
        ax.set_ylabel("Signature score (per-patient z)", fontsize=9)
        short = sig_short.get(sig_name, sig_name)
        ax.set_title(f"{short}\nER subtype  —  R²_CC = {r2_val:.3f}",
                     fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.8, ls="--")

        # Annotate G1→G2M delta direction
        med_g1  = np.median(data_by_phase[0])
        med_g2m = np.median(data_by_phase[2])
        direction = "↑ cycling cells score higher" if med_g2m > med_g1 else "↔ no phase bias"
        ax.text(0.5, 0.97, direction, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, style="italic", color="dimgray")

    # ── Top row col 2: scatter stem_consensus vs ESC_ASSOU (ER, subsampled) ─
    ax_sc = fig.add_subplot(gs[0, 2])
    # subsample for speed
    rng   = np.random.default_rng(42)
    n_sub = min(8000, len(er))
    idx   = rng.choice(len(er), n_sub, replace=False)
    sub   = er.iloc[idx].copy()

    z_bad  = zscore_per_patient(sub[f"sig_{BAD_SIG}"],  sub["sample"])
    z_good = zscore_per_patient(sub[f"sig_{BEST_SIG}"], sub["sample"])

    sc_ax = ax_sc.scatter(z_bad, z_good, c=sub["G2M_score"].values,
                          cmap="Reds", alpha=0.35, s=4, rasterized=True,
                          vmin=-0.1, vmax=0.8)
    plt.colorbar(sc_ax, ax=ax_sc, label="G2M score", shrink=0.8)
    ax_sc.axvline(0, color="gray", lw=0.7, ls="--")
    ax_sc.axhline(0, color="gray", lw=0.7, ls="--")
    ax_sc.set_xlabel(f"{sig_short[BAD_SIG]} (z)", fontsize=9)
    ax_sc.set_ylabel(f"{sig_short[BEST_SIG]} (z)", fontsize=9)
    ax_sc.set_title("ER — direct comparison\n(coloured by G2M score)",
                    fontsize=10, fontweight="bold")

    # ── Bottom row: R²_CC bar chart for all signatures (ER) ──────────────
    ax_bar = fig.add_subplot(gs[1, :])
    bar_colors = ["#E63946" if s == sig_short[BAD_SIG]
                  else "#1A73E8" if s == sig_short[BEST_SIG]
                  else "#AAAAAA"
                  for s in r2_er.index]
    bars = ax_bar.barh(r2_er.index, r2_er.values, color=bar_colors, edgecolor="white")
    ax_bar.set_xlabel("R²_CC  (variance in signature score explained by S + G2M)", fontsize=9)
    ax_bar.set_title("All 9 signatures — CC contamination (ER subtype, Visvader)", fontsize=10)
    ax_bar.axvline(0, color="black", lw=0.5)
    for bar, val in zip(bars, r2_er.values):
        ax_bar.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax_bar.set_xlim(0, r2_er.max() * 1.25)

    # Legend for bar chart colours
    legend_elems = [
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#E63946",
               markersize=10, label=f"{sig_short[BAD_SIG]} (selected bad example)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#1A73E8",
               markersize=10, label=f"{sig_short[BEST_SIG]} (selected for downstream)"),
    ]
    ax_bar.legend(handles=legend_elems, fontsize=8, loc="lower right")

    fig.suptitle(f"{dataset_name} — Stemness signature contamination by cell cycle",
                 fontsize=13, fontweight="bold", y=1.01)
    path = os.path.join(outdir, "figA_cc_contamination.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared hexbin drawing function — matches 07_hipoxia_tim3.py style exactly
# ══════════════════════════════════════════════════════════════════════════════
def hexbin_panel(ax, x, y, title, xlabel, ylabel,
                 gridsize=90, title_color="black", add_scie_nsi=True):
    """
    Hexbin panel styled to match the existing pipeline figures:
      - cmap RdYlBu_r, white background, light gray grid
      - 0.5–99.5 percentile clip with 20% padding
      - dashed black lines at x=0, y=0
      - Pearson r / p / n annotation bottom-right
      - optional SCIE / NSI corner labels
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        ax.set_visible(False)
        return None

    xmin, xmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ymin, ymax = np.percentile(y, 0.5), np.percentile(y, 99.5)
    xpad = 0.20 * (xmax - xmin)
    ypad = 0.20 * (ymax - ymin)
    xl = (xmin - xpad, xmax + xpad)
    yl = (ymin - ypad, ymax + ypad)

    ax.set_facecolor("white")
    ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="RdYlBu_r",
                   mincnt=1, linewidths=0.02,
                   extent=[xl[0], xl[1], yl[0], yl[1]], zorder=2)
    ax.axhline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=0.9, alpha=0.5, zorder=3)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", color=title_color)

    r, p = pearsonr(x, y)
    pstr = f"p={p:.1e}" if p >= 1e-300 else "p<1e-300"
    ax.text(0.97, 0.03, f"r={r:.3f}\n{pstr}\nn={len(x):,}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    if add_scie_nsi:
        ax.text(0.96, 0.08, "SCIE", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, fontweight="bold",
                color="#1A73E8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1A73E8", alpha=0.85))
        ax.text(0.04, 0.92, "NSI", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                color="#D32F2F",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#D32F2F", alpha=0.85))
    return hb


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Stemness × Immune landscape (pooled, z-scored)
# ══════════════════════════════════════════════════════════════════════════════
def fig_B_stem_immune_landscape(df_v, df_k, outdir):
    """
    Hexbin of (stemness, immune) with per-patient z-score, for both datasets
    × all subtypes. Layout: 2 rows (datasets) × 3 cols (subtypes).
    """
    datasets = [
        ("Visvader",   df_v, "sample"),
        ("Klughammer", df_k, "sample"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(5.5 * 3, 5.5 * 2))

    for row, (ds_name, df, pat_col) in enumerate(datasets):
        for col, subtype in enumerate(SUBTYPES):
            ax = axes[row, col]
            sub = df[df["subtype"] == subtype].copy()
            if len(sub) < MIN_CELLS:
                ax.set_visible(False)
                continue

            stem_z = zscore_per_patient(sub[f"sig_{BEST_SIG}"], sub[pat_col])
            imm_z  = zscore_per_patient(sub["sig_immune"],       sub[pat_col])

            hb = hexbin_panel(
                ax,
                stem_z.values, imm_z.values,
                title=f"{ds_name} — {subtype}  (n={len(sub):,})",
                xlabel=f"{BEST_SIG} score (per-patient z)",
                ylabel="Immune score (per-patient z)",
                title_color=SUBTYPE_COLORS.get(subtype, "black"),
            )
            if hb is not None:
                plt.colorbar(hb, ax=ax, shrink=0.8, label="Cell count")

    fig.suptitle(
        f"Stemness × Immune landscape  —  {BEST_SIG} signature\n"
        "(per-patient z-score normalisation;  "
        "SCIE = high stem / low immune;  NSI = low stem / high immune)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(outdir, "figB_stem_immune_landscape.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Per-patient consistency (all subtypes, both datasets)
# ══════════════════════════════════════════════════════════════════════════════
def fig_C_per_patient(df_v, df_k, outdir):
    """
    One figure per subtype (ER, HER2, TNBC). Each figure shows all qualifying
    patients from Visvader and Klughammer, up to 9 per dataset, each panel
    independently z-scored.
    """
    MAX_PER_DS = 9

    datasets = [
        ("Visvader",   df_v, "sample"),
        ("Klughammer", df_k, "sample"),
    ]

    def zsc(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    for subtype in SUBTYPES:
        panels = []
        for ds_name, df, pat_col in datasets:
            sub_ds = df[df["subtype"] == subtype]
            top_pts = (sub_ds[pat_col].value_counts()
                         .head(MAX_PER_DS).index.tolist())
            for pid in top_pts:
                sub = sub_ds[sub_ds[pat_col] == pid].copy()
                if len(sub) < 100:
                    continue
                short = pid.split("_", 1)[-1] if "_" in pid else pid
                panels.append((ds_name, short,
                                zsc(sub[f"sig_{BEST_SIG}"]).values,
                                zsc(sub["sig_immune"]).values,
                                len(sub)))

        if not panels:
            continue

        ncols = 3
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 5.0 * nrows))
        axes_flat = np.array(axes).flatten()

        for i, (ds_name, short_id, stem_z, imm_z, n) in enumerate(panels):
            ax = axes_flat[i]
            hb = hexbin_panel(
                ax, stem_z, imm_z,
                title=f"{ds_name} — {short_id}  (n={n:,})",
                xlabel=f"{BEST_SIG} score (z)",
                ylabel="Immune score (z)",
                gridsize=35,
                title_color=SUBTYPE_COLORS.get(subtype, "black"),
            )
            if hb is not None:
                plt.colorbar(hb, ax=ax, shrink=0.7, label="Cell count")

        for j in range(len(panels), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Per-patient consistency — {subtype} subtype\n"
            f"{BEST_SIG} signature  (within-patient z-score;  "
            "SCIE = high stem / low immune;  NSI = low stem / high immune)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(outdir, f"figC_per_patient_{subtype}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Signature quality map  (CC contamination × CytoTRACE2 signal)
# ══════════════════════════════════════════════════════════════════════════════
SIG_SHORT = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BENPORATH_ES1"                    : "BENPORATH",
    "EMP"                              : "EMP",
    "fMaSC_over_aMaSC"                 : "fMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"         : "LIM_MaSC",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
    "LIM_MAMMARY_LUMINAL_MATURE_UP"    : "LIM_LumMat",
    "stem_consensus"                   : "stem_consensus",
}

def fig_D_signature_quality(df_v, outdir):
    """
    Left (large): bubble scatter of r_cyto_partial (x) vs R²_CC (y) for all
    signatures × both datasets × all subtypes. Ideal region annotated.
    Right top/bottom: violin by CC phase for stem_consensus (worst CC) vs EMP
    (cleanest CC), ER subtype — the most extreme contrast.
    """
    # ── Load ranking CSVs ────────────────────────────────────────────────────
    rv = pd.read_csv("Visvader/figures/part1/ranking_with_immune_Visvader.csv")
    rk = pd.read_csv("Klughammer/figures/part1/ranking_with_immune_Klughammer.csv")
    rv["dataset"] = "Visvader"
    rk["dataset"] = "Klughammer"
    ranking = pd.concat([rv, rk])
    ranking = ranking[ranking["subtype"].isin(SUBTYPES)].copy()
    ranking["sig_label"] = ranking["signature"].map(SIG_SHORT).fillna(ranking["signature"])

    sig_names  = list(SIG_SHORT.keys())
    cmap_sigs  = plt.cm.tab10(np.linspace(0, 1, len(sig_names)))
    sig_color  = {s: cmap_sigs[i] for i, s in enumerate(sig_names)}
    ds_marker  = {"Visvader": "o", "Klughammer": "^"}
    st_size    = {"ER": 120, "HER2": 70, "TNBC": 95}

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.7, 1],
                             hspace=0.45, wspace=0.30)

    # ── Panel A: quality scatter ─────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[:, 0])

    for _, row in ranking.iterrows():
        sig = row["signature"]
        if sig not in sig_color:
            continue
        ax_sc.scatter(
            row["r_cyto_partial"], row["R2_CC"],
            c=[sig_color[sig]],
            marker=ds_marker[row["dataset"]],
            s=st_size.get(row["subtype"], 80),
            alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3,
        )

    # Reference lines & ideal region
    ax_sc.axhline(0.15, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_sc.axvline(0.15, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_sc.fill_between([0.15, ax_sc.get_xlim()[1] if ax_sc.get_xlim()[1] > 0.15 else 1.0],
                        0, 0.15, color="#E8F5E9", alpha=0.4, zorder=0,
                        label="Ideal region")
    ax_sc.text(0.97, 0.03,
               "← IDEAL\nhigh stemness signal\nlow CC contamination",
               transform=ax_sc.transAxes, ha="right", va="bottom",
               fontsize=8, color="#2E7D32", style="italic")

    ax_sc.set_xlabel("Partial r (CytoTRACE2 | CC corrected)\n"
                     "→ stemness signal independent of cell cycle", fontsize=9)
    ax_sc.set_ylabel("R²_CC  (variance explained by S + G2M scores)\n"
                     "← cell cycle contamination", fontsize=9)
    ax_sc.set_title("Signature quality map — all 9 signatures, both datasets, all subtypes",
                    fontsize=10, fontweight="bold")
    ax_sc.set_facecolor("white")
    ax_sc.grid(True, color="#DDDDDD", lw=0.5, zorder=0)

    # Signature colour legend
    leg_sig = [Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=sig_color[s], markersize=8,
                       label=SIG_SHORT[s])
               for s in sig_names if s in sig_color]
    leg_ds  = [Line2D([0],[0], marker=m, color="gray", ls="",
                       markersize=8, label=ds)
               for ds, m in ds_marker.items()]
    leg_st  = [Line2D([0],[0], marker="o", color="gray", ls="",
                       markersize=np.sqrt(sz/4), label=st)
               for st, sz in st_size.items()]
    l1 = ax_sc.legend(handles=leg_sig, fontsize=7, loc="upper left",
                      title="Signature", title_fontsize=8, framealpha=0.9)
    ax_sc.add_artist(l1)
    ax_sc.legend(handles=leg_ds + leg_st, fontsize=7, loc="upper right",
                 title="Dataset / Subtype", title_fontsize=8, framealpha=0.9)

    # ── Panels B & C: violins — stem_consensus vs EMP (ER, Visvader) ─────────
    er = df_v[df_v["subtype"] == "ER"].copy()
    r2_table = {
        "stem_consensus": 0.297,
        "EMP":            0.000,
    }

    for ax_row, sig_name in enumerate(["stem_consensus", "EMP"]):
        ax = fig.add_subplot(gs[ax_row, 1])
        sig_col = f"sig_{sig_name}"
        if sig_col not in er.columns:
            ax.set_visible(False)
            continue

        z      = zscore_per_patient(er[sig_col], er["sample"])
        phases = ["G1", "S", "G2M"]
        data   = [z[er["phase"] == p].values for p in phases]
        colors = [PHASE_COLORS[p] for p in phases]

        parts = ax.violinplot(data, positions=[0, 1, 2],
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(phases, fontsize=9)
        ax.set_ylabel("Score (per-patient z)", fontsize=8)
        r2 = r2_table.get(sig_name, "?")
        ax.set_title(f"{SIG_SHORT.get(sig_name, sig_name)}"
                     f"  —  R²_CC = {r2:.3f}",
                     fontsize=9, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_facecolor("white")
        ax.grid(True, color="#DDDDDD", lw=0.5, axis="y", zorder=0)

        delta = np.median(data[2]) - np.median(data[0])
        ax.text(0.97, 0.96,
                f"Δ median(G2M − G1) = {delta:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#C62828" if delta > 0.05 else "#1565C0")

    fig.suptitle(
        "Signature selection: CytoTRACE2 correlation vs cell cycle contamination\n"
        "Visvader + Klughammer  ·  ER / HER2 / TNBC subtypes",
        fontsize=11, fontweight="bold",
    )
    path = os.path.join(outdir, "figD_signature_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — SCIE/NSI enrichment across all signatures
# ══════════════════════════════════════════════════════════════════════════════
def fig_E_scie_nsi_general(outdir):
    """
    Purely from pre-computed subpop_metrics CSVs — no raw data needed.

    Top row:
      Left  — heatmap: SCIE fold enrichment (signatures × dataset×subtype)
      Right — heatmap: NSI  fold enrichment (signatures × dataset×subtype)
    Bottom row:
      Centre — scatter: mean SCIE fold (x) vs mean NSI fold (y) per
               signature, averaged across all subtypes and datasets.
               Ideal signatures sit in the top-right quadrant (both enriched).
               Bubble size = % patients passing chi² test.
    """
    rv = pd.read_csv("Visvader/figures/part1/subpop_metrics_Visvader.csv")
    rk = pd.read_csv("Klughammer/figures/part1/subpop_metrics_Klughammer.csv")
    rv["dataset"] = "Visvader"
    rk["dataset"] = "Klughammer"
    df = pd.concat([rv, rk])
    df = df[df["subtype"].isin(SUBTYPES)].copy()
    df["sig_label"] = df["signature"].map(SIG_SHORT).fillna(df["signature"])

    # Column order for heatmap: Visvader ER/HER2/TNBC | Klughammer ER/HER2/TNBC
    col_order  = [(ds, st) for ds in ["Visvader","Klughammer"]
                            for st in SUBTYPES]
    col_labels = [f"{ds}\n{st}" for ds, st in col_order]

    # Row order: sort by mean SCIE fold descending
    sig_order = (df.groupby("sig_label")["fold_scie_mean"]
                   .mean()
                   .sort_values(ascending=False)
                   .index.tolist())

    def build_matrix(metric):
        rows = []
        for sig in sig_order:
            row_vals = []
            for ds, st in col_order:
                mask = (df["sig_label"] == sig) & (df["dataset"] == ds) & (df["subtype"] == st)
                sub  = df[mask]
                row_vals.append(sub[metric].values[0] if len(sub) else np.nan)
            rows.append(row_vals)
        return np.array(rows)

    mat_scie = build_matrix("fold_scie_mean")
    mat_nsi  = build_matrix("fold_nsi_mean")
    mat_chi2 = build_matrix("chi2_pct_sig")   # % patients significant

    import matplotlib.colors as mcolors
    # Diverging colormap centred at 1.0
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
    cmap = plt.cm.RdBu_r

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1.6, 1],
                             width_ratios=[1, 1, 0.8],
                             hspace=0.5, wspace=0.40)

    def draw_heatmap(ax, mat, title, show_ylabel=True):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=0, ha="center")
        ax.set_yticks(range(len(sig_order)))
        if show_ylabel:
            ax.set_yticklabels(sig_order, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        # Annotate cells
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                if np.isnan(val):
                    continue
                chi = mat_chi2[r, c] if mat is mat_scie else mat_chi2[r, c]
                star = "**" if chi >= 0.75 else ("*" if chi >= 0.5 else "")
                txt  = f"{val:.2f}{star}"
                color = "white" if abs(val - 1.0) > 0.4 else "black"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=6.5, color=color, fontweight="bold")
        # Separator between Visvader and Klughammer columns
        ax.axvline(2.5, color="black", lw=1.5)
        return im

    ax_scie = fig.add_subplot(gs[0, 0])
    ax_nsi  = fig.add_subplot(gs[0, 1])
    im = draw_heatmap(ax_scie, mat_scie,
                      "SCIE fold enrichment\n(stem-high / immune-low corner)",
                      show_ylabel=True)
    draw_heatmap(ax_nsi,  mat_nsi,
                 "NSI fold enrichment\n(stem-low / immune-high corner)",
                 show_ylabel=False)

    # Shared colorbar
    ax_cb = fig.add_subplot(gs[0, 2])
    ax_cb.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_cb, fraction=0.8, pad=0.05,
                        label="Fold enrichment\n(1 = expected under independence)")
    cbar.ax.tick_params(labelsize=8)

    # ── Bottom: SCIE vs NSI scatter per signature ─────────────────────────────
    ax_dot = fig.add_subplot(gs[1, :])

    # Mean across all subtypes × datasets per signature
    mean_df = (df.groupby("sig_label")[["fold_scie_mean","fold_nsi_mean","chi2_pct_sig"]]
                 .mean()
                 .reset_index())

    cmap_sigs = plt.cm.tab10(np.linspace(0, 1, len(mean_df)))
    for i, row in mean_df.iterrows():
        ax_dot.scatter(row["fold_scie_mean"], row["fold_nsi_mean"],
                       s=200 + 400 * row["chi2_pct_sig"],
                       color=cmap_sigs[i], alpha=0.85,
                       edgecolors="white", linewidths=0.8, zorder=3)
        ax_dot.annotate(row["sig_label"],
                        (row["fold_scie_mean"], row["fold_nsi_mean"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, fontweight="bold")

    ax_dot.axvline(1, color="gray", lw=0.9, ls="--", alpha=0.6)
    ax_dot.axhline(1, color="gray", lw=0.9, ls="--", alpha=0.6)
    # Shade ideal quadrant
    xlim = ax_dot.get_xlim(); ylim = ax_dot.get_ylim()
    ax_dot.fill_between([1, max(xlim[1], 1.5)], 1, max(ylim[1], 1.5),
                         color="#E8F5E9", alpha=0.35, zorder=0)
    ax_dot.text(max(xlim[1], 1.5) - 0.01, max(ylim[1], 1.5) - 0.01,
                "Both enriched\n(ideal)",
                ha="right", va="top", fontsize=8,
                color="#2E7D32", style="italic")
    ax_dot.set_xlabel("Mean SCIE fold enrichment  (stem-high / immune-low)",
                      fontsize=9)
    ax_dot.set_ylabel("Mean NSI fold enrichment\n(stem-low / immune-high)",
                      fontsize=9)
    ax_dot.set_title(
        "Summary: which signatures detect both SCIE and NSI subpopulations?\n"
        "(mean across ER/HER2/TNBC × Visvader/Klughammer;  "
        "bubble size = % patients with significant chi² enrichment)",
        fontsize=9,
    )
    ax_dot.set_facecolor("white")
    ax_dot.grid(True, color="#DDDDDD", lw=0.5, zorder=0)

    fig.suptitle(
        "SCIE / NSI subpopulation enrichment across all 9 signatures\n"
        "(fold enrichment of corner tertiles vs independence null;  "
        "* ≥50% / ** ≥75% patients with significant chi² test)",
        fontsize=11, fontweight="bold",
    )
    path = os.path.join(outdir, "figE_scie_nsi_enrichment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTDIR_V, exist_ok=True)
    os.makedirs(OUTDIR_K, exist_ok=True)

    df_v = load_visvader()
    df_k = load_klughammer()

    print("\nFigure D — Signature quality map …")
    fig_D_signature_quality(df_v, OUTDIR_V)

    print("\nFigure E — SCIE/NSI enrichment (all signatures) …")
    fig_E_scie_nsi_general(OUTDIR_V)

    print("\nDone. Figures saved to:", OUTDIR_V)
