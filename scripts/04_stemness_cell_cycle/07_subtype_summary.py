"""
One summary figure per dataset combining:
  - Panel grid (left): r_CT2 and R²(CC) per subtype × signature
  - Heatmap (right): per-patient r_CT2 z-scores with outliers labelled

Outputs: figures/part1/summary_Visvader.png
         figures/part1/summary_Klughammer.png
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from sklearn.linear_model import LinearRegression
from itertools import combinations
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT     = "figures/part1"
MIN_CEL = 50

DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_processed.h5ad",
                   "batch": "sample",   "subtype_col": "subtype"},
    "Klughammer": {"file": "klughammer_malignant_processed.h5ad",
                   "batch": "sampleid", "subtype_col": "subtype"},
}

STEM_SIGS = [
    "ESC_ASSOU", "ESC_WONG", "BENPORATH_ES1", "EMP",
    "fMaSC_over_aMaSC", "LIM_MAMMARY_STEM_CELL_UP",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP", "LIM_MAMMARY_LUMINAL_MATURE_UP",
    "stem_consensus",
]
SIG_LABELS = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BENPORATH_ES1"                    : "BenPorath",
    "EMP"                              : "EMP",
    "fMaSC_over_aMaSC"                 : "fMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"         : "LIM_Stem",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
    "LIM_MAMMARY_LUMINAL_MATURE_UP"    : "LIM_LumMat",
    "stem_consensus"                   : "stem_cons",
}
N_SIGS = len(STEM_SIGS)

SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
SUBTYPE_ORDER  = ["ER", "HER2", "TNBC"]

BG     = "#FFFFFF"
LITE   = "#F9FAFB"
DARK   = "#1F2937"
GREY   = "#6B7280"
BORDER = "#E5E7EB"

plt.rcParams.update({"font.family": "DejaVu Sans"})


# ── helpers ───────────────────────────────────────────────────────────────────
def safe_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20: return np.nan
    return float(spearmanr(x[m], y[m])[0])

def r2_cc(sig, s, g2m):
    m = np.isfinite(sig) & np.isfinite(s) & np.isfinite(g2m)
    if m.sum() < 20: return np.nan
    X = np.column_stack([s[m], g2m[m]])
    return float(LinearRegression().fit(X, sig[m]).score(X, sig[m]))

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def pairwise_mw(groups):
    result = {}
    for a, b in combinations(groups.keys(), 2):
        va, vb = groups[a], groups[b]
        if len(va) < 3 or len(vb) < 3:
            result[(a, b)] = np.nan; continue
        _, p = mannwhitneyu(va, vb, alternative="two-sided")
        result[(a, b)] = p
    return result


# ── compute per-patient metrics ───────────────────────────────────────────────
all_pat = {}
for ds_name, cfg in DATASETS.items():
    print(f"Loading {ds_name} …")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs.copy(); del adata
    sigs  = [s for s in STEM_SIGS if f"score_{s}" in obs.columns]
    rows  = []
    for pat in obs[cfg["batch"]].unique():
        g = obs[obs[cfg["batch"]] == pat]
        if len(g) < MIN_CEL: continue
        st  = str(g[cfg["subtype_col"]].iloc[0])
        ss  = g["S.Score"].values.astype(float)
        g2m = g["G2M.Score"].values.astype(float)
        ct2 = g["CytoTRACE2"].values.astype(float)
        for sig in sigs:
            sv = g[f"score_{sig}"].values.astype(float)
            rows.append({"patient": pat, "subtype": st, "sig": sig,
                         "r_CT2": safe_r(sv, ct2),
                         "R2_CC": r2_cc(sv, ss, g2m)})
    all_pat[ds_name] = pd.DataFrame(rows)
    n_sub = obs.drop_duplicates(cfg["batch"])[cfg["subtype_col"]].value_counts()
    print(f"  {n_sub.to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
# Draw one summary figure per dataset
# ══════════════════════════════════════════════════════════════════════════════
def draw_strip_metric(ax, df, metric, hline, ylim):
    """Strip+box plot: x = subtype groups (one mini-cluster per signature)."""
    tick_xs, tick_labels = [], []
    gap_between = 0.6   # gap between signature clusters
    cluster_w   = len(SUBTYPE_ORDER) * 0.55

    for ci, sig in enumerate(STEM_SIGS):
        x_base = ci * (cluster_w + gap_between)
        sub_df = df[df["sig"] == sig][["subtype", metric, "patient"]].dropna()

        groups = {st: sub_df[sub_df["subtype"] == st][metric].values
                  for st in SUBTYPE_ORDER}

        # significance test (TNBC vs ER most relevant)
        pw = pairwise_mw(groups)

        for si, st in enumerate(SUBTYPE_ORDER):
            vals = groups[st]
            xi   = x_base + si * 0.55
            col  = SUBTYPE_COLORS[st]
            if len(vals) == 0: continue

            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr  = q3 - q1
            lo_f = q1 - 1.5 * iqr
            hi_f = q3 + 1.5 * iqr
            bw   = 0.22

            ax.fill_betweenx([q1, q3], xi - bw/2, xi + bw/2,
                             color=col, alpha=0.22, zorder=2)
            ax.plot([xi - bw/2, xi + bw/2], [med, med],
                    color=col, lw=2.0, zorder=3)
            for wh in ([vals.min(), q1], [q3, vals.max()]):
                ax.plot([xi, xi], wh, color=col, lw=0.8, zorder=2)

            rng = np.random.default_rng(42 + ci * 3 + si)
            jit = rng.uniform(-0.10, 0.10, len(vals))
            pat_rows = sub_df[sub_df["subtype"] == st][["patient", metric]].dropna()
            for (_, row_v), jv in zip(pat_rows.iterrows(), jit):
                v   = row_v[metric]
                out = v < lo_f or v > hi_f
                ax.scatter(xi + jv, v, color=col,
                           s=22 if not out else 50,
                           alpha=0.80 if not out else 1.0,
                           zorder=4,
                           edgecolors="black" if out else "white",
                           linewidths=0.7 if out else 0.25)
                if out:
                    # short patient label
                    label = row_v["patient"].split("-")[-1] if "-" in row_v["patient"] \
                            else row_v["patient"][:8]
                    ax.text(xi + jv, v + (iqr * 0.18 if v > med else -iqr * 0.18),
                            label, fontsize=5.5, color="#374151",
                            ha="center", va="bottom" if v > med else "top",
                            style="italic")

        # significance bracket: TNBC vs ER
        tnbc_er_p = pw.get(("ER", "TNBC"), pw.get(("TNBC", "ER"), np.nan))
        s = stars(tnbc_er_p) if not np.isnan(tnbc_er_p) else ""
        if s not in ("", "ns"):
            x1 = x_base + SUBTYPE_ORDER.index("ER")   * 0.55
            x2 = x_base + SUBTYPE_ORDER.index("TNBC") * 0.55
            y_br = (ylim[1] - (ylim[1] - ylim[0]) * 0.08)
            ax.plot([x1, x1, x2, x2],
                    [y_br, y_br + (ylim[1]-ylim[0])*0.03,
                     y_br + (ylim[1]-ylim[0])*0.03, y_br],
                    color="#374151", lw=0.8)
            ax.text((x1+x2)/2, y_br + (ylim[1]-ylim[0])*0.04, s,
                    ha="center", va="bottom", fontsize=7.5, color="#374151")

        mid = x_base + (len(SUBTYPE_ORDER) - 1) * 0.55 / 2
        tick_xs.append(mid)
        tick_labels.append(SIG_LABELS[sig])

    if hline is not None:
        ax.axhline(hline, color=GREY, ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, (N_SIGS - 1) * (len(SUBTYPE_ORDER) * 0.55 + gap_between)
                + len(SUBTYPE_ORDER) * 0.55 + 0.3)
    ax.set_facecolor(LITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=8.5)
    ax.grid(axis="y", color=BORDER, linewidth=0.7, zorder=0)


def draw_heatmap(ax, df, ds_name):
    """Z-scored r_CT2 heatmap: rows=patients, cols=signatures."""
    sigs_p = [s for s in STEM_SIGS if s in df["sig"].unique()]
    pivot  = df[df["sig"].isin(sigs_p)].pivot(
        index="patient", columns="sig", values="r_CT2"
    )[sigs_p]
    z = (pivot - pivot.mean()) / pivot.std()

    sub_map = df.drop_duplicates("patient").set_index("patient")["subtype"]
    z["_st"]   = z.index.map(sub_map)
    z["_mean"] = z[sigs_p].mean(axis=1)
    z = z.sort_values(["_st", "_mean"])
    subtypes = z.pop("_st")
    z.pop("_mean")

    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")

    # subtype colour stripe on left
    for ri, st in enumerate(subtypes):
        ax.add_patch(plt.Rectangle(
            (-1.0, ri - 0.5), 0.55, 1.0,
            color=SUBTYPE_COLORS.get(st, GREY), alpha=0.4,
            clip_on=False, transform=ax.transData))

    # subtype dividers
    prev = None
    for ri, st in enumerate(subtypes):
        if prev is not None and st != prev:
            ax.axhline(ri - 0.5, color=GREY, lw=1.2, ls="--", alpha=0.5)
        prev = st

    # annotate outlier cells
    for ri, pat in enumerate(z.index):
        for ci, sig in enumerate(sigs_p):
            val = z.loc[pat, sig]
            if abs(val) >= 2.0:
                ax.text(ci, ri, f"{val:+.1f}",
                        ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold")

    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index, fontsize=6.0)
    for yi, (pat, st) in enumerate(zip(z.index, subtypes)):
        ax.get_yticklabels()[yi].set_color(SUBTYPE_COLORS.get(st, GREY))

    ax.set_xticks(range(len(sigs_p)))
    ax.set_xticklabels([SIG_LABELS[s] for s in sigs_p],
                       rotation=35, ha="right", fontsize=8.5)
    ax.set_facecolor(BG)
    ax.spines[["top", "right", "left", "bottom"]].set_color(BORDER)

    return im


for ds_name, df in all_pat.items():
    n_pat = df["patient"].nunique()
    n_sub = df.drop_duplicates("patient")["subtype"].value_counts()
    sub_str = "  ·  ".join(f"{st} n={n_sub.get(st,0)}" for st in SUBTYPE_ORDER)

    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.6, 1],
        height_ratios=[1, 1],
        hspace=0.52, wspace=0.28,
        left=0.05, right=0.97, top=0.88, bottom=0.12,
    )

    ax_r   = fig.add_subplot(gs[0, 0])   # r_CT2 subtype strips
    ax_r2  = fig.add_subplot(gs[1, 0])   # R2_CC subtype strips
    ax_hm  = fig.add_subplot(gs[:, 1])   # heatmap

    fig.text(0.5, 0.965,
             f"Part 1 — {ds_name}  ({n_pat} patients)   subtype & outlier summary",
             ha="center", va="top", fontsize=14, fontweight="bold", color=DARK)
    fig.text(0.5, 0.930,
             sub_str + "   ·   brackets = TNBC vs ER (Mann-Whitney)",
             ha="center", va="top", fontsize=9, color=GREY, style="italic")

    # ── strip panels ─────────────────────────────────────────────────────────
    draw_strip_metric(ax_r,  df, "r_CT2", hline=0, ylim=(-0.70, 0.95))
    ax_r.set_ylabel("r(sig, CytoTRACE2)\nper patient", fontsize=9, color=DARK)
    ax_r.set_title("A   Potency correlation by subtype",
                   fontsize=10, fontweight="bold", color=DARK, loc="left", pad=4)
    ax_r.set_xticklabels([])  # shared x with bottom panel

    draw_strip_metric(ax_r2, df, "R2_CC", hline=None, ylim=(-0.02, 0.95))
    ax_r2.set_ylabel("R²(CC → sig)\nper patient", fontsize=9, color=DARK)
    ax_r2.set_title("B   CC contamination by subtype",
                    fontsize=10, fontweight="bold", color=DARK, loc="left", pad=4)

    # ── heatmap ──────────────────────────────────────────────────────────────
    im = draw_heatmap(ax_hm, df, ds_name)
    ax_hm.set_title("C   r(sig, CT2) z-scores  —  outliers labelled",
                    fontsize=10, fontweight="bold", color=DARK, loc="left", pad=4)
    cb = plt.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.12,
                      label="z-score")
    cb.ax.tick_params(labelsize=7)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=SUBTYPE_COLORS[st], label=st, alpha=0.8)
        for st in SUBTYPE_ORDER
    ] + [
        mpatches.Patch(edgecolor="black", facecolor="white", lw=0.8,
                       label="outlier (1.5 × IQR)"),
    ]
    fig.legend(handles=legend_handles, fontsize=9, loc="lower center", ncol=4,
               bbox_to_anchor=(0.32, 0.01), framealpha=0.9, edgecolor=BORDER)

    outfile = f"summary_{ds_name}.png"
    fig.savefig(os.path.join(OUT, outfile),
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ {outfile}")
