"""
Per-patient subtype comparison and outlier detection.

Outputs (figures/part1/):
  subtype_r_CT2.png     — r(sig, CytoTRACE2) per subtype, all signatures
  subtype_R2_CC.png     — R²(CC→sig) per subtype, all signatures
  outlier_heatmap.png   — per-patient r_CT2 z-scores, outliers labelled
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
SUBTYPE_ORDER  = ["ER", "HER2", "TNBC"]

BG     = "#FFFFFF"
LITE   = "#F9FAFB"
DARK   = "#1F2937"
GREY   = "#6B7280"
BORDER = "#E5E7EB"

plt.rcParams.update({"font.family": "DejaVu Sans"})


# ══════════════════════════════════════════════════════════════════════════════
# Compute per-patient metrics
# ══════════════════════════════════════════════════════════════════════════════
def safe_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20: return np.nan
    return float(spearmanr(x[m], y[m])[0])

def r2_cc(sig, s, g2m):
    m = np.isfinite(sig) & np.isfinite(s) & np.isfinite(g2m)
    if m.sum() < 20: return np.nan
    X = np.column_stack([s[m], g2m[m]])
    return float(LinearRegression().fit(X, sig[m]).score(X, sig[m]))

all_pat = {}   # ds_name → DataFrame

for ds_name, cfg in DATASETS.items():
    print(f"Loading {ds_name} …")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs.copy(); del adata
    sigs  = [s for s in STEM_SIGS if f"score_{s}" in obs.columns]
    rows  = []
    for pat in obs[cfg["batch"]].unique():
        g = obs[obs[cfg["batch"]] == pat]
        if len(g) < MIN_CEL: continue
        st   = str(g[cfg["subtype_col"]].iloc[0])
        ss   = g["S.Score"].values.astype(float)
        g2m  = g["G2M.Score"].values.astype(float)
        ct2  = g["CytoTRACE2"].values.astype(float)
        for sig in sigs:
            sv = g[f"score_{sig}"].values.astype(float)
            rows.append({
                "patient": pat, "subtype": st, "sig": sig,
                "r_CT2" : safe_r(sv, ct2),
                "r_S"   : safe_r(sv, ss),
                "R2_CC" : r2_cc(sv, ss, g2m),
            })
    all_pat[ds_name] = pd.DataFrame(rows)
    print(f"  {obs[cfg['batch']].nunique()} patients, {len(sigs)} sigs")


# ══════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def kruskal_pairwise(groups):
    """Return dict of (st1,st2) → p-value (Mann-Whitney, two-sided)."""
    result = {}
    for a, b in combinations(groups.keys(), 2):
        va, vb = groups[a], groups[b]
        if len(va) < 3 or len(vb) < 3:
            result[(a, b)] = np.nan; continue
        _, p = mannwhitneyu(va, vb, alternative="two-sided")
        result[(a, b)] = p
    return result

def draw_sig_bracket(ax, x1, x2, y, p, y_range):
    """Draw a significance bracket between x1 and x2 at height y."""
    if np.isnan(p): return
    s = stars(p)
    if s == "ns": return
    h = y_range * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color="#374151", lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.1, s,
            ha="center", va="bottom", fontsize=7.5, color="#374151")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 & 2: subtype strip+box per metric
# ══════════════════════════════════════════════════════════════════════════════
def plot_subtype_metric(metric, ylabel, ylim, outfile, hline=None):
    fig, axes = plt.subplots(
        2, len(STEM_SIGS),
        figsize=(18, 7),
        sharey="row",
    )
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Per-patient {ylabel}  —  by breast cancer subtype\n"
        "Each dot = one patient  ·  box = IQR  ·  line = median",
        fontsize=12, fontweight="bold", color=DARK, y=1.01,
    )

    for row_i, (ds_name, df) in enumerate(all_pat.items()):
        sigs_present = [s for s in STEM_SIGS if s in df["sig"].unique()]
        vals_all = df[metric].dropna()
        y_range  = vals_all.max() - vals_all.min()

        for col_i, sig in enumerate(STEM_SIGS):
            ax = axes[row_i, col_i]
            ax.set_facecolor(LITE)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(BORDER)
            ax.tick_params(colors="#374151", labelsize=7)
            if hline is not None:
                ax.axhline(hline, color=GREY, ls="--", lw=0.8, alpha=0.6)

            sub_df = df[df["sig"] == sig][["subtype", metric, "patient"]].dropna()

            groups = {}
            for st in SUBTYPE_ORDER:
                g_vals = sub_df[sub_df["subtype"] == st][metric].values
                groups[st] = g_vals

            # box + strip per subtype
            x_pos = {st: i for i, st in enumerate(SUBTYPE_ORDER)}
            for st in SUBTYPE_ORDER:
                vals = groups[st]
                xi   = x_pos[st]
                col  = SUBTYPE_COLORS[st]
                if len(vals) == 0: continue

                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                bw = 0.30
                ax.fill_betweenx([q1, q3], xi - bw/2, xi + bw/2,
                                 color=col, alpha=0.25, zorder=2)
                ax.plot([xi - bw/2, xi + bw/2], [med, med],
                        color=col, lw=2.2, zorder=3)
                ax.plot([xi, xi], [vals.min(), q1], color=col, lw=0.9, zorder=2)
                ax.plot([xi, xi], [q3, vals.max()], color=col, lw=0.9, zorder=2)

                rng = np.random.default_rng(42 + col_i)
                jit = rng.uniform(-0.14, 0.14, len(vals))

                # outlier detection per subtype
                iqr  = q3 - q1
                lo_f = q1 - 1.5 * iqr
                hi_f = q3 + 1.5 * iqr

                pat_vals = sub_df[sub_df["subtype"] == st][["patient", metric]].dropna()
                for (_, row_v), jv in zip(pat_vals.iterrows(), jit):
                    v   = row_v[metric]
                    out = v < lo_f or v > hi_f
                    ax.scatter(xi + jv, v, color=col,
                               s=28 if not out else 55,
                               alpha=0.75 if not out else 1.0,
                               zorder=4,
                               edgecolors="black" if out else "white",
                               linewidths=0.8 if out else 0.3)
                    if out:
                        ax.text(xi + jv + 0.15, v, row_v["patient"],
                                fontsize=5.5, color="#374151", va="center",
                                style="italic")

            # significance brackets — only ER vs TNBC and TNBC vs HER2
            pw = kruskal_pairwise(groups)
            y_top = vals_all.max() + y_range * 0.05
            pairs_to_show = [("ER", "TNBC"), ("HER2", "TNBC"), ("ER", "HER2")]
            offsets = [0, y_range * 0.08, y_range * 0.16]
            for (a, b), offset in zip(pairs_to_show, offsets):
                p = pw.get((a, b), pw.get((b, a), np.nan))
                if not np.isnan(p) and stars(p) != "ns":
                    draw_sig_bracket(ax, x_pos[a], x_pos[b],
                                     y_top + offset, p, y_range)

            ax.set_xticks(range(len(SUBTYPE_ORDER)))
            ax.set_xticklabels(SUBTYPE_ORDER, fontsize=7.5)
            ax.set_xlim(-0.6, len(SUBTYPE_ORDER) - 0.4)
            if ylim: ax.set_ylim(*ylim)

            if col_i == 0:
                ax.set_ylabel(f"{ds_name}\n{ylabel}", fontsize=8, color=DARK)
            if row_i == 0:
                ax.set_title(SIG_LABELS[sig], fontsize=8.5,
                             fontweight="bold", color=DARK, pad=3)

    legend_handles = [
        mpatches.Patch(color=SUBTYPE_COLORS[st], label=st)
        for st in SUBTYPE_ORDER
    ] + [
        mpatches.Patch(edgecolor="black", facecolor="white", lw=0.8,
                       label="outlier (1.5×IQR)"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9, edgecolor=BORDER)

    plt.tight_layout(w_pad=0.4, h_pad=1.5)
    fig.savefig(os.path.join(OUT, outfile),
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ {outfile}")


plot_subtype_metric("r_CT2", "r(sig, CytoTRACE2)", (-0.65, 0.90),
                    "subtype_r_CT2.png", hline=0)
plot_subtype_metric("R2_CC", "R²(CC → sig)",        (-0.02, 0.90),
                    "subtype_R2_CC.png", hline=None)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Outlier heatmap — r_CT2 z-scores per patient × signature
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Per-patient r(sig, CytoTRACE2)  —  z-scored within each signature\n"
    "Outliers (|z| > 2) are labelled  ·  rows sorted by subtype then mean z-score",
    fontsize=12, fontweight="bold", color=DARK, y=1.01,
)

THRESHOLD = 2.0

for ax, (ds_name, df) in zip(axes, all_pat.items()):
    sigs_present = [s for s in STEM_SIGS if s in df["sig"].unique()]
    pivot = df[df["sig"].isin(sigs_present)].pivot(
        index="patient", columns="sig", values="r_CT2"
    )[sigs_present]

    # z-score each column
    z = (pivot - pivot.mean()) / pivot.std()

    # attach subtype
    sub_map = (df.drop_duplicates("patient")
                  .set_index("patient")["subtype"])
    subtype_ord = ["ER", "HER2", "TNBC"]
    z["_st"]  = z.index.map(sub_map)
    z["_mean"] = z[sigs_present].mean(axis=1)
    z = z.sort_values(["_st", "_mean"])
    subtypes = z.pop("_st")
    z.pop("_mean")

    vmax = 2.5
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="z-score of r(sig, CT2)")

    # y-axis patient labels coloured by subtype
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index, fontsize=6.5)
    for yi, (pat, st) in enumerate(zip(z.index, subtypes)):
        ax.get_yticklabels()[yi].set_color(SUBTYPE_COLORS.get(st, GREY))

    # x-axis
    ax.set_xticks(range(len(sigs_present)))
    ax.set_xticklabels([SIG_LABELS[s] for s in sigs_present],
                       rotation=35, ha="right", fontsize=8.5)

    # annotate outlier cells and flag patient
    for ri, pat in enumerate(z.index):
        for ci, sig in enumerate(sigs_present):
            val = z.loc[pat, sig]
            if abs(val) >= THRESHOLD:
                ax.text(ci, ri, f"{val:+.1f}",
                        ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold")

    # subtype colour bar on right margin
    for ri, (pat, st) in enumerate(zip(z.index, subtypes)):
        ax.add_patch(plt.Rectangle(
            (len(sigs_present) - 0.45, ri - 0.5), 0.45, 1.0,
            color=SUBTYPE_COLORS.get(st, GREY), alpha=0.35,
            clip_on=False, transform=ax.transData,
        ))

    # subtype dividers
    prev = None
    for ri, st in enumerate(subtypes):
        if prev is not None and st != prev:
            ax.axhline(ri - 0.5, color=GREY, lw=1.2, ls="--", alpha=0.6)
        prev = st

    ax.set_title(f"{ds_name}", fontsize=11, fontweight="bold", color=DARK)

subtype_handles = [mpatches.Patch(color=SUBTYPE_COLORS[st], label=st, alpha=0.7)
                   for st in subtype_ord]
fig.legend(handles=subtype_handles, fontsize=9,
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03),
           framealpha=0.9, edgecolor=BORDER)

plt.tight_layout(w_pad=3.0)
fig.savefig(os.path.join(OUT, "outlier_heatmap.png"),
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("✓ outlier_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Text report: outlier patients and subtype test summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("OUTLIER PATIENTS  (r_CT2 for ESC_WONG, IQR method)")
print("=" * 65)
for ds_name, df in all_pat.items():
    sub = df[df["sig"] == "ESC_WONG"][["patient", "subtype", "r_CT2"]].dropna()
    q1, q3 = sub["r_CT2"].quantile([0.25, 0.75])
    iqr = q3 - q1
    out = sub[(sub["r_CT2"] < q1 - 1.5*iqr) | (sub["r_CT2"] > q3 + 1.5*iqr)]
    print(f"\n{ds_name}  (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}):")
    for _, row in out.sort_values("r_CT2").iterrows():
        print(f"  {row['patient']:30s}  {row['subtype']:6s}  r_CT2={row['r_CT2']:+.3f}")

print("\n" + "=" * 65)
print("SUBTYPE KRUSKAL-WALLIS — r_CT2 for ESC_WONG")
print("=" * 65)
for ds_name, df in all_pat.items():
    sub = df[df["sig"] == "ESC_WONG"][["subtype", "r_CT2"]].dropna()
    groups = {st: sub[sub["subtype"] == st]["r_CT2"].values
              for st in SUBTYPE_ORDER if st in sub["subtype"].values}
    if len(groups) < 2: continue
    stat, p = kruskal(*groups.values())
    print(f"\n{ds_name}: H={stat:.2f}, p={p:.4f}  ({stars(p)})")
    for (a, b), pw_p in kruskal_pairwise(groups).items():
        ma = np.median(groups[a]) if a in groups else np.nan
        mb = np.median(groups[b]) if b in groups else np.nan
        print(f"  {a:6s} (med={ma:+.3f}) vs {b:6s} (med={mb:+.3f})  p={pw_p:.4f}  {stars(pw_p)}")
