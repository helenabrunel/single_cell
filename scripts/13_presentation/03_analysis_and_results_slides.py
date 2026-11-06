"""
Generate two presentation-quality figures for part1:
  fig_slide1_analysis.png  — workflow / analysis design
  fig_slide2_results.png   — key results summary
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

OUT = "figures/part1"
os.makedirs(OUT, exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────────
C_BLUE   = "#2C6FAC"
C_TEAL   = "#2A9D8F"
C_ORANGE = "#E76F51"
C_PURPLE = "#6A4C93"
C_GOLD   = "#E9C46A"
C_GREY   = "#6B7280"
C_LIGHT  = "#F8F9FA"
C_WHITE  = "#FFFFFF"
C_DARK   = "#1F2937"

FONTS = {"family": "DejaVu Sans"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — ANALYSIS WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def rounded_box(ax, x, y, w, h, color, alpha=1.0, lw=1.5,
                edgecolor="white", radius=0.04, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha, zorder=zorder,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    return box


def text_ax(ax, x, y, s, size=10, color="white", weight="normal",
            ha="center", va="center", zorder=5, wrap=False, style="normal"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size,
            color=color, fontweight=weight, ha=ha, va=va,
            zorder=zorder, fontstyle=style,
            multialignment="center")


def arrow(ax, x0, y0, x1, y1, color=C_GREY, lw=1.5, zorder=3,
          style="->", mutation_scale=12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=mutation_scale),
                zorder=zorder)


fig1, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
fig1.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)

# ── Title ──────────────────────────────────────────────────────────────────────
text_ax(ax, 0.5, 0.95,
        "Part 1 — Stemness signature evaluation & CC-free index",
        size=17, weight="bold", color=C_GOLD)
text_ax(ax, 0.5, 0.89,
        "Which stemness signature best captures developmental potency, independently of cell cycle?",
        size=11, color="#D1D5DB", style="italic")

# ── SECTION LABELS ─────────────────────────────────────────────────────────────
for lx, label, col in [(0.03, "DATA", C_BLUE),
                        (0.295, "ANALYSIS", C_TEAL),
                        (0.61, "METRICS", C_ORANGE),
                        (0.845, "OUTPUT", C_PURPLE)]:
    text_ax(ax, lx + 0.06, 0.83, label, size=9, weight="bold", color=col, ha="center")
    ax.plot([lx, lx + 0.12], [0.815, 0.815],
            transform=ax.transAxes, color=col, lw=1.5, alpha=0.6)

# ── DATA boxes ─────────────────────────────────────────────────────────────────
# Visvader
rounded_box(ax, 0.03, 0.60, 0.13, 0.20, C_BLUE, radius=0.025)
text_ax(ax, 0.095, 0.76, "Visvader", size=11, weight="bold")
text_ax(ax, 0.095, 0.71, "Primary tumour", size=9, color="#DBEAFE")
text_ax(ax, 0.095, 0.67, "121,632 cells", size=9, color="#DBEAFE")
text_ax(ax, 0.095, 0.63, "31 patients", size=9, color="#DBEAFE")
text_ax(ax, 0.095, 0.60, "ER · HER2 · TNBC", size=8, color="#BFDBFE")

# Klughammer
rounded_box(ax, 0.03, 0.34, 0.13, 0.20, C_BLUE, radius=0.025, alpha=0.75)
text_ax(ax, 0.095, 0.50, "Klughammer", size=11, weight="bold")
text_ax(ax, 0.095, 0.45, "Metastatic biopsy", size=9, color="#DBEAFE")
text_ax(ax, 0.095, 0.41, "83,490 cells", size=9, color="#DBEAFE")
text_ax(ax, 0.095, 0.37, "19 patients", size=9, color="#DBEAFE")
text_ax(ax, 0.095, 0.34, "ER · HER2 · TNBC", size=8, color="#BFDBFE")

# 9 sigs label
rounded_box(ax, 0.03, 0.13, 0.13, 0.17, "#374151", radius=0.025)
text_ax(ax, 0.095, 0.265, "9 stemness\nsignatures", size=10, weight="bold", color=C_GOLD)
sigs_short = ["ESC_ASSOU","ESC_WONG","BenPorath",
              "EMP","fMaSC","LIM_Stem",
              "LIM_LumProg","LIM_LumMat","stem_consensus"]
for i, s in enumerate(sigs_short):
    col = i % 3
    row = i // 3
    text_ax(ax, 0.042 + col*0.042, 0.215 - row*0.025, s,
            size=6.5, color="#9CA3AF", ha="left")

# ── ANALYSIS block ─────────────────────────────────────────────────────────────
arrow(ax, 0.165, 0.70, 0.29, 0.70, color=C_GREY, lw=1.8)
arrow(ax, 0.165, 0.44, 0.29, 0.44, color=C_GREY, lw=1.8)

rounded_box(ax, 0.29, 0.30, 0.30, 0.50, "#1A3A4A", radius=0.025)
text_ax(ax, 0.44, 0.785,
        "Per-patient analysis", size=11, weight="bold", color=C_TEAL)
text_ax(ax, 0.44, 0.755,
        "(≥50 cells / patient — both datasets)", size=8.5, color="#9CA3AF")

# patient schematic
pat_y = 0.72
for i, (pid, nc, col) in enumerate([
        ("Patient 1", "n=3,403", "#60A5FA"),
        ("Patient 2", "n=5,512", "#34D399"),
        ("···", "",          "#9CA3AF"),
        ("Patient 31", "n=808",  "#F472B6"),
]):
    bx = 0.30 + i * 0.073
    rounded_box(ax, bx, pat_y - 0.04, 0.065, 0.055, col, alpha=0.25, radius=0.012)
    text_ax(ax, bx + 0.033, pat_y - 0.005, pid, size=6.5, color=col, weight="bold")
    text_ax(ax, bx + 0.033, pat_y - 0.025, nc,  size=6,   color="#9CA3AF")

# within-patient step
arrow(ax, 0.44, pat_y - 0.05, 0.44, 0.62, color=C_TEAL, lw=1.4)
rounded_box(ax, 0.305, 0.55, 0.27, 0.065, "#0F3D3A", radius=0.015)
text_ax(ax, 0.44, 0.585,
        "Spearman r  (sig vs CT2 / S.Score / G2M.Score / immune)",
        size=8.5, color="#6EE7B7")
text_ax(ax, 0.44, 0.563,
        "computed independently per patient",
        size=7.5, color="#9CA3AF")

# pooling step
arrow(ax, 0.44, 0.55, 0.44, 0.488, color=C_TEAL, lw=1.4)
rounded_box(ax, 0.305, 0.43, 0.27, 0.055, "#0F3D3A", radius=0.015)
text_ax(ax, 0.44, 0.462,
        "Fisher z meta-analysis  →  pooled r ± 95% CI",
        size=8.5, color="#6EE7B7")
text_ax(ax, 0.44, 0.440,
        "weighted by √(n−3) per patient",
        size=7.5, color="#9CA3AF")

# pooled analysis
arrow(ax, 0.44, 0.43, 0.44, 0.375, color=C_TEAL, lw=1.4)
rounded_box(ax, 0.305, 0.315, 0.27, 0.055, "#0F3D3A", radius=0.015)
text_ax(ax, 0.44, 0.347,
        "R²(CC→sig) · partial r(CT2 | CC) · LMM β",
        size=8.5, color="#6EE7B7")
text_ax(ax, 0.44, 0.325,
        "pooled — patient random effect in LMM",
        size=7.5, color="#9CA3AF")

# CC-free step
arrow(ax, 0.44, 0.315, 0.44, 0.255, color=C_ORANGE, lw=1.8)
rounded_box(ax, 0.295, 0.195, 0.29, 0.058, "#3D1A0F", radius=0.015)
text_ax(ax, 0.44, 0.230,
        "CC-free residualisation  (per patient)",
        size=9, weight="bold", color=C_ORANGE)
text_ax(ax, 0.44, 0.208,
        "OLS: sig ~ S.Score + G2M.Score → residuals",
        size=8, color="#FCA5A5")

# scie check
arrow(ax, 0.44, 0.195, 0.44, 0.138, color=C_ORANGE, lw=1.8)
rounded_box(ax, 0.295, 0.082, 0.29, 0.052, "#3D1A0F", radius=0.015)
text_ax(ax, 0.44, 0.113,
        "SCIE axis check:  r(residual, immune)  vs  r(sig, immune)",
        size=8.5, weight="bold", color=C_GOLD)
text_ax(ax, 0.44, 0.093,
        "does the stem × immune anti-correlation survive CC removal?",
        size=7.5, color="#FCA5A5")

# ── METRICS column ─────────────────────────────────────────────────────────────
arrow(ax, 0.59, 0.44, 0.61, 0.44, color=C_GREY, lw=1.8)

metrics = [
    (C_TEAL,   "r(sig, CytoTRACE2)",        "developmental potency alignment"),
    (C_TEAL,   "partial r(CT2 | CC)",        "potency signal after CC control"),
    ("#F59E0B", "R²(CC → signature)",         "fraction of variance = cell cycle"),
    ("#F59E0B", "r(sig, S.Score / G2M)",      "S-phase & G2M contamination"),
    (C_ORANGE,  "r(sig, immunogenic)",        "SCIE axis relevance"),
    (C_PURPLE,  "dCor(sig, CytoTRACE2)",     "nonlinear potency association"),
    (C_PURPLE,  "LMM β  CytoTRACE2",         "patient-corrected coefficient"),
]

my = 0.75
for i, (col, name, sub) in enumerate(metrics):
    by = my - i * 0.087
    rounded_box(ax, 0.615, by - 0.035, 0.22, 0.070, "#111827", radius=0.012)
    ax.plot([0.618, 0.622], [by, by],
            transform=ax.transAxes, color=col, lw=3)
    text_ax(ax, 0.727, by + 0.010, name,  size=8.5, color="white",  ha="center")
    text_ax(ax, 0.727, by - 0.013, sub,   size=7,   color="#9CA3AF", ha="center")

# composite score
rounded_box(ax, 0.615, 0.082, 0.22, 0.055, "#1F2D1F", radius=0.012, lw=2,
            edgecolor=C_GOLD)
text_ax(ax, 0.727, 0.115, "Composite rank score", size=9, weight="bold", color=C_GOLD)
text_ax(ax, 0.727, 0.095, "0.5×r(CT2) + 0.5×partial_r − R²(CC)", size=7.5, color="#FDE68A")

# ── OUTPUT column ──────────────────────────────────────────────────────────────
arrow(ax, 0.84, 0.44, 0.86, 0.44, color=C_GREY, lw=1.8)

outputs = [
    (C_PURPLE,  "Heatmap",      "9 sigs × 8 metrics × subtypes"),
    (C_PURPLE,  "Forest plot",  "pooled r ± CI  (CT2, CC, immune)"),
    (C_PURPLE,  "Tradeoff plot","R²_CC vs partial r — best sig region"),
    (C_ORANGE,  "CC-free panel","before/after residualisation"),
    (C_ORANGE,  "SCIE survival","r_immune: original vs CC-free"),
    (C_GOLD,    "Ranking table","CSV  ·  2 datasets  ·  18 rows"),
    (C_BLUE,    "CC-free h5ads","score_{sig}_ccfree in obs"),
]

oy = 0.76
for i, (col, name, sub) in enumerate(outputs):
    by = oy - i * 0.083
    rounded_box(ax, 0.86, by - 0.032, 0.125, 0.060, "#1F1035", radius=0.010)
    text_ax(ax, 0.924, by + 0.008, name,  size=8.5, color=col,     weight="bold", ha="center")
    text_ax(ax, 0.924, by - 0.014, sub,   size=6.8, color="#9CA3AF", ha="center")

# ── legend strip at bottom ─────────────────────────────────────────────────────
for lx, lc, lt in [
        (0.30, C_TEAL,   "Potency"),
        (0.42, "#F59E0B","Cell-cycle"),
        (0.54, C_ORANGE, "SCIE axis"),
        (0.66, C_PURPLE, "Non-linear / LMM"),
]:
    ax.plot([lx, lx + 0.02], [0.028, 0.028],
            transform=ax.transAxes, color=lc, lw=4)
    text_ax(ax, lx + 0.03, 0.028, lt, size=8, color=lc, ha="left")

plt.tight_layout(pad=0.3)
out1 = os.path.join(OUT, "fig_slide1_analysis.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig1)
print(f"✓ {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("figures/part1/ranking_table.csv")

SIG_ORDER = ["ESC_ASSOU","ESC_WONG","BenPorath_ES1",
             "EMP","fMaSC/aMaSC","LIM_Stem",
             "LIM_LumProg","LIM_LumMat","stem_consensus"]

# colour per signature (consistent across panels)
SIG_COLORS = {s: c for s, c in zip(
    SIG_ORDER,
    ["#60A5FA","#34D399","#A78BFA","#FBBF24","#F472B6",
     "#38BDF8","#FB923C","#A3E635","#F87171"]
)}

fig2 = plt.figure(figsize=(16, 9))
fig2.patch.set_facecolor(C_DARK)

gs = fig2.add_gridspec(2, 3, left=0.05, right=0.97,
                        top=0.88, bottom=0.08,
                        wspace=0.38, hspace=0.42)

ax_title = fig2.add_axes([0, 0.90, 1, 0.10])
ax_title.axis("off"); ax_title.set_facecolor(C_DARK)
ax_title.text(0.5, 0.65,
    "Part 1 — Results: which stemness signature captures potency independently of cell cycle?",
    ha="center", va="center", fontsize=15, fontweight="bold",
    color=C_GOLD, transform=ax_title.transAxes)
ax_title.text(0.5, 0.15,
    "Per-patient Fisher z meta-analysis  ·  Visvader (31 patients)  &  Klughammer (19 patients)  ·  Global (all subtypes)",
    ha="center", va="center", fontsize=9.5, color="#9CA3AF",
    transform=ax_title.transAxes)

# helper to style axes
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.set_title(title, color="white", fontsize=9.5, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color="#9CA3AF", fontsize=8.5)
    ax.set_ylabel(ylabel, color="#9CA3AF", fontsize=8.5)
    ax.grid(color="#1F2937", linewidth=0.8, zorder=0)


# ── Panel A: Tradeoff R²_CC vs partial_r_CT2 ─────────────────────────────────
axA = fig2.add_subplot(gs[0, 0])
style_ax(axA, "A   CC contamination vs net potency signal",
         "R²(cell cycle → signature)   [contamination →]",
         "Partial r(sig, CT2 | CC)   [potency ↑]")

for ds, mk in [("Visvader", "o"), ("Klughammer", "D")]:
    sub = df[df["dataset"] == ds]
    for _, row in sub.iterrows():
        x, y = row["R2_CC"], row["partial_r_CT2"]
        sig  = row["signature"]
        col  = SIG_COLORS.get(sig, "white")
        axA.scatter(x, y, color=col, s=90, marker=mk, zorder=4,
                    edgecolors="white" if ds == "Klughammer" else col,
                    linewidths=1.2 if ds == "Klughammer" else 0)
        if ds == "Visvader":
            axA.annotate(sig.replace("BenPorath_ES1","BenPorath")
                           .replace("fMaSC/aMaSC","fMaSC")
                           .replace("LIM_MAMMARY_STEM_CELL_UP","LIM_Stem")
                           .replace("LIM_MAMMARY_LUMINAL_PROGENITOR_UP","LumProg")
                           .replace("LIM_MAMMARY_LUMINAL_MATURE_UP","LumMat")
                           .replace("stem_consensus","stem_cons"),
                         (x, y), xytext=(5, 3), textcoords="offset points",
                         fontsize=6.5, color=col, zorder=5)

axA.axhline(0, color="#4B5563", ls="--", lw=0.8)
axA.axvline(0, color="#4B5563", ls="--", lw=0.8)
axA.set_xlim(-0.02, 0.62); axA.set_ylim(-0.6, 0.68)
# ideal quadrant annotation
axA.fill_betweenx([-0.02, 0.68], -0.02, 0.10,
                   color=C_TEAL, alpha=0.06, zorder=0)
axA.text(0.005, 0.62, "← ideal", color=C_TEAL, fontsize=7.5, alpha=0.8)

handles = [
    plt.scatter([], [], marker="o", color="white", s=60, label="Visvader"),
    plt.scatter([], [], marker="D", color="none", edgecolors="white",
                linewidths=1.5, s=60, label="Klughammer"),
]
axA.legend(handles=handles, fontsize=7.5, framealpha=0.2,
           facecolor="#1F2937", edgecolor="#374151", labelcolor="white")


# ── Panel B: Ranked bar — partial_r_CT2 (both datasets) ──────────────────────
axB = fig2.add_subplot(gs[0, 1])
style_ax(axB, "B   Potency signal after CC control",
         "Partial r(sig, CytoTRACE2 | CC)   [pooled across patients]",
         "")

vis  = df[df["dataset"] == "Visvader"].set_index("signature")
klug = df[df["dataset"] == "Klughammer"].set_index("signature")

y = np.arange(len(SIG_ORDER))
bar_h = 0.35
for i, sig in enumerate(SIG_ORDER):
    col = SIG_COLORS.get(sig, "grey")
    v_v = vis.loc[sig, "partial_r_CT2"] if sig in vis.index else np.nan
    v_k = klug.loc[sig, "partial_r_CT2"] if sig in klug.index else np.nan
    if not np.isnan(v_v):
        axB.barh(y[i] + bar_h/2, v_v, bar_h, color=col, alpha=0.9, zorder=3)
        axB.text(max(v_v + 0.01, 0.01), y[i] + bar_h/2,
                 f"{v_v:+.2f}", va="center", fontsize=6.5, color=col)
    if not np.isnan(v_k):
        axB.barh(y[i] - bar_h/2, v_k, bar_h, color=col, alpha=0.45,
                 hatch="//", zorder=3, edgecolor=col, linewidth=0)
        axB.text(max(v_k + 0.01, 0.01), y[i] - bar_h/2,
                 f"{v_k:+.2f}", va="center", fontsize=6.5, color=col, alpha=0.75)

axB.set_yticks(y)
short = {"ESC_ASSOU":"ESC_ASSOU","ESC_WONG":"ESC_WONG",
         "BenPorath_ES1":"BenPorath","EMP":"EMP","fMaSC/aMaSC":"fMaSC",
         "LIM_Stem":"LIM_Stem","LIM_LumProg":"LIM_LumProg",
         "LIM_LumMat":"LIM_LumMat","stem_consensus":"stem_cons"}
axB.set_yticklabels([short.get(s, s) for s in SIG_ORDER],
                    fontsize=8, color="white")
axB.axvline(0, color="#6B7280", lw=1)
axB.set_xlim(-0.65, 0.75)
pat_v = mpatches.Patch(color="white", alpha=0.9, label="Visvader (solid)")
pat_k = mpatches.Patch(color="white", alpha=0.45, hatch="//", label="Klughammer (hatched)")
axB.legend(handles=[pat_v, pat_k], fontsize=7, framealpha=0.2,
           facecolor="#1F2937", edgecolor="#374151", labelcolor="white",
           loc="lower right")


# ── Panel C: R²_CC heatmap ───────────────────────────────────────────────────
axC = fig2.add_subplot(gs[0, 2])
style_ax(axC, "C   Cell-cycle variance explained  (R²)",
         "Dataset", "")

data_r2 = np.array([
    [vis.loc[s, "R2_CC"] if s in vis.index else np.nan for s in SIG_ORDER],
    [klug.loc[s, "R2_CC"] if s in klug.index else np.nan for s in SIG_ORDER],
])
im = axC.imshow(data_r2, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.65,
                 origin="upper")
axC.set_xticks(range(len(SIG_ORDER)))
axC.set_xticklabels([short.get(s, s) for s in SIG_ORDER],
                     rotation=45, ha="right", fontsize=7.5, color="white")
axC.set_yticks([0, 1])
axC.set_yticklabels(["Visvader", "Klughammer"], fontsize=8.5, color="white")
for j in range(data_r2.shape[1]):
    for i in range(data_r2.shape[0]):
        v = data_r2[i, j]
        if not np.isnan(v):
            txt_col = "white" if v > 0.35 else "black"
            axC.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=7, color=txt_col, fontweight="bold")
plt.colorbar(im, ax=axC, shrink=0.6, pad=0.03).ax.tick_params(
    colors="white", labelsize=7)
axC.tick_params(colors="#9CA3AF")


# ── Panel D: SCIE axis survival ───────────────────────────────────────────────
axD = fig2.add_subplot(gs[1, 0])
style_ax(axD, "D   SCIE axis survival after CC residualisation",
         "r(sig, immunogenic)  — original",
         "r(sig_ccfree, immunogenic)  — CC-free")

for ds, mk, msize in [("Visvader","o",90), ("Klughammer","D",90)]:
    sub = df[df["dataset"] == ds]
    for _, row in sub.iterrows():
        x   = row["r_immune"]
        y_v = row["r_immune_ccfree"]
        sig = row["signature"]
        col = SIG_COLORS.get(sig, "white")
        if not (np.isnan(x) or np.isnan(y_v)):
            axD.scatter(x, y_v, color=col, s=msize, marker=mk, zorder=4,
                        edgecolors="white" if ds == "Klughammer" else col,
                        linewidths=1.2 if ds == "Klughammer" else 0)
            axD.annotate(short.get(sig, sig), (x, y_v),
                         xytext=(4, 2), textcoords="offset points",
                         fontsize=6.5, color=col, zorder=5)

lim = 0.25
axD.plot([-lim, lim], [-lim, lim], color="#6B7280", ls="--", lw=1, alpha=0.7)
axD.axhline(0, color="#4B5563", ls=":", lw=0.6)
axD.axvline(0, color="#4B5563", ls=":", lw=0.6)
axD.set_xlim(-lim, lim); axD.set_ylim(-lim, lim)
axD.text(-lim + 0.01, lim - 0.02,
         "y = x\n(no change)", fontsize=7, color="#6B7280", style="italic")
axD.text(0.03, -lim + 0.03,
         "SCIE axis weakened\nby CC removal", fontsize=7, color=C_ORANGE,
         style="italic")
axD.text(-lim + 0.01, 0.03,
         "SCIE axis\nstrengthened", fontsize=7, color=C_TEAL, style="italic")


# ── Panel E: Before/after r_CT2 ──────────────────────────────────────────────
axE = fig2.add_subplot(gs[1, 1])
style_ax(axE, "E   CytoTRACE2 alignment before & after CC removal",
         "", "r(sig, CytoTRACE2)")

y = np.arange(len(SIG_ORDER))
for i, sig in enumerate(SIG_ORDER):
    col = SIG_COLORS.get(sig, "grey")
    for ds, xoffset, mk, alpha in [
            ("Visvader",   -0.12, "o", 1.0),
            ("Klughammer", +0.12, "D", 1.0),
    ]:
        sub = df[df["dataset"] == ds]
        if sub.empty or sig not in sub["signature"].values:
            continue
        row = sub[sub["signature"] == sig].iloc[0]
        r_o = row["r_CT2"]
        r_c = row["r_CT2_ccfree"]
        if np.isnan(r_o) or np.isnan(r_c):
            continue
        yi = y[i] + xoffset
        axE.scatter(r_o, yi, color=col, s=55, marker=mk, zorder=4, alpha=0.9)
        axE.scatter(r_c, yi, color=col, s=55, marker=mk, zorder=4,
                    edgecolors=col, facecolors="none", lw=1.5, alpha=0.9)
        axE.annotate("", xy=(r_c, yi), xytext=(r_o, yi),
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                     lw=1.2, alpha=0.6,
                                     mutation_scale=8))

axE.axvline(0, color="#6B7280", lw=1, ls="--")
axE.set_yticks(y)
axE.set_yticklabels([short.get(s, s) for s in SIG_ORDER], fontsize=8, color="white")
axE.set_xlim(-0.35, 0.70)
axE.text(0.62, len(SIG_ORDER) - 0.6,
         "filled = original\nopen = CC-free",
         fontsize=7, color="#9CA3AF", va="top")


# ── Panel F: Summary text ──────────────────────────────────────────────────────
axF = fig2.add_subplot(gs[1, 2])
axF.set_facecolor("#111827")
axF.axis("off")
for spine in axF.spines.values():
    spine.set_edgecolor("#374151")

axF.text(0.5, 0.97, "F   Key takeaways", ha="center", va="top",
         fontsize=9.5, fontweight="bold", color="white",
         transform=axF.transAxes)

findings = [
    (C_TEAL,   "ESC_WONG best overall",
               "Highest partial r(CT2|CC) in both datasets\n"
               "(0.46 Vis · 0.52 Klug)  with moderate R²_CC (0.30)"),
    (C_TEAL,   "ESC_ASSOU close second",
               "Strong partial r but higher CC contamination\n"
               "(R²_CC 0.31–0.38 vs 0.30 for ESC_WONG)"),
    ("#F59E0B", "stem_consensus — severe CC contamination",
               "R²_CC = 0.55 in both datasets\n"
               "Partial r drops substantially; cannot use as-is"),
    ("#F59E0B", "TNBC subtypes most CC-contaminated",
               "ESC_ASSOU R²_CC = 0.53 in TNBC (Visvader)\n"
               "Proliferative biology confounds stemness signal"),
    (C_ORANGE,  "SCIE axis survives CC removal",
               "r(sig_ccfree, immune) ≈ r(sig, immune)\n"
               "Anti-correlation not driven by cell cycle"),
    (C_GOLD,   "Recommendation",
               "Use ESC_WONG as primary stemness axis\n"
               "Or ESC_WONG_ccfree for CC-controlled analyses"),
]

fy = 0.88
for col, title, body in findings:
    axF.plot([0.02, 0.06], [fy, fy], transform=axF.transAxes,
             color=col, lw=3.5)
    axF.text(0.09, fy, title, transform=axF.transAxes,
             fontsize=8.5, fontweight="bold", color=col, va="center")
    fy -= 0.055
    axF.text(0.09, fy, body, transform=axF.transAxes,
             fontsize=7.2, color="#D1D5DB", va="center")
    fy -= 0.100

plt.tight_layout(rect=[0, 0, 1, 0.90])
out2 = os.path.join(OUT, "fig_slide2_results.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig2)
print(f"✓ {out2}")
