"""
Regenerate presentation slides v2:
  fig_slide1_analysis.png  — white background, clean aligned layout
  fig_slide2_results.png   — composite of real generated figures
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
import os

OUT = "figures/part1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ── Palette (white-background friendly) ───────────────────────────────────────
W_BG     = "#FFFFFF"
W_PANEL  = "#F3F4F6"   # light grey panel background
W_TEXT   = "#111827"   # near-black
W_SUB    = "#6B7280"   # secondary text
W_LINE   = "#D1D5DB"   # gridlines / dividers

C_DATA   = "#1D4ED8"   # blue
C_ANAL   = "#0F766E"   # teal – within-patient steps
C_POOL   = "#6D28D9"   # purple – pooled steps
C_CC     = "#B45309"   # amber – CC-free steps
C_SCIE   = "#B91C1C"   # red   – SCIE axis
C_OUT    = "#374151"   # dark grey – outputs


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def rbox(ax, x, y, w, h, fc, ec="none", lw=1.2, alpha=1.0,
         radius=0.02, zorder=2, clip=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder, clip_on=clip)
    ax.add_patch(p)

def T(ax, x, y, s, size=9, color=W_TEXT, weight="normal",
      ha="center", va="center", zorder=6, style="normal", alpha=1.0,
      wrap=False):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder, fontstyle=style, alpha=alpha,
            multialignment=ha if ha != "center" else "center")

def arr(ax, x0, y0, x1, y1, color="#9CA3AF", lw=1.6, style="->",
        ms=10, zorder=4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=ms),
                zorder=zorder)

def hline(ax, x0, x1, y, color=W_LINE, lw=0.8, ls="--"):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=1)

def section_header(ax, x, y, w, label, color):
    rbox(ax, x, y, w, 0.38, fc=color, ec="none", radius=0.04, zorder=3)
    T(ax, x + w/2, y + 0.19, label, size=8.5, color="white",
      weight="bold", zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1  —  ANALYSIS WORKFLOW  (white background)
# ══════════════════════════════════════════════════════════════════════════════

FW, FH = 16, 9   # figure inches
fig1, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis("off")
fig1.patch.set_facecolor(W_BG)
ax.set_facecolor(W_BG)

# ── Column definitions ─────────────────────────────────────────────────────────
#   DATA      x: 0.25 – 2.90   w=2.65
#   ANALYSIS  x: 3.20 – 10.0   w=6.80
#   METRICS   x: 10.30 – 13.10  w=2.80
#   OUTPUT    x: 13.40 – 15.75  w=2.35

C1x, C1w = 0.25, 2.65
C2x, C2w = 3.20, 6.80
C3x, C3w = 10.30, 2.80
C4x, C4w = 13.40, 2.35

YMAX = 8.10   # content top
YMIN = 0.55   # content bottom

# ── Title ──────────────────────────────────────────────────────────────────────
T(ax, FW/2, 8.65,
  "Part 1 — Stemness signature evaluation & CC-free synthetic index",
  size=16, weight="bold", color=W_TEXT)
T(ax, FW/2, 8.28,
  "Which stemness signature best captures developmental potency, independently of cell cycle?",
  size=10, color=W_SUB, style="italic")

# thin rule under title
ax.plot([0.3, 15.7], [8.12, 8.12], color=W_LINE, lw=1.0)

# ── Column headers ─────────────────────────────────────────────────────────────
for x, w, label, col in [
    (C1x, C1w, "INPUT",    C_DATA),
    (C2x, C2w, "ANALYSIS", C_ANAL),
    (C3x, C3w, "METRICS",  C_POOL),
    (C4x, C4w, "OUTPUT",   C_OUT),
]:
    section_header(ax, x, YMAX - 0.36, w, label, col)

# vertical dividers
for xd in [C2x - 0.14, C3x - 0.14, C4x - 0.14]:
    ax.plot([xd, xd], [YMIN, YMAX], color=W_LINE, lw=0.7, zorder=1)

# ── COLUMN 1: INPUT ────────────────────────────────────────────────────────────
def data_box(y0, y1, title, lines, col):
    rbox(ax, C1x, y0, C1w, y1-y0, fc=col+"18", ec=col, lw=1.4, radius=0.08)
    T(ax, C1x + C1w/2, y1 - 0.28, title, size=10, weight="bold", color=col)
    for i, line in enumerate(lines):
        T(ax, C1x + 0.18, y1 - 0.58 - i*0.28, line, size=8.2,
          color=W_SUB, ha="left")

data_box(5.35, 7.68,
         "Visvader",
         ["Primary breast tumour",
          "121,632 cells", "31 patients",
          "ER · HER2 · TNBC",
          "Raw counts available"],
         C_DATA)

data_box(2.70, 5.10,
         "Klughammer",
         ["Metastatic biopsy",
          "83,490 cells", "19 patients",
          "ER · HER2 · TNBC",
          "Log-norm only (no counts)"],
         C_DATA)

# 9 signatures box
rbox(ax, C1x, YMIN, C1w, 2.00, fc="#FEF3C720", ec=C_CC, lw=1.2, radius=0.08)
T(ax, C1x + C1w/2, YMIN + 1.78, "9 stemness signatures", size=9.5,
  weight="bold", color=C_CC)
sigs = ["ESC_ASSOU · ESC_WONG · BenPorath_ES1",
        "EMP · fMaSC/aMaSC",
        "LIM_Stem · LIM_LumProg · LIM_LumMat",
        "stem_consensus"]
for i, s in enumerate(sigs):
    T(ax, C1x + C1w/2, YMIN + 1.38 - i*0.28, s, size=7.5, color=W_SUB)

# arrows data → analysis
arr(ax, C1x + C1w, 6.52, C2x, 6.52, color=C_DATA)
arr(ax, C1x + C1w, 3.90, C2x, 3.90, color=C_DATA)

# ── COLUMN 2: ANALYSIS ─────────────────────────────────────────────────────────
# Step positions (top of each step box)
STEPS = [
    # (y_top, height, color, title, body_lines, tag)
    (7.68, 1.16, C_ANAL,
     "Per-patient iteration  (≥50 cells / patient)",
     ["For each patient independently:"],
     "within-patient"),
    (6.22, 1.12, C_ANAL,
     "Spearman r  (×4 targets, within patient)",
     ["r(sig, CytoTRACE2)  ·  r(sig, S.Score)",
      "r(sig, G2M.Score)   ·  r(sig, immune)"],
     "within-patient"),
    (4.80, 1.08, C_ANAL,
     "Fisher z meta-analysis  →  pooled r ± 95% CI",
     ["Patients weighted by √(n − 3)",
      "Gives: pooled estimate + CI + % patients r > 0"],
     "within-patient"),
    (3.48, 1.00, C_POOL,
     "R²(CC→sig)  ·  partial r(CT2 | CC)  ·  LMM β",
     ["OLS variance explained by S.Score + G2M.Score",
      "Partial r controls for CC; LMM adds patient random effect"],
     "pooled"),
    (2.20, 0.96, C_CC,
     "CC-free residualisation  (per patient)",
     ["OLS per patient:  sig ~ S.Score + G2M.Score",
      "Residuals stored as  score_{sig}_ccfree"],
     "cc-free"),
    (0.92, 0.96, C_SCIE,
     "SCIE axis check",
     ["r(residual, immune)  vs  r(sig, immune)",
      "Does stem × immune anti-corr survive CC removal?"],
     "scie"),
]

TAG_COLORS = {
    "within-patient": C_ANAL,
    "pooled"        : C_POOL,
    "cc-free"       : C_CC,
    "scie"          : C_SCIE,
}
TAG_LABELS = {
    "within-patient": "per patient",
    "pooled"        : "pooled",
    "cc-free"       : "per patient",
    "scie"          : "validation",
}

for i, (yt, h, col, title, body, tag) in enumerate(STEPS):
    yb = yt - h
    rbox(ax, C2x, yb, C2w, h - 0.08,
         fc=col+"12", ec=col, lw=1.3, radius=0.07, zorder=2)
    # left accent stripe
    rbox(ax, C2x, yb, 0.14, h - 0.08, fc=col, ec="none", radius=0.04, zorder=3)
    # title
    T(ax, C2x + 0.26, yt - 0.28, title,
      size=9.5, weight="bold", color=col, ha="left", zorder=5)
    # body lines
    for j, line in enumerate(body):
        T(ax, C2x + 0.26, yt - 0.56 - j*0.25, line,
          size=8, color=W_SUB, ha="left", zorder=5)
    # tag badge
    tc = TAG_COLORS[tag]
    rbox(ax, C2x + C2w - 1.45, yt - 0.41, 1.35, 0.26,
         fc=tc+"22", ec=tc, lw=0.9, radius=0.06, zorder=4)
    T(ax, C2x + C2w - 0.775, yt - 0.28, TAG_LABELS[tag],
      size=7, color=tc, weight="bold", zorder=5)
    # arrow to next step
    if i < len(STEPS) - 1:
        ay = yb + 0.04
        acol = col if tag == STEPS[i+1][5] else "#9CA3AF"
        arr(ax, C2x + C2w/2, ay, C2x + C2w/2, ay - 0.14,
            color=acol, lw=1.4, ms=9)

# patient mini-schematic inside step 1
PAT_Y = 7.68 - 0.75
for pi, (pid, nc, col) in enumerate([
    ("Patient 1", "n=5,512", "#60A5FA"),
    ("Patient 2", "n=3,403", "#34D399"),
    ("···",       "",        "#9CA3AF"),
    ("Pat. 31",   "n=808",   "#F472B6"),
]):
    px = C2x + 0.18 + pi * 1.50
    rbox(ax, px, PAT_Y - 0.22, 1.30, 0.40,
         fc=col+"22", ec=col, lw=0.9, radius=0.05, zorder=3)
    T(ax, px + 0.65, PAT_Y - 0.02, pid, size=7.5,
      color=col, weight="bold", zorder=5)
    T(ax, px + 0.65, PAT_Y - 0.22, nc, size=7,
      color=W_SUB, zorder=5)

# ── COLUMN 3: METRICS ──────────────────────────────────────────────────────────
METRICS = [
    (C_ANAL,  "r(sig, CytoTRACE2)",       "developmental potency alignment"),
    (C_ANAL,  "partial r(CT2 | CC)",       "net potency after CC control"),
    (C_POOL,  "R²(CC → signature)",        "fraction of variance = cell cycle"),
    (C_POOL,  "r(sig, S.Score / G2M)",     "S-phase & G2M contamination"),
    (C_SCIE,  "r(sig, immunogenic)",       "SCIE axis relevance"),
    (C_POOL,  "dCor(sig, CytoTRACE2)",    "nonlinear potency association"),
    (C_POOL,  "LMM β  CytoTRACE2",        "patient-corrected coefficient"),
]
MH     = (YMAX - 0.38 - YMIN) / len(METRICS)   # height per metric row
for mi, (col, name, sub) in enumerate(METRICS):
    my = YMAX - 0.38 - (mi + 1) * MH
    rbox(ax, C3x, my + 0.06, C3w, MH - 0.12,
         fc=W_PANEL, ec=W_LINE, lw=0.8, radius=0.06, zorder=2)
    rbox(ax, C3x, my + 0.06, 0.10, MH - 0.12, fc=col, ec="none",
         radius=0.04, zorder=3)
    T(ax, C3x + 0.22, my + MH/2 + 0.07, name,
      size=8.5, weight="bold", color=W_TEXT, ha="left")
    T(ax, C3x + 0.22, my + MH/2 - 0.12, sub,
      size=7.2, color=W_SUB, ha="left")

# composite score box
rbox(ax, C3x, YMIN, C3w, 0.55,
     fc="#FEF9C3", ec="#CA8A04", lw=1.4, radius=0.08, zorder=2)
T(ax, C3x + C3w/2, YMIN + 0.38, "Composite rank score", size=9,
  weight="bold", color="#92400E")
T(ax, C3x + C3w/2, YMIN + 0.16,
  "0.5 × r(CT2) + 0.5 × partial_r − R²(CC)",
  size=7.5, color="#A16207")

# arrow analysis → metrics
arr(ax, C2x + C2w, (YMAX + YMIN)/2, C3x, (YMAX + YMIN)/2, color="#9CA3AF")

# ── COLUMN 4: OUTPUT ───────────────────────────────────────────────────────────
OUTPUTS = [
    (C_ANAL,  "Heatmap",           "9 sigs × 8 metrics × subtypes"),
    (C_ANAL,  "Forest plot",       "pooled r ± 95% CI  (CT2, CC, immune)"),
    (C_POOL,  "Tradeoff plot",     "R²_CC vs partial r — best sig region"),
    (C_CC,    "CC-free comparison","before vs after residualisation"),
    (C_SCIE,  "SCIE survival",     "r_immune: original vs CC-free"),
    ("#CA8A04","Ranking table",    "CSV · 2 datasets · 18 rows"),
    (C_DATA,  "CC-free h5ads",     "score_{sig}_ccfree in .obs"),
]
OH = (YMAX - 0.38 - YMIN) / len(OUTPUTS)
for oi, (col, name, sub) in enumerate(OUTPUTS):
    oy = YMAX - 0.38 - (oi + 1) * OH
    rbox(ax, C4x, oy + 0.05, C4w, OH - 0.10,
         fc=W_PANEL, ec=W_LINE, lw=0.8, radius=0.06, zorder=2)
    rbox(ax, C4x, oy + 0.05, 0.09, OH - 0.10, fc=col, ec="none",
         radius=0.04, zorder=3)
    T(ax, C4x + 0.20, oy + OH/2 + 0.06, name,
      size=8.5, weight="bold", color=W_TEXT, ha="left")
    T(ax, C4x + 0.20, oy + OH/2 - 0.13, sub,
      size=7, color=W_SUB, ha="left")

# arrow metrics → output
arr(ax, C3x + C3w, (YMAX + YMIN)/2, C4x, (YMAX + YMIN)/2, color="#9CA3AF")

# ── Legend strip ───────────────────────────────────────────────────────────────
ly = 0.22
for lx, col, lbl in [
    (0.40, C_ANAL, "Within-patient"),
    (2.40, C_POOL, "Pooled (all patients)"),
    (5.00, C_CC,   "CC-free residualisation"),
    (7.40, C_SCIE, "SCIE axis validation"),
]:
    rbox(ax, lx, ly - 0.08, 0.28, 0.18, fc=col, ec="none",
         radius=0.03, zorder=3)
    T(ax, lx + 0.38, ly + 0.01, lbl, size=8, color=W_SUB, ha="left")

out1 = os.path.join(OUT, "fig_slide1_analysis.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight", facecolor=W_BG)
plt.close(fig1)
print(f"✓ {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2  —  RESULTS  (composite of real figures)
# ══════════════════════════════════════════════════════════════════════════════

def load_img(path):
    return mpimg.imread(path)

# Load figure files
imgs = {
    "tradeoff_v" : load_img(f"{OUT}/Visvader_tradeoff.png"),
    "tradeoff_k" : load_img(f"{OUT}/Klughammer_tradeoff.png"),
    "forest_v"   : load_img(f"{OUT}/Visvader_forest.png"),
    "forest_k"   : load_img(f"{OUT}/Klughammer_forest.png"),
    "scie_v"     : load_img(f"{OUT}/Visvader_scie_survival.png"),
    "scie_k"     : load_img(f"{OUT}/Klughammer_scie_survival.png"),
    "ccfree_v"   : load_img(f"{OUT}/Visvader_ccfree_comparison.png"),
    "ccfree_k"   : load_img(f"{OUT}/Klughammer_ccfree_comparison.png"),
}

# Layout: title + 3-row × 2-col grid
# Row 1: tradeoff Vis | tradeoff Klug
# Row 2: forest Vis | forest Klug
# Row 3: scie Vis | scie Klug

BG2 = "#F9FAFB"

fig2 = plt.figure(figsize=(18, 11))
fig2.patch.set_facecolor(BG2)

# Title
fig2.text(0.5, 0.975,
          "Part 1 — Results: which stemness signature captures potency independently of cell cycle?",
          ha="center", va="top", fontsize=15, fontweight="bold", color=W_TEXT)
fig2.text(0.5, 0.952,
          "Per-patient Fisher z meta-analysis  ·  Visvader (31 patients, primary)  "
          "and  Klughammer (19 patients, metastatic)  ·  Global (all subtypes pooled)",
          ha="center", va="top", fontsize=9.5, color=W_SUB, style="italic")

# Dataset column headers
fig2.text(0.275, 0.928, "Visvader  (primary tumour)", ha="center",
          fontsize=11, fontweight="bold", color=C_DATA)
fig2.text(0.725, 0.928, "Klughammer  (metastatic)", ha="center",
          fontsize=11, fontweight="bold", color="#7C3AED")

# thin rule
line_ax = fig2.add_axes([0.02, 0.918, 0.96, 0.002])
line_ax.set_facecolor(W_LINE); line_ax.axis("off")

# Row labels (left margin)
row_labels = [
    (0.775, "CC contamination\nvs potency signal",
     "R² (cell cycle → sig)  vs  partial r (sig, CT2 | CC)\nBest signatures: top-left quadrant (low CC, high potency)"),
    (0.490, "SCIE axis survival\nafter CC residualisation",
     "r(sig, immunogenic)  original  vs  CC-free\nPoints near diagonal: anti-correlation is not driven by cell cycle"),
    (0.205, "Before → after\nCC residualisation",
     "Effect of OLS residualisation on r(CT2), r(S.Score), r(G2M), r(immune)\nFilled = original  ·  Open = CC-free  ·  Arrows show direction of change"),
]

# Grid: 3 rows × 2 cols
# Axes positions: [left, bottom, width, height]
col_left  = [0.03, 0.50]
col_w     = 0.45
row_bot   = [0.63, 0.345, 0.06]
row_h     = 0.265

panel_imgs = [
    [imgs["tradeoff_v"],  imgs["tradeoff_k"]],
    [imgs["scie_v"],      imgs["scie_k"]],
    [imgs["ccfree_v"],    imgs["ccfree_k"]],
]
panel_labels = [
    ["A  Visvader — CC contamination vs net potency",
     "B  Klughammer — CC contamination vs net potency"],
    ["C  Visvader — SCIE axis survival",
     "D  Klughammer — SCIE axis survival"],
    ["E  Visvader — Before vs after CC residualisation",
     "F  Klughammer — Before vs after CC residualisation"],
]

for ri in range(3):
    for ci in range(2):
        ax_pos = [col_left[ci], row_bot[ri], col_w, row_h]
        ax_img = fig2.add_axes(ax_pos)
        ax_img.imshow(panel_imgs[ri][ci])
        ax_img.axis("off")
        # Panel label
        ax_img.set_title(panel_labels[ri][ci],
                         fontsize=9, fontweight="bold",
                         color=W_TEXT, pad=4,
                         loc="left")
        # Border
        for spine in ax_img.spines.values():
            spine.set_edgecolor(W_LINE)
            spine.set_linewidth(0.6)
            spine.set_visible(True)

# Row annotations on the right
for yfrac, row_title, row_note in row_labels:
    fig2.text(0.975, yfrac, row_title,
              ha="right", va="center", fontsize=9, fontweight="bold",
              color=W_TEXT, rotation=90)

# Bottom caption
fig2.text(0.5, 0.025,
          "Composite rank score = 0.5 × r(CytoTRACE2) + 0.5 × partial r(CT2|CC) − R²(CC)  "
          "·  ESC_WONG ranked #1 in both datasets  ·  stem_consensus excluded: R²_CC = 0.55",
          ha="center", va="bottom", fontsize=8.5, color=W_SUB, style="italic")

out2 = os.path.join(OUT, "fig_slide2_results.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor=BG2)
plt.close(fig2)
print(f"✓ {out2}")
