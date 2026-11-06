"""
Conceptual analysis slide — clean 3-panel didactic diagram.
Output: figures/part1/fig_slide1_concept.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT = "figures/part1"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# colours
C1 = "#DC2626"; C1_L = (0.86, 0.08, 0.15, 0.07)   # red
C2 = "#2563EB"; C2_L = (0.15, 0.39, 0.92, 0.07)   # blue
C3 = "#059669"; C3_L = (0.02, 0.59, 0.41, 0.07)   # green
BG   = "#FFFFFF"
GREY = "#9CA3AF"
DARK = "#1F2937"
LITE = "#F3F4F6"

fig, axes = plt.subplots(1, 3, figsize=(18, 9))
fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 10.5)
    ax.axis("off")
    ax.set_facecolor(BG)


def box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad={radius}", facecolor=fc,
        edgecolor=ec, linewidth=lw, zorder=2, clip_on=False))

def T(ax, x, y, s, size=10, color=DARK, weight="normal",
      ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, fontstyle=style, multialignment="center",
            zorder=5, linespacing=1.5, clip_on=False)

def badge(ax, x, y, n, color):
    ax.add_patch(plt.Circle((x, y), 0.42, color=color, zorder=6, clip_on=False))
    T(ax, x, y, str(n), 15, "white", "bold")

def dn_arrow(ax, x, y0, y1, color, lw=1.8):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=5)


# ── Title ──────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         "Are stemness signatures measuring stem-like identity — or just proliferation?",
         ha="center", va="top", fontsize=17, fontweight="bold", color=DARK)
fig.text(0.5, 0.925,
         "Cell-cycle contamination check  ·  9 signatures  ·  31 + 19 breast cancer patients",
         ha="center", va="top", fontsize=11, color=GREY, style="italic")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — THE PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
badge(ax, 0.85, 9.7, 1, C1)
T(ax, 5.0, 9.7, "The problem", 15, C1, "bold")
T(ax, 5.0, 9.05, "Cycling cells also score high\non stemness signatures", 11)

# scatter panel
box(ax, 0.4, 2.10, 9.2, 6.65, LITE, GREY, lw=0.8, radius=0.2)
T(ax, 5.0, 8.45, "Stemness score vs S.Score  (example patient)", 9.5, GREY)

np.random.seed(7)
n = 55
ss  = np.random.randn(n) * 0.28
sig = ss * 0.55 + np.random.randn(n) * 0.20

def sc(vx, vy):
    return (1.3 + (vx+0.82)/1.64*7.0,
            2.60 + (vy+0.62)/1.24*5.70)

for i in range(n):
    px, py = sc(ss[i], sig[i])
    col = C1 if ss[i]>0.15 else (C3 if ss[i]<-0.15 else GREY)
    ax.scatter(px, py, s=38, color=col, alpha=0.75, zorder=4,
               edgecolors="white", linewidths=0.4)
p0 = sc(-0.80, -0.44); p1 = sc(0.80, 0.44)
ax.plot([p0[0],p1[0]], [p0[1],p1[1]], color=C1, lw=2, ls="--", alpha=0.8)

T(ax, 5.0, 2.33, "S.Score  (cell-cycle phase)", 9.5, GREY)
ax.text(0.02, 5.45, "Stemness\nscore", fontsize=9.5, color=GREY,
        ha="center", va="center", rotation=90)
T(ax, 7.9, 7.95, "r ≈ +0.45", 11, C1, "bold")
ax.annotate("high CC &\nhigh stem?", xy=sc(0.52, 0.40),
            xytext=(7.0, 7.40), fontsize=9, color=C1,
            arrowprops=dict(arrowstyle="->", color=C1, lw=1.2), ha="center")
ax.annotate("low CC,\nstill stem", xy=sc(-0.44,-0.12),
            xytext=(2.2, 3.80), fontsize=9, color=C3,
            arrowprops=dict(arrowstyle="->", color=C3, lw=1.2), ha="center")

box(ax, 0.4, 0.20, 9.2, 1.70, C1_L, C1, lw=1.3, radius=0.2)
T(ax, 5.0, 1.08,
  "How much of each signature's variance\nis explained by cell cycle alone?",
  11, C1, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — THE APPROACH
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
badge(ax, 0.85, 9.7, 2, C2)
T(ax, 5.0, 9.7, "The approach", 15, C2, "bold")
T(ax, 5.0, 9.05, "Per patient, per signature", 11)

# Step A
box(ax, 0.4, 5.80, 9.2, 2.90, C2_L, C2, lw=1.4, radius=0.2)
T(ax, 5.0, 8.40, "A  —  Quantify CC contamination", 12, C2, "bold")
T(ax, 5.0, 7.78,
  "OLS:  sig ~ S.Score + G2M.Score   (per patient)",
  10.5, DARK, style="italic")
T(ax, 5.0, 7.15,
  "R²(CC→sig)  =  fraction of variance\nexplained by cell-cycle phase",
  10.5, GREY)
T(ax, 5.0, 6.22,
  "→  pooled across patients via Fisher z",
  9.5, C2)

dn_arrow(ax, 5.0, 5.77, 5.42, C2)

# Step B
box(ax, 0.4, 2.00, 9.2, 3.20, C2_L, C2, lw=1.4, radius=0.2)
T(ax, 5.0, 4.95, "B  —  Build a CC-free synthetic index", 12, C2, "bold")
T(ax, 5.0, 4.33, "Per-patient OLS residuals:", 10.5)
T(ax, 5.0, 3.70,
  "sig_ccfree  =  sig  −  β₁·S.Score  −  β₂·G2M",
  11, C2, "bold", style="italic")
T(ax, 5.0, 3.05,
  "Removes the CC component within each patient\nindependently — one value per cell",
  10.5, GREY)
T(ax, 5.0, 2.38,
  "Stored as  score_{sig}_ccfree  in the h5ad",
  9.5, GREY, style="italic")

box(ax, 0.4, 0.20, 9.2, 1.60, C2_L, C2, lw=1.3, radius=0.2)
T(ax, 5.0, 1.02,
  "Does genuine potency signal survive\nonce proliferation is removed?",
  11, C2, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — WHAT CHANGES?  (before / after concept)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
badge(ax, 0.85, 9.7, 3, C3)
T(ax, 5.0, 9.7, "What changes?", 15, C3, "bold")
T(ax, 5.0, 9.05, "Before vs after CC residualisation\n(same cells, same patient)", 11)

# synthetic data — two clusters: cycling (orange/red) and quiescent (blue/green)
np.random.seed(42)
n_cyc, n_qui = 28, 27
# cycling: high S.Score, moderate-high raw sig, but low TRUE stem state
ss_cyc  = np.random.randn(n_cyc) * 0.18 + 0.45
sig_cyc = ss_cyc * 0.62 + np.random.randn(n_cyc) * 0.12 - 0.10
# quiescent: low S.Score, variable raw sig, but high TRUE stem state
ss_qui  = np.random.randn(n_qui) * 0.18 - 0.42
sig_qui = ss_qui * 0.62 + np.random.randn(n_qui) * 0.12 + 0.25

ss_all  = np.concatenate([ss_cyc,  ss_qui])
sig_all = np.concatenate([sig_cyc, sig_qui])

# OLS residuals (simulate residualisation)
from numpy.polynomial import polynomial as P
beta = np.polyfit(ss_all, sig_all, 1)
sig_ccfree = sig_all - np.polyval(beta, ss_all)

col_cyc = C1   # red  = cycling
col_qui = C3   # green = quiescent / truly stem-like
cols = [col_cyc]*n_cyc + [col_qui]*n_qui

# helper to map data → axis coords
def sc3(vx, vy, x0, y0, W, H, xr, yr):
    px = x0 + (vx - xr[0]) / (xr[1] - xr[0]) * W
    py = y0 + (vy - yr[0]) / (yr[1] - yr[0]) * H
    return px, py

# ── BEFORE panel ──────────────────────────────────────────────────────────────
box(ax, 0.4, 5.50, 9.2, 3.10, LITE, GREY, lw=0.8, radius=0.2)
T(ax, 5.0, 8.30, "BEFORE  —  raw stemness score", 12, C1, "bold")

XR = (-0.85, 0.85); YR = (-0.70, 0.75)
X0, Y0, W, H = 0.9, 5.72, 8.2, 2.60
for i, (vx, vy) in enumerate(zip(ss_all, sig_all)):
    px, py = sc3(vx, vy, X0, Y0, W, H, XR, YR)
    ax.scatter(px, py, s=32, color=cols[i], alpha=0.72, zorder=4,
               edgecolors="white", linewidths=0.3)

# regression line
xl = np.array(XR)
yl = np.polyval(beta, xl)
p0b = sc3(xl[0], yl[0], X0, Y0, W, H, XR, YR)
p1b = sc3(xl[1], yl[1], X0, Y0, W, H, XR, YR)
ax.plot([p0b[0], p1b[0]], [p0b[1], p1b[1]],
        color=C1, lw=2, ls="--", alpha=0.8, zorder=5)

# axis ticks labels
T(ax, 5.0, 5.58, "S.Score  →", 8.5, GREY)
ax.text(0.55, 7.15, "sig ↑", fontsize=8.5, color=GREY,
        ha="center", va="center", rotation=90)
T(ax, 3.0, 6.05, "r ≈ +0.45", 10, C1, "bold")
ax.annotate("cycling cells\nscore high", xy=sc3(0.50, 0.40, X0, Y0, W, H, XR, YR),
            xytext=(8.2, 7.80), fontsize=8.5, color=C1, ha="center",
            arrowprops=dict(arrowstyle="->", color=C1, lw=1.1))
ax.annotate("quiescent\nbut stem-like", xy=sc3(-0.50, 0.22, X0, Y0, W, H, XR, YR),
            xytext=(2.2, 7.80), fontsize=8.5, color=C3, ha="center",
            arrowprops=dict(arrowstyle="->", color=C3, lw=1.1))

dn_arrow(ax, 5.0, 5.47, 5.12, C3)

# ── AFTER panel ───────────────────────────────────────────────────────────────
box(ax, 0.4, 1.85, 9.2, 3.15, C3_L, C3, lw=1.2, radius=0.2)
T(ax, 5.0, 4.70, "AFTER  —  CC-free residual  (sig_ccfree)", 12, C3, "bold")

YR2 = (-0.45, 0.45)
X0, Y0, W, H = 0.9, 2.07, 8.2, 2.55
for i, (vx, vy) in enumerate(zip(ss_all, sig_ccfree)):
    px, py = sc3(vx, vy, X0, Y0, W, H, XR, YR2)
    ax.scatter(px, py, s=32, color=cols[i], alpha=0.72, zorder=4,
               edgecolors="white", linewidths=0.3)

# flat line at 0
p0a = sc3(XR[0], 0, X0, Y0, W, H, XR, YR2)
p1a = sc3(XR[1], 0, X0, Y0, W, H, XR, YR2)
ax.plot([p0a[0], p1a[0]], [p0a[1], p1a[1]],
        color=C3, lw=2, ls="--", alpha=0.8, zorder=5)

T(ax, 5.0, 1.95, "S.Score  →", 8.5, GREY)
ax.text(0.55, 3.40, "sig_ccfree ↑", fontsize=8.5, color=GREY,
        ha="center", va="center", rotation=90)
T(ax, 3.0, 2.42, "r ≈ 0", 10, C3, "bold")
ax.annotate("cycling cells:\nresidual ≈ 0", xy=sc3(0.50, -0.05, X0, Y0, W, H, XR, YR2),
            xytext=(8.2, 4.30), fontsize=8.5, color=C1, ha="center",
            arrowprops=dict(arrowstyle="->", color=C1, lw=1.1))
ax.annotate("quiescent remain\nhigh sig_ccfree", xy=sc3(-0.50, 0.30, X0, Y0, W, H, XR, YR2),
            xytext=(2.2, 4.30), fontsize=8.5, color=C3, ha="center",
            arrowprops=dict(arrowstyle="->", color=C3, lw=1.1))

# legend dots
ax.scatter(1.05, 1.30, s=42, color=C1, zorder=5, edgecolors="white", lw=0.4)
T(ax, 2.45, 1.30, "cycling (high S.Score)", 9, C1, ha="left")
ax.scatter(5.50, 1.30, s=42, color=C3, zorder=5, edgecolors="white", lw=0.4)
T(ax, 6.90, 1.30, "quiescent / stem-like", 9, C3, ha="left")

box(ax, 0.4, 0.20, 9.2, 1.45, C3_L, C3, lw=1.5, radius=0.2)
T(ax, 5.0, 0.95,
  "The residual separates true stemness from proliferation noise",
  11.5, C3, "bold")


# ── inter-panel arrows ─────────────────────────────────────────────────────────
for xf in [0.358, 0.671]:
    fig.add_artist(mpatches.FancyArrowPatch(
        posA=(xf-0.010, 0.50), posB=(xf+0.010, 0.50),
        transform=fig.transFigure, figure=fig,
        arrowstyle="-|>", mutation_scale=22,
        color="#D1D5DB", linewidth=2.5, zorder=10))

plt.tight_layout(rect=[0, 0, 1, 0.91], w_pad=2.0)
fig.savefig(os.path.join(OUT, "fig_slide1_concept.png"),
            dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("✓ fig_slide1_concept.png")
