"""
Standalone result figures, one per dataset:
  Visvader_before_after.png
  Klughammer_before_after.png
  Visvader_tradeoff_clean.png
  Klughammer_tradeoff_clean.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = "figures/part1"
plt.rcParams.update({"font.family": "DejaVu Sans"})

BG     = "#FFFFFF"
LITE   = "#F9FAFB"
DARK   = "#1F2937"
GREY   = "#6B7280"
BORDER = "#E5E7EB"

SIG_LABELS = {
    "ESC_ASSOU"    : "ESC_ASSOU",
    "ESC_WONG"     : "ESC_WONG",
    "BenPorath_ES1": "BenPorath",
    "EMP"          : "EMP",
    "fMaSC/aMaSC"  : "fMaSC",
    "LIM_Stem"     : "LIM_Stem",
    "LIM_LumProg"  : "LIM_LumProg",
    "LIM_LumMat"   : "LIM_LumMat",
    "stem_consensus": "stem_cons",
}
SIG_COLORS = {
    "ESC_ASSOU"    : "#2563EB",
    "ESC_WONG"     : "#059669",
    "BenPorath_ES1": "#7C3AED",
    "EMP"          : "#64748B",
    "fMaSC/aMaSC"  : "#F59E0B",
    "LIM_Stem"     : "#94A3B8",
    "LIM_LumProg"  : "#CBD5E1",
    "LIM_LumMat"   : "#374151",
    "stem_consensus": "#DC2626",
}

df   = pd.read_csv(os.path.join(OUT, "ranking_table.csv"))
vis  = df[df["dataset"] == "Visvader"].set_index("signature")
klug = df[df["dataset"] == "Klughammer"].set_index("signature")
sigs = list(SIG_LABELS.keys())


# ── helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax, xgrid=False, ygrid=False):
    ax.set_facecolor(LITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=9)
    if xgrid:
        ax.grid(axis="x", color=BORDER, linewidth=0.8, zorder=0)
    if ygrid:
        ax.grid(axis="y", color=BORDER, linewidth=0.8, zorder=0)


# ══════════════════════════════════════════════════════════════════════════════
# Before / after bar charts
# ══════════════════════════════════════════════════════════════════════════════
def plot_before_after(ds_df, ds_name, n_pat, outfile):
    # rank ascending → highest partial_r_CT2 ends at top (highest yi)
    order = [sigs[i] for i in
             np.argsort([ds_df.loc[s, "partial_r_CT2"] for s in sigs])]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    style_ax(ax, xgrid=True)

    bar_h, gap = 0.30, 0.07
    for yi, sig in enumerate(order):
        col    = SIG_COLORS[sig]
        before = ds_df.loc[sig, "r_CT2"]
        after  = ds_df.loc[sig, "partial_r_CT2"]
        y_bef  = yi + gap / 2 + bar_h / 2
        y_aft  = yi - gap / 2 - bar_h / 2

        ax.barh(y_bef, before, height=bar_h, color=col, alpha=0.35, zorder=3)
        ax.barh(y_aft,  after, height=bar_h, color=col, alpha=0.90, zorder=3)

        for val, yb in [(before, y_bef), (after, y_aft)]:
            pad = 0.008
            ha  = "left" if val >= 0 else "right"
            ax.text(val + (pad if val >= 0 else -pad), yb, f"{val:+.2f}",
                    fontsize=8, color="#374151", va="center", ha=ha, zorder=5)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SIG_LABELS[s] for s in order], fontsize=10)
    ax.axvline(0, color=GREY, lw=1.1, zorder=4)
    ax.set_xlabel("Spearman r(sig, CytoTRACE2)   [pooled across patients, Fisher z]",
                  fontsize=10, color="#374151")
    ax.set_xlim(-0.68, 0.80)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title(
        f"{ds_name} ({n_pat} patients)\n"
        "Potency signal: before vs after CC residualisation",
        fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.35,
                       label="before  —  r(sig, CytoTRACE2)"),
        mpatches.Patch(facecolor=GREY, alpha=0.90,
                       label="after   —  partial r(sig, CytoTRACE2 | CC)"),
    ], fontsize=9, loc="upper left", framealpha=0.9, edgecolor=BORDER)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, outfile),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ {outfile}")

plot_before_after(vis,  "Visvader",   31, "Visvader_before_after.png")
plot_before_after(klug, "Klughammer", 19, "Klughammer_before_after.png")


# ══════════════════════════════════════════════════════════════════════════════
# Tradeoff scatter — annotated for readability
# ══════════════════════════════════════════════════════════════════════════════
def plot_tradeoff_clean(ds_df, ds_name, n_pat, outfile):
    """
    x = R²(CC → sig)   — how CC-contaminated is the signature?
    y = r(sig, CT2)     — raw correlation with CytoTRACE2 (before CC removal)
    Message: signatures that look best on y (high CT2 correlation) are also
    the most CC-contaminated on x — you can't tell what's real without correction.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    style_ax(ax, xgrid=True, ygrid=True)

    # problem zone: high CC AND high CT2 → confounded
    ax.axhspan(0.20, 0.65, xmin=0.35, xmax=1.0,
               color="#FEE2E2", alpha=0.40, zorder=1)
    ax.text(0.54, 0.58, "high CT2 correlation\nbut also high CC contamination\n→ confounded",
            fontsize=8, color="#DC2626", alpha=0.75, ha="center", va="top", style="italic")

    # clean zone: high CT2, low CC → genuinely stem-like
    ax.axhspan(0.20, 0.65, xmin=0, xmax=0.35,
               color="#D1FAE5", alpha=0.40, zorder=1)
    ax.text(0.005, 0.60, "high CT2\nlow CC\n→ genuine",
            fontsize=8, color="#059669", alpha=0.85, va="top", style="italic")

    ax.axhline(0, color=GREY, lw=1.0, ls="--", alpha=0.5, zorder=2)
    ax.axvline(0, color=GREY, lw=1.0, ls="--", alpha=0.5, zorder=2)

    for sig in sigs:
        col = SIG_COLORS[sig]
        x   = ds_df.loc[sig, "R2_CC"]
        y   = ds_df.loc[sig, "r_CT2"]
        ms  = 180 if sig in ("ESC_WONG", "ESC_ASSOU", "stem_consensus") else 110
        ax.scatter(x, y, s=ms, color=col, zorder=4,
                   edgecolors="white", linewidths=1.0)

        offsets = {
            "ESC_WONG"     : ( 0.012,  0.0),
            "ESC_ASSOU"    : ( 0.012,  0.018),
            "BenPorath_ES1": ( 0.012,  0.0),
            "stem_consensus": ( 0.012, 0.0),
            "LIM_LumProg"  : ( 0.012,  0.0),
            "LIM_Stem"     : ( 0.012, -0.020),
            "EMP"          : ( 0.012,  0.0),
            "fMaSC/aMaSC"  : ( 0.012,  0.0),
            "LIM_LumMat"   : ( 0.012,  0.0),
        }
        dx, dy = offsets.get(sig, (0.012, 0.0))
        ax.text(x + dx, y + dy, SIG_LABELS[sig],
                fontsize=9, color=col, va="center",
                fontweight="bold", zorder=5)

    ax.set_xlabel("R²(CC → signature)   [CC contamination]",
                  fontsize=10.5, color="#374151")
    ax.set_ylabel("r(sig, CytoTRACE2)   [raw potency correlation]",
                  fontsize=10.5, color="#374151")
    ax.set_title(
        f"{ds_name} ({n_pat} patients)\n"
        "CC contamination vs raw potency correlation",
        fontsize=12, fontweight="bold", color=DARK, pad=10)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, outfile),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ {outfile}")

plot_tradeoff_clean(vis,  "Visvader",   31, "Visvader_tradeoff_clean.png")
plot_tradeoff_clean(klug, "Klughammer", 19, "Klughammer_tradeoff_clean.png")
