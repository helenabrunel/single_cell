"""
Per-dataset result slides:
  fig_slide2a_visvader.png    — Visvader:   tradeoff scatter + before/after bars
  fig_slide2b_klughammer.png  — Klughammer: tradeoff scatter + before/after bars
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
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

# rank by partial_r_CT2 descending — highest potency on top
order_vis  = [sigs[i] for i in np.argsort([-vis.loc[s,  "partial_r_CT2"] for s in sigs])]
order_klug = [sigs[i] for i in np.argsort([-klug.loc[s, "partial_r_CT2"] for s in sigs])]


def draw_dataset_slide(ds_df, ds_name, n_pat, order, outfile, accent):
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1, 1],
                          height_ratios=[1, 0.32],
                          hspace=0.42, wspace=0.38,
                          left=0.06, right=0.97, top=0.87, bottom=0.05)

    ax_scatter   = fig.add_subplot(gs[0, 0])
    ax_bars      = fig.add_subplot(gs[0, 1])
    ax_takeaways = fig.add_subplot(gs[1, :])

    fig.text(0.5, 0.968,
             f"Part 1 — Results: {ds_name}  ({n_pat} patients)",
             ha="center", va="top", fontsize=15, fontweight="bold", color=DARK)
    fig.text(0.5, 0.922,
             "Which stemness signature captures developmental potency independently of cell cycle?  "
             "·  Per-patient Fisher z meta-analysis  ·  Global (all subtypes)",
             ha="center", va="top", fontsize=9.5, color=GREY, style="italic")

    # ── A: Tradeoff scatter ───────────────────────────────────────────────────
    ax = ax_scatter
    ax.set_facecolor(LITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=8.5)
    ax.grid(color=BORDER, linewidth=0.8, zorder=0)

    ax.axhspan(0.35, 0.70, xmin=0, xmax=0.42, color="#D1FAE5", alpha=0.50, zorder=1)
    ax.text(0.005, 0.62, "← ideal", fontsize=8.5, color="#059669", style="italic")

    for sig in sigs:
        col = SIG_COLORS[sig]
        x   = ds_df.loc[sig, "R2_CC"]
        y   = ds_df.loc[sig, "partial_r_CT2"]
        ms  = 160 if sig in ("ESC_WONG", "stem_consensus") else 110
        ax.scatter(x, y, s=ms, color=col, marker="o", zorder=4,
                   edgecolors="white", linewidths=0.9)
        ax.text(x + 0.012, y, SIG_LABELS[sig], fontsize=8, color=col,
                va="center", fontweight="bold", zorder=5)

    ax.axhline(0, color=GREY, lw=0.9, ls="--", alpha=0.6)
    ax.axvline(0, color=GREY, lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel("R²(CC → signature)   [contamination →]", fontsize=9.5, color="#374151")
    ax.set_ylabel("Partial r(sig, CytoTRACE2 | CC)   [net potency ↑]",
                  fontsize=9.5, color="#374151")
    ax.set_title("A   CC contamination vs net potency signal",
                 fontsize=11, fontweight="bold", color=DARK, loc="left")

    # ── B: Before / after bar chart ───────────────────────────────────────────
    ax = ax_bars
    ax.set_facecolor(LITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=9)
    ax.grid(axis="x", color=BORDER, linewidth=0.8, zorder=0)

    bar_h, gap = 0.28, 0.06
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
                    fontsize=7.5, color="#374151", va="center", ha=ha, zorder=5)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SIG_LABELS[s] for s in order], fontsize=9.5)
    ax.axvline(0, color=GREY, lw=1.0, zorder=4)
    ax.set_xlabel("r(sig, CytoTRACE2)   [pooled across patients]",
                  fontsize=9.5, color="#374151")
    ax.set_xlim(-0.68, 0.80)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title("B   Potency signal: before vs after CC residualisation",
                 fontsize=11, fontweight="bold", color=DARK, loc="left")
    ax.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.35, label="before  —  r(sig, CytoTRACE2)"),
        mpatches.Patch(facecolor=GREY, alpha=0.90,
                       label="after   —  partial r(sig, CytoTRACE2 | CC)"),
    ], fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor=BORDER)

    # ── Takeaways ─────────────────────────────────────────────────────────────
    ax = ax_takeaways
    ax.set_facecolor(LITE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    top1  = ds_df["partial_r_CT2"].idxmax()
    top1v = ds_df.loc[top1, "partial_r_CT2"]
    top1r = ds_df.loc[top1, "r_CT2"]
    top1c = ds_df.loc[top1, "R2_CC"]
    worst = ds_df["R2_CC"].idxmax()
    wv    = ds_df.loc[worst, "R2_CC"]

    takeaways = [
        (SIG_COLORS[top1],
         f"★  {SIG_LABELS[top1]} — best potency signal",
         f"partial r(CT2|CC) = {top1v:+.2f}  (raw r = {top1r:+.2f})\n"
         f"R²_CC = {top1c:.2f}  —  moderate CC contamination"),
        ("#2563EB" if top1 != "ESC_ASSOU" else "#059669",
         f"ESC_ASSOU — close second" if top1 == "ESC_WONG" else "ESC_WONG — close second",
         f"partial r = {ds_df.loc['ESC_ASSOU' if top1 == 'ESC_WONG' else 'ESC_WONG', 'partial_r_CT2']:+.2f}  "
         f"·  R²_CC = {ds_df.loc['ESC_ASSOU' if top1 == 'ESC_WONG' else 'ESC_WONG', 'R2_CC']:.2f}"),
        ("#DC2626",
         f"stem_consensus excluded",
         f"R²_CC = {ds_df.loc['stem_consensus', 'R2_CC']:.2f}  —  majority of variance is CC\n"
         f"partial r drops to {ds_df.loc['stem_consensus', 'partial_r_CT2']:+.2f} after correction"),
        ("#374151",
         "LIM_LumMat — anti-stemness marker",
         f"partial r = {ds_df.loc['LIM_LumMat', 'partial_r_CT2']:+.2f}  "
         "·  reliably marks differentiated cells"),
    ]

    for i, (col, title, body) in enumerate(takeaways):
        x = 0.01 + i * 0.25
        ax.plot([x, x + 0.22], [0.97, 0.97], color=col, lw=2.5,
                transform=ax.transAxes, clip_on=False)
        ax.text(x, 0.88, title, fontsize=9.5, color=col, fontweight="bold",
                va="top", transform=ax.transAxes)
        ax.text(x, 0.48, body, fontsize=8.2, color="#374151",
                va="top", transform=ax.transAxes, multialignment="left")

    fig.savefig(os.path.join(OUT, outfile),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ {outfile}")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2a — Two datasets, two rows
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(2, 2,
                      width_ratios=[1, 1],
                      height_ratios=[1, 1],
                      hspace=0.45, wspace=0.38,
                      left=0.06, right=0.97, top=0.94, bottom=0.04)

fig.text(0.5, 0.975,
         "Part 1 — Results: which stemness signature captures potency independently of cell cycle?",
         ha="center", va="top", fontsize=14, fontweight="bold", color=DARK)
fig.text(0.5, 0.948,
         "Per-patient Fisher z meta-analysis  ·  Global (all subtypes)",
         ha="center", va="top", fontsize=9.5, color=GREY, style="italic")

rows = [
    (vis,  "Visvader",   31, order_vis,  "#2563EB"),
    (klug, "Klughammer", 19, order_klug, "#7C3AED"),
]

for row_i, (ds_df, ds_name, n_pat, order, accent) in enumerate(rows):

    ax_sc  = fig.add_subplot(gs[row_i, 0])
    ax_bar = fig.add_subplot(gs[row_i, 1])

    # row label — placed inside the scatter title, no separate fig.text needed

    # ── Tradeoff scatter ──────────────────────────────────────────────────────
    ax = ax_sc
    ax.set_facecolor(LITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.grid(color=BORDER, linewidth=0.8, zorder=0)

    # problem zone: high CC AND high CT2 → confounded
    ax.axhspan(0.20, 0.65, xmin=0.35, xmax=1.0,
               color="#FEE2E2", alpha=0.40, zorder=1)
    ax.text(0.54, 0.56, "high CT2\nbut confounded\nby CC",
            fontsize=7.5, color="#DC2626", alpha=0.80, ha="center",
            va="top", style="italic")
    # clean zone: high CT2, low CC
    ax.axhspan(0.20, 0.65, xmin=0, xmax=0.35,
               color="#D1FAE5", alpha=0.40, zorder=1)
    ax.text(0.005, 0.58, "high CT2\nlow CC\n→ genuine",
            fontsize=7.5, color="#059669", alpha=0.85, va="top", style="italic")

    offsets = {
        "ESC_WONG"     : ( 0.011,  0.0),
        "ESC_ASSOU"    : ( 0.011,  0.018),
        "BenPorath_ES1": ( 0.011,  0.0),
        "stem_consensus": ( 0.011, 0.0),
        "LIM_LumProg"  : ( 0.011,  0.0),
        "LIM_Stem"     : ( 0.011, -0.018),
        "EMP"          : ( 0.011,  0.0),
        "fMaSC/aMaSC"  : ( 0.011,  0.0),
        "LIM_LumMat"   : ( 0.011,  0.0),
    }
    for sig in sigs:
        col = SIG_COLORS[sig]
        x   = ds_df.loc[sig, "R2_CC"]
        y   = ds_df.loc[sig, "r_CT2"]
        ms  = 150 if sig in ("ESC_WONG", "ESC_ASSOU", "stem_consensus") else 100
        ax.scatter(x, y, s=ms, color=col, zorder=4,
                   edgecolors="white", linewidths=0.8)
        dx, dy = offsets.get(sig, (0.011, 0.0))
        ax.text(x + dx, y + dy, SIG_LABELS[sig], fontsize=7.5, color=col,
                va="center", fontweight="bold", zorder=5)

    ax.axhline(0, color=GREY, lw=0.9, ls="--", alpha=0.6)
    ax.axvline(0, color=GREY, lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel("R²(CC → signature)   [CC contamination]", fontsize=8.5, color="#374151")
    ax.set_ylabel("r(sig, CytoTRACE2)   [raw potency correlation]",
                  fontsize=8.5, color="#374151")
    panel = "A" if row_i == 0 else "C"
    ax.set_title(f"{panel}   {ds_name} ({n_pat} pat.)  —  CC contamination vs raw CT2 correlation",
                 fontsize=10, fontweight="bold", color=DARK, loc="left")

    # ── Before / after bars ───────────────────────────────────────────────────
    ax = ax_bar
    ax.set_facecolor(LITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors="#374151", labelsize=8.5)
    ax.grid(axis="x", color=BORDER, linewidth=0.8, zorder=0)

    bar_h, gap = 0.28, 0.06
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
                    fontsize=7, color="#374151", va="center", ha=ha, zorder=5)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SIG_LABELS[s] for s in order], fontsize=8.5)
    ax.axvline(0, color=GREY, lw=1.0, zorder=4)
    ax.set_xlabel("r(sig, CytoTRACE2)   [pooled across patients]",
                  fontsize=8.5, color="#374151")
    ax.set_xlim(-0.68, 0.80)
    ax.set_ylim(-0.6, len(order) - 0.4)
    panel = "B" if row_i == 0 else "D"
    ax.set_title(f"{panel}   {ds_name} ({n_pat} pat.)  —  potency before vs after CC correction",
                 fontsize=10, fontweight="bold", color=DARK, loc="left")

    if row_i == 0:
        ax.legend(handles=[
            mpatches.Patch(facecolor=GREY, alpha=0.35, label="before  —  r(sig, CytoTRACE2)"),
            mpatches.Patch(facecolor=GREY, alpha=0.90,
                           label="after   —  partial r(sig, CytoTRACE2 | CC)"),
        ], fontsize=8, loc="lower right", framealpha=0.9, edgecolor=BORDER)

fig.savefig(os.path.join(OUT, "fig_slide2a_results.png"),
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("✓ fig_slide2a_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2c — Per-patient distributions
# ══════════════════════════════════════════════════════════════════════════════
panels = [
    ("per_patient_r_CT2.png",
     "A   r(sig, CytoTRACE2) per patient",
     "How consistently does each signature track developmental potency across patients?"),
    ("per_patient_R2_CC.png",
     "B   R²(CC → sig) per patient",
     "How much of each signature's variance is explained by cell cycle — and how variable is this between patients?"),
    ("per_patient_R2_CT2.png",
     "C   R²(CT2 → sig) per patient",
     "How much of each signature's variance is explained by developmental potency?"),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 15))
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.988,
         "Part 1 — Per-patient distributions across 9 stemness signatures",
         ha="center", va="top", fontsize=15, fontweight="bold", color=DARK)
fig.text(0.5, 0.963,
         "Each dot = one patient  ·  coloured by subtype (ER / HER2 / TNBC)  ·  "
         "Box = IQR  ·  Line = median  ·  Visvader (31 pat.)  &  Klughammer (19 pat.)",
         ha="center", va="top", fontsize=9.5, color=GREY, style="italic")

for ax, (fname, panel_title, subtitle) in zip(axes, panels):
    img = mpimg.imread(os.path.join(OUT, fname))
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    ax.set_title(f"{panel_title}\n{subtitle}",
                 fontsize=10.5, fontweight="bold", color=DARK,
                 loc="left", pad=5)

plt.tight_layout(rect=[0, 0, 1, 0.958], h_pad=2.0)
fig.savefig(os.path.join(OUT, "fig_slide2c_per_patient.png"),
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("✓ fig_slide2c_per_patient.png")
