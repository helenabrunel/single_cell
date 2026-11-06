"""
Generate a presentation slide listing all 9 stemness signatures,
grouped by conceptual origin and order of inclusion in the project.
fig_slide0_signatures.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, pandas as pd

OUT = "figures/part1"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── Colours ────────────────────────────────────────────────────────────────────
C_DARK   = "#1F2937"
C_ESC    = "#2563EB"   # blue  — ESC / pluripotency
C_PLAST  = "#059669"   # green — mammary plasticity / progenitor
C_LIM    = "#7C3AED"   # purple — LIM mammary hierarchy
C_CONS   = "#B45309"   # amber — consensus

C_BAD    = "#EF4444"   # red   — CC contamination
C_GOOD   = "#10B981"   # teal  — potency alignment

GROUPS = [
    {
        "phase"  : "Phase 1",
        "concept": "Pluripotency / ESC",
        "note"   : "Transcriptional programs of embryonic stem cells",
        "color"  : C_ESC,
        "sigs"   : [
            {
                "key"  : "ESC_ASSOU",
                "label": "ESC_ASSOU",
                "ref"  : "Assou et al. 2007",
                "genes": "645 / 637",
                "desc" : "Transcriptional programme of\nhuman embryonic stem cells",
                "r2_vis": 0.310, "r2_klug": 0.383,
                "rct2_vis": 0.286, "rct2_klug": 0.443,
                "pct2_vis": 0.483, "pct2_klug": 0.422,
            },
            {
                "key"  : "ESC_WONG",
                "label": "ESC_WONG",
                "ref"  : "Wong et al. 2008",
                "genes": "289 / 285",
                "desc" : "Core ESC identity module;\nbest CytoTRACE2 alignment",
                "r2_vis": 0.296, "r2_klug": 0.303,
                "rct2_vis": 0.287, "rct2_klug": 0.488,
                "pct2_vis": 0.464, "pct2_klug": 0.524,
                "star": True,
            },
            {
                "key"  : "BENPORATH_ES1",
                "label": "BenPorath_ES1",
                "ref"  : "Ben-Porath et al. 2008",
                "genes": "318 / 314",
                "desc" : "ES-like module co-expressed\nin embryonic & cancer stem cells",
                "r2_vis": 0.365, "r2_klug": 0.459,
                "rct2_vis": 0.231, "rct2_klug": 0.417,
                "pct2_vis": 0.443, "pct2_klug": 0.385,
            },
        ],
    },
    {
        "phase"  : "Phase 2",
        "concept": "Mammary progenitor / Plasticity",
        "note"   : "Stem-like states specific to mammary biology",
        "color"  : C_PLAST,
        "sigs"   : [
            {
                "key"  : "EMP",
                "label": "EMP",
                "ref"  : "— ",
                "genes": "13 / 14",
                "desc" : "Epithelial–mesenchymal plasticity\nminimal CC contamination",
                "r2_vis": 0.004, "r2_klug": 0.003,
                "rct2_vis": 0.082, "rct2_klug": 0.121,
                "pct2_vis": 0.078, "pct2_klug": 0.272,
            },
            {
                "key"  : "fMaSC_over_aMaSC",
                "label": "fMaSC / aMaSC",
                "ref"  : "—",
                "genes": "1,701 / 1,675",
                "desc" : "Genes enriched in fetal vs adult\nmammary stem cells",
                "r2_vis": 0.154, "r2_klug": 0.288,
                "rct2_vis": 0.142, "rct2_klug": 0.181,
                "pct2_vis": 0.115, "pct2_klug": 0.036,
            },
        ],
    },
    {
        "phase"  : "Phase 3",
        "concept": "LIM mammary hierarchy",
        "note"   : "FACS-sorted cell types from normal breast (Lim et al.)",
        "color"  : C_LIM,
        "sigs"   : [
            {
                "key"  : "LIM_MAMMARY_STEM_CELL_UP",
                "label": "LIM_Stem",
                "ref"  : "Lim et al. 2009/2010",
                "genes": "465 / 478",
                "desc" : "Mammary stem cell (MaSC)\ntranscriptome — basal compartment",
                "r2_vis": 0.014, "r2_klug": 0.005,
                "rct2_vis": 0.063, "rct2_klug": -0.064,
                "pct2_vis": 0.231, "pct2_klug": 0.287,
                "warn": "orthogonal to CytoTRACE2",
            },
            {
                "key"  : "LIM_MAMMARY_LUMINAL_PROGENITOR_UP",
                "label": "LIM_LumProg",
                "ref"  : "Lim et al. 2009/2010",
                "genes": "58 / 57",
                "desc" : "Luminal progenitor transcriptome\n— intermediate state",
                "r2_vis": 0.054, "r2_klug": 0.008,
                "rct2_vis": 0.007, "rct2_klug": -0.093,
                "pct2_vis": 0.269, "pct2_klug": 0.126,
            },
            {
                "key"  : "LIM_MAMMARY_LUMINAL_MATURE_UP",
                "label": "LIM_LumMat",
                "ref"  : "Lim et al. 2009/2010",
                "genes": "108 / 109",
                "desc" : "Mature luminal transcriptome\n— fully differentiated (control)",
                "r2_vis": 0.135, "r2_klug": 0.023,
                "rct2_vis": -0.196, "rct2_klug": -0.255,
                "pct2_vis": -0.475, "pct2_klug": -0.463,
                "warn": "differentiation marker (negative)",
            },
        ],
    },
]

CONSENSUS = {
    "key"  : "stem_consensus",
    "label": "stem_consensus",
    "ref"  : "Hand-curated",
    "genes": "55 / 55",
    "desc" : "Consensus of stem-like genes across multiple signatures",
    "r2_vis": 0.550, "r2_klug": 0.551,
    "rct2_vis": 0.181, "rct2_klug": 0.424,
    "pct2_vis": 0.251, "pct2_klug": 0.312,
    "warn": "R²_CC = 0.55 — severe cell-cycle contamination",
    "color": C_CONS,
}


# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18); ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)

def rbox(x, y, w, h, fc, ec="white", lw=0.8, alpha=1.0, radius=0.15, zorder=2):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(b)

def txt(x, y, s, size=9, color="white", weight="normal",
        ha="center", va="center", zorder=5, style="normal", alpha=1.0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder, fontstyle=style, alpha=alpha,
            multialignment="center")

def mini_bar(x, y, value, vmin=-0.6, vmax=0.6, width=1.4, height=0.10,
             label="", low_bad=True):
    """Draw a small horizontal bar showing a metric value."""
    # background track
    rbox(x, y, width, height, "#374151", ec="none", alpha=0.8, radius=0.03, zorder=3)
    # zero line
    zero_x = x + width * (-vmin) / (vmax - vmin)
    ax.plot([zero_x, zero_x], [y, y + height], color="#6B7280", lw=0.6, zorder=4)
    # filled portion
    if not np.isnan(value):
        norm = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        fill_w = norm * width
        if low_bad:  # red = bad (contamination), green = good
            col = C_BAD if value > 0.2 else ("#FBBF24" if value > 0.08 else C_GOOD)
        else:        # green = good (positive), red = bad (negative)
            col = C_GOOD if value > 0.05 else (C_BAD if value < -0.05 else "#FBBF24")
        rbox(x, y, fill_w, height, col, ec="none", alpha=0.85, radius=0.03, zorder=4)
        val_x = x + fill_w + 0.05 if fill_w < width - 0.2 else x + fill_w - 0.25
        txt(val_x, y + height / 2, f"{value:+.2f}", size=6.5,
            color="white", ha="left", zorder=5)
    txt(x - 0.05, y + height / 2, label, size=6.5, color="#9CA3AF", ha="right", zorder=5)


# ── Title & subtitle ───────────────────────────────────────────────────────────
txt(9, 9.65, "Stemness signatures used in the SCIE project",
    size=17, weight="bold", color="#FCD34D")
txt(9, 9.30,
    "9 gene sets capturing different facets of stem-like identity in breast cancer  ·  "
    "n genes shown as Visvader / Klughammer",
    size=10, color="#9CA3AF", style="italic")

# ── Phase timeline strip ───────────────────────────────────────────────────────
phases = [
    (1.2,  6.2, "Phase 1", "ESC / Pluripotency", C_ESC),
    (7.2,  4.6, "Phase 2", "Mammary progenitor", C_PLAST),
    (12.2, 6.2, "Phase 3", "LIM mammary hierarchy", C_LIM),
]
arrow_kw = dict(arrowstyle="-|>", color="#4B5563", lw=1.5, mutation_scale=12)
for i in range(len(phases) - 1):
    x1 = phases[i][0] + 4.5
    x2 = phases[i+1][0] - 0.3
    ax.annotate("", xy=(x2, 9.05), xytext=(x1, 9.05),
                arrowprops=arrow_kw)
for px, pw, plabel, pconcept, pc in phases:
    rbox(px - 0.2, 8.82, pw + 0.4, 0.48, pc, alpha=0.18, ec=pc, lw=1.2, radius=0.12)
    txt(px + pw/2, 9.08, plabel, size=9.5, weight="bold", color=pc)
    txt(px + pw/2, 8.90, pconcept, size=8, color=pc, alpha=0.8)


# ── Draw signature cards ───────────────────────────────────────────────────────
# Layout: col_starts for each group
group_x   = [0.20, 6.40, 11.90]   # x start per group
group_w   = 5.70                   # group width
card_gap  = 0.18                   # gap between cards within group
card_top  = 8.55                   # y top of first card row

# Number of rows: max sigs in any group = 3, so 3 rows
ROW_H = 2.52   # height per card row

for g_idx, group in enumerate(GROUPS):
    gx   = group_x[g_idx]
    gcol = group["color"]
    n    = len(group["sigs"])
    # group label band
    rbox(gx, card_top - len(group["sigs"]) * ROW_H - 0.05,
         group_w, len(group["sigs"]) * ROW_H + 0.05,
         gcol, alpha=0.04, ec=gcol, lw=1.0, radius=0.2)

    for s_idx, sig in enumerate(group["sigs"]):
        cy   = card_top - (s_idx + 1) * ROW_H + card_gap
        ch   = ROW_H - card_gap * 2

        # Card background
        rbox(gx + 0.05, cy, group_w - 0.10, ch,
             "#111827", ec=gcol, lw=1.2, alpha=1.0, radius=0.15)

        # Coloured left stripe
        rbox(gx + 0.05, cy, 0.18, ch, gcol, ec="none", alpha=0.85, radius=0.10)

        # Star for recommended
        if sig.get("star"):
            txt(gx + 0.50, cy + ch - 0.18, "★ Recommended", size=7.5,
                color="#FCD34D", weight="bold", ha="left")

        # Label + ref
        txt(gx + 0.50, cy + ch - 0.42, sig["label"],
            size=12, weight="bold", color=gcol, ha="left")
        txt(gx + 0.50, cy + ch - 0.68, sig["ref"],
            size=7.5, color="#6B7280", ha="left", style="italic")

        # Description
        txt(gx + 0.50, cy + ch - 1.02, sig["desc"],
            size=8, color="#D1D5DB", ha="left")

        # n genes
        rbox(gx + group_w - 1.55, cy + ch - 0.52, 1.40, 0.30,
             "#1F2937", ec="#374151", lw=0.8, radius=0.06)
        txt(gx + group_w - 0.85, cy + ch - 0.37,
            f"{sig['genes']} genes", size=7, color="#9CA3AF")

        # Warning badge
        if sig.get("warn"):
            rbox(gx + 0.50, cy + 0.09, group_w - 0.75, 0.24,
                 "#7F1D1D", ec="#EF4444", lw=0.6, alpha=0.7, radius=0.06)
            txt(gx + 0.50 + (group_w - 0.75) / 2, cy + 0.21,
                f"⚠  {sig['warn']}", size=6.8, color="#FCA5A5")
            metric_bottom = cy + 0.38
        else:
            metric_bottom = cy + 0.12

        # Mini metrics — R²_CC (left) and r(CT2) (right)
        bar_w = (group_w - 0.80) / 2 - 0.10
        # R²_CC
        txt(gx + 0.50 + bar_w / 2, metric_bottom + 0.42,
            "R²(CC→sig)", size=6.5, color="#9CA3AF")
        mini_bar(gx + 0.50, metric_bottom + 0.24,
                 sig["r2_vis"], vmin=0, vmax=0.65, width=bar_w,
                 label="Vis", low_bad=True)
        mini_bar(gx + 0.50, metric_bottom + 0.12,
                 sig["r2_klug"], vmin=0, vmax=0.65, width=bar_w,
                 label="Klg", low_bad=True)

        # partial r(CT2|CC)
        x2 = gx + 0.50 + bar_w + 0.20
        txt(x2 + bar_w / 2, metric_bottom + 0.42,
            "partial r(CT2|CC)", size=6.5, color="#9CA3AF")
        mini_bar(x2, metric_bottom + 0.24,
                 sig["pct2_vis"], vmin=-0.6, vmax=0.6, width=bar_w,
                 label="Vis", low_bad=False)
        mini_bar(x2, metric_bottom + 0.12,
                 sig["pct2_klug"], vmin=-0.6, vmax=0.6, width=bar_w,
                 label="Klg", low_bad=False)


# ── stem_consensus — bottom spanning card ─────────────────────────────────────
cons_y = 0.15
cons_h = 1.28
cons_x = 0.20
cons_w = 17.60
gcol   = CONSENSUS["color"]

rbox(cons_x, cons_y, cons_w, cons_h, "#111827", ec=gcol, lw=1.4,
     alpha=1.0, radius=0.15)
rbox(cons_x, cons_y, 0.18, cons_h, gcol, ec="none", alpha=0.85, radius=0.10)

txt(cons_x + 0.40, cons_y + cons_h - 0.28, "stem_consensus",
    size=12, weight="bold", color=gcol, ha="left")
txt(cons_x + 0.40, cons_y + cons_h - 0.52,
    "Hand-curated  ·  55 genes  ·  consensus of overlapping stem-like genes across multiple sources",
    size=8.5, color="#D1D5DB", ha="left")
txt(cons_x + 0.40, cons_y + cons_h - 0.76,
    "Captures a shared transcriptional core of stemness — but heavily contaminated by cell-cycle genes",
    size=8, color="#9CA3AF", ha="left", style="italic")

# warning
rbox(cons_x + 0.40, cons_y + 0.10, 4.0, 0.28,
     "#7F1D1D", ec="#EF4444", lw=0.7, alpha=0.8, radius=0.06)
txt(cons_x + 0.40 + 2.0, cons_y + 0.24,
    "⚠  R²_CC = 0.55 — unusable without CC residualisation",
    size=7.5, color="#FCA5A5")

# mini bars
for i, (ds, r2, pct2) in enumerate([
        ("Vis", CONSENSUS["r2_vis"],  CONSENSUS["pct2_vis"]),
        ("Klg", CONSENSUS["r2_klug"], CONSENSUS["pct2_klug"]),
]):
    by = cons_y + 0.50 - i * 0.13
    txt(cons_x + 5.0, by + 0.05, "R²(CC→sig)", size=6.5, color="#9CA3AF")
    mini_bar(cons_x + 5.5, by, r2, vmin=0, vmax=0.65, width=2.0,
             label=ds, low_bad=True)
    txt(cons_x + 8.0, by + 0.05, "partial r(CT2|CC)", size=6.5, color="#9CA3AF")
    mini_bar(cons_x + 8.5, by, pct2, vmin=-0.6, vmax=0.6, width=2.0,
             label=ds, low_bad=False)

# consensus label badge
rbox(cons_x + 12.5, cons_y + 0.32, 4.8, 0.70,
     "#1C1506", ec=gcol, lw=0.9, radius=0.10)
txt(cons_x + 14.9, cons_y + 0.88,
    "Consensus across:", size=7.5, color=gcol, weight="bold")
for i, src in enumerate(["ESC programs", "Cancer stem cell modules",
                          "Mammary stem cell markers"]):
    txt(cons_x + 14.9, cons_y + 0.70 - i * 0.15, f"• {src}",
        size=7, color="#D1D5DB")


# ── Legend ────────────────────────────────────────────────────────────────────
lx = 0.25
for col, label in [(C_BAD,  "high CC contamination (bad)"),
                   (C_GOOD, "positive signal (good)"),
                   ("#FBBF24","intermediate")]:
    rbox(lx, 0.02, 0.25, 0.09, col, ec="none", alpha=0.85, radius=0.02)
    txt(lx + 0.30, 0.065, label, size=7.5, color="#9CA3AF", ha="left")
    lx += 2.0

txt(17.8, 0.065, "Vis = Visvader  ·  Klg = Klughammer",
    size=7.5, color="#6B7280", ha="right")


plt.tight_layout(pad=0)
out = os.path.join(OUT, "fig_slide0_signatures.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig)
print(f"✓ {out}")
