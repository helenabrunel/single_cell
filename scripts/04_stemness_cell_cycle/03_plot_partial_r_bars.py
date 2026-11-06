"""
Standalone partial r(CT2|CC) ranked bar chart.
Output: figures/part1/partial_r_CT2_ranked.png
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

df = pd.read_csv(os.path.join(OUT, "ranking_table.csv"))

SIG_LABELS = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BenPorath_ES1"                    : "BenPorath",
    "EMP"                              : "EMP",
    "fMaSC/aMaSC"                      : "fMaSC",
    "LIM_Stem"                         : "LIM_Stem",
    "LIM_LumProg"                      : "LIM_LumProg",
    "LIM_LumMat"                       : "LIM_LumMat",
    "stem_consensus"                   : "stem_cons",
}

SIG_COLORS = {
    "ESC_ASSOU"   : "#2563EB",
    "ESC_WONG"    : "#059669",
    "BenPorath_ES1": "#7C3AED",
    "EMP"         : "#64748B",
    "fMaSC/aMaSC" : "#F59E0B",
    "LIM_Stem"    : "#94A3B8",
    "LIM_LumProg" : "#CBD5E1",
    "LIM_LumMat"  : "#374151",
    "stem_consensus": "#DC2626",
}

vis  = df[df["dataset"] == "Visvader"].set_index("signature")
klug = df[df["dataset"] == "Klughammer"].set_index("signature")

sigs = list(SIG_LABELS.keys())
mean_partial = [(vis.loc[s, "partial_r_CT2"] + klug.loc[s, "partial_r_CT2"]) / 2
                for s in sigs]
order = [sigs[i] for i in np.argsort(mean_partial)]   # ascending → bottom = best when flipped

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F9FAFB")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#D1D5DB")
ax.tick_params(colors="#374151")
ax.grid(axis="x", color="#E5E7EB", linewidth=0.8, zorder=0)

bar_h   = 0.32
gap     = 0.06
y_ticks = []

for yi, sig in enumerate(order):
    y_vis  = yi + gap/2 + bar_h/2
    y_klug = yi - gap/2 - bar_h/2
    col    = SIG_COLORS[sig]

    v_val = vis.loc[sig,  "partial_r_CT2"]
    k_val = klug.loc[sig, "partial_r_CT2"]

    # Visvader — solid
    ax.barh(y_vis, v_val, height=bar_h, color=col,
            alpha=0.85, zorder=3, left=0)
    # Klughammer — hatched / lighter
    ax.barh(y_klug, k_val, height=bar_h, color=col,
            alpha=0.38, zorder=3, left=0,
            hatch="////", edgecolor=col, linewidth=0.4)

    # value labels
    pad = 0.008
    for val, yb in [(v_val, y_vis), (k_val, y_klug)]:
        ha = "left" if val >= 0 else "right"
        xpos = val + (pad if val >= 0 else -pad)
        ax.text(xpos, yb, f"{val:+.2f}", fontsize=8,
                color="#374151", va="center", ha=ha, zorder=5)

    y_ticks.append(yi)

ax.set_yticks(y_ticks)
ax.set_yticklabels([SIG_LABELS[s] for s in order], fontsize=10)
ax.axvline(0, color="#6B7280", lw=1.2, zorder=4)
ax.set_xlabel("Partial r (sig, CytoTRACE2 | CC)   [pooled across patients]",
              fontsize=10, color="#374151")
ax.set_xlim(-0.65, 0.75)
ax.set_ylim(-0.65, len(order) - 0.35)

ax.set_title("Potency signal after CC control\n"
             "Partial Spearman r(sig, CytoTRACE2 | S.Score, G2M)  —  ranked by mean",
             fontsize=12, fontweight="bold", color="#111827", pad=14)

legend_handles = [
    mpatches.Patch(facecolor="#6B7280", alpha=0.85, label="Visvader  (solid)"),
    mpatches.Patch(facecolor="#6B7280", alpha=0.38, hatch="////",
                   edgecolor="#6B7280", label="Klughammer  (hatched)"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
          framealpha=0.9, edgecolor="#D1D5DB")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "partial_r_CT2_ranked.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("✓ partial_r_CT2_ranked.png")
