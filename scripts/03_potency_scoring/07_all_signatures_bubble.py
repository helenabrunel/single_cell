"""
13_allsigs_bubble.py
=====================
Bubble-plot summary of median per-patient correlation between each stemness
signature (signatures_final.json + stem_consensus) and the two potency
readouts (CytoTRACE2, LandSCENT SR), across the three Visvader subtypes.

Data: Visvader/figures/V11/allsigs_vs_scent_cytotrace_perpatient.csv
      (produced inline in the analysis session; median_r_CT2 / median_r_SR,
      per subtype x signature, computed only over patients with >=500 cells)

Output: figures/V11/allsigs_bubble_summary.png
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import os

matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

FIG_DIR = "/home/hbrunel/single_cell/Visvader/figures/V11"
pp = pd.read_csv(os.path.join(FIG_DIR, "allsigs_vs_scent_cytotrace_perpatient.csv"))

SIG_ORDER = ["ESC_WONG", "ESC_ASSOU", "BENPORATH_ES1", "stem_consensus",
             "fMaSC", "EMP", "LIM_STEM", "LIM_LumProg", "LIM_LumMat"]
SUBTYPE_ORDER = ["ER", "HER2", "TNBC"]
METRICS = [("median_r_CT2", "vs CytoTRACE2"), ("median_r_SR", "vs LandSCENT SR")]

# Validated diverging pair (scib skill validator: PASS on lightness/chroma/CVD/contrast)
NEG, MID, POS = "#B2182B", "#F0F0F0", "#2166AC"   # low(-1) -> 0 -> high(+1)
cmap = LinearSegmentedColormap.from_list("div_rb", [NEG, MID, POS], N=256)
norm = Normalize(vmin=-1, vmax=1)

def size_for(r):
    # area scales with |r|; floor so near-zero r is still a visible (small) dot
    return 25 + 750 * (abs(r) ** 1.15)

fig, axes = plt.subplots(1, 2, figsize=(11, 7.8), sharey=True)

for ax, (col, title) in zip(axes, METRICS):
    for yi, sig in enumerate(SIG_ORDER):
        for xi, subtype in enumerate(SUBTYPE_ORDER):
            row = pp[(pp.signature == sig) & (pp.subtype == subtype)]
            if row.empty:
                continue
            r = row[col].values[0]
            ax.scatter(xi, yi, s=size_for(r), c=[cmap(norm(r))],
                       edgecolors="white", linewidths=0.8, zorder=3)
            # small bubbles: label offset below in muted ink; large bubbles: label inside
            if abs(r) >= 0.35:
                txt_color = "white" if abs(r) > 0.6 else "#222222"
                ax.text(xi, yi, f"{r:+.2f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color, zorder=4)
            else:
                ax.text(xi, yi - 0.30, f"{r:+.2f}", ha="center", va="center",
                        fontsize=6.5, color="#666666", zorder=4)

    ax.set_xticks(range(len(SUBTYPE_ORDER)))
    ax.set_xticklabels(SUBTYPE_ORDER, fontsize=10)
    ax.set_xlim(-0.6, len(SUBTYPE_ORDER) - 0.4)
    ax.set_ylim(-0.7, len(SIG_ORDER) - 0.3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#e6e6e6", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

axes[0].set_yticks(range(len(SIG_ORDER)))
axes[0].set_yticklabels(SIG_ORDER, fontsize=10)

fig.suptitle("Signature alignment with CytoTRACE2 and LandSCENT SR — Visvader\n"
             "median per-patient Pearson r (patients ≥ 500 cells)",
             fontsize=12, fontweight="bold", y=1.02)

# Colorbar (diverging, shared) — lower block
cbar_ax = fig.add_axes([0.93, 0.12, 0.018, 0.40])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label("median per-patient r", fontsize=9)
cb.ax.tick_params(labelsize=8)

# Size legend — separate block, well above the colorbar, extra vertical room per bubble
legend_ax = fig.add_axes([0.90, 0.60, 0.10, 0.30])
legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
legend_ax.axis("off")
legend_ax.set_title("magnitude", fontsize=8.5, pad=6)
for i, r_ex in enumerate([0.2, 0.5, 0.8]):
    y = 0.82 - i * 0.38
    legend_ax.scatter([0.30], [y], s=size_for(r_ex), c="#999999",
                       edgecolors="white", linewidths=1.0, clip_on=False)
    legend_ax.text(0.62, y, f"|r|={r_ex}", fontsize=7.5, va="center")
legend_ax.set_title("magnitude", fontsize=8)

plt.subplots_adjust(right=0.90, top=0.86, wspace=0.08)
out = os.path.join(FIG_DIR, "allsigs_bubble_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
