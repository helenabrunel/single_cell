"""
07_composition_plot.py
========================
Approach 2 (unbiased) presentation figure: stacked bar chart of CD8
cluster composition per patient, patients ORDERED by their malignant
LCOR_OE_malignant score, with a dashed line separating the LCOR-low half
from the LCOR-high half. This format suits this exact design
(unbiased cluster discovery + composition-by-LCOR-level display).

Each cluster is labeled with its post-hoc Tcell_UP24H/UP96H character (so
the reader can see, e.g., "cluster 3 -- high UP96H/TRM-like" directly in
the legend) without having presupposed that label before clustering.

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/07_composition_plot/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/07_composition_plot"
os.makedirs(OUTDIR, exist_ok=True)

DATASETS = {
    "Visvader": "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/02_visvader_analysis",
    "Klughammer": "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/03_klughammer_analysis",
}

for name, d in DATASETS.items():
    print(f"\n=== {name} ===")
    prop = pd.read_csv(f"{d}/persample_cluster_proportions.csv", index_col=0)
    scores = pd.read_csv(f"{d}/persample_scores.csv")
    cluster_char = pd.read_csv(f"{d}/cluster_characterization.csv", index_col=0)

    cluster_cols = [c for c in prop.columns]
    scores_hl = scores[scores["sample"].isin(prop.index)].copy()
    scores_hl = scores_hl.set_index("sample").loc[prop.index]
    order = scores_hl["mean_LCOR_OE_score"].sort_values().index
    prop = prop.loc[order]
    scores_hl = scores_hl.loc[order]

    n_low = (scores_hl["LCOR_group"] == "Low").sum()
    n_mid = (scores_hl["LCOR_group"] == "Mid").sum()

    labels = []
    for cl in cluster_cols:
        up96h = cluster_char.loc[int(cl), "Tcell_UP96H_score"] if int(cl) in cluster_char.index else cluster_char.loc[cl, "Tcell_UP96H_score"]
        up24h = cluster_char.loc[int(cl), "Tcell_UP24H_score"] if int(cl) in cluster_char.index else cluster_char.loc[cl, "Tcell_UP24H_score"]
        tag = "TRM-like(96h+)" if up96h == cluster_char["Tcell_UP96H_score"].max() else (
              "act./prolif.(24h+)" if up24h == cluster_char["Tcell_UP24H_score"].max() else "")
        labels.append(f"Cluster {cl}" + (f" [{tag}]" if tag else ""))

    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(prop)), 5.5))
    cmap = cm.get_cmap("tab10", len(cluster_cols))
    bottom = np.zeros(len(prop))
    for i, cl in enumerate(cluster_cols):
        ax.bar(range(len(prop)), prop[cl].values, bottom=bottom, color=cmap(i), label=labels[i], width=0.8)
        bottom += prop[cl].values

    ax.axvline(n_low - 0.5, color="black", ls="--", lw=1.2)
    ax.axvline(n_low + n_mid - 0.5, color="black", ls=":", lw=1)
    ax.set_xticks(range(len(prop)))
    ax.set_xticklabels(prop.index, rotation=90, fontsize=7)
    ax.set_ylabel("Proportion of CD8 T cells", fontsize=10)
    ax.set_xlabel("Patient (ordered by malignant LCOR_OE score, low -> high)", fontsize=10)
    ax.set_title(f"{name}: CD8 cluster composition by patient, ordered by LCOR program score\n"
                 f"(dashed = Low/Mid-High boundary, dotted = Mid/High boundary; unbiased clustering, no assumed marker)",
                 fontsize=10, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{name}_composition_by_lcor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTDIR}/{name}_composition_by_lcor.png")

print("\nDONE")
