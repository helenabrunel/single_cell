"""
02_composition_plot.py
========================
The FULLY unbiased view, complementary to 01's TRM%/exhaustion summary
scores: the complete CellTypist state composition (all ~50 states, top 8
+ Other) of the T-cell compartment, per patient, ordered by LCOR score --
no pre-selection of which state "counts". Two-panel figure (top: actual
LCOR score value, so the gradient is visible; bottom: stacked composition
bars), one per {dataset x LCOR_raw/LCOR_OE}.

Reuses the per-cell CellTypist labels already computed in 03_tcell_scores
(full T-cell gate: CD8+CD4+Treg+Other, not restricted to any lineage).

Output: 05_unbiased_analysis/figures/{name}_composition_by_{LCOR_raw,LCOR_OE}.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTDIR = "05_unbiased_analysis/figures"
os.makedirs(OUTDIR, exist_ok=True)

# Fixed category -> color mapping, same order and colors in BOTH datasets
# (previously colors were reassigned by each dataset's own abundance rank,
# so e.g. orange meant CD8-TRM in one plot and CD4 in the other -- fixed now)
CATEGORY_COLORS = {
    "CD8 - TRM": "#1f77b4",
    "CD8 - non-TRM": "#9467bd",
    "CD4": "#ff7f0e",
    "Treg": "#2ca02c",
    "Other (non-T / unconventional T)": "#7f7f7f",
}


def within_subtype_tercile(g, col):
    q1, q2 = g[col].quantile([1 / 3, 2 / 3])
    return np.select([g[col] <= q1, g[col] >= q2], ["Low", "High"], default="Mid")


def composition_plot(prop, score_map, group_map, score_label, title_prefix, fname, score_label_short=None):
    score_label_short = score_label_short or score_label
    scored_patients = [s for s in prop.index if s in score_map]
    order = pd.Series(score_map).loc[scored_patients].sort_values().index
    prop_ord = prop.loc[order]
    groups_ord = pd.Series(group_map).loc[order]
    n_low = (groups_ord == "Low").sum()
    n_mid = (groups_ord == "Mid").sum()

    # fixed category set and order (not sorted by abundance), same in every figure
    cat_order = list(CATEGORY_COLORS.keys())
    for c in cat_order:
        if c not in prop_ord.columns:
            prop_ord[c] = 0.0
    prop_plot = prop_ord[cat_order].copy()

    fig, (ax_score, ax) = plt.subplots(
        2, 1, figsize=(max(9, 0.5 * len(prop_plot)), 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.06})

    score_vals = pd.Series(score_map).loc[order].values
    ax_score.plot(range(len(prop_plot)), score_vals, color="black", marker="o", markersize=4, lw=1.5)
    ax_score.set_ylabel(score_label_short, fontsize=9)
    ax_score.tick_params(axis="y", labelsize=8)
    ax_score.grid(axis="y", alpha=0.2)

    bottom = np.zeros(len(prop_plot))
    for label in prop_plot.columns:
        ax.bar(range(len(prop_plot)), prop_plot[label].values, bottom=bottom,
               color=CATEGORY_COLORS[label], label=label, width=0.8)
        bottom += prop_plot[label].values

    for a in (ax_score, ax):
        if n_low > 0:
            a.axvline(n_low - 0.5, color="black", ls="--", lw=1.2)
            a.axvline(n_low + n_mid - 0.5, color="black", ls="--", lw=1.2)

    ax.set_xticks([])
    ax.set_ylabel("Proportion of T cells", fontsize=10)
    ax.set_xlabel(f"Patients (ordered by {score_label}, low -> high)", fontsize=10)
    ax_score.set_title(f"{title_prefix}: T-cell compartment composition (CellTypist-derived lineage), by patient\n"
                        f"CD8 (split TRM/non-TRM) / CD4 / Treg / non-T -- ordered by {score_label} "
                        f"(dashed lines = Low/Mid and Mid/High tercile boundaries, within-subtype terciles)",
                        fontsize=10, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTDIR}/{fname}.png")


for name in ["Visvader", "Klughammer"]:
    print(f"\n=== {name} ===")
    percell = pd.read_csv(f"03_tcell_scores/figures/{name}_percell_tcell_labels.csv")
    mal = pd.read_csv(f"02_malignant_scores/figures/{name}_persample_malignant_scores.csv")

    mal["LCOR_OE_group"] = mal.groupby("subtype", group_keys=False).apply(
        lambda g: pd.Series(within_subtype_tercile(g, "LCOR_OE_score"), index=g.index))
    mal["LCOR_raw_group"] = mal.groupby("subtype", group_keys=False).apply(
        lambda g: pd.Series(within_subtype_tercile(g, "LCOR_raw_pseudobulk"), index=g.index))

    # Explicit CD8/CD4/Treg/Other lineage categories (CellTypist's own fine
    # labels -- "...cytotoxic"="CD8", "...helper"="CD4" -- aggregated up, since
    # no CellTypist model outputs "CD8"/"CD4" directly: Immune_All_High only
    # has one pooled "T cells" category, no CD4/CD8 split at all). CD8 is
    # further split into TRM vs non-TRM, the project's key phenotype axis.
    def display_category(row):
        if row["lineage_celltypist"] == "CD8":
            return "CD8 - TRM" if row["is_trm_celltypist"] else "CD8 - non-TRM"
        if row["lineage_celltypist"] == "CD4":
            return "CD4"
        if row["lineage_celltypist"] == "Treg":
            return "Treg"
        return "Other (non-T / unconventional T)"

    percell["category"] = percell.apply(display_category, axis=1)
    prop = pd.crosstab(percell["sample"], percell["category"], normalize="index")
    print(f"  {prop.shape[0]} patients")
    print(f"  Overall composition: {prop.mean(axis=0).round(3).to_dict()}")

    composition_plot(prop, dict(zip(mal["sample"], mal["LCOR_OE_score"])),
                      dict(zip(mal["sample"], mal["LCOR_OE_group"])),
                      "LCOR_OE signature score (malignant)", name, f"{name}_composition_by_LCOR_OE",
                      score_label_short="LCOR_OE score")
    composition_plot(prop, dict(zip(mal["sample"], mal["LCOR_raw_pseudobulk"])),
                      dict(zip(mal["sample"], mal["LCOR_raw_group"])),
                      "LCOR (raw, malignant)", name, f"{name}_composition_by_LCOR_raw",
                      score_label_short="LCOR (raw)")

print("\nDONE")
