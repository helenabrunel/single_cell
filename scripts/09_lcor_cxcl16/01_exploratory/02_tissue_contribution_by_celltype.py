"""
02_tissue_contribution_by_celltype.py
=====================================
Linear-scale expression and aggregate tissue-signal contribution of CXCL16
(and LCOR) by cell type, dataset-wide, per subtype, and per patient.

WHY LINEAR SCALE, NOT expm1(mean(logcounts))
----------------------------------------------
logcounts = log1p(CP10K-normalized counts) (verified: expm1(logcounts) row
sums == 10000.0, adata.uns['log1p']['base'] is None -> natural log). Taking
exp(mean(log(x))) is the geometric mean, not the arithmetic mean, and
systematically UNDERSTATES true average expression for right-skewed,
zero-inflated data like this. Every "mean expression" and every
"contribution" below is computed as an arithmetic mean/sum of a LINEAR
per-cell value:
  - raw counts (adata.layers['counts']) is used as THE primary linear metric
    throughout -- it's the most literal measure of "how much CXCL16 signal
    was actually captured from this cell type in this tissue sample", with no
    normalization assumption baked in. It is NOT depth-corrected: cell types
    with systematically higher per-cell total counts will look more
    "productive" partly for technical reasons. This matters most for
    section 1 (per-cell type/per-cell averages); it matters much less for
    sections 2-4 (total tissue contribution), where "how many reads actually
    came from this cell type" is arguably the more literal quantity anyway.

CELL GROUPS -- same 13-group scheme as 01_celltype_specificity_overview.py (Epithelial | 11 immune
subtypes via the precomputed score_* columns in compartment_immune.h5ad,
winner-takes-all, see 07_hipoxia_tim3.py | Stromal).

MYELOID DEFINITION (user-specified) -- Macro_M1 + Macro_M2 + Dendritic.
(Mast excluded -- user's definition, not this script's choice.)

OUTPUTS
-------
figures/19_CXCL16_LCOR_linear_contribution/
    linear_expression_by_group.csv     -- Q1: gene x group x subtype, mean raw
                                           count/cell, n_cells
    cxcl16_contribution_by_celltype.csv -- Q2: CXCL16 total-signal % by group,
                                           dataset-wide + per subtype
    cxcl16_contribution_by_patient.csv -- Q3+Q4 (exact schema requested):
        patient_id, subtype, pct_cxcl16_from_epithelial,
        pct_cxcl16_from_myeloid, pct_cxcl16_from_other, n_epithelial,
        n_myeloid, myeloid_epithelial_ratio
    fig_patient_contribution_distribution.png -- Q3: per-patient distribution
                                                  of %epithelial / %myeloid,
                                                  by subtype
    fig_myeloid_ratio_vs_contribution.png     -- Q4: consistency check scatter
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/01_exploratorio/scripts/02_tissue_contribution_by_celltype.py
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Config ───────────────────────────────────────────────────────────────────
OUTDIR = "LCOR-CXCL16/01_exploratorio/figures/02_tissue_contribution_by_celltype"
GENES = ["CXCL16", "LCOR"]
MYELOID_GROUPS = ["Macro_M1", "Macro_M2", "Dendritic"]
SUBTYPES = ["All", "ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
CONTRIB_COLORS = {"Epithelial": "#E41A1C", "Myeloid": "#377EB8", "Other": "#999999"}
os.makedirs(OUTDIR, exist_ok=True)

# Immune subtype scoring -- copied verbatim from 07_hipoxia_tim3.py (and 01_celltype_specificity_overview.py)
IMMUNE_CELL_TYPES = {
    "CD8_T"      : "score_CD8_T",
    "CD4_T"      : "score_CD4_T",
    "Treg"       : "score_Treg",
    "NK"         : "score_NK",
    "Macro_M1"   : "score_Macro_M1",
    "Macro_M2"   : "score_Macro_M2",
    "B_cell"     : "score_B_cell",
    "Plasmablast": "score_Plasmablast",
    "Dendritic"  : "score_Dendritic",
    "Mast"       : "score_Mast",
}
GROUP_ORDER = ["Epithelial"] + list(IMMUNE_CELL_TYPES.keys()) + ["Unknown", "Stromal"]


def assign_immune_type(obs):
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items() if col in obs.columns}
    score_matrix = obs[[col for col in cols_present.values()]].values
    cell_types   = list(cols_present.keys())
    best_idx     = score_matrix.argmax(axis=1)
    best_score   = score_matrix.max(axis=1)
    return np.array([cell_types[i] if s > 0 else "Unknown"
                      for i, s in zip(best_idx, best_score)])


def extract(h5ad_file, genes):
    print(f"  Loading {h5ad_file}...")
    adata = sc.read_h5ad(h5ad_file)
    print(f"    Cells: {adata.n_obs:,}")
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise SystemExit(f"  Gene(s) not found in {h5ad_file}: {missing}")

    idx = [adata.var_names.get_loc(g) for g in genes]
    counts = adata.layers["counts"][:, idx]
    counts = np.asarray(counts.todense()) if issparse(counts) else np.asarray(counts)

    df = pd.DataFrame({
        "sample": adata.obs["sample"].astype(str).values,
        "subtype": adata.obs["subtype"].astype(str).values,
    })
    for i, g in enumerate(genes):
        df[f"{g}_counts"] = counts[:, i].astype(float)
    return df, adata.obs


def subset(df, subtype):
    return df if subtype == "All" else df[df["subtype"] == subtype]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD + BUILD THE 13-GROUP TABLE (raw counts only -- no logcounts used below)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print(" 02 -- CXCL16 / LCOR -- linear-scale expression & tissue contribution")
print("="*65)

epi_df, _ = extract("compartment_epithelial.h5ad", GENES)
epi_df["group"] = "Epithelial"

imm_df, imm_obs = extract("compartment_immune.h5ad", GENES)
imm_df["group"] = assign_immune_type(imm_obs)

str_df, _ = extract("compartment_stromal.h5ad", GENES)
str_df["group"] = "Stromal"

combined = pd.concat([epi_df, imm_df, str_df], ignore_index=True)
combined["is_epithelial"] = combined["group"] == "Epithelial"
combined["is_myeloid"] = combined["group"].isin(MYELOID_GROUPS)
print(f"\n  Combined: {len(combined):,} cells, {combined['sample'].nunique()} patients, "
      f"{combined['group'].nunique()} groups")


# ══════════════════════════════════════════════════════════════════════════════
# Q1 -- linear mean expression (raw counts) by gene x group x subtype
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Q1 -- mean raw-count expression by group x subtype (arithmetic mean)")
print("="*65)

rows1 = []
for subtype in SUBTYPES:
    d = subset(combined, subtype)
    for grp in GROUP_ORDER:
        dg = d[d["group"] == grp]
        n = len(dg)
        for g in GENES:
            mean_raw = dg[f"{g}_counts"].mean() if n > 0 else np.nan
            rows1.append({"subtype": subtype, "group": grp, "gene": g,
                          "n_cells": n, "mean_raw_count": mean_raw})
df1 = pd.DataFrame(rows1)
df1.to_csv(f"{OUTDIR}/linear_expression_by_group.csv", index=False)
print(f"  Saved: {OUTDIR}/linear_expression_by_group.csv")

for g in GENES:
    print(f"\n  {g} -- mean raw count per cell, by group (All subtypes):")
    piv = df1[(df1.subtype == "All") & (df1.gene == g)].set_index("group")["mean_raw_count"]
    print(piv.reindex(GROUP_ORDER).round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Q2 -- total estimated CXCL16 contribution by cell type (n x mean, as % of total)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Q2 -- CXCL16 total tissue-signal contribution by cell type")
print("="*65)

rows2 = []
for subtype in SUBTYPES:
    d = subset(combined, subtype)
    total_signal = d["CXCL16_counts"].sum()
    for grp in GROUP_ORDER:
        dg = d[d["group"] == grp]
        n = len(dg)
        mean_raw = dg["CXCL16_counts"].mean() if n > 0 else 0.0
        contrib = n * mean_raw  # == dg["CXCL16_counts"].sum(), by construction
        pct = 100 * contrib / total_signal if total_signal > 0 else np.nan
        rows2.append({"subtype": subtype, "group": grp, "n_cells": n,
                      "mean_raw_count": mean_raw, "total_cxcl16_counts": contrib,
                      "pct_of_total_cxcl16_signal": pct})
df2 = pd.DataFrame(rows2)
df2.to_csv(f"{OUTDIR}/cxcl16_contribution_by_celltype.csv", index=False)
print(f"  Saved: {OUTDIR}/cxcl16_contribution_by_celltype.csv")

print("\n  CXCL16 contribution ranking, All subtypes combined (n x mean, % of total):")
print(df2[df2.subtype == "All"].sort_values("pct_of_total_cxcl16_signal", ascending=False)
      [["group", "n_cells", "mean_raw_count", "pct_of_total_cxcl16_signal"]]
      .round(2).to_string(index=False))

myeloid_pct_all = df2[(df2.subtype == "All") & (df2.group.isin(MYELOID_GROUPS))]["pct_of_total_cxcl16_signal"].sum()
epi_pct_all = df2[(df2.subtype == "All") & (df2.group == "Epithelial")]["pct_of_total_cxcl16_signal"].values[0]
print(f"\n  Collapsed (All subtypes): Epithelial={epi_pct_all:.1f}%, "
      f"Myeloid(Macro_M1+M2+Dendritic)={myeloid_pct_all:.1f}%, "
      f"Other={100-epi_pct_all-myeloid_pct_all:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Q3 + Q4 -- per-patient CXCL16 contribution breakdown, and consistency check
# against the myeloid:epithelial cell-count ratio
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Q3 -- per-patient CXCL16 contribution (Epithelial vs Myeloid vs Other)")
print("="*65)

patient_rows = []
for sample, dp in combined.groupby("sample"):
    total = dp["CXCL16_counts"].sum()
    epi = dp.loc[dp["is_epithelial"], "CXCL16_counts"].sum()
    mye = dp.loc[dp["is_myeloid"], "CXCL16_counts"].sum()
    other = total - epi - mye
    n_epi = int(dp["is_epithelial"].sum())
    n_mye = int(dp["is_myeloid"].sum())
    patient_rows.append({
        "patient_id": sample,
        "subtype": dp["subtype"].iloc[0],
        "pct_cxcl16_from_epithelial": 100 * epi / total if total > 0 else np.nan,
        "pct_cxcl16_from_myeloid":    100 * mye / total if total > 0 else np.nan,
        "pct_cxcl16_from_other":      100 * other / total if total > 0 else np.nan,
        "n_epithelial": n_epi,
        "n_myeloid": n_mye,
        "myeloid_epithelial_ratio": n_mye / n_epi if n_epi > 0 else np.nan,
    })

df3 = pd.DataFrame(patient_rows)
df3["subtype"] = pd.Categorical(df3["subtype"], categories=["ER", "HER2", "TNBC"], ordered=True)
df3 = df3.sort_values(["subtype", "patient_id"]).reset_index(drop=True)
df3.to_csv(f"{OUTDIR}/cxcl16_contribution_by_patient.csv", index=False)
print(f"  Saved: {OUTDIR}/cxcl16_contribution_by_patient.csv  (n={len(df3)} patients)")

print("\n  Distribution across patients (this is the point: NOT just one global number):")
for col in ["pct_cxcl16_from_epithelial", "pct_cxcl16_from_myeloid", "pct_cxcl16_from_other"]:
    desc = df3[col].describe(percentiles=[0.25, 0.5, 0.75])
    print(f"    {col}: mean={desc['mean']:.1f}  median={desc['50%']:.1f}  "
          f"std={desc['std']:.1f}  min={desc['min']:.1f}  max={desc['max']:.1f}  "
          f"IQR=[{desc['25%']:.1f}, {desc['75%']:.1f}]")

print("\n  Same, by subtype:")
print(df3.groupby("subtype", observed=True)[["pct_cxcl16_from_epithelial", "pct_cxcl16_from_myeloid"]]
      .agg(["mean", "std", "min", "max"]).round(1).to_string())

print("\n" + "="*65)
print(" Q4 -- myeloid:epithelial cell-count ratio vs % CXCL16 from myeloid")
print("="*65)

valid = df3.dropna(subset=["myeloid_epithelial_ratio", "pct_cxcl16_from_myeloid"])
valid = valid[np.isfinite(valid["myeloid_epithelial_ratio"])]
r_p, p_p = stats.pearsonr(valid["myeloid_epithelial_ratio"], valid["pct_cxcl16_from_myeloid"])
r_s, p_s = stats.spearmanr(valid["myeloid_epithelial_ratio"], valid["pct_cxcl16_from_myeloid"])
print(f"  Pearson  r   = {r_p:.3f}, p={p_p:.3g}  (n={len(valid)})")
print(f"  Spearman rho = {r_s:.3f}, p={p_s:.3g}  (n={len(valid)})")
print("  (Positive correlation expected as an internal-consistency check: patients")
print("   with relatively more myeloid cells should show relatively more of their")
print("   CXCL16 signal coming from myeloid cells.)")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building plots...")

# Fig 1 -- per-patient distribution of %epithelial / %myeloid, by subtype
rng = np.random.RandomState(0)
fig, axes = plt.subplots(1, len(SUBTYPES), figsize=(3.0 * len(SUBTYPES), 4.2), sharey=True)
for ax, subtype in zip(axes, SUBTYPES):
    d = subset(df3, subtype)
    data = [d["pct_cxcl16_from_epithelial"].dropna().values,
            d["pct_cxcl16_from_myeloid"].dropna().values]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False,
                     patch_artist=True, zorder=2)
    for patch, key in zip(bp["boxes"], ["Epithelial", "Myeloid"]):
        patch.set_facecolor(CONTRIB_COLORS[key]); patch.set_alpha(0.25)
        patch.set_edgecolor(CONTRIB_COLORS[key])
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#444444")
    for xi, vals, key in zip([0, 1], data, ["Epithelial", "Myeloid"]):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + jitter, vals, s=22, color=CONTRIB_COLORS[key],
                   edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Epithelial", "Myeloid"], fontsize=8.5)
    title_color = SUBTYPE_COLORS.get(subtype, "black")
    ax.set_title(f"{subtype} (n={d['patient_id'].nunique()})", fontsize=10,
                 fontweight="bold", color=title_color)
    ax.set_ylim(-3, 103)
    ax.grid(axis="y", color="#EAEAEA", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("% of patient's total CXCL16 signal", fontsize=9)
fig.suptitle("Per-patient CXCL16 contribution: Epithelial vs Myeloid (Macro_M1+M2+Dendritic)\n"
             "each point = one patient", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_patient_contribution_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_patient_contribution_distribution.png")

# Fig 2 -- consistency check: myeloid:epithelial cell ratio vs % CXCL16 from myeloid
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for st, color in SUBTYPE_COLORS.items():
    d = valid[valid["subtype"] == st]
    ax.scatter(d["myeloid_epithelial_ratio"], d["pct_cxcl16_from_myeloid"],
               color=color, label=st, s=45, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Myeloid : Epithelial cell-count ratio (log scale)", fontsize=9)
ax.set_ylabel("% of patient's CXCL16 signal from Myeloid", fontsize=9)
ax.set_title(f"Pearson r={r_p:.2f} (p={p_p:.2g})   Spearman ρ={r_s:.2f} (p={p_s:.2g})   n={len(valid)}",
             fontsize=9.5, fontweight="bold")
ax.legend(fontsize=8, title="Subtype")
ax.grid(alpha=0.25, zorder=0)
fig.suptitle("Internal-consistency check: cell-count ratio vs signal-contribution %", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_myeloid_ratio_vs_contribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_myeloid_ratio_vs_contribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
readme = [
    "02 -- CXCL16 / LCOR -- linear-scale expression & tissue contribution",
    "="*70,
    "",
    "ALL VALUES ARE ARITHMETIC MEANS/SUMS OF RAW COUNTS (adata.layers['counts']),",
    "NEVER expm1(mean(logcounts)) -- see script docstring for why that matters.",
    "",
    "CELL GROUPS: same 13-group scheme as 01_celltype_specificity_overview.py (Epithelial | 11 immune",
    "subtypes via precomputed score_* columns in compartment_immune.h5ad,",
    "winner-takes-all -- see 07_hipoxia_tim3.py | Stromal).",
    "MYELOID = Macro_M1 + Macro_M2 + Dendritic (user-specified; Mast excluded).",
    "",
    "Q1 -- linear_expression_by_group.csv",
    "-"*70,
    "  Mean raw CXCL16/LCOR count per cell, by group x subtype.",
    "",
    "Q2 -- cxcl16_contribution_by_celltype.csv",
    "-"*70,
    "  total_cxcl16_counts (= n_cells x mean_raw_count = sum of raw counts)",
    "  per group, expressed as % of the total CXCL16 signal in that subtype",
    "  (or All). Answers: which cell type supplies most of the CXCL16 actually",
    "  detected in a whole-tissue sample.",
    f"  Collapsed (All subtypes): Epithelial={epi_pct_all:.1f}%, "
    f"Myeloid={myeloid_pct_all:.1f}%, Other={100-epi_pct_all-myeloid_pct_all:.1f}%",
    "",
    "Q3 -- cxcl16_contribution_by_patient.csv / fig_patient_contribution_distribution.png",
    "-"*70,
    "  Per-patient % of that patient's own total CXCL16 signal coming from",
    "  Epithelial vs Myeloid vs Other. See the printed per-patient describe()",
    "  stats above for the actual spread (mean/median/std/min/max/IQR) -- the",
    "  question was whether this % varies a lot between patients, not just a",
    "  single pooled number.",
    "",
    "Q4 -- fig_myeloid_ratio_vs_contribution.png",
    "-"*70,
    f"  myeloid_epithelial_ratio (n_myeloid/n_epithelial per patient) vs",
    f"  pct_cxcl16_from_myeloid: Pearson r={r_p:.3f} (p={p_p:.3g}), "
    f"Spearman rho={r_s:.3f} (p={p_s:.3g}), n={len(valid)}.",
    "  A positive correlation is the expected internal-consistency result: more",
    "  myeloid cells relative to epithelial ones should mean more of a patient's",
    "  CXCL16 signal comes from myeloid cells, all else equal.",
    "",
    "CAVEATS",
    "-"*70,
    "  - Raw counts are NOT sequencing-depth-corrected. If myeloid and",
    "    epithelial cells differ systematically in total captured counts/cell",
    "    (RNA content, capture efficiency), that alone could shift contribution",
    "    % independent of true biology. Cross-check against",
    "    linear_expression_by_group.csv (mean_raw_count) if this matters for a",
    "    specific conclusion.",
    "  - 'Other' includes CD8_T/CD4_T/Treg/NK/B_cell/Plasmablast/Mast/Unknown/",
    "    Stromal -- a large, heterogeneous bucket, not a single cell type.",
    "  - This is a per-patient, cross-sectional description, not a test of",
    "    the CXCL16->CXCR6 interaction itself (see 01_liana_interaction_vs_lcor.py for that).",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Output: {OUTDIR}/
============================================================
""")
