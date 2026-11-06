"""
01_pseudobulk_myeloid_headline.py
==================================
Per-patient PSEUDOBULK of LCOR vs CXCL16, restricted to MYELOID cells
(Macro_M1 + Macro_M2 + Dendritic combined), correlated across patients.
Direct myeloid analogue of 03_pseudobulk/01_pseudobulk_by_patient_headline.py,
which ran the same test in the malignant (epithelial) compartment and found
it null (Spearman rho=-0.11, n=34, p=0.53; TNBC rho=0.24, n=8, p=0.57).

WHY THIS SCRIPT EXISTS
-----------------------
01_exploratorio/01_celltype_specificity_overview.py showed CXCL16 is most
specifically expressed in myeloid cells, not malignant epithelial cells:
detection is ~3x higher in Macro_M1 (39.6%), Macro_M2 (46.7%) and Dendritic
(44.5%) than in Epithelial (13.7%). LCOR detection is also somewhat higher
in myeloid cells (13.6-17.3%) than Epithelial (15.5%, comparable). Restricting
the LCOR-CXCL16 test to myeloid cells -- where CXCL16 is actually most
expressed -- might reveal a relationship the malignant-compartment tests
missed. An earlier informal check lacked the depth/patient corrections
developed in 02_correlacion/ and 03_pseudobulk/; this reruns the question
with that full methodology.

CELL SELECTION
--------------
compartment_immune.h5ad already carries precomputed per-cell marker scores
(score_Macro_M1, score_Macro_M2, score_Dendritic, ...). Winner-takes-all
assignment (assign_immune_type, copied verbatim from
01_exploratorio/01_celltype_specificity_overview.py) is used to label every
cell, then MYELOID = cells assigned to Macro_M1, Macro_M2, or Dendritic.
These three are pooled into one "Myeloid" group for the primary (best
statistical power) test, since per-patient myeloid cell counts are much
lower than malignant cell counts (median ~180 vs thousands); a secondary
breakdown by Macro_M1/Macro_M2/Dendritic separately is also produced for
patients with enough cells in each subgroup.

PSEUDOBULK METHOD
-----------------
Identical to 03_pseudobulk/01: pseudobulk_CP10K = 1e4 * sum(raw counts) /
sum(total_counts) across all myeloid cells of a patient (arithmetic
sum-then-normalize, not mean-of-logcounts). log1p-transformed for
correlation. MIN_CELLS = 20 myeloid cells/patient (same threshold as the
malignant-compartment script, for direct comparability) -- retains 32/34
patients (see cell-count check below).

DEPTH- AND CELL-COUNT-CONFOUND CHECKS
---------------------------------------
Same two checks as 03_pseudobulk/01, since the malignant-compartment result
there showed no depth confound but this is a different compartment with
different per-cell UMI depth: (a) does patient median depth predict either
gene's pseudobulk value, with a depth-adjusted residual correlation; (b)
does the correlation strengthen in better-powered (higher n_cells) patients
(sensitivity to a rising n_cells threshold).

Input:  compartment_immune.h5ad
Output: LCOR-CXCL16/05_myeloid/figures/01_pseudobulk_myeloid_headline/
    persample_pseudobulk_myeloid.csv
    persample_pseudobulk_subgroups.csv
    correlation_table.csv
    ncell_sensitivity.csv
    fig_pseudobulk_scatter.png
    fig_depth_confound_check.png
    fig_ncell_confound_check.png
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/05_myeloid/scripts/01_pseudobulk_myeloid_headline.py
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
H5AD_FILE   = "compartment_immune.h5ad"
OUTDIR      = "LCOR-CXCL16/05_myeloid/figures/01_pseudobulk_myeloid_headline"
GENE_X, GENE_Y = "LCOR", "CXCL16"
MIN_CELLS   = 20
MIN_CELLS_SUBGROUP = 20
SUBTYPES    = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
MYELOID_GROUPS = ["Macro_M1", "Macro_M2", "Dendritic"]
os.makedirs(OUTDIR, exist_ok=True)

# Winner-takes-all immune typing -- copied verbatim from
# 01_exploratorio/01_celltype_specificity_overview.py (07_hipoxia_tim3.py origin)
IMMUNE_CELL_TYPES = {
    "CD8_T": "score_CD8_T", "CD4_T": "score_CD4_T", "Treg": "score_Treg", "NK": "score_NK",
    "Macro_M1": "score_Macro_M1", "Macro_M2": "score_Macro_M2", "B_cell": "score_B_cell",
    "Plasmablast": "score_Plasmablast", "Dendritic": "score_Dendritic", "Mast": "score_Mast",
}


def assign_immune_type(obs):
    cols_present = {ct: col for ct, col in IMMUNE_CELL_TYPES.items() if col in obs.columns}
    score_matrix = obs[[col for col in cols_present.values()]].values
    cell_types = list(cols_present.keys())
    best_idx = score_matrix.argmax(axis=1)
    best_score = score_matrix.max(axis=1)
    return np.array([cell_types[i] if s > 0 else "Unknown"
                      for i, s in zip(best_idx, best_score)])


print("="*65)
print(f" 01 -- {GENE_X} vs {GENE_Y} -- PATIENT PSEUDOBULK, myeloid cells only")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}   Patients: {adata.obs['sample'].nunique()}")

missing = [g for g in (GENE_X, GENE_Y) if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

celltype = assign_immune_type(adata.obs)
is_myeloid = np.isin(celltype, MYELOID_GROUPS)
print(f"  Myeloid cells ({'+'.join(MYELOID_GROUPS)}): {is_myeloid.sum():,} / {adata.n_obs:,}")

idx = [adata.var_names.get_loc(g) for g in (GENE_X, GENE_Y)]
raw = adata.layers["counts"][:, idx]
raw = np.asarray(raw.todense()) if issparse(raw) else np.asarray(raw)
logc = adata.layers["logcounts"][:, idx] if "logcounts" in adata.layers else adata.X[:, idx]
logc = np.asarray(logc.todense()) if issparse(logc) else np.asarray(logc)

obs = pd.DataFrame({
    "sample":  adata.obs["sample"].astype(str).values,
    "subtype": adata.obs["subtype"].astype(str).values,
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "celltype": celltype,
    f"{GENE_X}_raw": raw[:, 0].astype(float),
    f"{GENE_Y}_raw": raw[:, 1].astype(float),
    f"{GENE_X}_log": logc[:, 0],
    f"{GENE_Y}_log": logc[:, 1],
})
del adata
obs_myeloid = obs[is_myeloid].copy()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PER-PATIENT PSEUDOBULK -- myeloid, combined
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building per-patient pseudobulk (CP10K sum-then-normalize)...")


def pseudobulk_table(d, min_cells):
    rows = []
    for s, g in d.groupby("sample"):
        n = len(g)
        if n < min_cells:
            continue
        lib_size = g["total_counts"].sum()
        row = {"sample": s, "subtype": g["subtype"].iloc[0], "n_cells": n,
               "median_depth": g["total_counts"].median()}
        for gene in (GENE_X, GENE_Y):
            sum_raw = g[f"{gene}_raw"].sum()
            cp10k = 1e4 * sum_raw / lib_size
            row[f"{gene}_pseudobulk_CP10K"] = cp10k
            row[f"{gene}_log1p_CP10K"] = np.log1p(cp10k)
            row[f"{gene}_pct_detected"] = 100 * (g[f"{gene}_raw"] > 0).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["subtype"] = pd.Categorical(out["subtype"], categories=SUBTYPES, ordered=True)
        out = out.sort_values(["subtype", "sample"]).reset_index(drop=True)
    return out


df = pseudobulk_table(obs_myeloid, MIN_CELLS)
df.to_csv(f"{OUTDIR}/persample_pseudobulk_myeloid.csv", index=False)
print(f"  Patients retained (>={MIN_CELLS} myeloid cells): {len(df)} / {obs['sample'].nunique()}")
print(f"  Saved: {OUTDIR}/persample_pseudobulk_myeloid.csv")
print("\n  Per-subtype patient counts:")
print(df["subtype"].value_counts().reindex(SUBTYPES).to_string())

# Secondary breakdown: each myeloid subgroup separately (diagnostic, lower n)
sub_rows = []
for grp in MYELOID_GROUPS:
    d_grp = pseudobulk_table(obs_myeloid[obs_myeloid["celltype"] == grp], MIN_CELLS_SUBGROUP)
    d_grp["myeloid_subgroup"] = grp
    sub_rows.append(d_grp)
df_subgroups = pd.concat(sub_rows, ignore_index=True) if sub_rows else pd.DataFrame()
df_subgroups.to_csv(f"{OUTDIR}/persample_pseudobulk_subgroups.csv", index=False)
print(f"\n  Saved: {OUTDIR}/persample_pseudobulk_subgroups.csv")
print("  Patients retained per myeloid subgroup (>= {} cells):".format(MIN_CELLS_SUBGROUP))
if len(df_subgroups):
    print(df_subgroups.groupby("myeloid_subgroup").size().to_string())


def corr_group(d, xcol, ycol):
    if len(d) < 4:
        return dict(n=len(d), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_rho=np.nan, spearman_p=np.nan)
    r, p = stats.pearsonr(d[xcol], d[ycol])
    rho, p_s = stats.spearmanr(d[xcol], d[ycol])
    return dict(n=len(d), pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=p_s)


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATIONS -- combined myeloid, All + per subtype; subgroups separately
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Correlations: LCOR vs CXCL16 pseudobulk, across patients (myeloid, combined)")
print("="*65)

corr_rows = []
xcol, ycol = f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K"
res_all = corr_group(df, xcol, ycol)
corr_rows.append({"group": "All (myeloid combined)", **res_all})
print(f"\n  All patients (n={res_all['n']}): Pearson r={res_all['pearson_r']:.3f} "
      f"(p={res_all['pearson_p']:.3g}), Spearman rho={res_all['spearman_rho']:.3f} "
      f"(p={res_all['spearman_p']:.3g})")
for st in SUBTYPES:
    d_st = df[df["subtype"] == st]
    res_st = corr_group(d_st, xcol, ycol)
    corr_rows.append({"group": st, **res_st})
    print(f"    {st} (n={res_st['n']}): Pearson r={res_st['pearson_r']:.3f} "
          f"(p={res_st['pearson_p']:.3g}), Spearman rho={res_st['spearman_rho']:.3f} "
          f"(p={res_st['spearman_p']:.3g})")

print("\n  Same test within each myeloid subgroup separately (diagnostic, lower n):")
subgroup_corr_rows = []
for grp in MYELOID_GROUPS:
    d_grp = df_subgroups[df_subgroups["myeloid_subgroup"] == grp]
    res_grp = corr_group(d_grp, xcol, ycol)
    subgroup_corr_rows.append({"myeloid_subgroup": grp, **res_grp})
    print(f"    {grp} (n={res_grp['n']}): Pearson r={res_grp['pearson_r']:.3f} "
          f"(p={res_grp['pearson_p']:.3g}), Spearman rho={res_grp['spearman_rho']:.3f} "
          f"(p={res_grp['spearman_p']:.3g})")
df_subgroup_corr = pd.DataFrame(subgroup_corr_rows)

# ══════════════════════════════════════════════════════════════════════════════
# DEPTH-CONFOUND CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Depth-confound check")
print("="*65)

df["log_median_depth"] = np.log(df["median_depth"])
depth_rows = []
for gene, col in [(GENE_X, f"{GENE_X}_log1p_CP10K"), (GENE_Y, f"{GENE_Y}_log1p_CP10K")]:
    rho, p = stats.spearmanr(df["log_median_depth"], df[col])
    depth_rows.append({"gene": gene, "spearman_rho_vs_depth": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient median depth: Spearman rho={rho:.3f}, p={p:.3g}")


def resid_vs_depth(y):
    slope, intercept, r, p, se = stats.linregress(df["log_median_depth"], y)
    return y - (intercept + slope * df["log_median_depth"])


df[f"{GENE_X}_resid"] = resid_vs_depth(df[f"{GENE_X}_log1p_CP10K"])
df[f"{GENE_Y}_resid"] = resid_vs_depth(df[f"{GENE_Y}_log1p_CP10K"])
rho_adj, p_adj = stats.spearmanr(df[f"{GENE_X}_resid"], df[f"{GENE_Y}_resid"])
print(f"\n  DEPTH-ADJUSTED (residuals after regressing each gene on log median depth):")
print(f"    Spearman(LCOR resid, CXCL16 resid) = {rho_adj:.3f}, p={p_adj:.3g}, n={len(df)}")
corr_rows.append({"group": "All_depth_adjusted_resid", "n": len(df),
                   "pearson_r": np.nan, "pearson_p": np.nan,
                   "spearman_rho": rho_adj, "spearman_p": p_adj})
df_corr = pd.DataFrame(corr_rows)

# ══════════════════════════════════════════════════════════════════════════════
# CELL-COUNT CONFOUND CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(" Cell-count confound check")
print("="*65)

ncell_rows = []
for gene, col in [(GENE_X, f"{GENE_X}_log1p_CP10K"), (GENE_Y, f"{GENE_Y}_log1p_CP10K")]:
    rho, p = stats.spearmanr(df["n_cells"], df[col])
    ncell_rows.append({"gene": gene, "spearman_rho_vs_ncells": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient n_cells: Spearman rho={rho:.3f}, p={p:.3g}")

NCELL_THRESHOLDS = [MIN_CELLS, 50, 100, 200]
print(f"\n  LCOR-CXCL16 correlation at rising n_cells thresholds {NCELL_THRESHOLDS}:")
sensitivity_rows = []
for min_n in NCELL_THRESHOLDS:
    d = df[df["n_cells"] >= min_n]
    res = corr_group(d, xcol, ycol)
    sensitivity_rows.append({"min_n_cells": min_n, **res})
    print(f"    n_cells>={min_n}: n_patients={res['n']}, Spearman rho={res['spearman_rho']:.3f}, "
          f"p={res['spearman_p']:.3g}")
df_ncell_sensitivity = pd.DataFrame(sensitivity_rows)
df_ncell_sensitivity.to_csv(f"{OUTDIR}/ncell_sensitivity.csv", index=False)
print(f"  Saved: {OUTDIR}/ncell_sensitivity.csv")

valid_sens = [r for r in sensitivity_rows if r["n"] >= 4]
ncells_strengthen = (len(valid_sens) >= 2 and
                      abs(valid_sens[-1]["spearman_rho"]) > abs(valid_sens[0]["spearman_rho"]))

df_corr.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)
print(f"\n  Saved: {OUTDIR}/correlation_table.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- pseudobulk scatter, All + per subtype (myeloid combined)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building figures...")

fig, axes = plt.subplots(1, 1 + len(SUBTYPES), figsize=(5.2 * (1 + len(SUBTYPES)), 5))
ax = axes[0]
for st, color in SUBTYPE_COLORS.items():
    d = df[df["subtype"] == st]
    ax.scatter(d[xcol], d[ycol], color=color, label=st, s=55, alpha=0.85, edgecolor="white", linewidth=0.6)
ax.set_title(f"All (n={res_all['n']})\nPearson r={res_all['pearson_r']:.2f} (p={res_all['pearson_p']:.3g})\n"
             f"Spearman rho={res_all['spearman_rho']:.2f} (p={res_all['spearman_p']:.3g})",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=8, title="Subtype")
ax.set_xlabel(f"{GENE_X} pseudobulk (log1p CP10K)", fontsize=9)
ax.set_ylabel(f"{GENE_Y} pseudobulk (log1p CP10K)", fontsize=9)
ax.grid(alpha=0.25)

for i, st in enumerate(SUBTYPES):
    ax = axes[i + 1]
    d = df[df["subtype"] == st]
    res_st = corr_group(d, xcol, ycol)
    ax.scatter(d[xcol], d[ycol], color=SUBTYPE_COLORS[st], s=55, alpha=0.9, edgecolor="white", linewidth=0.6)
    if len(d) >= 2:
        try:
            z = np.polyfit(d[xcol], d[ycol], 1)
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=SUBTYPE_COLORS[st], lw=1.2, alpha=0.6, ls="--")
        except Exception:
            pass
    ax.set_title(f"{st} (n={res_st['n']})\nPearson r={res_st['pearson_r']:.2f} (p={res_st['pearson_p']:.3g})\n"
                 f"Spearman rho={res_st['spearman_rho']:.2f} (p={res_st['spearman_p']:.3g})",
                 fontsize=9, fontweight="bold", color=SUBTYPE_COLORS[st])
    ax.set_xlabel(f"{GENE_X} pseudobulk (log1p CP10K)", fontsize=9)
    ax.grid(alpha=0.25)

fig.suptitle(f"Per-patient pseudobulk, myeloid cells only (Macro_M1+Macro_M2+Dendritic) -- {GENE_X} vs {GENE_Y}",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_pseudobulk_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_pseudobulk_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- depth confound check
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
for ax, gene, col in zip(axes[:2], (GENE_X, GENE_Y), (xcol, ycol)):
    for st, color in SUBTYPE_COLORS.items():
        d = df[df["subtype"] == st]
        ax.scatter(d["median_depth"], d[col], color=color, label=st, s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
    row = next(r for r in depth_rows if r["gene"] == gene)
    ax.set_xscale("log")
    ax.set_xlabel("Patient median per-cell depth (total_counts)", fontsize=8.5)
    ax.set_ylabel(f"{gene} pseudobulk (log1p CP10K)", fontsize=8.5)
    ax.set_title(f"{gene} vs patient depth\nSpearman rho={row['spearman_rho_vs_depth']:.2f}, p={row['p']:.3g}",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)

ax = axes[2]
for st, color in SUBTYPE_COLORS.items():
    d = df[df["subtype"] == st]
    ax.scatter(d[f"{GENE_X}_resid"], d[f"{GENE_Y}_resid"], color=color, label=st, s=50, alpha=0.9, edgecolor="white", linewidth=0.6)
ax.set_xlabel(f"{GENE_X} residual (depth-adjusted)", fontsize=8.5)
ax.set_ylabel(f"{GENE_Y} residual (depth-adjusted)", fontsize=8.5)
ax.set_title(f"Depth-adjusted: {GENE_X} vs {GENE_Y} residuals\nSpearman rho={rho_adj:.2f}, p={p_adj:.3g}, n={len(df)}",
             fontsize=9, fontweight="bold")
ax.axhline(0, color="gray", lw=0.8, alpha=0.5); ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
ax.legend(fontsize=7.5, title="Subtype")
ax.grid(alpha=0.25)

fig.suptitle("Depth-confound check (myeloid pseudobulk): patient sequencing depth vs pseudobulk, depth-adjusted residuals",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_depth_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_depth_confound_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 -- cell-count confound check
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
for ax, gene, col in zip(axes[:2], (GENE_X, GENE_Y), (xcol, ycol)):
    for st, color in SUBTYPE_COLORS.items():
        d = df[df["subtype"] == st]
        ax.scatter(d["n_cells"], d[col], color=color, label=st, s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
    row = next(r for r in ncell_rows if r["gene"] == gene)
    ax.set_xscale("log")
    ax.set_xlabel("Myeloid cells contributing to this patient's pseudobulk", fontsize=8.5)
    ax.set_ylabel(f"{gene} pseudobulk (log1p CP10K)", fontsize=8.5)
    ax.set_title(f"{gene} vs n_cells\nSpearman rho={row['spearman_rho_vs_ncells']:.2f}, p={row['p']:.3g}",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, title="Subtype")

ax = axes[2]
ax.plot(df_ncell_sensitivity["min_n_cells"], df_ncell_sensitivity["spearman_rho"],
        "o-", color="#377EB8", lw=2, markersize=7)
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
for _, r in df_ncell_sensitivity.iterrows():
    ax.annotate(f"n={int(r['n'])}", (r["min_n_cells"], r["spearman_rho"]),
                textcoords="offset points", xytext=(0, 8), fontsize=7.5, ha="center")
ax.set_xscale("log")
ax.set_xlabel("Min. n_cells threshold (patients below excluded)", fontsize=8.5)
ax.set_ylabel(f"Spearman rho ({GENE_X} vs {GENE_Y})", fontsize=8.5)
ax.set_title("Does the correlation strengthen in\nbetter-powered (higher n_cells) patients?",
             fontsize=9, fontweight="bold")
ax.grid(alpha=0.25)

fig.suptitle("Cell-count confound check (myeloid pseudobulk): patient n_cells vs pseudobulk, threshold sensitivity",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_ncell_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_ncell_confound_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════════════
tnbc_row = df_corr[df_corr.group == "TNBC"].iloc[0]

readme = [
    f"01 -- {GENE_X} vs {GENE_Y} -- per-patient pseudobulk, MYELOID cells only",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "03_pseudobulk/01_pseudobulk_by_patient_headline.py found LCOR-CXCL16",
    "pseudobulk correlation null in the malignant compartment (rho=-0.11,",
    "n=34, p=0.53; TNBC rho=0.24, n=8, p=0.57). 01_exploratorio/",
    "01_celltype_specificity_overview.py showed CXCL16 is far more",
    "cell-type-specific to myeloid cells (Macro_M1/Macro_M2/Dendritic,",
    "~40-47% detection) than to malignant epithelial cells (13.7%), so the",
    "LCOR-CXCL16 test might behave",
    "differently in the compartment where CXCL16 is actually most",
    "expressed -- this reruns the same pseudobulk methodology as",
    "03_pseudobulk/01, restricted to myeloid cells instead of malignant.",
    "",
    "CELL SELECTION",
    "-"*70,
    "  Winner-takes-all argmax over score_Macro_M1/score_Macro_M2/",
    "  score_Dendritic (+ other immune scores) in compartment_immune.h5ad,",
    "  copied verbatim from 01_exploratorio/01_celltype_specificity_overview.py.",
    f"  MYELOID = cells assigned to {' + '.join(MYELOID_GROUPS)} ({is_myeloid.sum():,} cells,",
    f"  {obs['sample'].nunique()} patients total). Pooled for the primary test",
    "  (power); each subgroup also tested separately (see below, lower n).",
    "",
    "PSEUDOBULK METHOD",
    "-"*70,
    "  Identical to 03_pseudobulk/01: pseudobulk_CP10K = 1e4 * sum(raw counts) /",
    "  sum(total_counts), summed across all myeloid cells of a patient.",
    "  log1p-transformed for correlation.",
    f"  MIN_CELLS = {MIN_CELLS} myeloid cells/patient (same threshold as the",
    f"  malignant-compartment script) -- {len(df)}/{obs['sample'].nunique()} patients retained.",
    "",
    "RESULTS -- headline correlation (myeloid, combined)",
    "-"*70,
    f"  All patients (n={res_all['n']}): Pearson r={res_all['pearson_r']:.3f} "
    f"(p={res_all['pearson_p']:.3g}), Spearman rho={res_all['spearman_rho']:.3f} "
    f"(p={res_all['spearman_p']:.3g})",
    f"  TNBC only (n={tnbc_row['n']}): Spearman rho={tnbc_row['spearman_rho']:.3f} "
    f"(p={tnbc_row['spearman_p']:.3g})",
    "  Full ER/HER2/TNBC breakdown: correlation_table.csv",
    "",
    "RESULTS -- by myeloid subgroup separately (diagnostic, lower n each)",
    "-"*70,
] + [
    f"  {r['myeloid_subgroup']} (n={r['n']}): Pearson r={r['pearson_r']:.3f} "
    f"(p={r['pearson_p']:.3g}), Spearman rho={r['spearman_rho']:.3f} (p={r['spearman_p']:.3g})"
    for r in subgroup_corr_rows
] + [
    "",
    "DEPTH-CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient median depth: Spearman rho={r['spearman_rho_vs_depth']:.3f}, p={r['p']:.3g}"
    for r in depth_rows
] + [
    f"  Depth-adjusted residual correlation: Spearman rho={rho_adj:.3f}, p={p_adj:.3g}",
    "",
    "CELL-COUNT CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient n_cells: Spearman rho={r['spearman_rho_vs_ncells']:.3f}, p={r['p']:.3g}"
    for r in ncell_rows
] + [
    "  Sensitivity to a stricter n_cells threshold:",
] + [
    f"    n_cells>={int(r['min_n_cells'])}: n_patients={int(r['n'])}, Spearman rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.3g}"
    for _, r in df_ncell_sensitivity.iterrows()
] + [
    "",
    "OUTPUT FILES",
    "-"*70,
    "  persample_pseudobulk_myeloid.csv    -- per-patient pseudobulk (myeloid combined), n_cells, depth, subtype",
    "  persample_pseudobulk_subgroups.csv  -- same, split by Macro_M1/Macro_M2/Dendritic",
    "  correlation_table.csv               -- Pearson/Spearman, All/ER/HER2/TNBC + depth-adjusted",
    "  ncell_sensitivity.csv               -- correlation at each n_cells threshold",
    "  fig_pseudobulk_scatter.png          -- myeloid combined, All + per subtype",
    "  fig_depth_confound_check.png",
    "  fig_ncell_confound_check.png",
    "",
    "CAVEATS",
    "-"*70,
    f"  - n={len(df)} patients, and myeloid cell counts per patient are much lower",
    "    than malignant cell counts (median ~180 vs thousands) -- pseudobulk",
    "    per patient is noisier here even with CXCL16's higher detection rate.",
    "  - Myeloid subgroup breakdown (Macro_M1/M2/Dendritic separately) has",
    "    fewer patients meeting MIN_CELLS and should be read as exploratory,",
    "    not a confirmatory test on its own.",
    "  - Correlational, cross-sectional; cannot establish causality/direction.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Patients: {len(df)} (All) / TNBC n={tnbc_row['n']}
  All (myeloid combined): Pearson r={res_all['pearson_r']:.3f} (p={res_all['pearson_p']:.3g})  Spearman rho={res_all['spearman_rho']:.3f} (p={res_all['spearman_p']:.3g})
  TNBC:                   Spearman rho={tnbc_row['spearman_rho']:.3f} (p={tnbc_row['spearman_p']:.3g})
  Depth-adjusted residual rho: {rho_adj:.3f} (p={p_adj:.3g})
  Output: {OUTDIR}/
============================================================
""")
