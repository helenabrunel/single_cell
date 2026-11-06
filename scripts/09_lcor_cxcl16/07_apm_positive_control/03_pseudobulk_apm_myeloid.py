"""
03_pseudobulk_apm_myeloid.py
=============================
POSITIVE-CONTROL check, MYELOID compartment (Macro_M1+Macro_M2+Dendritic):
per-patient PSEUDOBULK of LCOR vs the APM meta-gene (TAP1+TAP2+TAPBP+HLA-A+HLA-B+
HLA-C+PSMB8+PSMB9+B2M raw counts summed per cell). Direct myeloid
analogue of 01_pseudobulk_apm_malignant.py (this project's malignant-
compartment positive control) and of 05_myeloid/01_pseudobulk_myeloid_
headline.py (LCOR vs CXCL16, myeloid, null: rho=0.20, n=32, p=0.28).

WHY THIS SCRIPT EXISTS
-----------------------
The positive-control question (see 01's docstring for full rationale)
was raised specifically after the MYELOID LCOR-CXCL16 null -- this is the
myeloid-compartment version of the same check, using the identical cell
selection (winner-takes-all immune typing) as 05_myeloid.

Input:  compartment_immune.h5ad
Output: LCOR-CXCL16/07_apm_positive_control/figures/03_pseudobulk_apm_myeloid/

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/07_apm_positive_control/scripts/03_pseudobulk_apm_myeloid.py
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

H5AD_FILE   = "compartment_immune.h5ad"
OUTDIR      = "LCOR-CXCL16/07_apm_positive_control/figures/03_pseudobulk_apm_myeloid"
GENE_X      = "LCOR"
APM_GENES   = ["TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "PSMB8", "PSMB9", "B2M"]
GENE_Y      = "APM_score"
MIN_CELLS   = 20
SUBTYPES    = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
MYELOID_GROUPS = ["Macro_M1", "Macro_M2", "Dendritic"]
os.makedirs(OUTDIR, exist_ok=True)

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
print(f" 03 -- {GENE_X} vs {GENE_Y} (positive control) -- PATIENT PSEUDOBULK, myeloid cells")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}   Patients: {adata.obs['sample'].nunique()}")

missing = [g for g in [GENE_X] + APM_GENES if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

celltype = assign_immune_type(adata.obs)
is_myeloid = np.isin(celltype, MYELOID_GROUPS)
print(f"  Myeloid cells ({'+'.join(MYELOID_GROUPS)}): {is_myeloid.sum():,} / {adata.n_obs:,}")

idx_x = adata.var_names.get_loc(GENE_X)
idx_apm = [adata.var_names.get_loc(g) for g in APM_GENES]
raw_x = adata.layers["counts"][:, idx_x]
raw_x = np.asarray(raw_x.todense()).ravel() if issparse(raw_x) else np.asarray(raw_x).ravel()
raw_apm_matrix = adata.layers["counts"][:, idx_apm]
raw_apm_matrix = np.asarray(raw_apm_matrix.todense()) if issparse(raw_apm_matrix) else np.asarray(raw_apm_matrix)
raw_apm = raw_apm_matrix.sum(axis=1)

pct_detected_per_gene = {g: 100 * (raw_apm_matrix[:, i] > 0).mean() for i, g in enumerate(APM_GENES)}
print(f"\n  Per-cell detection rate, {GENE_X}: {100*(raw_x>0).mean():.1f}%")
print(f"  Per-cell detection rate, {GENE_Y} (>=1 of 9 genes): {100*(raw_apm>0).mean():.1f}%")
for g, pct in pct_detected_per_gene.items():
    print(f"    {g}: {pct:.1f}%")

obs = pd.DataFrame({
    "sample":  adata.obs["sample"].astype(str).values,
    "subtype": adata.obs["subtype"].astype(str).values,
    "total_counts": adata.obs["total_counts"].values.astype(float),
    "celltype": celltype,
    f"{GENE_X}_raw": raw_x.astype(float),
    f"{GENE_Y}_raw": raw_apm.astype(float),
})
del adata
obs_myeloid = obs[is_myeloid].copy()

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
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["subtype"] = pd.Categorical(out["subtype"], categories=SUBTYPES, ordered=True)
        out = out.sort_values(["subtype", "sample"]).reset_index(drop=True)
    return out

df = pseudobulk_table(obs_myeloid, MIN_CELLS)
df.to_csv(f"{OUTDIR}/persample_pseudobulk_myeloid.csv", index=False)
print(f"  Patients retained (>={MIN_CELLS} myeloid cells): {len(df)} / {obs['sample'].nunique()}")

def corr_group(d, xcol, ycol):
    if len(d) < 4:
        return dict(n=len(d), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_rho=np.nan, spearman_p=np.nan)
    r, p = stats.pearsonr(d[xcol], d[ycol])
    rho, p_s = stats.spearmanr(d[xcol], d[ycol])
    return dict(n=len(d), pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=p_s)

xcol, ycol = f"{GENE_X}_log1p_CP10K", f"{GENE_Y}_log1p_CP10K"
print("\n" + "="*65)
print(f" Correlations: {GENE_X} vs {GENE_Y} pseudobulk, across patients (myeloid)")
print("="*65)
corr_rows = []
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

print("\n" + "="*65)
print(" Depth-confound check")
print("="*65)
df["log_median_depth"] = np.log(df["median_depth"])
depth_rows = []
for gene, col in [(GENE_X, xcol), (GENE_Y, ycol)]:
    rho, p = stats.spearmanr(df["log_median_depth"], df[col])
    depth_rows.append({"gene": gene, "spearman_rho_vs_depth": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient median depth: Spearman rho={rho:.3f}, p={p:.3g}")

def resid_vs_depth(y):
    slope, intercept, r, p, se = stats.linregress(df["log_median_depth"], y)
    return y - (intercept + slope * df["log_median_depth"])

df[f"{GENE_X}_resid"] = resid_vs_depth(df[xcol])
df[f"{GENE_Y}_resid"] = resid_vs_depth(df[ycol])
rho_adj, p_adj = stats.spearmanr(df[f"{GENE_X}_resid"], df[f"{GENE_Y}_resid"])
print(f"\n  DEPTH-ADJUSTED: Spearman(LCOR resid, APM resid) = {rho_adj:.3f}, p={p_adj:.3g}, n={len(df)}")
corr_rows.append({"group": "All_depth_adjusted_resid", "n": len(df),
                   "pearson_r": np.nan, "pearson_p": np.nan,
                   "spearman_rho": rho_adj, "spearman_p": p_adj})
df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(f"{OUTDIR}/correlation_table.csv", index=False)

print("\n" + "="*65)
print(" Cell-count confound check")
print("="*65)
ncell_rows = []
for gene, col in [(GENE_X, xcol), (GENE_Y, ycol)]:
    rho, p = stats.spearmanr(df["n_cells"], df[col])
    ncell_rows.append({"gene": gene, "spearman_rho_vs_ncells": rho, "p": p})
    print(f"  {gene} pseudobulk vs patient n_cells: Spearman rho={rho:.3f}, p={p:.3g}")
NCELL_THRESHOLDS = [MIN_CELLS, 50, 100, 200]
sensitivity_rows = []
for min_n in NCELL_THRESHOLDS:
    d = df[df["n_cells"] >= min_n]
    res = corr_group(d, xcol, ycol)
    sensitivity_rows.append({"min_n_cells": min_n, **res})
    print(f"    n_cells>={min_n}: n_patients={res['n']}, Spearman rho={res['spearman_rho']:.3f}, "
          f"p={res['spearman_p']:.3g}")
df_ncell_sensitivity = pd.DataFrame(sensitivity_rows)
df_ncell_sensitivity.to_csv(f"{OUTDIR}/ncell_sensitivity.csv", index=False)
print(f"\n  Saved: {OUTDIR}/correlation_table.csv, {OUTDIR}/ncell_sensitivity.csv")

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
fig.suptitle(f"Per-patient pseudobulk, myeloid cells (Macro_M1+Macro_M2+Dendritic) -- {GENE_X} vs {GENE_Y} (positive control)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_pseudobulk_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_pseudobulk_scatter.png")

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
fig.suptitle("Depth-confound check (positive control, myeloid pseudobulk)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_depth_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_depth_confound_check.png")

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
ax.set_title("Sensitivity to n_cells threshold", fontsize=9, fontweight="bold")
ax.grid(alpha=0.25)
fig.suptitle("Cell-count confound check (positive control, myeloid)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig_ncell_confound_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/fig_ncell_confound_check.png")

tnbc_row = df_corr[df_corr.group == "TNBC"].iloc[0]
readme = [
    f"03 -- {GENE_X} vs {GENE_Y} (POSITIVE CONTROL) -- per-patient pseudobulk, MYELOID cells",
    "="*70,
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "Myeloid-compartment positive control (see 01_pseudobulk_apm_malignant.py",
    "for full rationale) -- the question was raised specifically after",
    "the myeloid LCOR-CXCL16 null (05_myeloid), so this directly checks",
    "method sensitivity in that same compartment/cell-typing pipeline.",
    "",
    "APM GENE SET",
    "-"*70,
    "  " + ", ".join(APM_GENES),
    "",
    "RESULTS -- headline correlation (myeloid, combined)",
    "-"*70,
    f"  All patients (n={res_all['n']}): Pearson r={res_all['pearson_r']:.3f} "
    f"(p={res_all['pearson_p']:.3g}), Spearman rho={res_all['spearman_rho']:.3f} "
    f"(p={res_all['spearman_p']:.3g})",
    f"  TNBC only (n={tnbc_row['n']}): Spearman rho={tnbc_row['spearman_rho']:.3f} "
    f"(p={tnbc_row['spearman_p']:.3g})",
    "",
    "DEPTH-CONFOUND CHECK",
    "-"*70,
] + [
    f"  {r['gene']} pseudobulk vs patient median depth: Spearman rho={r['spearman_rho_vs_depth']:.3f}, p={r['p']:.3g}"
    for r in depth_rows
] + [
    f"  Depth-adjusted residual correlation: Spearman rho={rho_adj:.3f}, p={p_adj:.3g}",
    "",
    "OUTPUT FILES",
    "-"*70,
    "  persample_pseudobulk_myeloid.csv, correlation_table.csv, ncell_sensitivity.csv",
    "  fig_pseudobulk_scatter.png, fig_depth_confound_check.png, fig_ncell_confound_check.png",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Patients: {len(df)} (All) / TNBC n={tnbc_row['n']}
  All: Pearson r={res_all['pearson_r']:.3f} (p={res_all['pearson_p']:.3g})  Spearman rho={res_all['spearman_rho']:.3f} (p={res_all['spearman_p']:.3g})
  Depth-adjusted residual rho: {rho_adj:.3f} (p={p_adj:.3g})
  Output: {OUTDIR}/
============================================================
""")
