"""
03_klughammer_analysis.py
============================
Same test as 02_visvader_analysis.py, in Klughammer/HTAPP (independent
atlas, metastatic breast cancer). Much better powered for the T-cell side:
25,262 T cells / 27 donors here vs 5,213 CD8 cells / 16 usable patients in
Visvader.

CD8 IDENTIFICATION
-------------------
klughammer_Lymphoid.h5ad only labels cells broadly as "T"/"NK"/"B"/
"B_plasma" (author_cell_type) -- no CD8/CD4 split provided. Built here:
restrict to author_cell_type=="T", then CD8_T = CD8 marker score
(mean logcounts CD8A,CD8B) > CD4 logcounts (winner-takes-all, same spirit
as the malignant/immune marker calls used elsewhere in this project).

Malignant LCOR_OE_malignant score: klughammer_Malignant.h5ad (same file
family as the malignant-only object used in 07/08, here the compartment-
specific version with 'compartments'=='Malignant' cells only).

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/03_klughammer_analysis/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

SIG_FILE = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/01_build_signatures/signatures.json"
OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/03_klughammer_analysis"
MAL_FILE = "/home/hbrunel/single_cell/Klughammer/klughammer_Malignant.h5ad"
LYM_FILE = "/home/hbrunel/single_cell/Klughammer/klughammer_Lymphoid.h5ad"
os.makedirs(OUTDIR, exist_ok=True)

with open(SIG_FILE) as f:
    signatures = json.load(f)

readme = [
    "03 -- LCOR program vs CD8 T-cell phenotype, Klughammer (independent atlas)",
    "="*70,
    "",
    "SIGNATURES USED",
    "-"*70,
    f"  LCOR_OE_malignant: {len(signatures['LCOR_OE_malignant'])} genes",
    f"  Tcell_UP24H: {len(signatures['Tcell_UP24H'])} genes",
    f"  Tcell_UP96H: {len(signatures['Tcell_UP96H'])} genes",
]

def get_symbol_matrix(adata, genes):
    """Handles this dataset's mixed var_names (gene symbols direct where unique,
    Ensembl fallback otherwise) -- falls back to raw.var['feature_name'] lookup
    for any gene not found directly in var_names."""
    present_direct = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    resolved = {g: g for g in present_direct}
    if missing and adata.raw is not None and "feature_name" in adata.raw.var.columns:
        sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
        for g in missing:
            if g in sym2ens.index:
                ens = sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]
                if ens in adata.var_names:
                    resolved[g] = ens
    return resolved

# ============================================================================
# STEP 1 -- malignant cells: LCOR_OE_malignant score, per patient
# ============================================================================
print(f"Loading {MAL_FILE}...")
adata_mal = sc.read_h5ad(MAL_FILE)
print(f"  Malignant cells: {adata_mal.n_obs:,}, donors: {adata_mal.obs['donor_id'].nunique()}")
resolved = get_symbol_matrix(adata_mal, signatures["LCOR_OE_malignant"])
missing = set(signatures["LCOR_OE_malignant"]) - set(resolved.keys())
print(f"  LCOR_OE_malignant genes found: {len(resolved)}/{len(signatures['LCOR_OE_malignant'])} (missing: {sorted(missing)})")
sc.tl.score_genes(adata_mal, list(resolved.values()), score_name="LCOR_OE_score", random_state=0, use_raw=False)

mal_obs = pd.DataFrame({
    "sample": adata_mal.obs["donor_id"].astype(str).values,
    "subtype": adata_mal.obs["subtype"].astype(str).values,
    "LCOR_OE_score": adata_mal.obs["LCOR_OE_score"].values,
})
del adata_mal
patient_lcor = mal_obs.groupby("sample").agg(
    subtype=("subtype", "first"), n_malignant=("LCOR_OE_score", "size"),
    mean_LCOR_OE_score=("LCOR_OE_score", "mean"),
).reset_index()
print(f"  Patients (malignant): {len(patient_lcor)}")
readme += ["", "STEP 1 -- LCOR_OE_malignant score, per patient",
           f"  Genes found: {len(resolved)}/{len(signatures['LCOR_OE_malignant'])}"]

# ============================================================================
# STEP 2 -- CD8 T cells: identify + score Tcell_UP24H / UP96H
# ============================================================================
print(f"\nLoading {LYM_FILE}...")
adata_lym = sc.read_h5ad(LYM_FILE)
is_t = adata_lym.obs["author_cell_type"].astype(str) == "T"
adata_t = adata_lym[is_t].copy()
del adata_lym
print(f"  T cells: {adata_t.n_obs:,}")

cd8_genes = get_symbol_matrix(adata_t, ["CD8A", "CD8B", "CD4"])
def get_log(g):
    col = adata_t[:, cd8_genes[g]].X
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()

cd8_score = np.mean(np.column_stack([get_log("CD8A"), get_log("CD8B")]), axis=1)
cd4_score = get_log("CD4")
is_cd8 = cd8_score > cd4_score
print(f"  CD8 T cells (CD8A/B score > CD4): {is_cd8.sum():,} / {adata_t.n_obs:,}")

adata_cd8 = adata_t[is_cd8].copy()
del adata_t

for sig_name in ["Tcell_UP24H", "Tcell_UP96H"]:
    resolved_t = get_symbol_matrix(adata_cd8, signatures[sig_name])
    miss = set(signatures[sig_name]) - set(resolved_t.keys())
    print(f"  {sig_name} genes found: {len(resolved_t)}/{len(signatures[sig_name])} (missing count={len(miss)})")
    sc.tl.score_genes(adata_cd8, list(resolved_t.values()), score_name=f"{sig_name}_score", random_state=0, use_raw=False)
    readme += [f"  {sig_name}: {len(resolved_t)}/{len(signatures[sig_name])} genes found, missing count={len(miss)}"]

cd8_obs = pd.DataFrame({
    "sample": adata_cd8.obs["donor_id"].astype(str).values,
    "subtype": adata_cd8.obs["subtype"].astype(str).values,
    "Tcell_UP24H_score": adata_cd8.obs["Tcell_UP24H_score"].values,
    "Tcell_UP96H_score": adata_cd8.obs["Tcell_UP96H_score"].values,
})
patient_cd8 = cd8_obs.groupby("sample").agg(
    n_cd8=("Tcell_UP24H_score", "size"), mean_UP24H=("Tcell_UP24H_score", "mean"),
    mean_UP96H=("Tcell_UP96H_score", "mean"),
).reset_index()

# ============================================================================
# APPROACH A -- per-patient correlation
# ============================================================================
print("\n" + "="*65)
print(" APPROACH A -- per-patient correlation")
print("="*65)
MIN_CD8 = 20
merged = patient_lcor.merge(patient_cd8, on="sample", how="inner")
merged = merged[merged["n_cd8"] >= MIN_CD8]
print(f"  Patients with >={MIN_CD8} CD8 T cells: {len(merged)}")
merged.to_csv(f"{OUTDIR}/persample_scores.csv", index=False)
print(f"  Subtype composition: {merged['subtype'].value_counts().to_dict()}")

readme += ["", "APPROACH A -- per-patient correlation",
           f"  Patients with >={MIN_CD8} CD8 T cells: {len(merged)}",
           f"  Subtype composition: {merged['subtype'].value_counts().to_dict()}"]
corr_results = []
for tcol in ["mean_UP24H", "mean_UP96H"]:
    rho, p = stats.spearmanr(merged["mean_LCOR_OE_score"], merged[tcol])
    print(f"  LCOR_OE_malignant vs {tcol}: Spearman rho={rho:.3f}, p={p:.3g}, n={len(merged)}")
    corr_results.append({"target": tcol, "rho": rho, "p": p, "n": len(merged)})
    readme += [f"  LCOR_OE_malignant vs {tcol}: rho={rho:.3f}, p={p:.3g}, n={len(merged)}"]

# partial correlation adjusting for subtype (residual approach, since Visvader showed subtype confounding)
print("\n  Subtype-adjusted (residuals after removing subtype means):")
merged["subtype_cat"] = pd.Categorical(merged["subtype"])
def residualize(y, cat):
    means = y.groupby(cat).transform("mean")
    return y - means
merged["LCOR_resid"] = residualize(merged["mean_LCOR_OE_score"], merged["subtype_cat"])
adj_results = []
for tcol in ["mean_UP24H", "mean_UP96H"]:
    merged[f"{tcol}_resid"] = residualize(merged[tcol], merged["subtype_cat"])
    rho, p = stats.spearmanr(merged["LCOR_resid"], merged[f"{tcol}_resid"])
    print(f"  [subtype-adjusted] LCOR_OE vs {tcol}: rho={rho:.3f}, p={p:.3g}")
    adj_results.append({"target": tcol, "rho": rho, "p": p})
    readme += [f"  [subtype-adjusted] LCOR_OE vs {tcol}: rho={rho:.3f}, p={p:.3g}"]

pd.DataFrame(corr_results).to_csv(f"{OUTDIR}/approachA_correlations.csv", index=False)
pd.DataFrame(adj_results).to_csv(f"{OUTDIR}/approachA_correlations_subtype_adjusted.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
for ax, tcol, res in zip(axes, ["mean_UP24H", "mean_UP96H"], corr_results):
    for st, color in subtype_colors.items():
        d = merged[merged.subtype == st]
        ax.scatter(d["mean_LCOR_OE_score"], d[tcol], s=50, alpha=0.85, edgecolor="white", color=color, label=st)
    ax.set_xlabel("Malignant LCOR_OE_malignant score (mean/patient)", fontsize=9)
    ax.set_ylabel(f"CD8 {tcol} score (mean/patient)", fontsize=9)
    ax.set_title(f"rho={res['rho']:.2f}, p={res['p']:.3g}, n={res['n']}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.suptitle("Klughammer -- Approach A: LCOR program vs CD8 T-cell phenotype, per patient", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/approachA_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/approachA_scatter.png")

# ============================================================================
# APPROACH B -- LCOR-high/low, WITHIN-SUBTYPE tercile split (avoids the
# subtype confound found in Visvader)
# ============================================================================
print("\n" + "="*65)
print(" APPROACH B -- LCOR-high vs LCOR-low (tercile WITHIN subtype)")
print("="*65)
def within_subtype_tercile(g):
    q1, q2 = g["mean_LCOR_OE_score"].quantile([1/3, 2/3])
    return np.select([g["mean_LCOR_OE_score"] <= q1, g["mean_LCOR_OE_score"] >= q2], ["Low", "High"], default="Mid")

merged["LCOR_group"] = merged.groupby("subtype", group_keys=False).apply(
    lambda g: pd.Series(within_subtype_tercile(g), index=g.index)
)
print(merged[["sample", "subtype", "mean_LCOR_OE_score", "LCOR_group"]].sort_values(["subtype", "mean_LCOR_OE_score"]).to_string(index=False))
merged.to_csv(f"{OUTDIR}/persample_scores.csv", index=False)
print(f"\n  LCOR_group x subtype cross-tab:\n{pd.crosstab(merged['subtype'], merged['LCOR_group'])}")

readme += ["", "APPROACH B -- LCOR-high/low tercile split WITHIN subtype (avoids Visvader's subtype confound)"]
bi_results = []
for tcol in ["mean_UP24H", "mean_UP96H"]:
    hi = merged.loc[merged.LCOR_group == "High", tcol]
    lo = merged.loc[merged.LCOR_group == "Low", tcol]
    u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    print(f"  {tcol}: High (n={len(hi)}) median={hi.median():.4f} vs Low (n={len(lo)}) median={lo.median():.4f}, p={p:.3g}")
    bi_results.append({"target": tcol, "high_n": len(hi), "high_median": hi.median(),
                        "low_n": len(lo), "low_median": lo.median(), "p": p})
    readme += [f"  {tcol}: High median={hi.median():.4f} (n={len(hi)}) vs Low median={lo.median():.4f} (n={len(lo)}), p={p:.3g}"]
pd.DataFrame(bi_results).to_csv(f"{OUTDIR}/approachB_score_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, tcol in zip(axes, ["mean_UP24H", "mean_UP96H"]):
    d = merged[merged.LCOR_group.isin(["High", "Low"])]
    for i, grp in enumerate(["Low", "High"]):
        vals = d.loc[d.LCOR_group == grp, tcol]
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)), vals, s=45, alpha=0.8)
        ax.scatter([i], [vals.median()], color="black", marker="_", s=400, linewidths=2, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LCOR-low", "LCOR-high"])
    ax.set_ylabel(f"CD8 {tcol} (per patient)", fontsize=9)
    ax.set_title(tcol, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Klughammer -- Approach B: CD8 phenotype, LCOR-high vs LCOR-low (within-subtype split)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/approachB_score_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/approachB_score_comparison.png")

# ============================================================================
# APPROACH B(ii) -- unbiased CD8 reclustering
# ============================================================================
print("\n" + "="*65)
print(" APPROACH B(ii) -- unbiased CD8 T-cell reclustering")
print("="*65)
lcor_group_map = dict(zip(merged["sample"], merged["LCOR_group"]))
adata_cd8.obs["LCOR_group"] = adata_cd8.obs["donor_id"].astype(str).map(lcor_group_map)
adata_cd8_sub = adata_cd8[adata_cd8.obs["LCOR_group"].isin(["High", "Low"])].copy()
print(f"  CD8 cells in High/Low patients: {adata_cd8_sub.n_obs:,}")

sc.pp.highly_variable_genes(adata_cd8_sub, n_top_genes=2000, flavor="seurat")
sc.pp.pca(adata_cd8_sub, n_comps=30, use_highly_variable=True, random_state=0)
sc.pp.neighbors(adata_cd8_sub, n_neighbors=15, random_state=0)
sc.tl.leiden(adata_cd8_sub, resolution=0.5, random_state=0, key_added="cd8_cluster")
sc.tl.umap(adata_cd8_sub, random_state=0)
print(f"  Clusters found: {adata_cd8_sub.obs['cd8_cluster'].nunique()}")

cluster_char = adata_cd8_sub.obs.groupby("cd8_cluster")[["Tcell_UP24H_score", "Tcell_UP96H_score"]].mean()
print("\n  Cluster characterization:")
print(cluster_char.to_string())
cluster_char.to_csv(f"{OUTDIR}/cluster_characterization.csv")

prop = pd.crosstab(adata_cd8_sub.obs["donor_id"].astype(str), adata_cd8_sub.obs["cd8_cluster"], normalize="index")
prop.to_csv(f"{OUTDIR}/persample_cluster_proportions.csv")
prop["LCOR_group"] = prop.index.map(lcor_group_map)
prop_long = prop.melt(id_vars="LCOR_group", var_name="cluster", value_name="proportion")
cluster_test = []
for cl in sorted(adata_cd8_sub.obs["cd8_cluster"].unique(), key=int):
    hi = prop_long.loc[(prop_long.cluster == cl) & (prop_long.LCOR_group == "High"), "proportion"]
    lo = prop_long.loc[(prop_long.cluster == cl) & (prop_long.LCOR_group == "Low"), "proportion"]
    if len(hi) >= 2 and len(lo) >= 2:
        u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    else:
        p = np.nan
    cluster_test.append({"cluster": cl, "high_median_prop": hi.median(), "low_median_prop": lo.median(),
                          "up24h_score": cluster_char.loc[cl, "Tcell_UP24H_score"],
                          "up96h_score": cluster_char.loc[cl, "Tcell_UP96H_score"], "p": p})
    print(f"  Cluster {cl}: High median prop={hi.median():.3f}, Low median prop={lo.median():.3f}, p={p:.3g} "
          f"(UP24H={cluster_char.loc[cl,'Tcell_UP24H_score']:.3f}, UP96H={cluster_char.loc[cl,'Tcell_UP96H_score']:.3f})")
df_cluster_test = pd.DataFrame(cluster_test)
df_cluster_test.to_csv(f"{OUTDIR}/approachB_cluster_proportions.csv", index=False)

readme += ["", "APPROACH B(ii) -- unbiased CD8 reclustering",
           f"  {adata_cd8_sub.n_obs:,} CD8 cells (High/Low), {adata_cd8_sub.obs['cd8_cluster'].nunique()} clusters"] + [
    f"    Cluster {r['cluster']}: High={r['high_median_prop']:.3f}, Low={r['low_median_prop']:.3f}, p={r['p']:.3g} "
    f"(UP24H={r['up24h_score']:.3f}, UP96H={r['up96h_score']:.3f})" for _, r in df_cluster_test.iterrows()
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
umap = adata_cd8_sub.obsm["X_umap"]
axes[0].scatter(umap[:, 0], umap[:, 1], c=adata_cd8_sub.obs["cd8_cluster"].astype(int), cmap="tab10", s=4, linewidths=0)
axes[0].set_title("CD8 clusters (unbiased)", fontsize=10, fontweight="bold")
for cl in adata_cd8_sub.obs["cd8_cluster"].unique():
    idx = adata_cd8_sub.obs["cd8_cluster"] == cl
    axes[0].annotate(cl, (umap[idx, 0].mean(), umap[idx, 1].mean()), fontsize=9, fontweight="bold")
for ax, sig in zip(axes[1:], ["Tcell_UP24H_score", "Tcell_UP96H_score"]):
    sca = ax.scatter(umap[:, 0], umap[:, 1], c=adata_cd8_sub.obs[sig].values, cmap="viridis", s=4, linewidths=0)
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title(sig, fontsize=10, fontweight="bold")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Klughammer -- Approach B(ii): unbiased CD8 clusters, characterized post-hoc", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/approachB_umap_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {OUTDIR}/approachB_umap_clusters.png")

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")
print("\nDONE")
