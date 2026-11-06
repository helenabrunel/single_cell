"""
02_visvader_analysis.py
=========================
Main test of the working hypothesis in Visvader: does a tumor's LCOR-program
(LCOR-OE signature, malignant cells) associate with its CD8 T cells having
the LCOR-induced phenotype (Tcell_UP24H / Tcell_UP96H signatures,
especially UP96H which includes CXCR6/ITGAE-CD103)?

APPROACH A -- per-patient signature-signature correlation
------------------------------------------------------------
Per patient: mean LCOR_OE_malignant score (malignant cells) vs mean
Tcell_UP24H / Tcell_UP96H score (CD8 T cells). Spearman correlation across
patients. Direct extension of the correlation approach already tried with
raw LCOR/CXCR6 (null) -- now using signatures instead of single genes.

APPROACH B -- unbiased: LCOR-high/low patients, compare CD8 T-cell phenotype
------------------------------------------------------------------------------
Patients split into top/bottom tercile of malignant LCOR_OE_malignant
score (same infiltration-style framework used earlier in this project).
Two complementary views:
  (i) Score-based: compare each patient's mean Tcell_UP24H/UP96H CD8 score
      between LCOR-high and LCOR-low groups (Mann-Whitney U).
  (ii) Unbiased: CD8 T cells re-clustered independently (PCA/neighbors/
      Leiden) with NO reference to LCOR status, then per-cluster
      proportion compared between LCOR-high and LCOR-low patients, and
      each cluster characterized post-hoc by its UP24H/UP96H scores (so
      cluster identity isn't presupposed).

Gene scoring: scanpy sc.tl.score_genes (control-gene-corrected mean
expression, the standard module-score method), on logcounts.

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/02_visvader_analysis/
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
OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/02_visvader_analysis"
os.makedirs(OUTDIR, exist_ok=True)

with open(SIG_FILE) as f:
    signatures = json.load(f)

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

readme = [
    "02 -- LCOR program vs CD8 T-cell phenotype, Visvader",
    "="*70,
    "",
    "SIGNATURES USED",
    "-"*70,
    f"  LCOR_OE_malignant: {len(signatures['LCOR_OE_malignant'])} genes",
    f"  Tcell_UP24H: {len(signatures['Tcell_UP24H'])} genes",
    f"  Tcell_UP96H: {len(signatures['Tcell_UP96H'])} genes",
]

# ============================================================================
# STEP 1 -- malignant cells: score LCOR_OE_malignant signature
# ============================================================================
print("Loading compartment_epithelial.h5ad...")
adata_mal = sc.read_h5ad("compartment_epithelial.h5ad")
adata_mal.X = adata_mal.layers["logcounts"]
genes_present = [g for g in signatures["LCOR_OE_malignant"] if g in adata_mal.var_names]
missing = set(signatures["LCOR_OE_malignant"]) - set(genes_present)
print(f"  LCOR_OE_malignant genes found: {len(genes_present)}/{len(signatures['LCOR_OE_malignant'])} (missing: {sorted(missing)})")
sc.tl.score_genes(adata_mal, genes_present, score_name="LCOR_OE_score", random_state=0)

mal_obs = pd.DataFrame({
    "sample": adata_mal.obs["sample"].astype(str).values,
    "subtype": adata_mal.obs["subtype"].astype(str).values,
    "LCOR_OE_score": adata_mal.obs["LCOR_OE_score"].values,
})
del adata_mal

patient_lcor = mal_obs.groupby("sample").agg(
    subtype=("subtype", "first"),
    n_malignant=("LCOR_OE_score", "size"),
    mean_LCOR_OE_score=("LCOR_OE_score", "mean"),
).reset_index()
print(f"  Patients: {len(patient_lcor)}")

readme += [
    "",
    "STEP 1 -- LCOR_OE_malignant score, per patient (malignant cells)",
    "-"*70,
    f"  Genes found: {len(genes_present)}/{len(signatures['LCOR_OE_malignant'])}, missing: {sorted(missing)}",
]

# ============================================================================
# STEP 2 -- CD8 T cells: score Tcell_UP24H / Tcell_UP96H
# ============================================================================
print("\nLoading compartment_immune.h5ad...")
adata_imm = sc.read_h5ad("compartment_immune.h5ad")
celltype = assign_immune_type(adata_imm.obs)
is_cd8 = celltype == "CD8_T"
print(f"  CD8 T cells: {is_cd8.sum():,} / {adata_imm.n_obs:,}")

adata_cd8 = adata_imm[is_cd8].copy()
adata_cd8.X = adata_cd8.layers["logcounts"]
del adata_imm

for sig_name in ["Tcell_UP24H", "Tcell_UP96H"]:
    genes_p = [g for g in signatures[sig_name] if g in adata_cd8.var_names]
    miss = set(signatures[sig_name]) - set(genes_p)
    print(f"  {sig_name} genes found: {len(genes_p)}/{len(signatures[sig_name])} (missing: {sorted(miss)[:10]}{'...' if len(miss)>10 else ''})")
    sc.tl.score_genes(adata_cd8, genes_p, score_name=f"{sig_name}_score", random_state=0)
    readme += [f"  {sig_name}: {len(genes_p)}/{len(signatures[sig_name])} genes found, missing count={len(miss)}"]

cd8_obs = pd.DataFrame({
    "sample": adata_cd8.obs["sample"].astype(str).values,
    "subtype": adata_cd8.obs["subtype"].astype(str).values,
    "Tcell_UP24H_score": adata_cd8.obs["Tcell_UP24H_score"].values,
    "Tcell_UP96H_score": adata_cd8.obs["Tcell_UP96H_score"].values,
}, index=adata_cd8.obs_names)

patient_cd8 = cd8_obs.groupby("sample").agg(
    n_cd8=("Tcell_UP24H_score", "size"),
    mean_UP24H=("Tcell_UP24H_score", "mean"),
    mean_UP96H=("Tcell_UP96H_score", "mean"),
).reset_index()

# ============================================================================
# APPROACH A -- per-patient correlation
# ============================================================================
print("\n" + "="*65)
print(" APPROACH A -- per-patient correlation: LCOR_OE score vs CD8 T-cell phenotype")
print("="*65)
MIN_CD8 = 20
merged = patient_lcor.merge(patient_cd8, on="sample", how="inner")
merged = merged[merged["n_cd8"] >= MIN_CD8]
print(f"  Patients with >={MIN_CD8} CD8 T cells: {len(merged)}")
merged.to_csv(f"{OUTDIR}/persample_scores.csv", index=False)

readme += [
    "",
    "APPROACH A -- per-patient correlation",
    "-"*70,
    f"  Patients with >={MIN_CD8} CD8 T cells: {len(merged)}",
]
corr_results = []
for tcol in ["mean_UP24H", "mean_UP96H"]:
    rho, p = stats.spearmanr(merged["mean_LCOR_OE_score"], merged[tcol])
    print(f"  LCOR_OE_malignant vs {tcol}: Spearman rho={rho:.3f}, p={p:.3g}, n={len(merged)}")
    corr_results.append({"target": tcol, "rho": rho, "p": p, "n": len(merged)})
    readme += [f"  LCOR_OE_malignant vs {tcol}: Spearman rho={rho:.3f}, p={p:.3g}, n={len(merged)}"]
pd.DataFrame(corr_results).to_csv(f"{OUTDIR}/approachA_correlations.csv", index=False)

subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, tcol, res in zip(axes, ["mean_UP24H", "mean_UP96H"], corr_results):
    for st, color in subtype_colors.items():
        d = merged[merged.subtype == st]
        ax.scatter(d["mean_LCOR_OE_score"], d[tcol], s=50, alpha=0.85, edgecolor="white", color=color, label=st)
    ax.set_xlabel("Malignant LCOR_OE_malignant score (mean/patient)", fontsize=9)
    ax.set_ylabel(f"CD8 {tcol} score (mean/patient)", fontsize=9)
    ax.set_title(f"rho={res['rho']:.2f}, p={res['p']:.3g}, n={res['n']}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.suptitle("Approach A: LCOR program (malignant) vs CD8 T-cell phenotype, per patient", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/approachA_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTDIR}/approachA_scatter.png")

# ============================================================================
# APPROACH B -- LCOR-high/low tercile split
# ============================================================================
print("\n" + "="*65)
print(" APPROACH B -- LCOR-high vs LCOR-low patients, CD8 T-cell phenotype")
print("="*65)
q1, q2 = merged["mean_LCOR_OE_score"].quantile([1/3, 2/3])
merged["LCOR_group"] = np.select(
    [merged["mean_LCOR_OE_score"] <= q1, merged["mean_LCOR_OE_score"] >= q2],
    ["Low", "High"], default="Mid"
)
print(merged[["sample", "mean_LCOR_OE_score", "LCOR_group"]].sort_values("mean_LCOR_OE_score").to_string(index=False))
merged.to_csv(f"{OUTDIR}/persample_scores.csv", index=False)

readme += [
    "",
    "APPROACH B(i) -- score comparison, LCOR-high vs LCOR-low (tercile split)",
    "-"*70,
    f"  Tercile cutoffs (LCOR_OE_malignant score): {q1:.4f} / {q2:.4f}",
]
bi_results = []
for tcol in ["mean_UP24H", "mean_UP96H"]:
    hi = merged.loc[merged.LCOR_group == "High", tcol]
    lo = merged.loc[merged.LCOR_group == "Low", tcol]
    u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    print(f"  {tcol}: High (n={len(hi)}) median={hi.median():.4f} vs Low (n={len(lo)}) median={lo.median():.4f}, Mann-Whitney p={p:.3g}")
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
fig.suptitle("Approach B(i): CD8 T-cell phenotype score, LCOR-high vs LCOR-low patients", fontsize=11, fontweight="bold")
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
adata_cd8.obs["LCOR_group"] = adata_cd8.obs["sample"].astype(str).map(lcor_group_map)
adata_cd8_sub = adata_cd8[adata_cd8.obs["LCOR_group"].isin(["High", "Low"])].copy()
print(f"  CD8 cells in High/Low patients: {adata_cd8_sub.n_obs:,}")

sc.pp.highly_variable_genes(adata_cd8_sub, n_top_genes=2000, flavor="seurat")
sc.pp.pca(adata_cd8_sub, n_comps=30, use_highly_variable=True, random_state=0)
sc.pp.neighbors(adata_cd8_sub, n_neighbors=15, random_state=0)
sc.tl.leiden(adata_cd8_sub, resolution=0.5, random_state=0, key_added="cd8_cluster")
sc.tl.umap(adata_cd8_sub, random_state=0)
print(f"  Clusters found: {adata_cd8_sub.obs['cd8_cluster'].nunique()}")

cluster_char = adata_cd8_sub.obs.groupby("cd8_cluster")[["Tcell_UP24H_score", "Tcell_UP96H_score"]].mean()
print("\n  Cluster characterization (mean signature scores):")
print(cluster_char.to_string())
cluster_char.to_csv(f"{OUTDIR}/cluster_characterization.csv")

prop = pd.crosstab(adata_cd8_sub.obs["sample"].astype(str), adata_cd8_sub.obs["cd8_cluster"], normalize="index")
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

readme += [
    "",
    "APPROACH B(ii) -- unbiased CD8 reclustering",
    "-"*70,
    f"  {adata_cd8_sub.n_obs:,} CD8 cells (High/Low patients only), {adata_cd8_sub.obs['cd8_cluster'].nunique()} Leiden clusters (res=0.5)",
    "  Cluster proportion High vs Low (Mann-Whitney across patients), and post-hoc signature characterization:",
] + [
    f"    Cluster {r['cluster']}: High median={r['high_median_prop']:.3f}, Low median={r['low_median_prop']:.3f}, "
    f"p={r['p']:.3g} (UP24H={r['up24h_score']:.3f}, UP96H={r['up96h_score']:.3f})"
    for _, r in df_cluster_test.iterrows()
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
umap = adata_cd8_sub.obsm["X_umap"]
sca = axes[0].scatter(umap[:, 0], umap[:, 1], c=adata_cd8_sub.obs["cd8_cluster"].astype(int), cmap="tab10", s=6, linewidths=0)
axes[0].set_title("CD8 clusters (unbiased)", fontsize=10, fontweight="bold")
for cl in adata_cd8_sub.obs["cd8_cluster"].unique():
    idx = adata_cd8_sub.obs["cd8_cluster"] == cl
    axes[0].annotate(cl, (umap[idx, 0].mean(), umap[idx, 1].mean()), fontsize=9, fontweight="bold")
for ax, sig in zip(axes[1:], ["Tcell_UP24H_score", "Tcell_UP96H_score"]):
    sca = ax.scatter(umap[:, 0], umap[:, 1], c=adata_cd8_sub.obs[sig].values, cmap="viridis", s=6, linewidths=0)
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title(sig, fontsize=10, fontweight="bold")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Approach B(ii): unbiased CD8 clusters, characterized post-hoc by LCOR-induced T-cell signatures",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/approachB_umap_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {OUTDIR}/approachB_umap_clusters.png")

with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print("\nDONE")
