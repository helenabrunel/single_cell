"""
09_celltypist_annotation_allT.py
=============================
Replaces the unsupervised Leiden reclustering in approach B(ii)
(02/03_..._analysis.py) with CellTypist reference annotation
(Immune_All_Low model), because the Leiden clusters turned out to be
dominated by patient identity (batch effect) rather than shared T-cell
states -- confirmed by most patients having ~100% of their CD8 cells in a
single cluster. CellTypist assigns each cell a NAMED, externally-defined
T-cell state (not derived from this data, so it cannot pick up a
patient-batch signal the way de novo clustering can), which is why
reference annotation was chosen.

Runs on the same T cells (CD4+CD8+Treg) identified in 02/03_..._analysis.py (Visvader:
compartment_immune.h5ad + assign_immune_type; Klughammer:
klughammer_Lymphoid.h5ad + CD8A/CD8B-vs-CD4 marker score), re-normalized to
log1p(CP10K) for CellTypist input. Per-cell direct model prediction
(majority_voting=False, since majority voting itself requires an internal
Leiden step -- the same issue we're trying to avoid).

Same composition-by-LCOR-score bar chart as 07_composition_plot.py, but
now labeled with real CellTypist T-cell states instead of anonymous
cluster numbers.

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/09_celltypist_annotation_allT/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import celltypist
from celltypist import models
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/09_celltypist_annotation_allT"
os.makedirs(OUTDIR, exist_ok=True)
models.download_models(force_update=False, model=["Immune_All_Low.pkl"])
MODEL = models.Model.load(model="Immune_All_Low.pkl")

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
    return np.array([cell_types[i] if s > 0 else "Unknown" for i, s in zip(best_idx, best_score)])

def get_symbol_matrix(adata, genes):
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


def annotate_dataset(name, adata_cd8, scores_path):
    print(f"\n=== {name} ===")
    scores = pd.read_csv(scores_path)

    # re-normalize to log1p(CP10K) from raw counts for CellTypist
    raw = adata_cd8.layers["counts"] if "counts" in adata_cd8.layers else adata_cd8.raw.X
    raw = raw.tocsr() if issparse(raw) else raw
    totals = np.asarray(raw.sum(axis=1)).ravel()
    adata_ct = adata_cd8.copy()
    if "counts" not in adata_cd8.layers:
        adata_ct = sc.AnnData(X=raw, obs=adata_cd8.obs.copy(),
                               var=pd.DataFrame(index=adata_cd8.raw.var["feature_name"].astype(str).values))
        adata_ct.var_names_make_unique()
    else:
        adata_ct.X = raw
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)

    print(f"  Running CellTypist on {adata_ct.n_obs:,} T cells (CD4+CD8+Treg)...")
    pred = celltypist.annotate(adata_ct, model=MODEL, majority_voting=False)
    labels = pred.predicted_labels["predicted_labels"].values
    print(f"  Predicted label counts:\n{pd.Series(labels).value_counts().to_string()}")

    # exhaustion score -- canonical markers, direct test of "less exhaustion"
    EXHAUSTION_GENES = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "ENTPD1"]
    exh_present = [g for g in EXHAUSTION_GENES if g in adata_ct.var_names]
    print(f"  Exhaustion genes found: {exh_present} ({len(exh_present)}/{len(EXHAUSTION_GENES)})")
    sc.tl.score_genes(adata_ct, exh_present, score_name="Exhaustion_score", random_state=0, use_raw=False)

    TRM_LABELS = {"Trm cytotoxic T cells", "Tem/Trm cytotoxic T cells"}
    is_trm = np.isin(labels, list(TRM_LABELS))

    sample_col = "sample" if "sample" in adata_cd8.obs.columns else "donor_id"
    df = pd.DataFrame({
        "sample": adata_cd8.obs[sample_col].astype(str).values,
        "label": labels,
        "is_trm": is_trm,
        "Exhaustion_score": adata_ct.obs["Exhaustion_score"].values,
    })
    df.to_csv(f"{OUTDIR}/{name}_percell_labels.csv", index=False)

    # per-patient TRM fraction + mean exhaustion score -- the direct hypothesis test
    patient_trm_exh = df.groupby("sample").agg(
        n_cells=("label", "size"), TRM_fraction=("is_trm", "mean"),
        mean_Exhaustion_score=("Exhaustion_score", "mean"),
    ).reset_index()
    patient_trm_exh.to_csv(f"{OUTDIR}/{name}_persample_TRM_exhaustion.csv", index=False)

    prop = pd.crosstab(df["sample"], df["label"], normalize="index")
    prop.to_csv(f"{OUTDIR}/{name}_persample_label_proportions.csv")

    # test High vs Low per label (LCOR_OE grouping, as before)
    lcor_group_map = dict(zip(scores["sample"], scores["LCOR_group"])) if "LCOR_group" in scores.columns else {}
    prop_hl = prop.copy()
    prop_hl["LCOR_group"] = prop_hl.index.map(lcor_group_map)
    test_results = []
    for label in prop.columns:
        hi = prop_hl.loc[prop_hl.LCOR_group == "High", label]
        lo = prop_hl.loc[prop_hl.LCOR_group == "Low", label]
        if len(hi) >= 2 and len(lo) >= 2:
            u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        else:
            p = np.nan
        test_results.append({"label": label, "high_median": hi.median(), "low_median": lo.median(),
                              "high_n": len(hi), "low_n": len(lo), "p": p})
    pd.DataFrame(test_results).sort_values("p").to_csv(f"{OUTDIR}/{name}_label_High_vs_Low.csv", index=False)

    # ------------------------------------------------------------------------
    # Load LCOR_raw + LCOR_OE per-patient data (from 06_scatterplots), merge
    # with TRM/exhaustion, and build a within-subtype tercile group for BOTH
    # metrics -- needed to make two composition plots (one ordered by each).
    # ------------------------------------------------------------------------
    master = pd.read_csv(f"LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/06_scatterplots/{name.lower()}_master_data.csv")
    merged = master.merge(patient_trm_exh, on="sample", how="inner")

    def within_subtype_tercile(g, col):
        q1, q2 = g[col].quantile([1 / 3, 2 / 3])
        return np.select([g[col] <= q1, g[col] >= q2], ["Low", "High"], default="Mid")

    merged["LCOR_OE_group"] = merged.groupby("subtype", group_keys=False).apply(
        lambda g: pd.Series(within_subtype_tercile(g, "mean_LCOR_OE_score"), index=g.index))
    merged["LCOR_raw_group"] = merged.groupby("subtype", group_keys=False).apply(
        lambda g: pd.Series(within_subtype_tercile(g, "LCOR_raw_pseudobulk"), index=g.index))
    merged.to_csv(f"{OUTDIR}/{name}_persample_merged.csv", index=False)

    # ------------------------------------------------------------------------
    # Composition plot, callable for either score column -- ordered by that
    # score, boundary lines from that score's own within-subtype tercile.
    # ------------------------------------------------------------------------
    TOP_N = 8

    def plot_composition(score_col, group_col, score_label, fname_suffix):
        score_map = dict(zip(merged["sample"], merged[score_col]))
        group_map = dict(zip(merged["sample"], merged[group_col]))
        scored_patients = [s for s in prop.index if s in score_map]
        order = pd.Series(score_map).loc[scored_patients].sort_values().index
        prop_ord = prop.loc[order]
        groups_ord = pd.Series(group_map).loc[order]
        n_low = (groups_ord == "Low").sum()
        n_mid = (groups_ord == "Mid").sum()

        overall_mean = prop_ord.mean(axis=0).sort_values(ascending=False)
        top_labels = list(overall_mean.head(TOP_N).index)
        other_cols = [c for c in prop_ord.columns if c not in top_labels]
        prop_plot = prop_ord[top_labels].copy()
        if other_cols:
            prop_plot["Other (<%.0f%% each, %d labels)" % (100 * overall_mean.iloc[TOP_N - 1], len(other_cols))] = prop_ord[other_cols].sum(axis=1)

        fig, (ax_score, ax) = plt.subplots(
            2, 1, figsize=(max(9, 0.5 * len(prop_plot)), 6.5), sharex=True,
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.06})

        # top panel: the actual score value as its own line plot -- separated
        # from the bars so the gradient is clearly visible, not crowded
        score_vals = pd.Series(score_map).loc[order].values
        ax_score.plot(range(len(prop_plot)), score_vals, color="black", marker="o", markersize=4, lw=1.5)
        ax_score.set_ylabel(score_label, fontsize=9)
        ax_score.tick_params(axis="y", labelsize=8)
        ax_score.grid(axis="y", alpha=0.2)

        # bottom panel: stacked composition bars
        n_colors = len(prop_plot.columns)
        colors = list(cm.get_cmap("tab10").colors[:min(n_colors, 9)])
        if n_colors > 9:
            colors.append((0.6, 0.6, 0.6, 1.0))
        bottom = np.zeros(len(prop_plot))
        for i, label in enumerate(prop_plot.columns):
            ax.bar(range(len(prop_plot)), prop_plot[label].values, bottom=bottom, color=colors[i], label=label, width=0.8)
            bottom += prop_plot[label].values

        for a in (ax_score, ax):
            if n_low > 0:
                a.axvline(n_low - 0.5, color="black", ls="--", lw=1.2)
                a.axvline(n_low + n_mid - 0.5, color="black", ls=":", lw=1)

        ax.set_xticks(range(len(prop_plot)))
        ax.set_xticklabels(prop_plot.index, rotation=90, fontsize=7)
        ax.set_ylabel("Proportion of T cells (CD4+CD8+Treg)", fontsize=10)
        ax.set_xlabel(f"Patient (ordered by {score_label}, low -> high)", fontsize=10)
        ax_score.set_title(f"{name}: T-cell state composition (CellTypist Immune_All_Low), by patient\n"
                            f"top {TOP_N} states + Other -- ordered by {score_label} "
                            f"(dashed=Low/Mid boundary, dotted=Mid/High boundary, within-subtype terciles)",
                            fontsize=10, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/{name}_composition_celltypist_{fname_suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUTDIR}/{name}_composition_celltypist_{fname_suffix}.png")

    plot_composition("mean_LCOR_OE_score", "LCOR_OE_group", "LCOR_OE signature score (malignant)", "by_LCOR_OE")
    plot_composition("LCOR_raw_pseudobulk", "LCOR_raw_group", "LCOR (raw, malignant)", "by_LCOR_raw")

    # ------------------------------------------------------------------------
    # Direct hypothesis test: LCOR (raw + OE) vs TRM_fraction and vs
    # Exhaustion_score, same Bassez-style scatter as 06_scatterplots.py
    # ------------------------------------------------------------------------

    subtype_colors = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}

    def bassez_plot(d, xcol, ycol, xlabel, ylabel, title, fname):
        dd = d.dropna(subset=[xcol, ycol])
        if len(dd) < 4:
            print(f"  SKIP {fname}: n={len(dd)}")
            return
        x, y = dd[xcol].values, dd[ycol].values
        r, p = stats.pearsonr(x, y)
        fig, ax = plt.subplots(figsize=(7, 5.3))
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        yhat = intercept + slope * xs
        n = len(x); dof = n - 2
        x_mean, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
        resid = y - (intercept + slope * x)
        s_err = np.sqrt(np.sum(resid ** 2) / dof) if dof > 0 else np.nan
        t_val = stats.t.ppf(0.975, dof) if dof > 0 else np.nan
        ci = t_val * s_err * np.sqrt(1 / n + (xs - x_mean) ** 2 / sxx) if dof > 0 else np.zeros_like(xs)
        ax.fill_between(xs, yhat - ci, yhat + ci, color="gray", alpha=0.3, linewidth=0)
        ax.plot(xs, yhat, color="black", lw=1.5)
        for st, color in subtype_colors.items():
            m = dd["subtype"].values == st
            if m.sum() == 0:
                continue
            ax.scatter(x[m], y[m], color=color, s=70, label=f"{st} (n={m.sum()})", edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(0.02, 0.97, f"R = {r:.2f}\np-value = {p:.2g}", transform=ax.transAxes, fontsize=12, va="top", ha="left")
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(title="Subtype", loc="best", fontsize=8, title_fontsize=9)
        ax.grid(alpha=0.15)
        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}.png  (R={r:.2f}, p={p:.3g}, n={n})")

    print(f"\n  --- Direct hypothesis test: LCOR vs TRM_fraction / Exhaustion_score ---")
    for xcol, xlabel in [("LCOR_raw_pseudobulk", "LCOR in malignant cells (log1p CP10K)"),
                          ("mean_LCOR_OE_score", "LCOR_OE signature score (malignant cells)")]:
        bassez_plot(merged, xcol, "TRM_fraction", xlabel, "Fraction of T cells (CD4+CD8+Treg) = TRM (CellTypist)",
                    f"{name}: {xlabel} vs TRM fraction", f"{name}_{xcol}_vs_TRM_fraction")
        bassez_plot(merged, xcol, "mean_Exhaustion_score", xlabel, "T-cell exhaustion score (PDCD1/LAG3/HAVCR2/TIGIT/CTLA4/TOX/ENTPD1)",
                    f"{name}: {xlabel} vs exhaustion score", f"{name}_{xcol}_vs_Exhaustion_score")


# ============================================================================
# Visvader
# ============================================================================
print("Loading compartment_immune.h5ad (Visvader)...")
adata_imm = sc.read_h5ad("compartment_immune.h5ad")
celltype = assign_immune_type(adata_imm.obs)
# all T cells: CD8_T + CD4_T + Treg (excludes NK/myeloid/B, which are not T cells)
adata_vis_t = adata_imm[np.isin(celltype, ["CD8_T", "CD4_T", "Treg"])].copy()
del adata_imm
annotate_dataset("Visvader", adata_vis_t,
                  "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/02_visvader_analysis/persample_scores.csv")
del adata_vis_t

# ============================================================================
# Klughammer
# ============================================================================
print("\nLoading klughammer_Lymphoid.h5ad...")
adata_lym = sc.read_h5ad("/home/hbrunel/single_cell/Klughammer/klughammer_Lymphoid.h5ad")
# all T cells: author_cell_type=="T" already covers CD4+CD8+Treg together
# (no further CD8A/B-vs-CD4 marker split -- that's what made 08 CD8-only)
is_t = adata_lym.obs["author_cell_type"].astype(str) == "T"
adata_klu_t = adata_lym[is_t].copy()
del adata_lym
annotate_dataset("Klughammer", adata_klu_t,
                  "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/03_klughammer_analysis/persample_scores.csv")

print("\nDONE")
