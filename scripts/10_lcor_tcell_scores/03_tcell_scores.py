"""
01_tcell_scores.py
====================
Per-patient T-cell-compartment scores, both datasets, non-integrated data
only (raw counts -> own log1p(CP10K); CellTypist used only for a NAMED,
externally-defined lineage/state call, not for expression values).

PIPELINE
--------
1. T-cell gate (coarse, "is this a T cell"): Visvader = CD8_T+CD4_T+Treg via
   assign_immune_type (marker-score argmax, already in the object).
   Klughammer = author_cell_type=="T" (already in the object). Neither gate
   is CD8-vs-CD4 -- just "T cell vs other immune cell".
2. CellTypist (Immune_All_Low, majority_voting=False) on that T-cell gate ->
   fine-grained state label per cell, from which CD8 / CD4 / Treg lineage
   is derived (label map below) -- CellTypist's own semantics are the
   lineage source of truth here, not a custom marker threshold.
3. Within CD8 cells (biased + unbiased readouts):
     - CXCR6 (raw pseudobulk) -- the direct hypothesis readout
     - ITGAE/CD103 (raw pseudobulk, pooled across all T cells: AllT) --
       second biased readout, tissue-residency marker from the working hypothesis
     - Tcell_UP24H / Tcell_UP96H signature scores (lab-derived; UP96H
       includes CXCR6/ITGAE) -- biased readouts
     - TRM_fraction (CellTypist Trm/Tem-Trm cytotoxic labels) -- unbiased
     - Exhaustion_score (PDCD1/LAG3/HAVCR2/TIGIT/CTLA4/TOX/ENTPD1) -- unbiased
4. Within CD4 cells: Exhaustion_score only (comparison axis -- TRM and the
   lab's CD8-centric signatures are not meaningful for CD4).
5. Independent of CellTypist: each T cell is also scored against 6 state
   gene sets from Zheng et al., Science 2021 (abe6474, pan-cancer T-cell
   atlas) -- Naive, Effector_cytotoxic, Effector_proliferative, Stem_Tpex,
   TRM, Exhausted -- and assigned to its argmax state (STATE_GENES_ABE6474
   below). A second, externally-anchored categorical state call alongside
   CellTypist, for the unbiased composition/multiscatter analysis.

LINEAGE MAP (CellTypist Immune_All_Low labels actually observed)
---------------------------------------------------------------------------
CD8:  Tem/Trm cytotoxic T cells, Trm cytotoxic T cells,
      Tem/Temra cytotoxic T cells, Tcm/Naive cytotoxic T cells,
      CD8a/a, CD8a/b(entry)
CD4:  Tcm/Naive helper T cells, Tem/Effector helper T cells,
      Follicular helper T cells, Type 1 helper T cells,
      Type 17 helper T cells, Memory CD4+ cytotoxic T cells
Treg: Regulatory T cells, Treg(diff)
(everything else -- gamma-delta, MAIT, NKT, cycling, thymocyte, non-T
 contamination -- excluded from the CD8/CD4 comparison)
TRM (within CD8): Trm cytotoxic T cells, Tem/Trm cytotoxic T cells

Output: 03_tcell_scores/figures/{Visvader,Klughammer}_persample_tcell_scores.csv
        03_tcell_scores/figures/{Visvader,Klughammer}_percell_tcell_labels.csv
"""
import scanpy as sc
import numpy as np
import pandas as pd
import json, os, warnings
import celltypist
from celltypist import models
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "03_tcell_scores/figures"
os.makedirs(OUTDIR, exist_ok=True)
models.download_models(force_update=False, model=["Immune_All_Low.pkl"])
MODEL = models.Model.load(model="Immune_All_Low.pkl")

with open("01_signatures/figures/signatures.json") as f:
    SIGNATURES = json.load(f)

CD8_LABELS = {"Tem/Trm cytotoxic T cells", "Trm cytotoxic T cells", "Tem/Temra cytotoxic T cells",
              "Tcm/Naive cytotoxic T cells", "CD8a/a", "CD8a/b(entry)"}
CD4_LABELS = {"Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Follicular helper T cells",
              "Type 1 helper T cells", "Type 17 helper T cells", "Memory CD4+ cytotoxic T cells"}
TREG_LABELS = {"Regulatory T cells", "Treg(diff)"}
TRM_LABELS = {"Trm cytotoxic T cells", "Tem/Trm cytotoxic T cells"}
EXHAUSTION_GENES = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "ENTPD1"]

# T-cell state gene sets from Zheng et al., Science 2021 (10.1126/science.abe6474,
# pan-cancer scRNA-seq T-cell atlas). Genes explicitly named in the main text for
# the corresponding metaclusters (cytotoxic effector molecules; TCF7+/stem-like
# Tex; ZNF683+CXCR6+ Trm; terminal Tex) are used as-is; genes not detailed in the
# main text (naive, proliferative -- the paper uses a computational "proliferation
# index" rather than naming genes) are filled in with standard literature markers.
STATE_GENES_ABE6474 = {
    "Naive": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R"],
    "Effector_cytotoxic": ["GZMB", "IFNG", "PRF1", "GZMA", "NKG7"],
    "Effector_proliferative": ["MKI67", "TOP2A", "STMN1"],
    "Stem_Tpex": ["TCF7", "CD200", "GNG4", "IGFBP4", "IGFL2", "CXCR5", "SLAMF6"],
    "TRM": ["ZNF683", "CXCR6", "ITGAE"],
    "Exhausted": ["TOX", "TIGIT", "CTLA4", "TNFRSF9", "ENTPD1", "HAVCR2", "LAG3", "PDCD1"],
}


def lineage_of(label):
    if label in CD8_LABELS:
        return "CD8"
    if label in CD4_LABELS:
        return "CD4"
    if label in TREG_LABELS:
        return "Treg"
    return "Other"


def get_symbol_ensembl_map(adata, genes):
    sym2ens = pd.Series(adata.raw.var.index, index=adata.raw.var["feature_name"].astype(str))
    resolved = {}
    for g in genes:
        if g in sym2ens.index:
            ens = sym2ens[g] if np.isscalar(sym2ens[g]) else sym2ens[g].iloc[0]
            if ens in adata.raw.var_names:
                resolved[g] = ens
    return resolved


def get_logcounts(adata_ct, gene):
    if gene not in adata_ct.var_names:
        return np.full(adata_ct.n_obs, np.nan)
    col = adata_ct[:, gene].X
    return np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()


def run_celltypist_and_score(adata_t, sample_col, build_log1p_cp10k_fullgenome):
    """adata_t: T-cell-gated AnnData (raw counts retrievable). Returns a
    per-cell dataframe with sample, CellTypist-based lineage/TRM AND a
    SECOND, independent marker-gene-based lineage/TRM call (CD8A/CD8B vs
    CD4 for lineage; ITGAE/CD103 positivity for TRM -- NOT CXCR6, to avoid
    circularity with the CXCR6 hypothesis test itself), plus UP24H/UP96H/
    Exhaustion scores and CXCR6 raw/total (for pseudobulk)."""
    adata_ct, cxcr6_raw, itgae_raw, total_counts = build_log1p_cp10k_fullgenome(adata_t)

    print(f"  Running CellTypist on {adata_ct.n_obs:,} T cells...")
    pred = celltypist.annotate(adata_ct, model=MODEL, majority_voting=False)
    labels = pred.predicted_labels["predicted_labels"].values
    lineage_ct = np.array([lineage_of(l) for l in labels])
    print(f"  Lineage counts (CellTypist): {pd.Series(lineage_ct).value_counts().to_dict()}")

    # SECOND, independent marker-based lineage + TRM call
    cd8a = get_logcounts(adata_ct, "CD8A")
    cd8b = get_logcounts(adata_ct, "CD8B")
    cd4 = get_logcounts(adata_ct, "CD4")
    itgae = get_logcounts(adata_ct, "ITGAE")
    cd8_marker_score = np.nanmean(np.column_stack([cd8a, cd8b]), axis=1)
    lineage_marker = np.where(cd8_marker_score > cd4, "CD8", np.where(cd4 > 0, "CD4", "Other"))
    is_trm_marker = itgae > 0
    print(f"  Lineage counts (marker CD8A/CD8B-vs-CD4): {pd.Series(lineage_marker).value_counts().to_dict()}")
    print(f"  Agreement CellTypist vs marker lineage (CD8/CD4 cells only): "
          f"{np.mean(lineage_ct[np.isin(lineage_ct,['CD8','CD4'])] == lineage_marker[np.isin(lineage_ct,['CD8','CD4'])]):.2%}")

    genes_up24h = [g for g in SIGNATURES["Tcell_UP24H"] if g in adata_ct.var_names]
    genes_up96h = [g for g in SIGNATURES["Tcell_UP96H"] if g in adata_ct.var_names]
    genes_exh = [g for g in EXHAUSTION_GENES if g in adata_ct.var_names]
    print(f"  UP24H genes: {len(genes_up24h)}/{len(SIGNATURES['Tcell_UP24H'])}, "
          f"UP96H genes: {len(genes_up96h)}/{len(SIGNATURES['Tcell_UP96H'])}, "
          f"Exhaustion genes: {len(genes_exh)}/{len(EXHAUSTION_GENES)}")
    sc.tl.score_genes(adata_ct, genes_up24h, score_name="UP24H_score", random_state=0, use_raw=False)
    sc.tl.score_genes(adata_ct, genes_up96h, score_name="UP96H_score", random_state=0, use_raw=False)
    sc.tl.score_genes(adata_ct, genes_exh, score_name="Exhaustion_score", random_state=0, use_raw=False)

    # Zheng et al. (abe6474) state scores -> each T cell assigned to its
    # highest-scoring state (same argmax logic as Visvader's own coarse
    # immune-type call), independent of CellTypist.
    abe_score_cols = []
    for state, genes in STATE_GENES_ABE6474.items():
        present = [g for g in genes if g in adata_ct.var_names]
        score_name = f"ABE_{state}_score"
        sc.tl.score_genes(adata_ct, present, score_name=score_name, random_state=0, use_raw=False)
        abe_score_cols.append(score_name)
        print(f"  [abe6474] {state}: {len(present)}/{len(genes)} genes")
    abe_score_mat = adata_ct.obs[abe_score_cols].values
    state_abe6474 = np.array(list(STATE_GENES_ABE6474.keys()))[abe_score_mat.argmax(axis=1)]
    print(f"  State counts (abe6474 argmax): {pd.Series(state_abe6474).value_counts().to_dict()}")

    df = pd.DataFrame({
        "sample": adata_t.obs[sample_col].astype(str).values,
        "label": labels,
        "lineage_celltypist": lineage_ct,
        "is_trm_celltypist": np.isin(labels, list(TRM_LABELS)),
        "lineage_marker": lineage_marker,
        "is_trm_marker": is_trm_marker,
        "UP24H_score": adata_ct.obs["UP24H_score"].values,
        "UP96H_score": adata_ct.obs["UP96H_score"].values,
        "Exhaustion_score": adata_ct.obs["Exhaustion_score"].values,
        "CXCR6_raw": cxcr6_raw,
        "ITGAE_raw": itgae_raw,
        "state_abe6474": state_abe6474,
        "total_counts": total_counts,
    })
    for col in abe_score_cols:
        df[col] = adata_ct.obs[col].values
    return df


def aggregate_persample(df):
    """Builds CD8/CD4 per-patient aggregates TWICE -- once using the
    CellTypist-based lineage call, once using the independent marker-based
    call -- so results can be compared side by side rather than committing
    to a single method."""
    out = None
    for method, lineage_col, trm_col in [("celltypist", "lineage_celltypist", "is_trm_celltypist"),
                                          ("marker", "lineage_marker", "is_trm_marker")]:
        cd8 = df[df[lineage_col] == "CD8"]
        cd4 = df[df[lineage_col] == "CD4"]

        cd8_agg = cd8.groupby("sample").apply(lambda g: pd.Series({
            f"n_CD8_{method}": len(g),
            f"CXCR6_pseudobulk_CD8_{method}": np.log1p(1e4 * g["CXCR6_raw"].sum() / g["total_counts"].sum()),
            f"UP24H_score_CD8_{method}": g["UP24H_score"].mean(),
            f"UP96H_score_CD8_{method}": g["UP96H_score"].mean(),
            f"TRM_fraction_CD8_{method}": g[trm_col].mean(),
            f"Exhaustion_score_CD8_{method}": g["Exhaustion_score"].mean(),
        })).reset_index()

        cd4_agg = cd4.groupby("sample").apply(lambda g: pd.Series({
            f"n_CD4_{method}": len(g),
            f"Exhaustion_score_CD4_{method}": g["Exhaustion_score"].mean(),
        })).reset_index()

        merged = cd8_agg.merge(cd4_agg, on="sample", how="outer")
        out = merged if out is None else out.merge(merged, on="sample", how="outer")

    # "T cells in general" -- the whole T-cell gate (CD8+CD4+Treg+Other),
    # i.e. no CellTypist/marker lineage split at all. For the biased
    # (hypothesis-driven) readouts only -- TRM is not meaningful pooled
    # across lineages, so it is intentionally omitted here.
    allt_agg = df.groupby("sample").apply(lambda g: pd.Series({
        "n_AllT": len(g),
        "CXCR6_pseudobulk_AllT": np.log1p(1e4 * g["CXCR6_raw"].sum() / g["total_counts"].sum()),
        "ITGAE_pseudobulk_AllT": np.log1p(1e4 * g["ITGAE_raw"].sum() / g["total_counts"].sum()),
        "UP24H_score_AllT": g["UP24H_score"].mean(),
        "UP96H_score_AllT": g["UP96H_score"].mean(),
        "Exhaustion_score_AllT": g["Exhaustion_score"].mean(),
    })).reset_index()
    out = out.merge(allt_agg, on="sample", how="outer")
    return out


# ============================================================================
# Visvader
# ============================================================================
print("\n=== Visvader (compartment_immune.h5ad) ===")
IMMUNE_CELL_TYPES = {
    "CD8_T": "score_CD8_T", "CD4_T": "score_CD4_T", "Treg": "score_Treg", "NK": "score_NK",
    "Macro_M1": "score_Macro_M1", "Macro_M2": "score_Macro_M2", "B_cell": "score_B_cell",
    "Plasmablast": "score_Plasmablast", "Dendritic": "score_Dendritic", "Mast": "score_Mast",
}
def assign_immune_type(obs):
    cols = {ct: col for ct, col in IMMUNE_CELL_TYPES.items() if col in obs.columns}
    mat = obs[[c for c in cols.values()]].values
    names = list(cols.keys())
    best_idx = mat.argmax(axis=1); best_score = mat.max(axis=1)
    return np.array([names[i] if s > 0 else "Unknown" for i, s in zip(best_idx, best_score)])

adata_imm = sc.read_h5ad("/home/hbrunel/single_cell/Visvader/compartment_immune.h5ad")
coarse_celltype = assign_immune_type(adata_imm.obs)
adata_vis_t = adata_imm[np.isin(coarse_celltype, ["CD8_T", "CD4_T", "Treg"])].copy()
del adata_imm
print(f"  T-cell gate (CD8_T+CD4_T+Treg): {adata_vis_t.n_obs:,} cells, {adata_vis_t.obs['sample'].nunique()} patients")

def build_visvader(adata_t):
    idx_cxcr6 = adata_t.var_names.get_loc("CXCR6")
    idx_itgae = adata_t.var_names.get_loc("ITGAE")
    raw_all = adata_t.layers["counts"]
    cxcr6_raw = np.asarray(raw_all[:, idx_cxcr6].todense()).ravel() if issparse(raw_all) else np.asarray(raw_all[:, idx_cxcr6]).ravel()
    itgae_raw = np.asarray(raw_all[:, idx_itgae].todense()).ravel() if issparse(raw_all) else np.asarray(raw_all[:, idx_itgae]).ravel()
    total_counts = adata_t.obs["total_counts"].values.astype(float)
    adata_ct = adata_t.copy()
    adata_ct.X = raw_all
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    return adata_ct, cxcr6_raw, itgae_raw, total_counts

df_vis = run_celltypist_and_score(adata_vis_t, "sample", build_visvader)
df_vis.drop(columns=["total_counts"]).to_csv(f"{OUTDIR}/Visvader_percell_tcell_labels.csv", index=False)
vis_agg = aggregate_persample(df_vis)
vis_agg.to_csv(f"{OUTDIR}/Visvader_persample_tcell_scores.csv", index=False)
print(f"  Saved: {OUTDIR}/Visvader_persample_tcell_scores.csv ({len(vis_agg)} patients)")
del adata_vis_t

# ============================================================================
# Klughammer
# ============================================================================
print("\n=== Klughammer (klughammer_Lymphoid.h5ad) ===")
adata_lym = sc.read_h5ad("/home/hbrunel/single_cell/Klughammer/klughammer_Lymphoid.h5ad")
adata_klu_t = adata_lym[adata_lym.obs["author_cell_type"].astype(str) == "T"].copy()
del adata_lym
print(f"  T-cell gate (author_cell_type=='T'): {adata_klu_t.n_obs:,} cells, {adata_klu_t.obs['donor_id'].nunique()} patients")

def build_klughammer(adata_t):
    needed = ["CXCR6", "ITGAE"]
    resolved = get_symbol_ensembl_map(adata_t, needed)
    col = adata_t.raw[:, resolved["CXCR6"]].X
    cxcr6_raw = np.asarray(col.todense()).ravel() if issparse(col) else np.asarray(col).ravel()
    col_itgae = adata_t.raw[:, resolved["ITGAE"]].X
    itgae_raw = np.asarray(col_itgae.todense()).ravel() if issparse(col_itgae) else np.asarray(col_itgae).ravel()
    total_counts = adata_t.obs["total_counts"].values.astype(float)
    raw_full = adata_t.raw.X
    raw_full = raw_full.tocsr() if issparse(raw_full) else raw_full
    feature_names = adata_t.raw.var["feature_name"].astype(str).values
    adata_ct = sc.AnnData(X=raw_full, obs=adata_t.obs.copy(), var=pd.DataFrame(index=feature_names))
    adata_ct.var_names_make_unique()
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
    return adata_ct, cxcr6_raw, itgae_raw, total_counts

df_klu = run_celltypist_and_score(adata_klu_t, "donor_id", build_klughammer)
df_klu.drop(columns=["total_counts"]).to_csv(f"{OUTDIR}/Klughammer_percell_tcell_labels.csv", index=False)
klu_agg = aggregate_persample(df_klu)
klu_agg.to_csv(f"{OUTDIR}/Klughammer_persample_tcell_scores.csv", index=False)
print(f"  Saved: {OUTDIR}/Klughammer_persample_tcell_scores.csv ({len(klu_agg)} patients)")

print("\nDONE")
