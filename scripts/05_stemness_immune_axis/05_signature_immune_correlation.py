"""
part1b_immune_correlation.py
=============================
Extension of Part 1: adds the immune consensus score as a target.

For each stemness signature × subtype × dataset, computes:
  - Per-patient Spearman r meta-analysis (Fisher z) vs immune score
  - Distance Correlation (dCor) vs immune score
  - Partial Spearman r vs immune controlling for CC
  - LMM: signature ~ immune_score + (1|patient)
  - Combined rank: r_cyto_partial − r_immune_pooled − R²_CC
    (ideal signature: high stemness, negative immune, low CC)

Joins with existing part1 ranking CSVs and produces:
  - Updated ranking CSV with immune metrics
  - Forest plot: per-patient r vs immune score (want negative)
  - 2D scatter: r_cyto_partial (x) vs r_immune_pooled (y) per subtype
    — ideal signatures sit in the top-left quadrant

Run from: ~/single_cell/
  python3 part1b_immune_correlation.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json, os, gc, warnings
from scipy.stats import spearmanr
from scipy.sparse import issparse
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
import dcor
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Immune consensus (APM + IFN α/γ selection) ───────────────────────────────
IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
]

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASETS = {
    "Visvader": {
        "h5ad"        : "Visvader/compartment_epithelial.h5ad",
        "sigfile"     : "Visvader/signatures_confirmed.json",
        "part1_csv"   : "Visvader/figures/part1/ranking_Visvader.csv",
        "patient_col" : "sample",
        "subtype_col" : "subtype",
        "cc_in_obs"   : False,
        "outdir"      : "Visvader/figures/part1",
    },
    "Klughammer": {
        "h5ad"        : "Klughammer/klughammer_Malignant.h5ad",
        "sigfile"     : "Klughammer/signatures_confirmed.json",
        "part1_csv"   : "Klughammer/figures/part1/ranking_Klughammer.csv",
        "patient_col" : "donor_id",
        "subtype_col" : "subtype",
        "cc_in_obs"   : True,
        "s_col"       : "S.Score",
        "g2m_col"     : "G2M.Score",
        "outdir"      : "Klughammer/figures/part1",
    },
}

# Cell cycle genes (Tirosh 2016) — needed for Visvader
S_GENES = [
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1",
    "GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51",
    "RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1",
    "CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
]
G2M_GENES = [
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","SMC4","CCNB2","CKAP2L",
    "CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP",
    "CDCA3","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5",
    "CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR",
    "CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA",
]

SUBTYPES     = ["ER", "HER2", "TNBC"]
MIN_CELLS    = 500
MIN_PATIENTS = 3
SUBSAMPLE_N  = 800
N_BINS_MI    = 15

SIG_LABELS = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BENPORATH_ES1"                    : "BENPORATH",
    "EMP"                              : "EMP",
    "fMaSC_over_aMaSC"                 : "fMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"         : "LIM_MaSC",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
    "LIM_MAMMARY_LUMINAL_MATURE_UP"    : "LIM_LumMat",
    "stem_consensus"                   : "stem_consensus",
}
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C",
                  "Global": "#888888"}
SIG_MARKERS    = ["o","s","D","^","v","<",">","P","*"]

# ── Helpers (same as part1) ───────────────────────────────────────────────────
def to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)

def compute_mi(x, y, n_bins=N_BINS_MI):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 30:
        return np.nan
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    x_d = enc.fit_transform(x.reshape(-1,1)).ravel().astype(int)
    y_d = enc.fit_transform(y.reshape(-1,1)).ravel().astype(int)
    return mutual_info_score(x_d, y_d)

def compute_dcor(x, y, n_sub=SUBSAMPLE_N, rng=42):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 30:
        return np.nan
    if len(x) > n_sub:
        idx = np.random.default_rng(rng).choice(len(x), n_sub, replace=False)
        x, y = x[idx], y[idx]
    return float(dcor.u_distance_correlation_sqr(x, y))

def partial_spearman(x, y, covariates):
    valid = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    x, y, cov = x[valid], y[valid], covariates[valid]
    if len(x) < 30:
        return np.nan, np.nan
    lr = LinearRegression()
    x_res = x - lr.fit(cov, x).predict(cov)
    y_res = y - lr.fit(cov, y).predict(cov)
    r, p = spearmanr(x_res, y_res)
    return float(r), float(p)

def per_patient_meta(patient_stats):
    valid = [(r, n) for r, n in patient_stats
             if r is not None and np.isfinite(r) and n > 3]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan
    zs = np.array([np.arctanh(np.clip(r, -0.9999, 0.9999)) for r, _ in valid])
    ws = np.array([max(np.sqrt(n - 3), 0) for _, n in valid])
    if ws.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    z_mean   = np.average(zs, weights=ws)
    se       = 1.0 / np.sqrt(ws.sum())
    pooled_r = float(np.tanh(z_mean))
    ci_lo    = float(np.tanh(z_mean - 1.96 * se))
    ci_hi    = float(np.tanh(z_mean + 1.96 * se))
    pct_neg  = float(np.mean([r < 0 for r, _ in valid]))
    return pooled_r, ci_lo, ci_hi, pct_neg

def fit_lmm(df, sig_col, target_col, patient_col, n_sub=SUBSAMPLE_N, rng=0):
    rng_gen = np.random.default_rng(rng)
    parts = []
    for _, grp in df.groupby(patient_col):
        if len(grp) > n_sub:
            grp = grp.sample(n_sub, random_state=int(rng_gen.integers(1e6)))
        parts.append(grp)
    sub = pd.concat(parts)[[sig_col, target_col, patient_col]].dropna()
    if sub[patient_col].nunique() < MIN_PATIENTS:
        return np.nan, np.nan
    sub = sub.rename(columns={sig_col:"sig", target_col:"target", patient_col:"patient"})
    try:
        res = smf.mixedlm("sig ~ target", sub, groups=sub["patient"]).fit(
            method="lbfgs", disp=False)
        return float(res.params["target"]), float(res.pvalues["target"])
    except Exception:
        return np.nan, np.nan

# ── Load data and score ───────────────────────────────────────────────────────
def load_and_score(cfg):
    print(f"  Loading {cfg['h5ad']} …")
    adata = sc.read_h5ad(cfg["h5ad"])

    # CC scores
    if not cfg["cc_in_obs"]:
        s_g   = [g for g in S_GENES   if g in adata.var_names]
        g2m_g = [g for g in G2M_GENES if g in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_g, g2m_genes=g2m_g)
        cfg["s_col"], cfg["g2m_col"] = "S_score", "G2M_score"

    # Immune score
    imm_present = [g for g in IMMUNOGENIC if g in adata.var_names]
    print(f"  Immune genes found: {len(imm_present)}/{len(IMMUNOGENIC)}")
    sc.tl.score_genes(adata, imm_present, score_name="immune_score", use_raw=False)

    # Stemness signatures
    with open(cfg["sigfile"]) as f:
        sigs = json.load(f)
    sig_cols = []
    for name, genes in sigs.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 5:
            continue
        sc.tl.score_genes(adata, present, score_name=f"sig_{name}", use_raw=False)
        sig_cols.append(f"sig_{name}")

    # Patient filter
    pat_col = cfg["patient_col"]
    counts  = adata.obs[pat_col].value_counts()
    keep    = counts[counts >= MIN_CELLS].index
    adata   = adata[adata.obs[pat_col].isin(keep)].copy()
    print(f"  After ≥{MIN_CELLS} filter: {adata.n_obs:,} cells, "
          f"{adata.obs[pat_col].nunique()} patients")

    obs_cols = ([cfg["subtype_col"], pat_col,
                 cfg["s_col"], cfg["g2m_col"], "immune_score"] + sig_cols)
    df = adata.obs[obs_cols].copy()
    df["CC"] = (df[cfg["s_col"]] + df[cfg["g2m_col"]]) / 2
    del adata; gc.collect()
    return df, sig_cols

# ── Compute immune metrics per signature × subtype ───────────────────────────
def compute_immune_metrics(df, sig_col, s_col, g2m_col, patient_col):
    sig     = df[sig_col].values.astype(float)
    imm     = df["immune_score"].values.astype(float)
    s       = df[s_col].values.astype(float)
    g2m     = df[g2m_col].values.astype(float)
    patients = df[patient_col].values
    m = {}

    # Per-patient meta-analysis vs immune
    ps = []
    dcor_vals, mi_vals = [], []
    for pid in np.unique(patients):
        mask = patients == pid
        if mask.sum() < 30:
            continue
        r, _ = spearmanr(sig[mask], imm[mask])
        ps.append((r, mask.sum()))
        dcor_vals.append(compute_dcor(sig[mask], imm[mask]))
        mi_vals.append(compute_mi(sig[mask], imm[mask]))

    pr, ci_lo, ci_hi, pct_neg = per_patient_meta(ps)
    m["r_immune_pooled"]  = pr
    m["r_immune_ci_lo"]   = ci_lo
    m["r_immune_ci_hi"]   = ci_hi
    m["pct_immune_neg"]   = pct_neg   # % patients showing negative r
    m["dcor_immune_mean"] = np.nanmean(dcor_vals) if dcor_vals else np.nan
    m["mi_immune_mean"]   = np.nanmean(mi_vals)   if mi_vals   else np.nan

    # Partial r vs immune controlling for CC
    cov = np.column_stack([s, g2m])
    r_p, p_p = partial_spearman(sig, imm, cov)
    m["r_immune_partial"] = r_p
    m["p_immune_partial"] = p_p

    # LMM: sig ~ immune + (1|patient)
    if df[patient_col].nunique() >= MIN_PATIENTS:
        beta, pval = fit_lmm(df, sig_col, "immune_score", patient_col)
        m["lmm_beta_immune"] = beta
        m["lmm_p_immune"]    = pval
    else:
        m["lmm_beta_immune"] = np.nan
        m["lmm_p_immune"]    = np.nan

    return m

# ── Main analysis ─────────────────────────────────────────────────────────────
def analyse(ds_name, cfg):
    print(f"\n{'='*65}\n {ds_name}\n{'='*65}")

    df_all, sig_cols = load_and_score(cfg)
    sub_col = cfg["subtype_col"]
    pat_col = cfg["patient_col"]
    s_col   = cfg["s_col"]
    g2m_col = cfg["g2m_col"]

    # Load existing part1 metrics
    part1 = pd.read_csv(cfg["part1_csv"])

    rows = []
    for subtype in SUBTYPES + ["Global"]:
        df_sub = df_all if subtype == "Global" \
                 else df_all[df_all[sub_col] == subtype]
        if len(df_sub) < MIN_CELLS or df_sub[pat_col].nunique() < 2:
            continue
        print(f"\n  ── {subtype}: {len(df_sub):,} cells, "
              f"{df_sub[pat_col].nunique()} patients")

        for sig_col in sig_cols:
            sig_name = sig_col[4:]
            m = compute_immune_metrics(df_sub, sig_col, s_col, g2m_col, pat_col)
            m.update({"dataset": ds_name, "subtype": subtype,
                      "signature": sig_name})
            rows.append(m)
            print(f"    {sig_name:<35} "
                  f"r_imm={m['r_immune_pooled']:+.3f}  "
                  f"r_imm_partial={m['r_immune_partial']:+.3f}  "
                  f"pct_neg={m['pct_immune_neg']:.0%}")

    imm_df = pd.DataFrame(rows)

    # Join with part1 results
    merged = part1.merge(
        imm_df[["dataset","subtype","signature",
                "r_immune_pooled","r_immune_ci_lo","r_immune_ci_hi",
                "pct_immune_neg","r_immune_partial","p_immune_partial",
                "dcor_immune_mean","mi_immune_mean",
                "lmm_beta_immune","lmm_p_immune"]],
        on=["dataset","subtype","signature"], how="left"
    )

    # Combined rank: high cyto + negative immune + low CC
    merged["rank_combined"] = (
        0.4 * merged["r_cyto_partial"].clip(-1,1)
        - 0.4 * merged["r_immune_pooled"].clip(-1,1)  # negative = good
        - 0.2 * merged["R2_CC"].clip(0,1)
    )

    out_csv = os.path.join(cfg["outdir"], f"ranking_with_immune_{ds_name}.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    return merged

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_immune_forest(merged, ds_name, outdir):
    """Forest plot: pooled r vs immune score per signature (want negative)."""
    subtypes_present = [s for s in SUBTYPES if s in merged["subtype"].values]
    sig_order  = [s for s in SIG_LABELS if s in merged["signature"].values]
    sig_labels = [SIG_LABELS.get(s, s) for s in sig_order]

    fig, axes = plt.subplots(1, len(subtypes_present),
                             figsize=(5 * len(subtypes_present), 5),
                             sharey=True)
    if len(subtypes_present) == 1:
        axes = [axes]

    for ax, subtype in zip(axes, subtypes_present):
        sub = merged[merged["subtype"] == subtype].copy()
        sub = sub.set_index("signature").reindex(sig_order).reset_index()
        y   = np.arange(len(sig_order))
        r   = sub["r_immune_pooled"].values.astype(float)
        lo  = sub["r_immune_ci_lo"].values.astype(float)
        hi  = sub["r_immune_ci_hi"].values.astype(float)
        pct = sub["pct_immune_neg"].values.astype(float)
        color = SUBTYPE_COLORS.get(subtype, "grey")

        for i in range(len(sig_order)):
            if np.isfinite(r[i]):
                ax.plot([lo[i], hi[i]], [y[i], y[i]],
                        color=color, lw=1.5, alpha=0.5)
                ax.scatter(r[i], y[i], color=color, s=60,
                           zorder=3, alpha=0.9)
                ax.text(hi[i] + 0.02, y[i],
                        f"{pct[i]:.0%} neg", va="center", fontsize=7.5,
                        color="dimgray")

        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-1, 1.3)
        ax.set_yticks(y)
        ax.set_yticklabels(sig_labels if subtype == subtypes_present[0] else [],
                           fontsize=9)
        ax.set_xlabel("Pooled Spearman r vs immune score\n(95% CI)", fontsize=9)
        ax.set_title(subtype, fontsize=11, fontweight="bold",
                     color=color)
        ax.axvspan(-1, 0, alpha=0.04, color="steelblue", label="negative (good)")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"{ds_name} — Stemness × Immune correlation\n"
                 "(negative = stem-high / immune-low → SCIE direction)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(outdir, f"immune_forest_{ds_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_2d_scatter(merged, ds_name, outdir):
    """
    2D scatter: r_cyto_partial (x) vs r_immune_pooled (y).
    Ideal signatures: right (high cyto) + bottom (negative immune).
    Bubble size encodes 1 - R2_CC (low CC contamination = larger).
    One panel per subtype.
    """
    subtypes_present = [s for s in SUBTYPES + ["Global"]
                        if s in merged["subtype"].values]
    fig, axes = plt.subplots(1, len(subtypes_present),
                             figsize=(4.5 * len(subtypes_present), 4.5),
                             sharey=True, sharex=True)
    if len(subtypes_present) == 1:
        axes = [axes]

    sig_order  = [s for s in SIG_LABELS if s in merged["signature"].values]
    sig_labels = [SIG_LABELS.get(s, s) for s in sig_order]
    markers    = {s: m for s, m in zip(sig_order, SIG_MARKERS)}

    for ax, subtype in zip(axes, subtypes_present):
        sub = merged[merged["subtype"] == subtype].copy()
        sub = sub.set_index("signature").reindex(sig_order)
        color = SUBTYPE_COLORS.get(subtype, "grey")

        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.4)

        # Shade ideal quadrant (high cyto, negative immune)
        ax.axvspan(0, 1,  ymin=0,   ymax=0.5, alpha=0.06,
                   color="green", label="SCIE-favourable")

        for sig in sig_order:
            row = sub.loc[sig] if sig in sub.index else None
            if row is None:
                continue
            xv = row["r_cyto_partial"]
            yv = row["r_immune_pooled"]
            r2 = row["R2_CC"]
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            size = max((1 - r2) * 300, 20)
            ax.scatter(xv, yv, s=size, color=color,
                       marker=markers.get(sig, "o"),
                       alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(SIG_LABELS.get(sig, sig), (xv, yv),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])

        ax.set_xlim(-0.6, 0.8)
        ax.set_ylim(-0.8, 0.6)
        ax.set_xlabel("r(CytoTRACE2 | CC)  →  more stemness", fontsize=9)
        if subtype == subtypes_present[0]:
            ax.set_ylabel("r(immune score)  ↓  more anti-immune", fontsize=9)
        ax.set_title(subtype, fontsize=11, fontweight="bold", color=color)
        ax.grid(alpha=0.25)

        # Annotate combined rank for top-2 in this panel
        if "rank_combined" in sub.columns:
            top2 = sub["rank_combined"].nlargest(2).index
            for sig in top2:
                if sig in sub.index:
                    xv = sub.loc[sig, "r_cyto_partial"]
                    yv = sub.loc[sig, "r_immune_pooled"]
                    rc = sub.loc[sig, "rank_combined"]
                    if np.isfinite(xv) and np.isfinite(yv):
                        ax.annotate(f"★ rank={rc:+.2f}",
                                    (xv, yv),
                                    textcoords="offset points",
                                    xytext=(5, -12), fontsize=7,
                                    color="navy", fontweight="bold")

    # Legend: bubble size = 1 - R2_CC
    for ax in axes:
        for r2_ex, label in [(0.0, "R²=0 (no CC)"),
                             (0.3, "R²=0.3"),
                             (0.6, "R²=0.6")]:
            ax.scatter([], [], s=max((1 - r2_ex) * 300, 20),
                       color="grey", alpha=0.5, label=label)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7,
                  title="Bubble = 1−R²_CC", title_fontsize=7)

    fig.suptitle(
        f"{ds_name} — Signature selection map\n"
        r"x = r(CytoTRACE2 | CC),  y = r(immune),  "
        r"size = 1−R²$_{CC}$",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(outdir, f"selection_map_{ds_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_combined_ranking(all_merged, outdir_base):
    """
    Side-by-side combined rank score across both datasets.
    """
    sig_order  = list(SIG_LABELS.keys())
    sig_labels = [SIG_LABELS.get(s, s) for s in sig_order]
    datasets   = list(all_merged.keys())
    subtypes   = SUBTYPES + ["Global"]

    fig, axes = plt.subplots(
        len(datasets), len(subtypes),
        figsize=(3.5 * len(subtypes), 3 * len(datasets)),
        sharey=True, squeeze=False
    )

    for r_idx, ds in enumerate(datasets):
        merged = all_merged[ds]
        for c_idx, subtype in enumerate(subtypes):
            ax = axes[r_idx, c_idx]
            sub = merged[merged["subtype"] == subtype].copy()
            sub = sub.set_index("signature").reindex(sig_order)

            rc  = sub["rank_combined"].values.astype(float)
            colors = ["#2ca02c" if v > 0 else "#d62728"
                      if np.isfinite(v) else "lightgrey" for v in rc]

            bars = ax.barh(range(len(sig_order)), rc, color=colors, alpha=0.8)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks(range(len(sig_labels)))
            ax.set_yticklabels(
                sig_labels if c_idx == 0 else [], fontsize=8
            )
            if r_idx == 0:
                ax.set_title(subtype, fontsize=10, fontweight="bold",
                             color=SUBTYPE_COLORS.get(subtype, "k"))
            if c_idx == 0:
                ax.set_ylabel(ds, fontsize=9, fontweight="bold")
            ax.set_xlim(-0.8, 0.8)
            ax.grid(axis="x", alpha=0.3)

            # Annotate values
            for i, v in enumerate(rc):
                if np.isfinite(v):
                    ax.text(v + (0.02 if v >= 0 else -0.02), i,
                            f"{v:+.2f}", va="center",
                            ha="left" if v >= 0 else "right", fontsize=7)

    fig.suptitle(
        "Combined rank score\n"
        r"0.4×r$_{cyto|CC}$ − 0.4×r$_{immune}$ − 0.2×R²$_{CC}$",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(outdir_base, "combined_ranking_both_datasets.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_merged = {}
    for ds_name, cfg in DATASETS.items():
        merged = analyse(ds_name, cfg)
        all_merged[ds_name] = merged

        print(f"\n  Generating figures for {ds_name} …")
        plot_immune_forest(merged, ds_name, cfg["outdir"])
        plot_2d_scatter(merged, ds_name, cfg["outdir"])

        print(f"\n  ── Top signatures by combined rank ({ds_name}) ──")
        for subtype in SUBTYPES + ["Global"]:
            sub = merged[merged["subtype"] == subtype].sort_values(
                "rank_combined", ascending=False)
            if sub.empty:
                continue
            top = sub[["signature","rank_combined","r_cyto_partial",
                        "r_immune_pooled","R2_CC"]].head(3)
            print(f"\n  {subtype}:")
            print(top.to_string(index=False))

        gc.collect()

    # Combined figure across both datasets
    plot_combined_ranking(all_merged,
                          outdir_base="Visvader/figures/part1")

    print("\n✓ Part 1b complete.")
