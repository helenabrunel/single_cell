"""
part1_stemness_vs_cellcycle.py
================================
Evaluate all 9 stemness signatures to determine which best captures true
developmental potency (stemness) independently of cell cycle (CC).
Second goal: build a CC-free synthetic stemness index and verify the SCIE axis
(stemness × immune anti-correlation) survives CC residualisation.

Inputs: pre-computed processed h5ads from part0
  - visvader_malignant_processed.h5ad
  - klughammer_malignant_processed.h5ad

Metrics computed per signature × subtype × dataset (all per-patient, then pooled):
  1. Spearman r (Fisher z meta-analysis with 95% CI) vs CytoTRACE2, CC, immune
  2. R² from OLS: sig ~ S.Score + G2M.Score          — CC variance explained
  3. Partial Spearman r: CytoTRACE2 | S.Score + G2M  — net stemness signal
  4. Distance Correlation (dCor) vs CytoTRACE2        — nonlinear association
  5. Linear Mixed Model: sig ~ CytoTRACE2 + (1|patient) — patient-corrected β

CC-free residualised index:
  Per patient OLS: sig ~ S.Score + G2M.Score → residuals stored as score_{sig}_ccfree
  Same metrics recomputed on residuals; key validation: does r(residual, immune) survive?

Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 part1_stemness_vs_cellcycle.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, gc, warnings
from scipy.stats import spearmanr
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import dcor
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUTDIR = "figures/part1"
os.makedirs(OUTDIR, exist_ok=True)

MIN_CELLS        = 50    # per-patient minimum for metric computation
MIN_PATIENTS_LMM = 3     # minimum patients to fit LMM
DCOR_MAX         = 800   # subsample cap for dCor (O(n²))
RANDOM           = 42

DATASETS = {
    "Visvader": {
        "file"       : "visvader_malignant_processed.h5ad",
        "batch_col"  : "sample",
        "subtype_col": "subtype",
        "outfile"    : "visvader_malignant_ccfree.h5ad",
    },
    "Klughammer": {
        "file"       : "klughammer_malignant_processed.h5ad",
        "batch_col"  : "sampleid",
        "subtype_col": "subtype",
        "outfile"    : "klughammer_malignant_ccfree.h5ad",
    },
}

STEM_SIGS = [
    "ESC_ASSOU", "ESC_WONG", "BENPORATH_ES1", "EMP",
    "fMaSC_over_aMaSC", "LIM_MAMMARY_STEM_CELL_UP",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP", "LIM_MAMMARY_LUMINAL_MATURE_UP",
    "stem_consensus",
]
SIG_LABELS = {
    "ESC_ASSOU"                        : "ESC_ASSOU",
    "ESC_WONG"                         : "ESC_WONG",
    "BENPORATH_ES1"                    : "BenPorath_ES1",
    "EMP"                              : "EMP",
    "fMaSC_over_aMaSC"                 : "fMaSC/aMaSC",
    "LIM_MAMMARY_STEM_CELL_UP"         : "LIM_Stem",
    "LIM_MAMMARY_LUMINAL_PROGENITOR_UP": "LIM_LumProg",
    "LIM_MAMMARY_LUMINAL_MATURE_UP"    : "LIM_LumMat",
    "stem_consensus"                   : "stem_consensus",
}
SUBTYPES       = ["ER", "HER2", "TNBC"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C",
                  "Global": "#333333"}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def partial_spearman(x, y, covariates):
    """Spearman r(x, y | covariates) via OLS residuals."""
    valid = (np.isfinite(x) & np.isfinite(y) &
             np.all(np.isfinite(covariates), axis=1))
    if valid.sum() < 20:
        return np.nan, np.nan
    x_, y_, cov_ = x[valid], y[valid], covariates[valid]
    lr = LinearRegression()
    rx = x_ - lr.fit(cov_, x_).predict(cov_)
    ry = y_ - lr.fit(cov_, y_).predict(cov_)
    r, p = spearmanr(rx, ry)
    return float(r), float(p)


def r2_cc(sig, s_score, g2m_score):
    """R² of OLS: sig ~ S.Score + G2M.Score."""
    valid = np.isfinite(sig) & np.isfinite(s_score) & np.isfinite(g2m_score)
    if valid.sum() < 20:
        return np.nan
    X = np.column_stack([s_score[valid], g2m_score[valid]])
    return float(LinearRegression().fit(X, sig[valid]).score(X, sig[valid]))


def cc_residuals_pat(sig, s_score, g2m_score):
    """Per-patient OLS residuals of sig ~ S.Score + G2M.Score."""
    X = np.column_stack([s_score, g2m_score])
    valid = np.isfinite(sig) & np.all(np.isfinite(X), axis=1)
    res = np.full_like(sig, np.nan)
    if valid.sum() < 20:
        return res
    reg = LinearRegression().fit(X[valid], sig[valid])
    res[valid] = sig[valid] - reg.predict(X[valid])
    return res


def safe_spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return np.nan
    r, _ = spearmanr(x[valid], y[valid])
    return float(r)


def safe_dcor(x, y, max_n=DCOR_MAX):
    valid = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[valid], y[valid]
    if len(xm) < 20:
        return np.nan
    if len(xm) > max_n:
        rng = np.random.default_rng(RANDOM)
        idx = rng.choice(len(xm), max_n, replace=False)
        xm, ym = xm[idx], ym[idx]
    try:
        return float(dcor.u_distance_correlation_sqr(xm, ym))
    except Exception:
        return np.nan


def per_patient_meta(patient_stats):
    """
    Fisher z meta-analysis from list of (r, n) tuples.
    Returns (pooled_r, ci_lo, ci_hi, pct_positive).
    """
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
    pct_pos  = float(np.mean([r > 0 for r, _ in valid]))
    return pooled_r, ci_lo, ci_hi, pct_pos


def fit_lmm(df, sig_col, target_col, patient_col):
    """LMM: sig_col ~ target_col + (1|patient_col). Returns (beta, p)."""
    sub = df[[sig_col, target_col, patient_col]].dropna()
    if sub[patient_col].nunique() < MIN_PATIENTS_LMM:
        return np.nan, np.nan
    # Subsample per patient for speed
    parts = []
    rng = np.random.default_rng(RANDOM)
    for _, grp in sub.groupby(patient_col):
        if len(grp) > DCOR_MAX:
            grp = grp.sample(DCOR_MAX, random_state=int(rng.integers(1e6)))
        parts.append(grp)
    sub = pd.concat(parts).rename(
        columns={sig_col: "sig_s", target_col: "target", patient_col: "patient"}
    )
    try:
        md  = smf.mixedlm("sig_s ~ target", sub, groups=sub["patient"])
        res = md.fit(method="lbfgs", disp=False)
        return float(res.params["target"]), float(res.pvalues["target"])
    except Exception:
        return np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(df, sig_col, batch_col, subtype_col=None,
                        subtype_filter=None, include_lmm=True):
    """
    All metrics for one signature on a subset of obs.
    df must have: sig_col, S.Score, G2M.Score, CytoTRACE2, score_immune, batch_col.
    """
    if subtype_filter and subtype_col:
        df = df[df[subtype_col] == subtype_filter]
    if len(df) < MIN_CELLS:
        return None

    sig  = df[sig_col].values.astype(float)
    s    = df["S.Score"].values.astype(float)
    g2m  = df["G2M.Score"].values.astype(float)
    ct2  = df["CytoTRACE2"].values.astype(float)
    imm  = df["score_immune"].values.astype(float)
    cc   = (s + g2m) / 2
    cov  = np.column_stack([s, g2m])
    pats = df[batch_col].values

    # ── Per-patient Spearman meta-analysis ────────────────────────────────
    def pat_meta(target):
        stats = []
        for pid in np.unique(pats):
            mask = pats == pid
            if mask.sum() < MIN_CELLS:
                continue
            r = safe_spearman(sig[mask], target[mask])
            stats.append((r, int(mask.sum())))
        return per_patient_meta(stats)

    r_ct2, ci_lo_ct2, ci_hi_ct2, pct_ct2 = pat_meta(ct2)
    r_s,   ci_lo_s,   ci_hi_s,   pct_s   = pat_meta(s)
    r_g2m, ci_lo_g2m, ci_hi_g2m, pct_g2m = pat_meta(g2m)
    r_cc,  ci_lo_cc,  ci_hi_cc,  pct_cc  = pat_meta(cc)
    r_imm, ci_lo_imm, ci_hi_imm, pct_imm = pat_meta(imm)

    # ── dCor vs CytoTRACE2 (per-patient, then mean) ───────────────────────
    dcor_vals = []
    for pid in np.unique(pats):
        mask = pats == pid
        if mask.sum() < MIN_CELLS:
            continue
        dcor_vals.append(safe_dcor(sig[mask], ct2[mask]))
    dcor_ct2 = float(np.nanmean(dcor_vals)) if dcor_vals else np.nan

    # ── Partial Spearman r: CT2 | CC ─────────────────────────────────────
    r_part, p_part = partial_spearman(sig, ct2, cov)

    # ── R²(CC → sig) ─────────────────────────────────────────────────────
    R2_cc = r2_cc(sig, s, g2m)

    # ── LMM: sig ~ CytoTRACE2 + (1|patient) ─────────────────────────────
    lmm_beta_ct2, lmm_p_ct2 = np.nan, np.nan
    lmm_beta_cc,  lmm_p_cc  = np.nan, np.nan
    if include_lmm and df[batch_col].nunique() >= MIN_PATIENTS_LMM:
        lmm_beta_ct2, lmm_p_ct2 = fit_lmm(df, sig_col, "CytoTRACE2", batch_col)
        df_tmp = df.copy()
        df_tmp["__cc"] = cc
        lmm_beta_cc, lmm_p_cc = fit_lmm(df_tmp, sig_col, "__cc", batch_col)

    # ── Composite rank score ──────────────────────────────────────────────
    rank_score = np.nan
    if np.isfinite(r_ct2) and np.isfinite(r_part) and np.isfinite(R2_cc):
        rank_score = float(0.5 * r_ct2 + 0.5 * r_part - R2_cc)

    return {
        "n_cells"         : len(df),
        "n_patients"      : int(df[batch_col].nunique()),
        # Spearman r vs CytoTRACE2
        "r_ct2"           : r_ct2,
        "r_ct2_ci_lo"     : ci_lo_ct2,
        "r_ct2_ci_hi"     : ci_hi_ct2,
        "pct_ct2_pos"     : pct_ct2,
        # Spearman r vs cell cycle
        "r_S"             : r_s,
        "r_G2M"           : r_g2m,
        "r_CC"            : r_cc,
        # Spearman r vs immune
        "r_immune"        : r_imm,
        "r_immune_ci_lo"  : ci_lo_imm,
        "r_immune_ci_hi"  : ci_hi_imm,
        # CC variance
        "R2_CC"           : R2_cc,
        # Partial r
        "partial_r_ct2"   : r_part,
        "partial_p_ct2"   : p_part,
        # dCor
        "dcor_ct2"        : dcor_ct2,
        # LMM
        "lmm_beta_ct2"    : lmm_beta_ct2,
        "lmm_p_ct2"       : lmm_p_ct2,
        "lmm_beta_cc"     : lmm_beta_cc,
        "lmm_p_cc"        : lmm_p_cc,
        # Composite
        "rank_score"      : rank_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CC-FREE RESIDUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_ccfree_scores(obs, sigs, batch_col):
    """
    Per patient, per signature: OLS residuals of sig ~ S.Score + G2M.Score.
    Returns dict: sig → ccfree_array (same length as obs).
    """
    ccfree = {}
    patients = obs[batch_col].values
    for sig in sigs:
        col    = f"score_{sig}"
        if col not in obs.columns:
            continue
        result = np.full(len(obs), np.nan)
        for pid in np.unique(patients):
            mask = patients == pid
            if mask.sum() < MIN_CELLS:
                continue
            res = cc_residuals_pat(
                obs[col].values[mask].astype(float),
                obs["S.Score"].values[mask].astype(float),
                obs["G2M.Score"].values[mask].astype(float),
            )
            result[mask] = res
        ccfree[sig] = result
    return ccfree


def compute_ccfree_metrics(obs, ccfree, sig, batch_col, subtype_col,
                           subtype_filter=None):
    """Compute r_ct2, r_CC, r_immune on the CC-free residuals of one signature."""
    col_name = f"__ccfree_{sig}"
    obs_tmp  = obs.copy()
    obs_tmp[col_name] = ccfree[sig]
    m = compute_all_metrics(
        obs_tmp, col_name, batch_col, subtype_col, subtype_filter,
        include_lmm=False,
    )
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

KEY_METRICS = [
    ("r_ct2",        "r(CytoTRACE2)",        "RdYlGn", -0.6, 0.6),
    ("partial_r_ct2","partial r(CT2|CC)",     "RdYlGn", -0.6, 0.6),
    ("R2_CC",        "R²(CC→sig)",           "YlOrRd",  0.0, 0.5),
    ("r_S",          "r(S.Score)",           "RdYlGn", -0.6, 0.6),
    ("r_G2M",        "r(G2M.Score)",         "RdYlGn", -0.6, 0.6),
    ("r_immune",     "r(immune)",            "RdYlGn", -0.6, 0.6),
    ("dcor_ct2",     "dCor(CytoTRACE2)",     "YlGn",    0.0, 0.3),
    ("rank_score",   "Rank score",           "RdYlGn", -0.5, 0.5),
]


def _draw_heatmap(ax, data, row_labels, col_labels, vmin, vmax,
                  cmap="RdBu_r", title=""):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v - (vmin+vmax)/2) >
                        0.35 * (vmax - vmin) else "black")
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)


def plot_heatmap(rows_by_subtype, ds_name):
    """Key-metrics heatmap per subtype + Global."""
    subtypes_present = [st for st in SUBTYPES + ["Global"]
                        if st in rows_by_subtype]
    sigs_ordered = [s for s in STEM_SIGS
                    if s in {r["signature"] for r in
                             next(iter(rows_by_subtype.values()), [])}]
    labels = [SIG_LABELS.get(s, s) for s in sigs_ordered]
    col_labels = [m[1] for m in KEY_METRICS]
    n_sub = len(subtypes_present)

    fig, axes = plt.subplots(1, n_sub, figsize=(len(KEY_METRICS) * 1.3 * n_sub + 1,
                                                 len(sigs_ordered) * 0.6 + 2))
    if n_sub == 1:
        axes = [axes]
    fig.suptitle(f"{ds_name} — Signature metrics (per-patient pooled)",
                 fontsize=11, fontweight="bold")

    for ci, subtype in enumerate(subtypes_present):
        rows = rows_by_subtype[subtype]
        df_s = pd.DataFrame(rows).set_index("signature").reindex(sigs_ordered)
        data = np.full((len(sigs_ordered), len(KEY_METRICS)), np.nan)
        for j, (col, _, _, vmin, vmax) in enumerate(KEY_METRICS):
            if col in df_s.columns:
                data[:, j] = df_s[col].values.astype(float)

        _draw_heatmap(axes[ci], data, labels, col_labels,
                      vmin=-0.6, vmax=0.6,
                      title=f"{subtype}  (n={df_s['n_patients'].iloc[0] if 'n_patients' in df_s.columns else '?'} pat.)")
        if ci > 0:
            axes[ci].set_yticks([])

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{ds_name}_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_forest(rows_global, ds_name):
    """Forest plot: pooled r ± 95% CI for CytoTRACE2 and CC, all sigs."""
    sigs   = [r["signature"] for r in rows_global]
    labels = [SIG_LABELS.get(s, s) for s in sigs]
    y      = np.arange(len(sigs))

    fig, axes = plt.subplots(1, 3, figsize=(15, len(sigs) * 0.5 + 2))
    fig.suptitle(f"{ds_name} — Pooled per-patient Spearman r (95% CI)",
                 fontsize=11, fontweight="bold")

    targets = [
        ("r_ct2", "r_ct2_ci_lo", "r_ct2_ci_hi",
         "r(sig, CytoTRACE2)", "steelblue"),
        ("r_CC", None, None,
         "r(sig, CC score)", "tomato"),
        ("r_immune", "r_immune_ci_lo", "r_immune_ci_hi",
         "r(sig, Immunogenic)", "darkorange"),
    ]

    df = pd.DataFrame(rows_global).set_index("signature").reindex(sigs)

    for ax, (r_col, lo_col, hi_col, title, color) in zip(axes, targets):
        for i, sig in enumerate(sigs):
            r  = df.loc[sig, r_col] if r_col in df.columns else np.nan
            lo = df.loc[sig, lo_col] if lo_col and lo_col in df.columns else np.nan
            hi = df.loc[sig, hi_col] if hi_col and hi_col in df.columns else np.nan
            if np.isfinite(float(r) if not pd.isna(r) else np.nan):
                if np.isfinite(float(lo) if not pd.isna(lo) else np.nan):
                    ax.plot([lo, hi], [y[i], y[i]], color=color,
                            linewidth=2, alpha=0.5)
                ax.scatter(r, y[i], color=color, s=60, zorder=3)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Pooled Spearman r", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(-0.7, 0.7)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{ds_name}_forest.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_tradeoff(rows_global, ds_name):
    """Scatter: R²_CC vs partial_r_CT2 — contamination vs net potency."""
    df     = pd.DataFrame(rows_global)
    sigs   = df["signature"].tolist()
    labels = [SIG_LABELS.get(s, s) for s in sigs]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sigs)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        x = row.get("R2_CC", np.nan)
        y = row.get("partial_r_ct2", np.nan)
        if np.isfinite(float(x)) and np.isfinite(float(y)):
            ax.scatter(x, y, color=colors[i], s=120, zorder=3,
                       edgecolors="black", linewidths=0.6)
            ax.annotate(labels[i], (x, y), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("R²(CC → signature)  [contamination ↑]", fontsize=10)
    ax.set_ylabel("Partial r(sig, CytoTRACE2 | CC)  [net potency ↑]", fontsize=10)
    ax.set_title(f"{ds_name} — CC contamination vs net potency\nIdeal: left + up",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{ds_name}_tradeoff.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_ccfree_comparison(orig_rows, ccf_rows, ds_name):
    """Before vs after CC residualisation: r_ct2, r_S, r_G2M, r_immune."""
    sigs   = [r["signature"] for r in orig_rows]
    labels = [SIG_LABELS.get(s, s) for s in sigs]
    y      = np.arange(len(sigs))

    df_o = pd.DataFrame(orig_rows).set_index("signature").reindex(sigs)
    df_c = pd.DataFrame(ccf_rows).set_index("signature").reindex(sigs)

    metrics_to_show = [
        ("r_ct2",    "r(CytoTRACE2)",   "steelblue"),
        ("r_S",      "r(S.Score)",      "tomato"),
        ("r_G2M",    "r(G2M.Score)",    "darkorange"),
        ("r_immune", "r(Immunogenic)",  "purple"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(16, len(sigs) * 0.4 + 2))
    fig.suptitle(f"{ds_name} — Effect of CC residualisation\n"
                 f"solid = original  |  dashed = CC-free",
                 fontsize=11, fontweight="bold")

    for ax, (col, title, color) in zip(axes, metrics_to_show):
        for i, sig in enumerate(sigs):
            r_o = df_o.loc[sig, col] if col in df_o.columns else np.nan
            r_c = df_c.loc[sig, col] if col in df_c.columns else np.nan
            r_o = float(r_o) if not pd.isna(r_o) else np.nan
            r_c = float(r_c) if not pd.isna(r_c) else np.nan
            if np.isfinite(r_o):
                ax.scatter(r_o, y[i], color=color, s=60, zorder=3)
            if np.isfinite(r_c):
                ax.scatter(r_c, y[i], color=color, s=60, zorder=3,
                           marker="D", alpha=0.7)
            if np.isfinite(r_o) and np.isfinite(r_c):
                ax.annotate("", xy=(r_c, y[i]), xytext=(r_o, y[i]),
                            arrowprops=dict(arrowstyle="->",
                                           color=color, alpha=0.5))
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Pooled Spearman r", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(-0.7, 0.7)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{ds_name}_ccfree_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_scie_axis_survival(orig_rows, ccf_rows, ds_name):
    """r_immune original vs CC-free: does the SCIE axis survive?"""
    df_o = pd.DataFrame(orig_rows)
    df_c = pd.DataFrame(ccf_rows)
    sigs = df_o["signature"].tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sigs)))

    x_vals, y_vals, labels_plot, cols = [], [], [], []
    for i, sig in enumerate(sigs):
        r_o = df_o.loc[df_o["signature"] == sig, "r_immune"].values
        r_c = df_c.loc[df_c["signature"] == sig, "r_immune"].values
        if not len(r_o) or not len(r_c):
            continue
        r_o, r_c = float(r_o[0]), float(r_c[0])
        if np.isfinite(r_o) and np.isfinite(r_c):
            x_vals.append(r_o); y_vals.append(r_c)
            labels_plot.append(SIG_LABELS.get(sig, sig))
            cols.append(colors[i])

    if not x_vals:
        return

    lim = max(abs(min(x_vals + y_vals, default=0)),
              abs(max(x_vals + y_vals, default=0.1))) + 0.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.axhline(0, color="grey", ls=":", lw=0.6, alpha=0.4)
    ax.axvline(0, color="grey", ls=":", lw=0.6, alpha=0.4)
    for x, y, lab, col in zip(x_vals, y_vals, labels_plot, cols):
        ax.scatter(x, y, color=col, s=130, zorder=3,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(lab, (x, y), xytext=(5, 3),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("r(sig, immunogenic) — original", fontsize=10)
    ax.set_ylabel("r(sig_ccfree, immunogenic) — CC-free", fontsize=10)
    ax.set_title(f"{ds_name}\nSCIE axis survival after CC residualisation\n"
                 "Points near y=x: immune anti-correlation is not driven by CC",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{ds_name}_scie_survival.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

all_tables = {}

for ds_name, cfg in DATASETS.items():
    print(f"\n{'='*65}")
    print(f" DATASET: {ds_name}")
    print(f"{'='*65}")

    adata       = sc.read_h5ad(cfg["file"])
    obs         = adata.obs.copy()
    batch_col   = cfg["batch_col"]
    subtype_col = cfg["subtype_col"]
    sigs        = [s for s in STEM_SIGS if f"score_{s}" in obs.columns]

    print(f"  {adata.n_obs:,} cells | {obs[batch_col].nunique()} patients")
    print(f"  Signatures available: {len(sigs)}")

    # ── Original metrics ─────────────────────────────────────────────────
    print("\n  Computing original metrics …")
    rows_by_subtype = {}
    for subtype in SUBTYPES + ["Global"]:
        sf = None if subtype == "Global" else subtype
        rows = []
        for sig in sigs:
            m = compute_all_metrics(
                obs, f"score_{sig}", batch_col, subtype_col, sf,
                include_lmm=True,
            )
            if m is None:
                continue
            m["signature"] = sig
            m["subtype"]   = subtype
            m["dataset"]   = ds_name
            rows.append(m)
            print(f"    {subtype:8s} | {SIG_LABELS[sig]:15s} | "
                  f"R²_CC={m['R2_CC']:.2f}  r_CT2={m['r_ct2']:+.3f}  "
                  f"partial_r={m['partial_r_ct2']:+.3f}  "
                  f"r_imm={m['r_immune']:+.3f}  rank={m['rank_score']:+.3f}")
        if rows:
            rows_by_subtype[subtype] = rows

    # ── CC-free scores ───────────────────────────────────────────────────
    print("\n  Building CC-free residualised scores …")
    ccfree = build_ccfree_scores(obs, sigs, batch_col)
    for sig, arr in ccfree.items():
        adata.obs[f"score_{sig}_ccfree"] = arr
    obs = adata.obs.copy()

    print("  Computing metrics on CC-free scores …")
    ccfree_rows = []
    for sig in sigs:
        col_name = f"score_{sig}_ccfree"
        tmp_col  = "__ccf_score"
        obs_tmp  = obs.copy()
        obs_tmp[tmp_col] = obs[col_name]
        m = compute_all_metrics(
            obs_tmp, tmp_col, batch_col, subtype_col, None,
            include_lmm=False,
        )
        if m is None:
            continue
        m["signature"] = sig
        m["subtype"]   = "Global"
        m["dataset"]   = ds_name
        ccfree_rows.append(m)

    # ── Save updated h5ad ───────────────────────────────────────────────
    adata.write_h5ad(cfg["outfile"])
    print(f"\n  ✓ Saved: {cfg['outfile']}")
    del adata; gc.collect()

    # ── Build combined table ─────────────────────────────────────────────
    all_rows = []
    for subtype, rows in rows_by_subtype.items():
        all_rows.extend(rows)
    df_results = pd.DataFrame(all_rows)
    df_ccfree  = pd.DataFrame(ccfree_rows).add_suffix("_ccfree").rename(
        columns={"signature_ccfree": "signature", "dataset_ccfree": "dataset",
                 "subtype_ccfree": "subtype"}
    )
    out_csv = os.path.join(OUTDIR, f"{ds_name}_metrics.csv")
    df_results.to_csv(out_csv, index=False)
    out_csv2 = os.path.join(OUTDIR, f"{ds_name}_ccfree_metrics.csv")
    df_ccfree.to_csv(out_csv2, index=False)
    print(f"  ✓ {out_csv}")
    print(f"  ✓ {out_csv2}")

    all_tables[ds_name] = {
        "rows_by_subtype": rows_by_subtype,
        "ccfree_rows"    : ccfree_rows,
        "df_results"     : df_results,
    }

    # ── Figures ─────────────────────────────────────────────────────────
    print(f"\n  Generating figures for {ds_name} …")
    plot_heatmap(rows_by_subtype, ds_name)

    global_rows = rows_by_subtype.get("Global", [])
    if global_rows:
        plot_forest(global_rows, ds_name)
        plot_tradeoff(global_rows, ds_name)
        plot_ccfree_comparison(global_rows, ccfree_rows, ds_name)
        plot_scie_axis_survival(global_rows, ccfree_rows, ds_name)

    # ── Top signatures summary ───────────────────────────────────────────
    print(f"\n  ── Top signatures by rank_score ({ds_name}, Global) ──")
    if global_rows:
        top = (pd.DataFrame(global_rows)
               .sort_values("rank_score", ascending=False)
               [["signature","rank_score","r_ct2","partial_r_ct2","R2_CC","r_immune"]]
               .head(5))
        top["signature"] = top["signature"].map(SIG_LABELS)
        print(top.to_string(index=False))

    gc.collect()


# ── Combined ranking CSV (both datasets) ─────────────────────────────────────
combined_rows = []
for ds_name, tbl in all_tables.items():
    global_rows = tbl["rows_by_subtype"].get("Global", [])
    ccfree_rows = tbl["ccfree_rows"]
    df_o = pd.DataFrame(global_rows).set_index("signature")
    # ccfree CSVs written with add_suffix → columns are r_ct2_ccfree, r_immune_ccfree, etc.
    csv_c = os.path.join(OUTDIR, f"{ds_name}_ccfree_metrics.csv")
    df_c  = (pd.read_csv(csv_c).set_index("signature")
             if os.path.exists(csv_c) else pd.DataFrame())

    def _g(row, key, default=np.nan):
        v = row.get(key, default) if hasattr(row, "get") else default
        try:
            return round(float(v), 3) if pd.notna(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    for sig in STEM_SIGS:
        if sig not in df_o.index:
            continue
        row_o = df_o.loc[sig]
        row_c = df_c.loc[sig] if sig in df_c.index else pd.Series(dtype=float)
        combined_rows.append({
            "dataset"          : ds_name,
            "signature"        : SIG_LABELS.get(sig, sig),
            "n_patients"       : _g(row_o, "n_patients"),
            "R2_CC"            : _g(row_o, "R2_CC"),
            "r_S"              : _g(row_o, "r_S"),
            "r_G2M"            : _g(row_o, "r_G2M"),
            "r_CT2"            : _g(row_o, "r_ct2"),
            "partial_r_CT2"    : _g(row_o, "partial_r_ct2"),
            "dCor_CT2"         : _g(row_o, "dcor_ct2"),
            "r_immune"         : _g(row_o, "r_immune"),
            "r_CT2_ccfree"     : _g(row_c, "r_ct2_ccfree"),
            "r_immune_ccfree"  : _g(row_c, "r_immune_ccfree"),
            "rank_score"       : _g(row_o, "rank_score"),
        })

combined_df = pd.DataFrame(combined_rows).sort_values(
    ["dataset", "rank_score"], ascending=[True, False]
)
out_rank = os.path.join(OUTDIR, "ranking_table.csv")
combined_df.to_csv(out_rank, index=False)
print(f"\n✓ Combined ranking: {out_rank}")
print("\n" + combined_df.to_string(index=False))

print(f"""
{'='*65}
 DONE — figures/part1/
{'='*65}
Per dataset:
  {{ds}}_heatmap.png          — key metrics × sigs × subtypes
  {{ds}}_forest.png           — pooled r ± 95%CI (CT2, CC, immune)
  {{ds}}_tradeoff.png         — R²_CC vs partial_r_CT2
  {{ds}}_ccfree_comparison.png— before/after CC residualisation (arrow)
  {{ds}}_scie_survival.png    — r_immune original vs CC-free (SCIE axis)
  {{ds}}_metrics.csv          — full per-metric table
  {{ds}}_ccfree_metrics.csv   — same for CC-free residuals

Combined:
  ranking_table.csv           — all sigs, both datasets

CC-free h5ads (new obs columns score_{{sig}}_ccfree):
  visvader_malignant_ccfree.h5ad
  klughammer_malignant_ccfree.h5ad
{'='*65}
""")
