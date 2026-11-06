"""
part1c_subpopulation_metrics.py
================================
For each stemness signature × subtype × dataset, computes per-patient
subpopulation detection metrics then meta-analyzes across patients:

  1. Quadrant enrichment (SCIE / NSI):
       - Within-patient tertile thresholds (not global)
       - SCIE = stem-high (t3) × immune-low (t1)
       - NSI  = stem-low  (t1) × immune-high (t3)
       - Fold-enrichment vs independence null
       - Chi-squared test of the full 3×3 contingency table
       - Meta: mean ± SD fold, % patients > 1.5×

  2. 2-component GMM per patient:
       - Fit on per-patient z-scored (stemness, immune) data
       - ΔBIC (BIC_1comp − BIC_2comp; positive = 2-component preferred)
       - Mahalanobis distance between centroids
       - Bhattacharyya coefficient (overlap; lower = better separation)

  3. Bimodality of stemness per patient:
       - Bimodality coefficient BC = (skew²+1)/(excess_kurtosis + correction)
       - BC > 5/9 ≈ 0.555 suggests bimodality
       - Hartigan's dip test (if `diptest` package is installed)

  4. Silhouette score per patient:
       - Labels: SCIE=0, NSI=1 (corner tertiles only; middle cells excluded)
       - Measures how clearly SCIE and NSI form distinct clusters

Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 part1c_subpopulation_metrics.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json, os, gc, warnings
from scipy.stats import chi2_contingency, skew, kurtosis
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ── Immune consensus (APM + IFN α/γ) ─────────────────────────────────────────
IMMUNOGENIC = [
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
]

# Cell cycle gene lists (Tirosh 2016 / Seurat)
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

# ── Dataset configuration ─────────────────────────────────────────────────────
DATASETS = {
    "Visvader": {
        "h5ad"        : "Visvader/compartment_epithelial.h5ad",
        "sigfile"     : "Visvader/signatures_confirmed.json",
        "part1b_csv"  : "Visvader/figures/part1/ranking_with_immune_Visvader.csv",
        "patient_col" : "sample",
        "subtype_col" : "subtype",
        "cc_in_obs"   : False,
        "outdir"      : "Visvader/figures/part1",
    },
    "Klughammer": {
        "h5ad"        : "Klughammer/klughammer_Malignant.h5ad",
        "sigfile"     : "Klughammer/signatures_confirmed.json",
        "part1b_csv"  : "Klughammer/figures/part1/ranking_with_immune_Klughammer.csv",
        "patient_col" : "donor_id",
        "subtype_col" : "subtype",
        "cc_in_obs"   : True,
        "s_col"       : "S.Score",
        "g2m_col"     : "G2M.Score",
        "outdir"      : "Klughammer/figures/part1",
    },
}

SUBTYPES    = ["ER", "HER2", "TNBC"]
MIN_CELLS   = 500
SUBSAMPLE_N = 800

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
SUBTYPE_COLORS = {
    "ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C", "Global": "#888888",
}

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _mahalanobis_centroids(mu1, s1, mu2, s2):
    sigma = (s1 + s2) / 2
    try:
        delta = mu1 - mu2
        return float(np.sqrt(delta @ np.linalg.inv(sigma) @ delta))
    except np.linalg.LinAlgError:
        return np.nan


def _bhattacharyya_coeff(mu1, s1, mu2, s2):
    """Bhattacharyya coefficient in [0,1]: 1 = complete overlap."""
    sigma = (s1 + s2) / 2
    try:
        delta = mu1 - mu2
        d_maha = delta @ np.linalg.inv(sigma) @ delta
        sg1, ld1 = np.linalg.slogdet(s1)
        sg2, ld2 = np.linalg.slogdet(s2)
        sgs, lds = np.linalg.slogdet(sigma)
        if sg1 <= 0 or sg2 <= 0 or sgs <= 0:
            return np.nan
        d_b = (1/8) * d_maha + 0.5 * (lds - 0.5 * (ld1 + ld2))
        return float(np.exp(-d_b))
    except np.linalg.LinAlgError:
        return np.nan

# ── Metric functions (all operate on per-patient arrays) ──────────────────────

def compute_quadrant_enrichment(stem, immune):
    """
    Within-patient tertile SCIE/NSI fold-enrichment vs independence null.
    SCIE = stem tertile-3 × immune tertile-1.
    NSI  = stem tertile-1 × immune tertile-3.
    """
    valid = np.isfinite(stem) & np.isfinite(immune)
    s, im = stem[valid], immune[valid]
    n = len(s)
    _nan = {"fold_scie": np.nan, "fold_nsi": np.nan,
            "chi2_pval": np.nan, "n_scie": 0, "n_nsi": 0}
    if n < 30:
        return _nan

    s_lo, s_hi = np.quantile(s, [1/3, 2/3])
    i_lo, i_hi = np.quantile(im, [1/3, 2/3])
    if s_lo >= s_hi or i_lo >= i_hi:
        return _nan

    s_t = np.digitize(s, [s_lo, s_hi])   # 0=low, 1=mid, 2=high
    i_t = np.digitize(im, [i_lo, i_hi])

    table = np.zeros((3, 3), dtype=float)
    for st, it in zip(s_t, i_t):
        table[st, it] += 1

    try:
        _, pval, _, _ = chi2_contingency(table)
    except Exception:
        pval = np.nan

    n_shi = (s_t == 2).sum()
    n_slo = (s_t == 0).sum()
    n_ilo = (i_t == 0).sum()
    n_ihi = (i_t == 2).sum()

    obs_scie = table[2, 0]
    obs_nsi  = table[0, 2]
    exp_scie = (n_shi / n) * (n_ilo / n) * n
    exp_nsi  = (n_slo / n) * (n_ihi / n) * n

    return {
        "fold_scie": obs_scie / exp_scie if exp_scie > 0 else np.nan,
        "fold_nsi" : obs_nsi  / exp_nsi  if exp_nsi  > 0 else np.nan,
        "chi2_pval": float(pval) if np.isfinite(pval) else np.nan,
        "n_scie"   : int(obs_scie),
        "n_nsi"    : int(obs_nsi),
    }


def compute_gmm_metrics(stem, immune, rng=42):
    """
    1-component vs 2-component GMM on per-patient z-scored (stem, immune).
    Returns ΔBIC (positive = 2-comp preferred), Mahalanobis, Bhattacharyya.
    """
    X = np.column_stack([stem, immune])
    ok = np.isfinite(X).all(axis=1)
    X  = X[ok]
    _nan = {"delta_bic": np.nan, "mahalanobis": np.nan, "bhattacharyya": np.nan}
    if len(X) < 30:
        return _nan
    if len(X) > SUBSAMPLE_N:
        X = X[np.random.default_rng(rng).choice(len(X), SUBSAMPLE_N, replace=False)]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    try:
        g1 = GaussianMixture(1, covariance_type="full",
                             random_state=rng, n_init=3).fit(X)
        g2 = GaussianMixture(2, covariance_type="full",
                             random_state=rng, n_init=5).fit(X)
        mu1, mu2 = g2.means_[0], g2.means_[1]
        s1,  s2  = g2.covariances_[0], g2.covariances_[1]
        return {
            "delta_bic"   : float(g1.bic(X) - g2.bic(X)),
            "mahalanobis" : _mahalanobis_centroids(mu1, s1, mu2, s2),
            "bhattacharyya": _bhattacharyya_coeff(mu1, s1, mu2, s2),
        }
    except Exception:
        return _nan


def compute_bimodality(stem, rng=42):
    """
    Bimodality coefficient BC = (skew²+1) / (excess_kurtosis + correction).
    BC > 5/9 ≈ 0.555 suggests bimodality (uniform distribution gives 5/9).
    """
    x = stem[np.isfinite(stem)]
    if len(x) < 30:
        return {"bimodality_coef": np.nan, "dip_pval": np.nan}
    if len(x) > SUBSAMPLE_N:
        x = x[np.random.default_rng(rng).choice(len(x), SUBSAMPLE_N, replace=False)]
    n  = len(x)
    sk = float(skew(x))
    ku = float(kurtosis(x, fisher=True))   # excess kurtosis
    # Small-sample correction for kurtosis denominator (SAS/STATA formula)
    corr = 3.0 * (n - 1)**2 / ((n - 2) * (n - 3)) if n > 3 else 3.0
    denom = ku + corr
    bc = float((sk**2 + 1) / denom) if abs(denom) > 1e-10 else np.nan

    dip_pval = np.nan
    try:
        import diptest as _dip
        _, dip_pval = _dip.diptest(x)
        dip_pval = float(dip_pval)
    except (ImportError, Exception):
        pass

    return {"bimodality_coef": bc, "dip_pval": dip_pval}


def compute_silhouette_quad(stem, immune, rng=42):
    """
    Silhouette score using within-patient SCIE/NSI corner-tertile labels.
    Only corner cells (SCIE or NSI) are used; middle cells are excluded.
    """
    valid = np.isfinite(stem) & np.isfinite(immune)
    s, im = stem[valid], immune[valid]
    if len(s) < 30:
        return np.nan

    s_lo, s_hi = np.quantile(s, [1/3, 2/3])
    i_lo, i_hi = np.quantile(im, [1/3, 2/3])
    if s_lo >= s_hi or i_lo >= i_hi:
        return np.nan

    s_t = np.digitize(s, [s_lo, s_hi])
    i_t = np.digitize(im, [i_lo, i_hi])

    lbl = np.full(len(s), -1)
    lbl[(s_t == 2) & (i_t == 0)] = 0   # SCIE
    lbl[(s_t == 0) & (i_t == 2)] = 1   # NSI

    mask = lbl >= 0
    if mask.sum() < 10 or len(np.unique(lbl[mask])) < 2:
        return np.nan

    X = np.column_stack([s[mask], im[mask]])
    y = lbl[mask]
    if len(X) > SUBSAMPLE_N:
        idx = np.random.default_rng(rng).choice(len(X), SUBSAMPLE_N, replace=False)
        X, y = X[idx], y[idx]
        if len(np.unique(y)) < 2:
            return np.nan
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    try:
        return float(silhouette_score(X, y))
    except Exception:
        return np.nan

# ── Per-patient loop ──────────────────────────────────────────────────────────

def per_patient_compute(df, sig_col, patient_col):
    rows = []
    for i, pid in enumerate(sorted(df[patient_col].unique())):
        sub = df[df[patient_col] == pid]
        if len(sub) < MIN_CELLS:
            continue
        s  = sub[sig_col].values.astype(float)
        im = sub["immune_score"].values.astype(float)
        rn = 42 + i

        row = {"patient_id": pid, "n_cells": len(sub)}
        row.update(compute_quadrant_enrichment(s, im))
        row.update(compute_gmm_metrics(s, im, rng=rn))
        row.update(compute_bimodality(s, rng=rn))
        row["silhouette"] = compute_silhouette_quad(s, im, rng=rn)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def meta_analyze(pat_df):
    """Summarize per-patient metrics: mean, SD, directional %, n patients."""
    if pat_df.empty or len(pat_df) < 2:
        return {}
    m = {}
    for col in ["fold_scie", "fold_nsi", "delta_bic",
                "mahalanobis", "bhattacharyya", "bimodality_coef", "silhouette"]:
        if col not in pat_df.columns:
            continue
        vals = pat_df[col].values.astype(float)
        vals = vals[np.isfinite(vals)]
        m[f"{col}_mean"]  = float(np.mean(vals))  if len(vals) > 0 else np.nan
        m[f"{col}_sd"]    = float(np.std(vals))   if len(vals) > 1 else np.nan
        m[f"{col}_n_pat"] = int(len(vals))
        if col in ("fold_scie", "fold_nsi"):
            m[f"{col}_pct15x"]  = float(np.mean(vals > 1.5)) if len(vals) > 0 else np.nan
        if col == "delta_bic":
            m[f"{col}_pct_pos"] = float(np.mean(vals > 0))   if len(vals) > 0 else np.nan
        if col == "silhouette":
            m[f"{col}_pct_pos"] = float(np.mean(vals > 0))   if len(vals) > 0 else np.nan

    if "chi2_pval" in pat_df.columns:
        pv = pat_df["chi2_pval"].dropna().values.astype(float)
        m["chi2_pct_sig"] = float(np.mean(pv < 0.05)) if len(pv) > 0 else np.nan
    if "dip_pval" in pat_df.columns:
        pv = pat_df["dip_pval"].dropna().values.astype(float)
        pv = pv[np.isfinite(pv)]
        m["dip_pct_sig"] = float(np.mean(pv < 0.05)) if len(pv) > 0 else np.nan

    return m

# ── Composite z-score ranking ─────────────────────────────────────────────────

def compute_composite(df):
    """
    Per-subtype z-score composite: z-normalize each metric within subtype,
    flip sign for Bhattacharyya (lower = better), then average.
    """
    _sign = {
        "fold_scie_mean": +1, "fold_nsi_mean": +1,
        "delta_bic_mean": +1, "mahalanobis_mean": +1,
        "bhattacharyya_mean": -1,                       # lower overlap = better
        "bimodality_coef_mean": +1, "silhouette_mean": +1,
    }
    df = df.copy()
    df["rank_subpop"] = np.nan
    for subtype in df["subtype"].unique():
        mask = df["subtype"] == subtype
        idx  = df.index[mask]
        z_sum   = np.zeros(mask.sum())
        n_valid = np.zeros(mask.sum())
        for col, sign in _sign.items():
            if col not in df.columns:
                continue
            vals = df.loc[idx, col].values.astype(float)
            fin  = np.isfinite(vals)
            if fin.sum() < 2:
                continue
            sd = np.nanstd(vals)
            if sd < 1e-10:
                continue
            z = sign * (vals - np.nanmean(vals)) / sd
            z_sum   += np.where(np.isfinite(z), z, 0)
            n_valid += fin.astype(float)
        df.loc[idx, "rank_subpop"] = np.where(
            n_valid > 0, z_sum / n_valid, np.nan
        )
    return df

# ── Load and score ────────────────────────────────────────────────────────────

def load_and_score(cfg):
    print(f"  Loading {cfg['h5ad']} …")
    adata = sc.read_h5ad(cfg["h5ad"])

    if not cfg["cc_in_obs"]:
        sg  = [g for g in S_GENES   if g in adata.var_names]
        g2m = [g for g in G2M_GENES if g in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=sg, g2m_genes=g2m)
        cfg["s_col"], cfg["g2m_col"] = "S_score", "G2M_score"

    imm = [g for g in IMMUNOGENIC if g in adata.var_names]
    print(f"  Immune genes found: {len(imm)}/{len(IMMUNOGENIC)}")
    sc.tl.score_genes(adata, imm, score_name="immune_score", use_raw=False)

    with open(cfg["sigfile"]) as f:
        sigs = json.load(f)
    sig_cols = []
    for name, genes in sigs.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 5:
            continue
        sc.tl.score_genes(adata, present, score_name=f"sig_{name}", use_raw=False)
        sig_cols.append(f"sig_{name}")

    pat_col = cfg["patient_col"]
    counts  = adata.obs[pat_col].value_counts()
    keep    = counts[counts >= MIN_CELLS].index
    adata   = adata[adata.obs[pat_col].isin(keep)].copy()
    print(f"  After ≥{MIN_CELLS} filter: {adata.n_obs:,} cells, "
          f"{adata.obs[pat_col].nunique()} patients")

    cols = ([cfg["subtype_col"], pat_col,
             cfg["s_col"], cfg["g2m_col"], "immune_score"] + sig_cols)
    df = adata.obs[cols].copy()
    del adata; gc.collect()
    return df, sig_cols

# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse(ds_name, cfg):
    print(f"\n{'='*65}\n {ds_name}\n{'='*65}")
    df_all, sig_cols = load_and_score(cfg)
    sub_col = cfg["subtype_col"]
    pat_col = cfg["patient_col"]

    rows = []
    for subtype in SUBTYPES + ["Global"]:
        df_sub = df_all if subtype == "Global" \
                 else df_all[df_all[sub_col] == subtype]
        n_pts  = df_sub[pat_col].nunique()

        if len(df_sub) < MIN_CELLS or n_pts < 2:
            print(f"\n  ── {subtype}: skipped (n_pts={n_pts})")
            continue
        print(f"\n  ── {subtype}: {len(df_sub):,} cells, {n_pts} patients")

        for sc_col in sig_cols:
            sig_name = sc_col[4:]
            print(f"    {sig_name:<35}", end="", flush=True)

            pat_df = per_patient_compute(df_sub, sc_col, pat_col)
            if pat_df.empty:
                print("  [no patients]")
                continue

            m = meta_analyze(pat_df)
            if not m:
                print("  [insufficient data]")
                continue

            m.update({"dataset": ds_name, "subtype": subtype,
                      "signature": sig_name, "n_patients": n_pts})
            rows.append(m)
            print(
                f"  SCIE={m.get('fold_scie_mean', np.nan):+.2f}"
                f"({m.get('fold_scie_pct15x', np.nan):.0%})"
                f"  NSI={m.get('fold_nsi_mean', np.nan):+.2f}"
                f"({m.get('fold_nsi_pct15x', np.nan):.0%})"
                f"  ΔBIC={m.get('delta_bic_mean', np.nan):+.1f}"
                f"  bimod={m.get('bimodality_coef_mean', np.nan):.3f}"
                f"  sil={m.get('silhouette_mean', np.nan):+.3f}"
            )

    if not rows:
        print(f"\n  No results for {ds_name}")
        return pd.DataFrame()

    result_df = compute_composite(pd.DataFrame(rows))

    os.makedirs(cfg["outdir"], exist_ok=True)
    out_csv = os.path.join(cfg["outdir"], f"subpop_metrics_{ds_name}.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    return result_df

# ── Plotting helpers ──────────────────────────────────────────────────────────

def _pivot(df, metric_col, sig_order, subtypes):
    """Pivot to signature×subtype matrix aligned to sig_order/subtypes."""
    if df.empty or metric_col not in df.columns:
        return np.full((len(sig_order), len(subtypes)), np.nan)
    pt = df.pivot_table(
        index="signature", columns="subtype",
        values=metric_col, aggfunc="first"
    ).reindex(index=sig_order, columns=subtypes)
    return pt.values.astype(float)


def _draw_heatmap(ax, data, row_labels, col_labels, cmap,
                  vmin=None, vmax=None, annot_data=None):
    """Draw a single (rows × cols) heatmap panel."""
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0, 1

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
    for xt, st in zip(ax.get_xticklabels(), col_labels):
        xt.set_color(SUBTYPE_COLORS.get(st, "k"))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            bg = (v - vmin) / (vmax - vmin + 1e-8)
            tc = "white" if bg > 0.65 else "black"
            txt = f"{v:.2f}"
            if annot_data is not None and np.isfinite(annot_data[i, j]):
                txt += f"\n{annot_data[i,j]:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=tc)

    return im


def _add_cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(label, fontsize=7)


# ── Figure 1: Enrichment heatmaps ────────────────────────────────────────────

def plot_enrichment_heatmaps(df, ds_name, outdir):
    sig_order  = [s for s in SIG_LABELS if s in df["signature"].values]
    sig_labels = [SIG_LABELS[s] for s in sig_order]
    subtypes   = [s for s in SUBTYPES + ["Global"] if s in df["subtype"].values]

    fig, axes = plt.subplots(1, 2, figsize=(3.8 * len(subtypes), 0.7 * len(sig_order) + 2),
                              squeeze=False)
    axes = axes[0]   # shape (2,)

    panels = [
        ("fold_scie_mean", "fold_scie_pct15x", "SCIE fold\n(stem-high × immune-low)",
         "Reds"),
        ("fold_nsi_mean",  "fold_nsi_pct15x",  "NSI fold\n(stem-low × immune-high)",
         "Blues"),
    ]

    for ax, (mcol, acol, title, cmap) in zip(axes, panels):
        data  = _pivot(df, mcol, sig_order, subtypes)
        annot = _pivot(df, acol, sig_order, subtypes)
        im = _draw_heatmap(ax, data, sig_labels, subtypes, cmap,
                           vmin=0.5, vmax=max(np.nanmax(data), 2.5),
                           annot_data=annot)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axhline(-0.5, color="white", lw=0)   # padding
        _add_cbar(fig, ax, im, "mean fold\n(% pts >1.5× annotated)")

    fig.suptitle(f"{ds_name} — SCIE & NSI quadrant enrichment (per-patient tertiles)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(outdir, f"subpop_enrichment_{ds_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: GMM metrics heatmaps ───────────────────────────────────────────

def plot_gmm_heatmaps(df, ds_name, outdir):
    sig_order  = [s for s in SIG_LABELS if s in df["signature"].values]
    sig_labels = [SIG_LABELS[s] for s in sig_order]
    subtypes   = [s for s in SUBTYPES + ["Global"] if s in df["subtype"].values]

    panels = [
        ("delta_bic_mean",    "ΔBIC\n(+ve = 2-comp preferred)",
         "PuRd",  None),
        ("mahalanobis_mean",  "Mahalanobis distance\n(higher = more separated)",
         "YlOrBr", None),
        ("bhattacharyya_mean","Bhattacharyya coeff\n(lower = less overlap)",
         "Blues_r", (0, 1)),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(3.8 * len(subtypes), 0.7 * len(sig_order) + 2),
        squeeze=False
    )
    axes = axes[0]

    for ax, (mcol, title, cmap, vrange) in zip(axes, panels):
        data = _pivot(df, mcol, sig_order, subtypes)
        vmin = vrange[0] if vrange else None
        vmax = vrange[1] if vrange else None
        im   = _draw_heatmap(ax, data, sig_labels, subtypes, cmap,
                             vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        _add_cbar(fig, ax, im, "mean across patients")

    fig.suptitle(f"{ds_name} — 2-component GMM metrics (per-patient z-scored data)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(outdir, f"subpop_gmm_{ds_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Bimodality + silhouette heatmaps ───────────────────────────────

def plot_bimod_sil_heatmaps(df, ds_name, outdir):
    sig_order  = [s for s in SIG_LABELS if s in df["signature"].values]
    sig_labels = [SIG_LABELS[s] for s in sig_order]
    subtypes   = [s for s in SUBTYPES + ["Global"] if s in df["subtype"].values]

    panels = [
        ("bimodality_coef_mean", None,
         "Bimodality coef\n(>0.555 = bimodal)", "YlOrRd", (0.3, 0.8)),
        ("silhouette_mean",       "silhouette_pct_pos",
         "Silhouette score\n(SCIE vs NSI clusters)", "Greens", (-0.3, 0.6)),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(3.8 * len(subtypes), 0.7 * len(sig_order) + 2),
        squeeze=False
    )
    axes = axes[0]

    for ax, (mcol, acol, title, cmap, vrange) in zip(axes, panels):
        data  = _pivot(df, mcol, sig_order, subtypes)
        annot = _pivot(df, acol, sig_order, subtypes) if acol else None
        im    = _draw_heatmap(ax, data, sig_labels, subtypes, cmap,
                              vmin=vrange[0], vmax=vrange[1],
                              annot_data=annot)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if acol:
            _add_cbar(fig, ax, im, "mean\n(% positive annotated)")
        else:
            # Add bimodality threshold line indicator
            _add_cbar(fig, ax, im, "mean")

    fig.suptitle(
        f"{ds_name} — Bimodality (stemness alone) & silhouette (SCIE vs NSI)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(outdir, f"subpop_bimod_sil_{ds_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Composite ranking bar chart ────────────────────────────────────

def plot_composite_rank(df, ds_name, outdir):
    sig_order  = [s for s in SIG_LABELS if s in df["signature"].values]
    sig_labels = [SIG_LABELS[s] for s in sig_order]
    subtypes   = [s for s in SUBTYPES + ["Global"] if s in df["subtype"].values]

    fig, axes = plt.subplots(1, len(subtypes), figsize=(4 * len(subtypes), 5),
                              sharey=True)
    if len(subtypes) == 1:
        axes = [axes]

    for ax, subtype in zip(axes, subtypes):
        sub = df[df["subtype"] == subtype].copy()
        sub = sub.set_index("signature").reindex(sig_order)
        rc  = sub["rank_subpop"].values.astype(float) \
              if "rank_subpop" in sub.columns else np.full(len(sig_order), np.nan)
        clrs = ["#2ca02c" if np.isfinite(v) and v > 0
                else "#d62728" if np.isfinite(v) else "lightgrey" for v in rc]
        ax.barh(range(len(sig_order)), rc, color=clrs, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(sig_order)))
        ax.set_yticklabels(sig_labels if subtype == subtypes[0] else [], fontsize=8)
        ax.set_title(subtype, fontsize=10, fontweight="bold",
                     color=SUBTYPE_COLORS.get(subtype, "k"))
        ax.set_xlabel("Composite z-score", fontsize=8)
        ax.set_xlim(-2.5, 2.5)
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(rc):
            if np.isfinite(v):
                ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=7)

    fig.suptitle(
        f"{ds_name} — Subpopulation composite ranking\n"
        r"↑ SCIE/NSI fold + ΔBIC + Mahalanobis + bimodality + silhouette − Bhattacharyya",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(outdir, f"subpop_ranking_{ds_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: SCIE × NSI scatter (both datasets) ─────────────────────────────

def plot_scie_nsi_scatter(all_results, outdir_base):
    """
    SCIE fold (x) vs NSI fold (y) per signature.
    Bubble size = % patients with SCIE fold > 1.5×.
    Ideal: top-right (both SCIE and NSI enriched simultaneously).
    """
    datasets   = list(all_results.keys())
    subtypes   = [s for ds_df in all_results.values()
                  for s in (SUBTYPES + ["Global"])
                  if not ds_df.empty and s in ds_df["subtype"].values]
    subtypes   = list(dict.fromkeys(subtypes))   # deduplicate, preserve order
    subtypes   = [s for s in SUBTYPES + ["Global"] if s in subtypes]

    sig_order  = list(SIG_LABELS.keys())
    _mkr       = dict(zip(sig_order, ["o","s","D","^","v","<",">","P","*"]))

    fig, axes = plt.subplots(
        len(datasets), len(subtypes),
        figsize=(3.8 * len(subtypes), 3.8 * len(datasets)),
        squeeze=False
    )

    for r, ds in enumerate(datasets):
        df = all_results[ds]
        for c, subtype in enumerate(subtypes):
            ax   = axes[r, c]
            color = SUBTYPE_COLORS.get(subtype, "grey")
            ax.axhline(1, color="grey", lw=0.7, ls="--", alpha=0.4)
            ax.axvline(1, color="grey", lw=0.7, ls="--", alpha=0.4)
            ax.axvspan(1, 4, ymin=0.45, alpha=0.05, color="green")

            if not df.empty and subtype in df["subtype"].values:
                sub = df[df["subtype"] == subtype]
                for sig in sig_order:
                    row = sub[sub["signature"] == sig]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    xv  = row.get("fold_scie_mean", np.nan)
                    yv  = row.get("fold_nsi_mean",  np.nan)
                    if not (np.isfinite(xv) and np.isfinite(yv)):
                        continue
                    pct = row.get("fold_scie_pct15x", 0.5)
                    sz  = max(float(pct) * 350, 20) if np.isfinite(pct) else 50
                    ax.scatter(xv, yv, s=sz, color=color,
                               marker=_mkr.get(sig, "o"),
                               alpha=0.85, zorder=3,
                               edgecolors="white", linewidths=0.5)
                    ax.annotate(
                        SIG_LABELS.get(sig, sig), (xv, yv),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                    )

            ax.set_xlim(0.4, 3.5)
            ax.set_ylim(0.4, 3.5)
            if r == len(datasets) - 1:
                ax.set_xlabel("SCIE fold enrichment", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{ds}\nNSI fold enrichment", fontsize=8,
                              fontweight="bold")
            if r == 0:
                ax.set_title(subtype, fontsize=10, fontweight="bold",
                             color=SUBTYPE_COLORS.get(subtype, "k"))
            ax.grid(alpha=0.2)

    fig.suptitle(
        "SCIE × NSI fold enrichment — both datasets\n"
        "Bubble = % patients with SCIE fold > 1.5×; "
        "top-right = both subpopulations enriched",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(outdir_base, "scie_nsi_scatter_both_datasets.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: Cross-dataset composite ranking ─────────────────────────────────

def plot_composite_both(all_results, outdir_base):
    sig_order  = list(SIG_LABELS.keys())
    sig_labels = [SIG_LABELS[s] for s in sig_order]
    datasets   = list(all_results.keys())
    subtypes   = SUBTYPES + ["Global"]

    fig, axes = plt.subplots(
        len(datasets), len(subtypes),
        figsize=(3.5 * len(subtypes), 3.0 * len(datasets)),
        sharey=True, squeeze=False
    )

    for r, ds in enumerate(datasets):
        df = all_results[ds]
        for c, subtype in enumerate(subtypes):
            ax  = axes[r, c]
            sub = df[df["subtype"] == subtype].copy() if not df.empty else pd.DataFrame()
            sub = sub.set_index("signature").reindex(sig_order) \
                  if not sub.empty else pd.DataFrame(index=sig_order)
            rc  = sub["rank_subpop"].values.astype(float) \
                  if ("rank_subpop" in sub.columns and not sub.empty) \
                  else np.full(len(sig_order), np.nan)
            clrs = ["#2ca02c" if np.isfinite(v) and v > 0
                    else "#d62728" if np.isfinite(v) else "lightgrey" for v in rc]
            ax.barh(range(len(sig_order)), rc, color=clrs, alpha=0.8)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks(range(len(sig_order)))
            ax.set_yticklabels(sig_labels if c == 0 else [], fontsize=8)
            if r == 0:
                ax.set_title(subtype, fontsize=10, fontweight="bold",
                             color=SUBTYPE_COLORS.get(subtype, "k"))
            if c == 0:
                ax.set_ylabel(ds, fontsize=9, fontweight="bold")
            ax.set_xlim(-2.5, 2.5)
            ax.grid(axis="x", alpha=0.3)
            for i, v in enumerate(rc):
                if np.isfinite(v):
                    ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}",
                            va="center", ha="left" if v >= 0 else "right", fontsize=6)

    fig.suptitle("Subpopulation composite ranking — both datasets",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(outdir_base, "subpop_ranking_both_datasets.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}

    for ds_name, cfg in DATASETS.items():
        result_df = analyse(ds_name, cfg)
        all_results[ds_name] = result_df

        if result_df.empty:
            continue

        print(f"\n  Generating figures for {ds_name} …")
        plot_enrichment_heatmaps(result_df, ds_name, cfg["outdir"])
        plot_gmm_heatmaps(result_df, ds_name, cfg["outdir"])
        plot_bimod_sil_heatmaps(result_df, ds_name, cfg["outdir"])
        plot_composite_rank(result_df, ds_name, cfg["outdir"])

        print(f"\n  ── Top signatures by subpop composite ({ds_name}) ──")
        for subtype in SUBTYPES + ["Global"]:
            sub = result_df[result_df["subtype"] == subtype].sort_values(
                "rank_subpop", ascending=False)
            if sub.empty:
                continue
            top = sub[["signature", "rank_subpop",
                        "fold_scie_mean", "fold_nsi_mean",
                        "delta_bic_mean", "bimodality_coef_mean",
                        "silhouette_mean"]].head(3)
            print(f"\n  {subtype}:")
            print(top.to_string(index=False))

        gc.collect()

    print("\n  Generating cross-dataset figures …")
    plot_scie_nsi_scatter(all_results, outdir_base="Visvader/figures/part1")
    plot_composite_both(all_results, outdir_base="Visvader/figures/part1")

    print("\n✓ Part 1c complete.")
