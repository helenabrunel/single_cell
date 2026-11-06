"""
10_TIM3_hypoxia_confident_positive.py
========================================
Sensitivity check on top of 08_TIM3_hypoxia_clean.py: what happens to the
HAVCR2(TIM3) x hypoxia result if we drop single-UMI "HAVCR2+" epithelial
cells before comparing groups, instead of using any raw count > 0 as the
positive definition?

WHY THIS EXISTS
----------------
Would removing HAVCR2+ cells with expression "close to 0"
clean up the depth artifact seen in earlier iterations (07-08)?
Diagnostic run first (see console output / project memory) found:

  compartment   %HAVCR2+   of those, %count==1   max count observed
  Immune        11.1%      70.5%                 13
  Stromal       1.4%       85.2%                 5
  Epithelial    0.36%      95.5%                 3

In immune cells HAVCR2 has real dynamic range (up to 13 UMIs), consistent
with genuine TIM3 expression. In epithelial cells, 425/445 "positive"
cells have exactly 1 UMI and the compartment-wide ceiling is 3 UMIs --
the signature of ambient RNA contamination from neighboring immune cells
rather than real epithelial expression.

NOTE ON MECHANICS: dropping low-count cells "before normalization" doesn't
change anything numerically here -- scanpy's per-cell normalization
(library-size scaling + log1p) is computed independently per cell, so a
retained cell's normalized value is identical regardless of which other
cells are excluded. The only real lever is which raw-count threshold
defines "positive": this script uses >=2 instead of >0, and drops the
count==1 cells entirely (they are neither confidently positive nor
negative) rather than folding them into HAVCR2-.

CAVEAT (read before interpreting results)
-------------------------------------------
Raising the threshold to raw count >= 2 leaves only 8 ER, 6 HER2, 6 TNBC
epithelial cells "confidently positive" (20 pooled). This is too small
for the magnitude-correlation analysis (HAVCR2 magnitude vs hypoxia,
within positive cells) to be interpretable per subtype -- a handful of
cells can flip a correlation sign. That panel is included for completeness
(pooled "All subtypes" only) but should be read as exploratory, not
confirmatory. The binary group comparison (confidently-positive vs
negative) is less fragile since the negative group is large (~121k),
but subtype-level n=6-8 positive cells is still a small-sample result --
treat any subtype-level significance with caution.

Output: figures/10_TIM3_confident_positive/<signature_name>/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, warnings, datetime
from scipy.sparse import issparse
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "compartment_epithelial.h5ad"
OUTDIR_BASE = "figures/10_TIM3_confident_positive"
os.makedirs(OUTDIR_BASE, exist_ok=True)

MIN_CONFIDENT_COUNT = 2  # raw count >= this => "confidently positive"; count==1 dropped

# Same signatures as 08_TIM3_hypoxia_clean.py
HYPOXIA_SIGNATURES = {
    "GO_0001666": list(dict.fromkeys([
        "HIF1A","EPAS1","HIF3A","ARNT","ARNT2","VEGFA","VEGFB","VEGFC","VEGFD","PGF",
        "LDHA","LDHB","PFKL","PFKM","PFKFB3","PFKFB4","SLC2A1","SLC2A3","HK1","HK2","ENO1","ENO2",
        "PDK1","PDK3","BNIP3","BNIP3L","CA9","CA12","ANGPT1","ANGPT2","ANGPTL4","EPO","EPOR",
        "NOS2","NOS3","HMOX1","HMOX2","ADM","ADM2","SERPINE1","SERPINE2","FN1","COL1A1","COL4A1",
        "MMP2","MMP9","MMP14","CXCR4","CXCL12","CXCL8","TNF","IL1A","IL1B","IL6","PTGS2",
        "TF","TFRC","FTH1","FTL","EGLN1","EGLN2","EGLN3","VHL","P4HA1","P4HA2","PLOD1","PLOD2","DDIT4","DDIT3",
        "ATF4","ATF3","SLC7A11","GPI","ALDOA","TPI1","GAPDH","PGK1","PGAM1","PKM","BCL2","BCL2L1","BAX","BAK1",
        "NOTCH1","NOTCH2","DLL4","JAG1","KDR","FLT1","TWIST1","SNAI1","SNAI2","ZEB1","VIM","CDH2",
        "MET","HGF","TGFB1","TGFB2","TGFBR1","MYC","PTEN","PIK3CA","AKT1","MTOR","MAPK1","MAPK3","STAT3","JAK1","JAK2",
        "NFE2L2","SOD1","SOD2","CAT","GPX1","GPX4",
    ])),
    "HALLMARK_HYPOXIA": list(dict.fromkeys([
        "ACKR3","ADM","ADORA2B","AK4","AKAP12","ALDOA","ALDOB","ALDOC","AMPD3","ANGPTL4",
        "ANKZF1","ANXA2","ATF3","ATP7A","B3GALT6","B4GALNT2","BCAN","BCL2","BGN","BHLHE40",
        "BNIP3L","BRS3","BTG1","CA12","CASP6","CAV1","CAVIN1","CAVIN3","CCN1","CCN2","CCN5",
        "CCNG2","CDKN1A","CDKN1B","CDKN1C","CHST2","CHST3","CITED2","COL5A1","CP","CSRP2",
        "CXCR4","DCN","DDIT3","DDIT4","DPYSL4","DTNA","DUSP1","EDN2","EFNA1","EFNA3","EGFR",
        "ENO1","ENO2","ENO3","ERO1A","ERRFI1","ETS1","EXT1","F3","FAM162A","FBP1","FOS","FOSL2",
        "FOXO3","GAA","GALK1","GAPDH","GAPDHS","GBE1","GCK","GCNT2","GLRX","GPC1","GPC3","GPC4",
        "GPI","GRHPR","GYS1","HAS1","HDLBP","HEXA","HK1","HK2","HMOX1","HOXB9","HS3ST1","HSPA5",
        "IDS","IER3","IGFBP1","IGFBP3","IL6","ILVBL","INHA","IRS2","ISG20","JMJD6","JUN","KDELR3",
        "KDM3A","KIF5A","KLF6","KLF7","KLHL24","LALBA","LARGE1","LDHA","LDHC","LOX","LXN","MAFF",
        "MAP3K1","MIF","MT1E","MT2A","MXI1","MYH9","NAGK","NCAN","NDRG1","NDST1","NDST2","NEDD4L",
        "NFIL3","NOCT","NR3C1","P4HA1","P4HA2","PAM","PCK1","PDGFB","PDK1","PDK3","PFKFB3","PFKL",
        "PFKP","PGAM2","PGF","PGK1","PGM1","PGM2","PHKG1","PIM1","PKLR","PKP1","PLAC8","PLAUR",
        "PLIN2","PNRC1","PPARGC1A","PPFIA4","PPP1R15A","PPP1R3C","PRDX5","PRKCA","PYGM","RBPJ",
        "RORA","RRAGD","S100A4","SAP30","SCARB1","SDC2","SDC3","SDC4","SELENBP1","SERPINE1",
        "SIAH2","SLC25A1","SLC2A1","SLC2A3","SLC2A5","SLC37A4","SLC6A6","SRPX","STBD1","STC1",
        "STC2","SULT2B1","TES","TGFB3","TGFBI","TGM2","TIPARP","TKTL1","TMEM45A","TNFAIP3","TPBG",
        "TPD52","TPI1","TPST2","UGP2","VEGFA","VHL","VLDLR","WSB1","XPNPEP1","ZFP36","ZNF292",
    ])),
}

SUBTYPES = ["ER", "HER2", "TNBC"]


def run_analysis_for_signature(sig_name, gene_list, adata):
    print(f"\n{'='*65}\n  Signature: {sig_name}\n{'='*65}")
    outdir = f"{OUTDIR_BASE}/{sig_name}"
    os.makedirs(outdir, exist_ok=True)

    hyp_genes = [g for g in gene_list if g in adata.var_names]
    print(f"  {len(hyp_genes)}/{len(gene_list)} signature genes found in data")
    sc.tl.score_genes(adata, hyp_genes, score_name="hypoxia_score", use_raw=False)
    hyp_raw = adata.obs["hypoxia_score"].values.astype(float)
    adata.obs["hypoxia_z"] = (hyp_raw - hyp_raw.mean()) / (hyp_raw.std() + 1e-8)

    df = pd.DataFrame({
        "hypoxia_z": adata.obs["hypoxia_z"].values,
        "subtype": adata.obs["subtype"].astype(str).values,
        "total_counts": adata.obs["total_counts"].values.astype(float),
        "HAVCR2_raw": adata.obs["HAVCR2_raw"].values,
    })
    df["log_total_counts"] = np.log(df["total_counts"])
    # 3-way split: drop count==1 (ambiguous) entirely from the group comparison
    df["HAVCR2_status"] = np.select(
        [df["HAVCR2_raw"] == 0, df["HAVCR2_raw"] >= MIN_CONFIDENT_COUNT],
        ["HAVCR2-", "HAVCR2+ (confident)"],
        default="ambiguous (count==1, dropped)",
    )
    df_clean = df[df["HAVCR2_status"] != "ambiguous (count==1, dropped)"].copy()
    n_dropped = (df["HAVCR2_status"] == "ambiguous (count==1, dropped)").sum()
    print(f"  Dropped {n_dropped:,} ambiguous count==1 cells; "
          f"{(df_clean['HAVCR2_status']=='HAVCR2+ (confident)').sum()} confidently positive "
          f"(raw count >= {MIN_CONFIDENT_COUNT}) remain")

    # ── Violin plots: HAVCR2- vs HAVCR2+ (confident), count==1 dropped ─────────
    print("  Generating violin plots (naive + depth-adjusted)...")
    groups = [("All subtypes", df_clean)] + [(st, df_clean[df_clean["subtype"] == st]) for st in SUBTYPES]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(
        f"Hypoxia in epithelial cells, HAVCR2+ (confident, count>={MIN_CONFIDENT_COUNT}) vs HAVCR2- ({sig_name})\n"
        f"count==1 cells dropped entirely (ambiguous / possible ambient noise)",
        fontsize=11, fontweight="bold")
    results = []
    for ax, (label, d) in zip(axes, groups):
        pos = d[d["HAVCR2_status"] == "HAVCR2+ (confident)"]["hypoxia_z"]
        neg = d[d["HAVCR2_status"] == "HAVCR2-"]["hypoxia_z"]
        low_power = len(pos) < 10
        if len(pos) < 3 or len(neg) < 3:
            ax.set_title(f"{label}\n(insufficient samples)", fontsize=9.5)
            continue

        u, p_naive = stats.mannwhitneyu(pos, neg, alternative="two-sided")

        d2 = d.copy()
        d2["HAVCR2_status"] = pd.Categorical(d2["HAVCR2_status"], categories=["HAVCR2-", "HAVCR2+ (confident)"])
        lm = smf.ols("hypoxia_z ~ HAVCR2_status + log_total_counts", data=d2).fit()
        p_adj = lm.pvalues.get("HAVCR2_status[T.HAVCR2+ (confident)]", np.nan)
        coef_adj = lm.params.get("HAVCR2_status[T.HAVCR2+ (confident)]", np.nan)

        results.append({
            "panel": label, "n_pos": len(pos), "n_neg": len(neg), "low_power_flag": low_power,
            "median_pos": pos.median(), "median_neg": neg.median(),
            "p_naive": p_naive, "coef_depth_adjusted": coef_adj, "p_depth_adjusted": p_adj,
        })

        parts = ax.violinplot([neg.values, pos.values], showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], ["#377EB8", "#E41A1C"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"HAVCR2-\n(n={len(neg):,})", f"HAVCR2+ conf.\n(n={len(pos):,})"], fontsize=9)
        ax.set_ylabel("Hypoxia score (z)", fontsize=9)
        sig_naive = "***" if p_naive < 0.001 else "**" if p_naive < 0.01 else "*" if p_naive < 0.05 else "ns"
        sig_adj = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        power_tag = "  [LOW POWER n<10]" if low_power else ""
        ax.set_title(
            f"{label}{power_tag}\nnaive p={p_naive:.2e} ({sig_naive})\n"
            f"depth-adjusted: p={p_adj:.2e} ({sig_adj})",
            fontsize=9, color=("darkorange" if low_power else "black")
        )
        ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#377EB8", alpha=0.65, label="HAVCR2-"),
                         Patch(facecolor="#E41A1C", alpha=0.65, label="HAVCR2+ (confident)")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f"{outdir}/violin_hypoxia_by_TIM3_status_confident.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir}/violin_hypoxia_by_TIM3_status_confident.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{outdir}/violin_stats_confident.csv", index=False)
    print(results_df.to_string(index=False))

    # ── Magnitude correlation, pooled only (subtype n too small) ──────────────
    print("\n  Generating naive vs depth-corrected correlation scatter (pooled only, n too small per subtype)...")
    hyp_resid_model = smf.ols("hypoxia_z ~ log_total_counts", data=df).fit()
    df["hypoxia_z_resid"] = hyp_resid_model.resid
    df["HAVCR2_norm"] = adata.obs["HAVCR2_norm"].values
    df["HAVCR2_depth_resid"] = adata.obs["HAVCR2_depth_resid"].values

    df_conf = df[df["HAVCR2_raw"] >= MIN_CONFIDENT_COUNT].copy()
    n_conf = len(df_conf)
    corr_rows = []
    if n_conf >= 5:
        r_naive, p_naive_c = stats.pearsonr(df_conf["hypoxia_z"], df_conf["HAVCR2_norm"])
        r_corr, p_corr_c = stats.pearsonr(df_conf["hypoxia_z_resid"], df_conf["HAVCR2_depth_resid"])
        corr_rows.append({"panel": "All subtypes (pooled)", "n": n_conf,
                           "r_naive": r_naive, "p_naive": p_naive_c,
                           "r_corrected": r_corr, "p_corrected": p_corr_c})

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(
            f"HAVCR2 magnitude vs hypoxia, confidently-positive cells only (count>={MIN_CONFIDENT_COUNT}, n={n_conf}) -- {sig_name}\n"
            "LOW POWER: pooled across subtypes, interpret with caution",
            fontsize=10.5, fontweight="bold", color="darkorange")
        for ax, (xcol, ycol, xlabel, ylabel, r, p, tag) in zip(axes, [
            ("hypoxia_z", "HAVCR2_norm", "Hypoxia score (z)", "HAVCR2 expression (normalized)", r_naive, p_naive_c, "naive"),
            ("hypoxia_z_resid", "HAVCR2_depth_resid", "Hypoxia score residual\n(depth-adjusted)", "HAVCR2 Pearson residual\n(depth-adjusted)", r_corr, p_corr_c, "depth-corrected"),
        ]):
            ax.scatter(df_conf[xcol], df_conf[ycol], s=30, alpha=0.75, color="#d62728",
                       edgecolors="white", linewidths=0.4)
            ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
            ax.axvline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
            if n_conf >= 3:
                z = np.polyfit(df_conf[xcol], df_conf[ycol], 1)
                xs = np.linspace(df_conf[xcol].min(), df_conf[xcol].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color="black", ls="--", lw=1.2, alpha=0.7)
            ax.set_title(f"{tag}: r={r:.3f}, p={p:.2e}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{outdir}/scatter_confident_positive_pooled.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outdir}/scatter_confident_positive_pooled.png")
    else:
        print(f"  Skipped correlation scatter: only n={n_conf} confidently-positive cells (need >=5)")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(f"{outdir}/correlation_confident_pooled.csv", index=False)

    return results_df, corr_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*65)
print(" 10 -- TIM3 x HYPOXIA -- CONFIDENT-POSITIVE SENSITIVITY CHECK")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

raw = adata.layers["counts"][:, adata.var_names.get_loc("HAVCR2")]
adata.obs["HAVCR2_raw"] = np.asarray(raw.todense()).flatten() if issparse(raw) else np.asarray(raw).flatten()
n_pos = (adata.obs["HAVCR2_raw"] > 0).sum()
n_conf = (adata.obs["HAVCR2_raw"] >= MIN_CONFIDENT_COUNT).sum()
n_singleton = ((adata.obs["HAVCR2_raw"] == 1)).sum()
print(f"  HAVCR2 raw count>0 : {n_pos} ({100*n_pos/adata.n_obs:.2f}%)")
print(f"    of which count==1 (dropped as ambiguous): {n_singleton} ({100*n_singleton/max(n_pos,1):.1f}%)")
print(f"    of which count>={MIN_CONFIDENT_COUNT} (confidently positive, kept): {n_conf}")
for st in SUBTYPES:
    m = adata.obs["subtype"].astype(str).values == st
    print(f"      {st}: raw>0={( (adata.obs['HAVCR2_raw'].values>0)&m).sum()}, "
          f"confident={((adata.obs['HAVCR2_raw'].values>=MIN_CONFIDENT_COUNT)&m).sum()}")

x = adata[:, "HAVCR2"].X
adata.obs["HAVCR2_norm"] = np.asarray(x.todense()).flatten() if issparse(x) else np.asarray(x).flatten()

print("\n  Fitting HAVCR2 depth-correction model (NB GLM, offset=log total_counts)...")
log_total_counts_all = np.log(adata.obs["total_counts"].values.astype(float))
havcr_model_df = pd.DataFrame({
    "havcr_raw": adata.obs["HAVCR2_raw"].values,
    "log_total_counts": log_total_counts_all,
})
havcr_model = smf.glm("havcr_raw ~ 1", data=havcr_model_df,
                       family=sm.families.NegativeBinomial(),
                       offset=havcr_model_df["log_total_counts"]).fit()
pred = havcr_model.predict(havcr_model_df, offset=havcr_model_df["log_total_counts"])
alpha = havcr_model.family.alpha if hasattr(havcr_model.family, "alpha") else 1.0
var = pred + alpha * pred**2
adata.obs["HAVCR2_depth_resid"] = ((havcr_model_df["havcr_raw"] - pred) / np.sqrt(var)).values
print(f"  Done (intercept={havcr_model.params['Intercept']:.4f})")

all_violin_results = {}
all_corr_results = {}
for sig_name, gene_list in HYPOXIA_SIGNATURES.items():
    v, c = run_analysis_for_signature(sig_name, gene_list, adata)
    all_violin_results[sig_name] = v
    all_corr_results[sig_name] = c

readme = [
    "10 -- TIM3 (HAVCR2) x Hypoxia -- confident-positive sensitivity check",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "MOTIVATION",
    "-"*60,
    "Follow-up on 08_TIM3_hypoxia_clean.py: does dropping single-UMI",
    "'HAVCR2+' epithelial cells (raw count==1, likely ambient RNA rather",
    "than real expression) clean up the result any further?",
    "",
    "DIAGNOSTIC THAT MOTIVATED THIS CHECK",
    "-"*60,
    "compartment   %HAVCR2+   of those %count==1   max count observed",
    "Immune        11.1%      70.5%                13",
    "Stromal       1.4%       85.2%                5",
    "Epithelial    0.36%      95.5%                3",
    "",
    "Immune cells show real dynamic range (up to 13 UMIs) -- consistent",
    "with genuine TIM3 expression. Epithelial 'positive' cells are almost",
    "all single-UMI with a compartment-wide ceiling of 3 -- consistent with",
    "ambient RNA contamination from neighboring immune cells, not real",
    "epithelial TIM3 expression.",
    "",
    "METHOD",
    "-"*60,
    f"Cells with raw HAVCR2 count==1 dropped entirely (neither positive nor",
    f"negative). 'Confidently positive' = raw count >= {MIN_CONFIDENT_COUNT}.",
    "Note: this does NOT require recomputing normalization -- scanpy's",
    "per-cell normalization is independent per cell, so dropping cells",
    "before/after normalization gives identical values for the cells kept.",
    "",
    "CAVEAT -- READ BEFORE TRUSTING ANY 'CONFIDENT POSITIVE' RESULT",
    "-"*60,
]
for st in SUBTYPES:
    m = adata.obs["subtype"].astype(str).values == st
    n_c = ((adata.obs["HAVCR2_raw"].values >= MIN_CONFIDENT_COUNT) & m).sum()
    readme.append(f"  {st}: only {n_c} confidently-positive epithelial cells")
readme += [
    f"  Pooled: only {n_conf} confidently-positive epithelial cells total",
    "",
    "This is too small for the magnitude-correlation panel to be reliable",
    "per subtype (a single outlier can flip r). Pooled correlation is shown",
    "for completeness but is exploratory, not confirmatory. The binary",
    "violin comparison is somewhat more robust (large negative group) but",
    "subtype-level positive n=6-8 should still be treated as low-power.",
    "",
    "SIGNATURES TESTED",
    "-"*60,
] + [f"  {name}: {len(genes)} genes" for name, genes in HYPOXIA_SIGNATURES.items()] + [
    "",
    "OUTPUT (per signature, in its own subfolder)",
    "-"*60,
    "  violin_hypoxia_by_TIM3_status_confident.png -- HAVCR2+ (conf.) vs HAVCR2-, count==1 dropped",
    "  violin_stats_confident.csv                  -- naive + depth-adjusted p per panel, low_power_flag",
    "  scatter_confident_positive_pooled.png       -- magnitude vs hypoxia, confident+ only, pooled",
    "  correlation_confident_pooled.csv            -- r/p, naive vs depth-corrected, pooled",
]
with open(f"{OUTDIR_BASE}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR_BASE}/README.txt")

print(f"""
============================================================
 DONE -- 10 confident-positive sensitivity check
============================================================
  Output: {OUTDIR_BASE}/
============================================================
""")
