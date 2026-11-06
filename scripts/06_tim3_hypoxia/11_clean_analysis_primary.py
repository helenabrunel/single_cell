"""
08_TIM3_hypoxia_clean.py
==========================
Clean restart of the TIM3 (HAVCR2) x hypoxia analysis in epithelial cells,
after the previous G1/G2 threshold-based design (07d-07g) turned out to
be confounded with sequencing depth, and the follow-up analyses (07f-07j)
became too layered/confusing to present.

DESIGN
------
1. Simple binary split: HAVCR2+ (raw count > 0) vs HAVCR2- (raw count = 0).
   No magnitude threshold -- avoids the depth artifact from normalizing a
   single-UMI-detection gene (see figures/07f_depth_adjusted/ for why that
   broke the original design).
2. Violin plots of hypoxia score by HAVCR2 status, per subtype. Each panel
   is annotated with both:
     - naive Mann-Whitney p-value
     - depth-adjusted p-value, from OLS: hypoxia_score ~ HAVCR2_status +
       log(total_counts) -- so the reader sees both the simple answer and
       whether it holds up once sequencing depth is accounted for.
3. Naive vs depth-corrected correlation scatter, within HAVCR2+ cells only:
   does the MAGNITUDE of HAVCR2 (not just detected/not) correlate with
   hypoxia score, per subtype -- shown both naively and after correcting
   both axes for per-cell sequencing depth (folded in from 07j, which
   showed this naive correlation is negative and mostly a depth artifact:
   it flips sign or disappears once corrected).
4. UMAP overview: subtype context, hypoxia score, and HAVCR2+ cells
   highlighted, so the biology can be inspected spatially.

DEPTH CORRECTION (for point 3)
----------------------------------------
  Hypoxia axis   : residuals of OLS(hypoxia_z ~ log(total_counts))
  HAVCR2 axis    : Pearson residuals, NB GLM HAVCR2_raw ~ offset(log(total_counts))
  Both models fit on all epithelial cells.

HYPOXIA SIGNATURES
--------------------
Add signatures to HYPOXIA_SIGNATURES below (name -> gene list) -- the
rest of the script runs once per signature automatically, no other
changes needed.

Output: figures/08_TIM3_hypoxia_clean/<signature_name>/
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
OUTDIR_BASE = "figures/08_TIM3_hypoxia_clean"
os.makedirs(OUTDIR_BASE, exist_ok=True)

# ── Hypoxia signatures ──────────────────────────────────────────────────────
# Add further signatures here as additional dict entries.
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
    # MSigDB signatures.
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
    "GAVISH_3CA_MALIGNANT_METAPROGRAM_6_HYPOXIA": list(dict.fromkeys([
        "ADM","AK4","AKAP12","ALDOA","ANGPTL4","ANKRD37","ARLN","BNIP3","BNIP3L","CA12","CA9",
        "CAV1","DDIT3","DDIT4","DNAJB9","EGLN3","ENO1","ENO2","EPB41L4A-AS1","ERO1A","ERRFI1",
        "FAM162A","GBE1","GPI","HILPDA","HK2","IGFBP3","IGFBP5","INSIG2","LDHA","LGALS3","MT1X",
        "MT2A","NDRG1","NDUFA4L2","NRN1","P4HA1","PDK1","PFKFB3","PFKP","PGK1","PLOD2","S100A10",
        "SLC16A3","SLC2A1","SLC3A2","TMEM45A","VEGFA","VIM","WSB1",
    ])),
    "GOBP_REGULATION_OF_CELLULAR_RESPONSE_TO_HYPOXIA": list(dict.fromkeys([
        "AJUBA","CHCHD2","COMMD1","DDAH1","DNMT3A","DRD2","EGLN1","ENO1","EPHA4","GRM5","HYOU1",
        "KCNK2","MAP2K1","MIR145","MIR21","MIR210","NFE2L2","NOL3","PIK3CB","PINK1","ROCK2",
        "STAT3","TMBIM6","USP19","VHL",
    ])),
    "GOBP_HIF1A_SIGNALING_PATHWAY": list(dict.fromkeys([
        "COMMD1","CYBB","EGLN1","HIF1A","MIR210","PDK1","PDK3","RWDD3",
    ])),
    "GOBP_NEG_REG_HYPOXIA_INDUCED_APOPTOTIC_SIGNALING": list(dict.fromkeys([
        "ENO1","HYOU1","MAP2K1","NOL3","PIK3CB","PINK1","TMBIM6",
    ])),
    "GOBP_POS_REG_CELLULAR_RESPONSE_TO_HYPOXIA": list(dict.fromkeys([
        "DNMT3A","GRM5","KCNK2","MIR145","MIR21",
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
        "HAVCR2_status": np.where(adata.obs["HAVCR2_raw"].values > 0, "HAVCR2+", "HAVCR2-"),
    })
    df["log_total_counts"] = np.log(df["total_counts"])

    # ── Violin plots ─────────────────────────────────────────────────────────
    print("  Generating violin plots (naive + depth-adjusted)...")
    groups = [("All subtypes", df)] + [(st, df[df["subtype"] == st]) for st in SUBTYPES]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(f"Hypoxia in epithelial cells, HAVCR2+ vs HAVCR2- ({sig_name})",
                 fontsize=12, fontweight="bold")
    results = []
    for ax, (label, d) in zip(axes, groups):
        pos = d[d["HAVCR2_status"] == "HAVCR2+"]["hypoxia_z"]
        neg = d[d["HAVCR2_status"] == "HAVCR2-"]["hypoxia_z"]
        if len(pos) < 3 or len(neg) < 3:
            ax.set_title(f"{label}\n(insufficient samples)", fontsize=9.5)
            continue

        # naive Mann-Whitney
        u, p_naive = stats.mannwhitneyu(pos, neg, alternative="two-sided")

        # depth-adjusted: OLS with log(total_counts) as covariate
        d2 = d.copy()
        d2["HAVCR2_status"] = pd.Categorical(d2["HAVCR2_status"], categories=["HAVCR2-", "HAVCR2+"])
        lm = smf.ols("hypoxia_z ~ HAVCR2_status + log_total_counts", data=d2).fit()
        p_adj = lm.pvalues.get("HAVCR2_status[T.HAVCR2+]", np.nan)
        coef_adj = lm.params.get("HAVCR2_status[T.HAVCR2+]", np.nan)

        results.append({
            "panel": label, "n_pos": len(pos), "n_neg": len(neg),
            "median_pos": pos.median(), "median_neg": neg.median(),
            "p_naive": p_naive, "coef_depth_adjusted": coef_adj, "p_depth_adjusted": p_adj,
        })

        parts = ax.violinplot([neg.values, pos.values], showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], ["#377EB8", "#E41A1C"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"HAVCR2-\n(n={len(neg):,})", f"HAVCR2+\n(n={len(pos):,})"], fontsize=9)
        ax.set_ylabel("Hypoxia score (z)", fontsize=9)
        sig_naive = "***" if p_naive < 0.001 else "**" if p_naive < 0.01 else "*" if p_naive < 0.05 else "ns"
        sig_adj = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        ax.set_title(
            f"{label}\nnaive p={p_naive:.2e} ({sig_naive})\n"
            f"depth-adjusted: p={p_adj:.2e} ({sig_adj})",
            fontsize=9
        )
        ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#377EB8", alpha=0.65, label="HAVCR2-"),
                         Patch(facecolor="#E41A1C", alpha=0.65, label="HAVCR2+")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f"{outdir}/violin_hypoxia_by_TIM3_status.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir}/violin_hypoxia_by_TIM3_status.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{outdir}/violin_stats.csv", index=False)
    print(results_df.to_string(index=False))

    # ── Naive vs depth-corrected correlation, within HAVCR2+ cells only ───────
    # Does the MAGNITUDE of HAVCR2 correlate with hypoxia, among cells that
    # express any HAVCR2 at all? Shown naively and depth-corrected, since the
    # naive version is confounded (see figures/07f_depth_adjusted/): a
    # shallow cell gets both an inflated HAVCR2 value and a deflated
    # hypoxia_z, which can manufacture a spurious negative correlation.
    print("\n  Generating naive vs depth-corrected correlation scatter...")
    hyp_resid_model = smf.ols("hypoxia_z ~ log_total_counts",
                               data=df.assign(log_total_counts=np.log(adata.obs["total_counts"].values))).fit()
    df["hypoxia_z_resid"] = hyp_resid_model.resid
    df["HAVCR2_norm"] = adata.obs["HAVCR2_norm"].values
    df["HAVCR2_depth_resid"] = adata.obs["HAVCR2_depth_resid"].values

    df_pos = df[df["HAVCR2_status"] == "HAVCR2+"].copy()

    def make_correlation_panel_plot(d, xcol, ycol, xlabel, ylabel, title, fname):
        panels = [("All HAVCR2+", d)] + [(f"{st} HAVCR2+", d[d["subtype"] == st]) for st in SUBTYPES]
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        colors = {"All HAVCR2+": "#4C72B0", "ER HAVCR2+": "#2ca02c",
                  "HER2 HAVCR2+": "#9467bd", "TNBC HAVCR2+": "#d62728"}
        rows = []
        for ax, (label, dd) in zip(axes, panels):
            if len(dd) < 3:
                ax.set_title(f"{label} (n={len(dd)})", fontsize=10)
                continue
            r, p = stats.pearsonr(dd[xcol], dd[ycol])
            rows.append({"panel": label, "r": r, "p": p, "n": len(dd)})
            ax.scatter(dd[xcol], dd[ycol], s=14, alpha=0.6, color=colors[label],
                       edgecolors="white", linewidths=0.3)
            ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
            ax.axvline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
            z = np.polyfit(dd[xcol], dd[ycol], 1)
            xs = np.linspace(dd[xcol].min(), dd[xcol].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="black", ls="--", lw=1.2, alpha=0.7)
            ax.set_title(f"{label} (n={len(dd)})\nr={r:.3f}, p={p:.2e}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outdir}/{fname}")
        return pd.DataFrame(rows)

    stats_naive = make_correlation_panel_plot(
        df_pos, "hypoxia_z", "HAVCR2_norm",
        "Hypoxia score (z)", "HAVCR2 expression (normalized)",
        f"NAIVE -- Hypoxia vs HAVCR2 magnitude, HAVCR2+ cells only ({sig_name})\n"
        "Not corrected for sequencing depth",
        "scatter_TIM3_hypoxia_naive.png"
    )
    stats_corrected = make_correlation_panel_plot(
        df_pos, "hypoxia_z_resid", "HAVCR2_depth_resid",
        "Hypoxia score residual\n(depth-adjusted)", "HAVCR2 Pearson residual\n(depth-adjusted)",
        f"DEPTH-CORRECTED -- Hypoxia vs HAVCR2 magnitude, HAVCR2+ cells only ({sig_name})\n"
        "Both axes adjusted for per-cell sequencing depth",
        "scatter_TIM3_hypoxia_depth_corrected.png"
    )
    corr_comparison = stats_naive.merge(stats_corrected, on=["panel", "n"], suffixes=("_naive", "_corrected"))
    corr_comparison = corr_comparison[["panel", "n", "r_naive", "p_naive", "r_corrected", "p_corrected"]]
    corr_comparison.to_csv(f"{outdir}/correlation_comparison.csv", index=False)
    print(corr_comparison.to_string(index=False))

    # ── UMAP overview ────────────────────────────────────────────────────────
    print("\n  Generating UMAP overview...")
    umap = adata.obsm["X_umap"]
    pos_mask = adata.obs["HAVCR2_raw"].values > 0

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(f"UMAP -- epithelial cells ({sig_name})", fontsize=12, fontweight="bold")

    ax = axes[0]
    subtype_colors = {"ER": "#4C72B0", "HER2": "#DD8452", "TNBC": "#55A868"}
    for st in SUBTYPES:
        m = adata.obs["subtype"].astype(str).values == st
        ax.scatter(umap[m, 0], umap[m, 1], s=2, alpha=0.4, color=subtype_colors[st], label=st,
                   rasterized=True)
    ax.set_title("Subtype", fontsize=10.5)
    ax.legend(fontsize=8.5, markerscale=4, loc="best")
    ax.set_xlabel("UMAP1", fontsize=9); ax.set_ylabel("UMAP2", fontsize=9)

    ax = axes[1]
    sc_ = ax.scatter(umap[:, 0], umap[:, 1], s=2, c=adata.obs["hypoxia_z"].values,
                      cmap="viridis", alpha=0.6, rasterized=True)
    plt.colorbar(sc_, ax=ax, label="Hypoxia score (z)", fraction=0.046, pad=0.04)
    ax.set_title("Hypoxia score", fontsize=10.5)
    ax.set_xlabel("UMAP1", fontsize=9); ax.set_ylabel("UMAP2", fontsize=9)

    ax = axes[2]
    ax.scatter(umap[~pos_mask, 0], umap[~pos_mask, 1], s=2, color="lightgray", alpha=0.4,
               label=f"HAVCR2- (n={(~pos_mask).sum():,})", rasterized=True)
    ax.scatter(umap[pos_mask, 0], umap[pos_mask, 1], s=14, color="#E41A1C", alpha=0.9,
               edgecolors="black", linewidths=0.3, label=f"HAVCR2+ (n={pos_mask.sum():,})",
               zorder=5)
    ax.set_title("HAVCR2+ cells", fontsize=10.5)
    ax.legend(fontsize=8.5, markerscale=2, loc="best")
    ax.set_xlabel("UMAP1", fontsize=9); ax.set_ylabel("UMAP2", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{outdir}/umap_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir}/umap_overview.png")

    return results_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*65)
print(" 08 -- TIM3 x HYPOXIA -- CLEAN RESTART")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

raw = adata.layers["counts"][:, adata.var_names.get_loc("HAVCR2")]
adata.obs["HAVCR2_raw"] = np.asarray(raw.todense()).flatten() if issparse(raw) else np.asarray(raw).flatten()
n_pos = (adata.obs["HAVCR2_raw"] > 0).sum()
print(f"  HAVCR2+ cells: {n_pos} ({100*n_pos/adata.n_obs:.2f}%)")

x = adata[:, "HAVCR2"].X
adata.obs["HAVCR2_norm"] = np.asarray(x.todense()).flatten() if issparse(x) else np.asarray(x).flatten()

# HAVCR2 depth-correction model, fit once on all epithelial cells (does not
# depend on the hypoxia signature, so it's shared across signatures below).
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
# .values on the RHS -- havcr_model_df has a default RangeIndex while
# adata.obs is indexed by cell barcode, so a plain Series assignment here
# would align by index label (all-NaN) instead of by row position.
adata.obs["HAVCR2_depth_resid"] = ((havcr_model_df["havcr_raw"] - pred) / np.sqrt(var)).values
print(f"  Done (intercept={havcr_model.params['Intercept']:.4f})")

all_results = {}
for sig_name, gene_list in HYPOXIA_SIGNATURES.items():
    all_results[sig_name] = run_analysis_for_signature(sig_name, gene_list, adata)

readme = [
    "08 -- TIM3 (HAVCR2) x Hypoxia -- clean restart",
    f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Input: {H5AD_FILE}",
    "="*60,
    "",
    "DESIGN",
    "-"*60,
    "HAVCR2+ (raw count > 0) vs HAVCR2- (raw count = 0) -- simple binary",
    "split, no magnitude threshold (avoids the depth artifact from the",
    "previous G1/G2 design -- see figures/07f_depth_adjusted/ for why).",
    "",
    "Each violin panel shows both a naive Mann-Whitney p-value and a",
    "depth-adjusted p-value (OLS: hypoxia_z ~ HAVCR2_status + log(total_counts)),",
    "so it's clear whether the result holds up once sequencing depth is",
    "accounted for.",
    "",
    "Within HAVCR2+ cells only, also compares whether the MAGNITUDE of",
    "HAVCR2 correlates with hypoxia, naively vs depth-corrected (both axes",
    "adjusted for per-cell sequencing depth). The naive version tends to be",
    "negative and depth-driven -- corrected, it's much weaker or flips sign.",
    "",
    "HYPOXIA SIGNATURES TESTED",
    "-"*60,
] + [f"  {name}: {len(genes)} genes" for name, genes in HYPOXIA_SIGNATURES.items()] + [
    "",
    "To add a new signature: edit HYPOXIA_SIGNATURES at the top of",
    "08_TIM3_hypoxia_clean.py and re-run -- no other changes needed.",
    "",
    "OUTPUT (per signature, in its own subfolder)",
    "-"*60,
    "  violin_hypoxia_by_TIM3_status.png        -- main result (HAVCR2+ vs HAVCR2-)",
    "  violin_stats.csv                         -- naive + depth-adjusted p per panel",
    "  scatter_TIM3_hypoxia_naive.png           -- HAVCR2 magnitude vs hypoxia, HAVCR2+ only",
    "  scatter_TIM3_hypoxia_depth_corrected.png -- same, both axes depth-adjusted",
    "  correlation_comparison.csv               -- r/p side by side, naive vs corrected",
    "  umap_overview.png                        -- subtype / hypoxia score / HAVCR2+ cells",
]
with open(f"{OUTDIR_BASE}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR_BASE}/README.txt")

print(f"""
============================================================
 DONE -- 08 clean restart
============================================================
  Output: {OUTDIR_BASE}/
============================================================
""")
