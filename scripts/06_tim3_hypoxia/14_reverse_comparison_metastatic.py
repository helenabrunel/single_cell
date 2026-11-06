"""
06_TIM3_hypoxia_reverse_comparison.py
========================================
Klughammer counterpart of Visvader/12_TIM3_hypoxia_reverse_comparison.py:
reverse direction of 04_TIM3_hypoxia_clean.py -- instead of comparing
hypoxia score between HAVCR2+ and HAVCR2- cells, split cells by hypoxia
level (tertiles) and compare the % of HAVCR2+ cells between hypoxia-high
and hypoxia-low groups.

DESIGN (identical to Visvader/12, see that script for full rationale)
------------------------------------------------------------------------
1. Per panel (All subtypes / ER / HER2 / TNBC), split cells into tertiles
   of hypoxia_z computed within that panel's own cells. Hypoxia-high = top
   tertile, hypoxia-low = bottom tertile; middle tertile dropped.
2. % HAVCR2+ (raw count > 0) per group, with Wilson 95% CI.
3. Naive test: Fisher's exact on the 2x2 table (odds ratio + p).
4. Depth-adjusted test: logistic regression HAVCR2_positive ~
   hypoxia_group + log(total_counts), fit on high+low cells only.

Output (added alongside existing 04 files, none of which are touched):
  figures/04_TIM3_hypoxia_clean/<signature_name>/
    barplot_HAVCR2pct_by_hypoxia_group.png
    proportions_stats.csv

Run from: ~/single_cell/Klughammer/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.proportion as smprop
import os, warnings
from scipy.sparse import issparse
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "klughammer_Malignant.h5ad"
OUTDIR_BASE = "figures/04_TIM3_hypoxia_clean"  # same folders as 04 -- adds new files only

# Same signatures as 04_TIM3_hypoxia_clean.py
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


def run_for_signature(sig_name, gene_list, adata):
    print(f"\n{'='*65}\n  Signature: {sig_name}\n{'='*65}")
    outdir = f"{OUTDIR_BASE}/{sig_name}"
    os.makedirs(outdir, exist_ok=True)

    hyp_genes = [g for g in gene_list if g in adata.var_names]
    sc.tl.score_genes(adata, hyp_genes, score_name="hypoxia_score", use_raw=False)
    hyp_raw = adata.obs["hypoxia_score"].values.astype(float)
    hypoxia_z = (hyp_raw - hyp_raw.mean()) / (hyp_raw.std() + 1e-8)

    df = pd.DataFrame({
        "hypoxia_z": hypoxia_z,
        "subtype": adata.obs["subtype"].astype(str).values,
        "total_counts": adata.obs["total_counts"].values.astype(float),
        "HAVCR2_pos": (adata.obs["HAVCR2_raw"].values > 0).astype(int),
    })
    df["log_total_counts"] = np.log(df["total_counts"])

    groups = [("All subtypes", df)] + [(st, df[df["subtype"] == st]) for st in SUBTYPES]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(
        f"% HAVCR2+ cells by hypoxia group (tertiles), Malignant cells ({sig_name}) -- Klughammer\n"
        "hypoxia-low = bottom tertile, hypoxia-high = top tertile (middle tertile dropped)",
        fontsize=11, fontweight="bold")
    results = []
    for ax, (label, d) in zip(axes, groups):
        q1, q2 = np.percentile(d["hypoxia_z"], [100/3, 200/3])
        d2 = d.copy()
        d2["hypoxia_group"] = np.select(
            [d2["hypoxia_z"] <= q1, d2["hypoxia_z"] >= q2],
            ["low", "high"], default="mid",
        )
        low = d2[d2["hypoxia_group"] == "low"]
        high = d2[d2["hypoxia_group"] == "high"]

        if len(low) < 20 or len(high) < 20:
            ax.set_title(f"{label}\n(insufficient samples)", fontsize=9.5)
            continue

        n_low, pos_low = len(low), low["HAVCR2_pos"].sum()
        n_high, pos_high = len(high), high["HAVCR2_pos"].sum()
        pct_low, pct_high = 100 * pos_low / n_low, 100 * pos_high / n_high
        ci_low = smprop.proportion_confint(pos_low, n_low, method="wilson")
        ci_high = smprop.proportion_confint(pos_high, n_high, method="wilson")

        table = [[pos_high, n_high - pos_high], [pos_low, n_low - pos_low]]
        odds_ratio, p_naive = stats.fisher_exact(table)

        d_hl = pd.concat([low, high]).copy()
        d_hl["hypoxia_group"] = pd.Categorical(d_hl["hypoxia_group"], categories=["low", "high"])
        glm = smf.glm("HAVCR2_pos ~ hypoxia_group + log_total_counts", data=d_hl,
                       family=sm.families.Binomial()).fit()
        coef_adj = glm.params.get("hypoxia_group[T.high]", np.nan)
        p_adj = glm.pvalues.get("hypoxia_group[T.high]", np.nan)
        or_adj = np.exp(coef_adj)

        results.append({
            "panel": label, "n_low": n_low, "n_high": n_high,
            "pct_HAVCR2pos_low": pct_low, "pct_HAVCR2pos_high": pct_high,
            "odds_ratio_naive": odds_ratio, "p_naive": p_naive,
            "odds_ratio_depth_adjusted": or_adj, "p_depth_adjusted": p_adj,
        })

        bar_x = [0, 1]
        heights = [pct_low, pct_high]
        errs_low = np.clip([pct_low - 100*ci_low[0], pct_high - 100*ci_high[0]], 0, None)
        errs_high = np.clip([100*ci_low[1] - pct_low, 100*ci_high[1] - pct_high], 0, None)
        ax.bar(bar_x, heights, color=["#377EB8", "#E41A1C"], alpha=0.75, width=0.6,
               yerr=[errs_low, errs_high], capsize=5, ecolor="black")
        ax.set_xticks(bar_x)
        ax.set_xticklabels([f"hypoxia-low\n(n={n_low:,})", f"hypoxia-high\n(n={n_high:,})"], fontsize=9)
        ax.set_ylabel("% HAVCR2+ cells", fontsize=9)
        sig_naive = "***" if p_naive < 0.001 else "**" if p_naive < 0.01 else "*" if p_naive < 0.05 else "ns"
        sig_adj = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        ax.set_title(
            f"{label}\nnaive OR={odds_ratio:.2f}, p={p_naive:.2e} ({sig_naive})\n"
            f"depth-adj. OR={or_adj:.2f}, p={p_adj:.2e} ({sig_adj})",
            fontsize=9
        )
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{outdir}/barplot_HAVCR2pct_by_hypoxia_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir}/barplot_HAVCR2pct_by_hypoxia_group.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{outdir}/proportions_stats.csv", index=False)
    print(results_df.to_string(index=False))
    return results_df


print("="*65)
print(" 06 -- TIM3 x HYPOXIA -- KLUGHAMMER -- REVERSE COMPARISON")
print("="*65)
print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)

havcr2_idx = adata.var_names.get_loc("HAVCR2")
raw = adata.raw.X[:, havcr2_idx]
adata.obs["HAVCR2_raw"] = np.asarray(raw.todense()).flatten() if issparse(raw) else np.asarray(raw).flatten()

for sig_name, gene_list in HYPOXIA_SIGNATURES.items():
    run_for_signature(sig_name, gene_list, adata)

print(f"\nDone. New files added to {OUTDIR_BASE}/<signature>/ (existing 04 files untouched).")
