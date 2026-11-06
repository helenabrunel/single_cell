"""
05_TIM3_hypoxia_clipped_violins.py
=====================================
Display-only follow-up to 04_TIM3_hypoxia_clean.py (Klughammer counterpart
of Visvader/11_TIM3_hypoxia_clipped_violins.py): same data, same hypoxia
scores, same HAVCR2+/- groups, same stats -- the only change is the violin
plot's y-axis, clipped to the pooled 1st-99th percentile of hypoxia_z
(with small padding) instead of the full range.

WHY: same check as on Visvader -- the "pinched near 0" look in the 04
violins is not zero-inflation or a residual depth confound in the hypoxia
score (verified on Visvader: 0 cells with score exactly 0; depth-corrected
residual violin virtually identical in shape to the raw one). It's a
handful of extreme-outlier cells forcing the y-axis wide. Clipping the
axis for display only -- no cells removed, stats unchanged.

Does NOT overwrite 04's output. Original violins remain at
figures/04_TIM3_hypoxia_clean/<signature>/violin_hypoxia_by_TIM3_status.png

Output: figures/05_TIM3_hypoxia_clipped/<signature_name>/violin_clipped.png
Run from: ~/single_cell/Klughammer/
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, warnings
from scipy.sparse import issparse
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE = "klughammer_Malignant.h5ad"
OUTDIR_BASE = "figures/05_TIM3_hypoxia_clipped"
os.makedirs(OUTDIR_BASE, exist_ok=True)

PCTILE_LO, PCTILE_HI = 1, 99
PAD_FRAC = 0.15

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


def clipped_ylim(values):
    lo, hi = np.percentile(values, [PCTILE_LO, PCTILE_HI])
    span = hi - lo
    return lo - PAD_FRAC * span, hi + PAD_FRAC * span


def run_for_signature(sig_name, gene_list, adata):
    print(f"  Signature: {sig_name}")
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
        "HAVCR2_status": np.where(adata.obs["HAVCR2_raw"].values > 0, "HAVCR2+", "HAVCR2-"),
    })
    df["log_total_counts"] = np.log(df["total_counts"])

    groups = [("All subtypes", df)] + [(st, df[df["subtype"] == st]) for st in SUBTYPES]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(
        f"Hypoxia in Malignant cells, HAVCR2+ vs HAVCR2- ({sig_name}) -- Klughammer\n"
        f"y-axis clipped to pooled {PCTILE_LO}-{PCTILE_HI}th percentile (display only -- no cells removed, stats unchanged)",
        fontsize=11, fontweight="bold")
    for ax, (label, d) in zip(axes, groups):
        pos = d[d["HAVCR2_status"] == "HAVCR2+"]["hypoxia_z"]
        neg = d[d["HAVCR2_status"] == "HAVCR2-"]["hypoxia_z"]
        if len(pos) < 3 or len(neg) < 3:
            ax.set_title(f"{label}\n(insufficient samples)", fontsize=9.5)
            continue

        u, p_naive = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        d2 = d.copy()
        d2["HAVCR2_status"] = pd.Categorical(d2["HAVCR2_status"], categories=["HAVCR2-", "HAVCR2+"])
        lm = smf.ols("hypoxia_z ~ HAVCR2_status + log_total_counts", data=d2).fit()
        p_adj = lm.pvalues.get("HAVCR2_status[T.HAVCR2+]", np.nan)

        parts = ax.violinplot([neg.values, pos.values], showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], ["#377EB8", "#E41A1C"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"HAVCR2-\n(n={len(neg):,})", f"HAVCR2+\n(n={len(pos):,})"], fontsize=9)
        ax.set_ylabel("Hypoxia score (z)", fontsize=9)
        ax.set_ylim(*clipped_ylim(d["hypoxia_z"].values))
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
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{outdir}/violin_clipped.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {outdir}/violin_clipped.png")


print("="*65)
print(" 05 -- TIM3 x HYPOXIA -- KLUGHAMMER -- CLIPPED-AXIS VIOLINS (display only)")
print("="*65)
print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)

# Raw counts live in adata.raw.X (Ensembl-indexed, positionally aligned
# with adata.var_names via adata.var['ensembl_id'] -- see 04's docstring).
havcr2_idx = adata.var_names.get_loc("HAVCR2")
raw = adata.raw.X[:, havcr2_idx]
adata.obs["HAVCR2_raw"] = np.asarray(raw.todense()).flatten() if issparse(raw) else np.asarray(raw).flatten()

for sig_name, gene_list in HYPOXIA_SIGNATURES.items():
    run_for_signature(sig_name, gene_list, adata)

print(f"\nDone. Output: {OUTDIR_BASE}/  (original 04 figures untouched)")
