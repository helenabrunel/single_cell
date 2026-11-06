"""
01_build_signatures.py
========================
Builds and saves the 3 gene signatures needed for the "LCOR program" /
T-cell phenotype analysis, from the lab's own experimental data:

1. LCOR_OE_malignant (top 50 genes by DESeq2 Wald statistic, log2FC>0,
   padj<0.05, from results_MDA_MB231.xlsx -- LCOR-OE vs control in
   MDA-MB-231 cells). LCOR and CXCL16 excluded: LCOR because it's the
   perturbed gene itself, CXCL16 because it will be used as a downstream
   readout later (avoid circularity).
2. Tcell_UP24H (117 genes, "FIRMAS Tcells induced by LCOR.xlsx", early/
   activation-heavy response -- includes many IEG/proliferation/metabolic
   genes, use with caution as it may partly reflect generic activation).
3. Tcell_UP96H (91 genes, same file, later/differentiated response --
   includes CXCR6 and ITGAE(CD103), the markers the working hypothesis
   centers on; likely the "91-gene Fig 2j" signature).

Output: LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/01_build_signatures/
    signatures.json
"""
import pandas as pd
import json
import os

OUTDIR = "LCOR-CXCL16/10_lcor_program_tcell_phenotype/figures/01_build_signatures"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================================
# 1. LCOR-OE malignant signature
# ============================================================================
df = pd.read_excel("/home/hbrunel/single_cell/results_MDA_MB231.xlsx",
                    sheet_name="results_annotated_DF_MDA_Lcorvs")
df.columns = ["symbol", "ensembl_id", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
df = df.dropna(subset=["symbol", "log2FoldChange", "padj", "stat"])
exclude = {"LCOR", "CXCL16"}
up = df[(~df["symbol"].isin(exclude)) & (df["log2FoldChange"] > 0) & (df["padj"] < 0.05)]
up = up.sort_values("stat", ascending=False)
lcor_oe_signature = up["symbol"].head(50).tolist()
print(f"LCOR_OE_malignant: {len(lcor_oe_signature)} genes")
print(lcor_oe_signature)

# ============================================================================
# 2/3. T-cell signatures (24h, 96h)
# ============================================================================
tdf = pd.read_excel("/home/hbrunel/single_cell/FIRMAS Tcells induced by LCOR.xlsx",
                     sheet_name="Hoja1", header=None)
tdf.columns = ["up24h", "up96h"]
tdf = tdf.iloc[1:]
tcell_24h = tdf["up24h"].dropna().astype(str).str.strip().tolist()
tcell_96h = tdf["up96h"].dropna().astype(str).str.strip().tolist()
print(f"\nTcell_UP24H: {len(tcell_24h)} genes")
print(f"Tcell_UP96H: {len(tcell_96h)} genes")
print(f"  Includes CXCR6: {'CXCR6' in tcell_96h}, ITGAE (CD103): {'ITGAE' in tcell_96h}")

signatures = {
    "LCOR_OE_malignant": lcor_oe_signature,
    "Tcell_UP24H": tcell_24h,
    "Tcell_UP96H": tcell_96h,
}
with open(f"{OUTDIR}/signatures.json", "w") as f:
    json.dump(signatures, f, indent=2)
print(f"\nSaved: {OUTDIR}/signatures.json")
