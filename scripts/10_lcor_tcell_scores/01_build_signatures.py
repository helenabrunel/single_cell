"""
01_build_signatures.py
========================
Builds the 3 gene signatures needed for this project, from the lab's own
experimental data (not from single-cell data -- these are independent,
external inputs):

1. LCOR_OE_malignant (50 genes): top genes by DESeq2 Wald statistic
   (log2FC>0, padj<0.05), LCOR-OE vs control in MDA-MB-231 cells
   (results_MDA_MB231.xlsx). LCOR and CXCL16 excluded from the signature
   itself (LCOR = the perturbed gene; CXCL16 = used as a downstream
   readout later, excluded to avoid circularity).
2. Tcell_UP24H (117 genes) / Tcell_UP96H (91 genes): genes induced in T
   cells by LCOR at 24h/96h ("FIRMAS Tcells induced by LCOR.xlsx").
   UP96H includes CXCR6 and ITGAE (CD103) -- the tissue-residency markers
   this project's hypothesis centers on. UP24H is activation/
   proliferation-heavy (MYC, cell cycle, IEGs) -- treat as a secondary,
   less specific readout.

Output: 01_signatures/figures/signatures.json
"""
import pandas as pd
import json
import os

OUTDIR = "01_signatures/figures"
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

# ============================================================================
# 2/3. T-cell signatures (24h, 96h)
# ============================================================================
tdf = pd.read_excel("/home/hbrunel/single_cell/FIRMAS Tcells induced by LCOR.xlsx",
                     sheet_name="Hoja1", header=None)
tdf.columns = ["up24h", "up96h"]
tdf = tdf.iloc[1:]
tcell_24h = tdf["up24h"].dropna().astype(str).str.strip().tolist()
tcell_96h = tdf["up96h"].dropna().astype(str).str.strip().tolist()
print(f"Tcell_UP24H: {len(tcell_24h)} genes")
print(f"Tcell_UP96H: {len(tcell_96h)} genes (includes CXCR6: {'CXCR6' in tcell_96h}, ITGAE: {'ITGAE' in tcell_96h})")

signatures = {
    "LCOR_OE_malignant": lcor_oe_signature,
    "Tcell_UP24H": tcell_24h,
    "Tcell_UP96H": tcell_96h,
}
with open(f"{OUTDIR}/signatures.json", "w") as f:
    json.dump(signatures, f, indent=2)
print(f"\nSaved: {OUTDIR}/signatures.json")
