"""
01b_add_clinical_metadata.py
=============================
Attach clinical/IHC metadata (tissue type, ER/PR/HER2 status, derived
subtype) to gse243526_raw.h5ad and gse243526_processed.h5ad.

This metadata is NOT in GSE243526_RAW.tar (only expression matrices are)
— it comes from the GEO series SOFT file (Sample_characteristics_ch1),
parsed once into gse243526_clinical_metadata.csv.

Key finding: "Patient_1".."Patient_4" are NORMAL breast tissue, not
tumor (confirmed by GEO tissue field) — despite the "Patient_" naming
looking uniform. Subtype (derived from ER/PR/HER2 IHC, standard rule:
HER2+ overrides > HR+ if ER+ or PR+ > TNBC if both ER-/PR-) is heavily
skewed: 5 HR+/HER2-, 5 HER2-equivocal (ambiguous IHC, not FISH-confirmed
HER2+), 1 TNBC, 0 confirmed HER2+. No clinical data exists for cell
lines (tagged "cell_line") — T47D has no RNA data at all (GEO lists a
GSM for it, but the file was never deposited in RAW.tar).

Input:  gse243526_clinical_metadata.csv, gse243526_raw.h5ad,
        gse243526_processed.h5ad
Output: same two .h5ad files, with tissue/er_ihc/pr_ihc/her2_ihc/subtype
        added to .obs (overwritten in place)

Run from: ~/single_cell/Regner/
  python3 01b_add_clinical_metadata.py
"""

import scanpy as sc
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

CLINICAL = "gse243526_clinical_metadata.csv"
FILES    = ["gse243526_raw.h5ad", "gse243526_processed.h5ad"]

clin = pd.read_csv(CLINICAL).set_index("sample_name")
clin_cols = ["tissue", "er_ihc", "pr_ihc", "her2_ihc", "subtype"]

print("\n" + "="*65)
print(" CLINICAL METADATA")
print("="*65)
print(clin[clin_cols].to_string())

for file in FILES:
    print(f"\n{'='*65}\n {file}\n{'='*65}")
    adata = sc.read_h5ad(file)

    for col in clin_cols:
        adata.obs[col] = adata.obs["sample_name"].map(clin[col]).astype("category")

    missing = adata.obs["tissue"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} cells with no clinical match")

    print(adata.obs[["sample_name", "tissue", "subtype"]]
          .drop_duplicates().sort_values("sample_name").to_string(index=False))

    adata.write_h5ad(file)
    print(f"  Saved: {file}")

print(f"""
══════════════════════════════════════════════════════════════
 DONE
══════════════════════════════════════════════════════════════
  obs columns added: tissue, er_ihc, pr_ihc, her2_ihc, subtype

  Tissue breakdown (samples):
{clin['tissue'].value_counts().to_string()}

  Subtype breakdown (tumor samples only):
{clin.loc[clin['tissue']=='primary breast tumor', 'subtype'].value_counts().to_string()}

  Next: python3 02c_subtype_overview.py
══════════════════════════════════════════════════════════════
""")
