# Potency scoring

Estimate cellular potency with CytoTRACE2 and LandSCENT and compare the two.

| Script | What it does |
|---|---|
| `01_cytotrace2_scoring.py` | Runs CytoTRACE2 on each compartment object, global and per subtype. |
| `02_cytotrace2_scoring_without_raw_counts.py` | Runs CytoTRACE2 when raw counts are missing, reverting log-normalized data to pseudo-counts per batch. |
| `03_export_for_landscent.py` | Exports log-count matrices and metadata to Matrix Market format for LandSCENT. |
| `04_landscent_scoring.R` | Computes LandSCENT signaling entropy (SR) per subtype on epithelial cells. |
| `05_landscent_cytotrace_umap.py` | Compares CytoTRACE2 and LandSCENT on UMAP. |
| `06_landscent_stemconsensus_umap.py` | Adds stem_consensus as a third potency axis to that comparison. |
| `07_all_signatures_bubble.py` | Summarizes per-patient correlations between each signature and both potency readouts as a bubble plot. |
| `08_cytotrace2_potency_umap.py` | Plots CytoTRACE2 potency category and score UMAPs per subtype. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
