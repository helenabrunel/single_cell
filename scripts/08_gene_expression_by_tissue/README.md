# Gene expression by tissue

Compare expression of genes of interest across lineages, tissues and subtypes.

| Script | What it does |
|---|---|
| `01_lcor_cxcl16_hexbin.py` | Plots hexbin correlation of LCOR and CXCL16 in malignant cells, overall and per subtype. |
| `02_lcor_epithelial_lineage.py` | Reproduces a reference layout: UMAPs by epithelial population and LCOR, plus a dot plot. |
| `03_lcor_lineage_three_categories.py` | Simplifies that figure to three epithelial lineage categories. |
| `04_lcor_malignant_only.py` | Restricts the figure to the malignant epithelial compartment. |
| `05_malignant_umap_by_subtype.py` | Colors the same malignant UMAP by breast cancer subtype. |
| `06_replot_large_text.py` | Replots those panels with larger text, without recomputing. |
| `07_recompute_umap_no_harmony.py` | Recomputes the malignant UMAP without Harmony integration. |
| `08_bach1_ppard_klf5_normal_vs_tumor.py` | Compares BACH1, PPARD and KLF5 between normal breast and primary tumor. |
| `09_bach1_ppard_klf5_with_cell_lines.py` | Makes the same comparison and adds breast cancer cell lines. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
