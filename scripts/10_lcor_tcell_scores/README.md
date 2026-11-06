# Lcor tcell scores

Relate tumor LCOR programs to T-cell phenotypes, using patient-level scores on non-integrated data.

| Script | What it does |
|---|---|
| `01_build_signatures.py` | Builds the three signatures from the lab's experimental data. |
| `02_malignant_scores.py` | Computes per-patient LCOR, LCOR_OE and CXCL16 scores in malignant cells on non-integrated data. |
| `03_tcell_scores.py` | Computes per-patient T-cell scores and labels on non-integrated data. |
| `04_hypothesis_scatterplots.py` | Hypothesis-driven test of LCOR against CXCL16, CXCR6 and the LCOR-induced T-cell signatures. |
| `05_unbiased_scatterplots.py` | Explores tissue residency and exhaustion beyond the CXCL16/CXCR6 hypothesis. |
| `06_celltypist_composition.py` | Shows the complete CellTypist T-cell state composition. |
| `07_celltypist_multiscatter.py` | Correlates LCOR with every CellTypist-derived lineage category symmetrically. |
| `08_zheng_states_composition.py` | Repeats the composition figure with pan-cancer T-cell states from Zheng et al. 2021. |
| `09_zheng_states_multiscatter.py` | Repeats the symmetric correlations for the Zheng et al. T-cell states. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
