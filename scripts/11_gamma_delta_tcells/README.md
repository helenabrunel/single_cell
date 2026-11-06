# Gamma delta tcells

Identify gamma-delta T cells, with TCR data and from expression alone.

## expression_only_call

| Script | What it does |
|---|---|
| `expression_only_call/01_extract_tnk.py` | Extracts the T/NK compartment with raw counts. |
| `expression_only_call/02_process_tnk.py` | Normalizes and embeds the T/NK compartment. |
| `expression_only_call/03_gd_calling_expression_only.py` | Calls gamma-delta T cells from expression alone, without TCR data. |
| `expression_only_call/04a_export_for_decontx.py` | Exports raw counts per sample for ambient RNA correction. |
| `expression_only_call/04b_decontx.R` | Corrects ambient RNA with decontX, one sample at a time. |
| `expression_only_call/04c_background_test.py` | Tests whether decontX makes TRDC and TRGC informative for gamma-delta cells. |
| `expression_only_call/05_export_for_report.py` | Exports tables for the feasibility report. |
| `expression_only_call/06_combined_rule_test.py` | Tests a combined TCR-gene rule and reviews CD3E expression in NK-labelled cells. |
| `expression_only_call/07_cd3e_by_celltype.py` | Investigates the source of CD3E signal in NK-labelled cells. |
| `expression_only_call/08_nk_subclusters.py` | Looks for T-like, possibly gamma-delta, cells among NK-labelled cells. |
| `expression_only_call/09_doublets.py` | Tests whether T-like NK subclusters are NK+T doublets, using Scrublet. |
| `expression_only_call/10_combined_score_call.py` | Calls gamma-delta cells with the combined TRD/TRG versus TRA/TRB score. |
| `expression_only_call/gd_expression_feasibility_report.Rmd` | Report on the feasibility of calling gamma-delta T cells without TCR data. |

## tcr_calibration

| Script | What it does |
|---|---|
| `tcr_calibration/01_build_platform_objects.py` | Builds separate AnnData objects for the two experimental platforms. |
| `tcr_calibration/02_process_and_extract_tnk.py` | Normalizes, integrates and clusters each platform, then extracts the T/NK compartment. |
| `tcr_calibration/03_gd_calling_with_tcr.py` | Identifies gamma-delta T cells using markers and TCR data. |
| `tcr_calibration/04_stcat_annotation.py` | Annotates T/NK cells with STCAT and compares the result with the marker/TCR call. |
| `tcr_calibration/05_export_for_report.py` | Exports lightweight tables for the R Markdown report. |
| `tcr_calibration/06_gd_signatures.py` | Scores functional-state signatures in gamma-delta T cells. |
| `tcr_calibration/07_constant_region_markers.py` | Assesses which TCR constant-region genes can mark gamma-delta cells without TCR sequencing. |
| `tcr_calibration/gd_tcr_calibration_report.Rmd` | Report on gamma-delta T-cell identification with TCR data. |

See [SCRIPT_INDEX](../../docs/SCRIPT_INDEX.md) for the folder to run each script from.
