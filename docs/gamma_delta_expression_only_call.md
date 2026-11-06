# γδ T cells in Klughammer et al. 2024 (HTAPP metastatic breast cancer) — feasibility test

**Result: γδ T cells cannot be identified at single-cell level in this dataset.** Work stopped after
the feasibility checks below; no signature/clinical analysis was run.

## Data
10x 3' v2/v3, no TCR sequencing; 30 patients, one metastatic biopsy each; 29,255 author-labelled
T/NK cells (28 patients). Raw counts from `.raw` of `../data/dataset.h5ad`.

## Scripts
| Script | What it does |
|---|---|
| `01_extract_TNK.py` | T/NK cells with raw counts + patient clinical table (`data/clinical_patients.csv`) |
| `02_process_TNK.py` | normalisation, HVG, PCA, Harmony (patient), UMAP, Leiden |
| `03_gd_calling.py` | expression-only rule calibrated on Azizi 10x (TRDC+ CD3+ CD4- CD8B- KLRF1- FCER1G-) |
| `04a_export_for_decontx.py`, `04b_decontx.R`, `04c_background_test.py` | ambient-RNA correction (decontX, conda env `decontx`) and background test |
| `06_combined_rule_test.py` | follow-up: combined TRDC/TRGC+ TRAC/TRBC- rule on Azizi TCR truth and Klughammer; CD3E in author-NK cells |

## Why it does not work
* TRDV genes are essentially undetected (≤ 0.4 % of TRDC+ cells) → no Vδ subsets.
* TRDC is detected in 83 % of NK cells (germline) and in 14 % of T cells.
* Background test (TRDC rate in DN candidates vs in sure-αβ T cells = CD4+ or CD8B+):
  Azizi 10x 5' ratio 5.0 (works, precision ≈ 0.80, consistent with TCR truth 0.88);
  Klughammer ratio 0.62 — TRDC is *more* frequent in sure-αβ cells, in every UMI-depth bin
  and in 21/22 patients.
* decontX (whole atlas, author cell types as clusters, per sample) barely changes TRDC
  (40/6,522 TRDC+ cells lose it); ratio unchanged.
* TRGC1, TRGC2, TRGC1|TRGC2, TRDC & TRGC, TRDC ≥ 2: none informative (ratio < 1). TRGC is not
  γδ-specific even in Azizi.
* TRDC background in αβ cells correlates with NK abundance per patient (Spearman ρ = 0.55,
  p = 0.007; 22 patients with ≥ 30 cells per group) and is higher in 3' v2 than v3; origin (ambient vs germline TRA/TRD-locus
  transcription) not resolvable.

Implication: any γδ annotation in this atlas based on TRDC (incl. automated tools such as STCAT)
would mostly contain αβ T cells.

## Follow-up (Oct 2026), after review: NK-annotated cells, CD3E, combined rule
* `06_combined_rule_test.py`: positive-marker choice and combined rule (TRDC/TRGC+ & TRAC=TRBC1=TRBC2=0
  & CD3E+ & CD4-). TCR-γδ cells in Azizi express TRAC/TRBC1/TRBC2 (81/70/79 %); only 2.1 % have all
  three at 0 → recall 0.01. Score γδ C − αβ C genes: AUC 0.83 vs TRDC alone 0.85. TRGC alone: precision
  0.10–0.19; TRDC & TRGC: 0.92 but recall 0.27. All Klughammer background ratios < 1.
* `07_cd3e_by_celltype.py`: CD3E/CD3D detection ratio 0.85 in T, 0.5–0.84 in other cell types, 1.33 in NK;
  14.8 % of NK without any other T transcript are CD3E+ (≤ 1 % elsewhere) → part of CD3E in NK is not
  from T cells.
* `08_nk_subclusters.py`: NK-annotated cells re-clustered → 2 T-like subclusters (516 cells, ≥ 18
  patients; CD3D ~69 %, TRAC ~62 %, CD8A 41 %, CD8B 27 %, TRGC1 38 %, KLRF1/FCER1G ~85 %, more UMIs).
  Not like Azizi γδ (IL7R 14 vs 83 %, CD5 3 vs 25 %, FCGR3A 74 vs 1 %); TRGC1 ≈ NK+T doublet
  expectation.
* `09_doublets.py`: Scrublet per channel flags ≤ 1.2 % doublets in every cell type (data likely
  pre-filtered; uninformative). UMI test: T-like cells ≈ other NK of the same patient (ratio 1.07) and
  ~half the NK+T doublet expectation (0.52) → single cells. Background test inside them: TRDC/TRGC as or
  more frequent in CD8B+ (αβ) than CD8B− → no γδ signal. Conclusion extends to NK-annotated cells.
* `10_combined_score_call.py`: combined score as a call (top 1–10 % of CD3E+CD4− cells). Azizi: precision
  0.41 at top 3 % (TRDC rule 0.88), background ratio < 1 (TRGC2 in αβ CD8 drives the top). Klughammer:
  55–78 % of called cells NK-annotated; non-NK ratio ≈ 1.
* The NK-marker filter of 03 (validated on Azizi 10x = sorted CD3+ T cells only) cut NK-annotated
  cells passing the rule from 1,279 to 29, undoing the inclusion of NK cells; the background test used
  author-T cells only.

Logs: `logs/03_gd_calling.log`, `logs/06_combined_rule_test.log`, `logs/04c_background_test.log`; tables in `figures/04_decontx/`.

Report: `report/klughammer_gdT_feasibility.html` (source `.Rmd`; tables from `05_export_for_report.py`).
