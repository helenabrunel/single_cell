# Part 0 — Dataset Description

## Visvader (Pal et al. 2021)

**Reference**: Pal et al. 2021, *Nature Genetics*. GEO accession: GSE161529.
**Compartment used**: Epithelial (malignant tumour cells).
**Patient column**: `sample`

### QC filter (≥500 cells per patient)

| Subtype | Total patients | Passing patients | Total cells | Cells after filter |
|---|---|---|---|---|
| ER | 20 | 20 | 74,287 | 74,287 |
| HER2 | 6 | 5 | 19,178 | 18,992 |
| TNBC | 8 | 6 | 28,636 | 28,353 |
| **Total** | **34** | **31** | **122,101** | **121,632** |

**Excluded patients (< 500 cells):**
- TN-MH0114-T2 (TNBC, n=210)
- HER2-MH0069 (HER2, n=186)
- TN-SH0106 (TNBC, n=73)

**Flagged patients (passing but notable QC):**
- mER-PM0178 (ER, n=6,865): mean genes = 455 — unusually low; likely low-complexity library
- TN-B1-Tum0554 (TNBC, n=1,418): mean mito = 18.3%
- ER-MH0163 (ER, n=5,512): mean mito = 14.7%

**Note on BRCA1-carrier TNBC samples**: four TNBC samples carry "B1" in their
name (TN-B1-MH4031, TN-B1-MH0131, TN-B1-Tum0554, TN-B1-MH0177), suggesting
possible BRCA1-carrier status. They were intentionally retained as TNBC but
should be flagged if TNBC results look anomalous.

### Per-patient cell counts (passing patients only)

| Patient | Subtype | N cells |
|---|---|---|
| TN-MH0135 | TNBC | 13,837 |
| ER-MH0029-9C | ER | 7,083 |
| mER-PM0178 | ER | 6,865 |
| ER-MH0125 | ER | 6,237 |
| ER-MH0114-T3 | ER | 5,981 |
| HER2-MH0176 | HER2 | 5,889 |
| TN-B1-MH0131 | TNBC | 5,577 |
| ER-MH0163 | ER | 5,512 |
| ER-MH0167-T | ER | 5,321 |
| TN-B1-MH4031 | TNBC | 5,259 |
| ER-MH0001 | ER | 5,206 |
| ER-MH0040 | ER | 5,174 |
| ER-MH0025 | ER | 4,819 |
| HER2-PM0337 | HER2 | 4,110 |
| HER2-MH0161 | HER2 | 3,929 |
| HER2-AH0308 | HER2 | 3,545 |
| ER-MH0043-T | ER | 3,403 |
| ER-MH0064-T | ER | 3,108 |
| ER-MH0056-T | ER | 3,088 |
| ER-MH0173-T | ER | 3,074 |
| ER-MH0042 | ER | 2,734 |
| ER-MH0029-7C | ER | 2,399 |
| HER2-MH0031 | HER2 | 1,519 |
| TN-B1-Tum0554 | TNBC | 1,418 |
| ER-PM0360 | ER | 1,411 |
| TN-MH0126 | TNBC | 1,201 |
| TN-B1-MH0177 | TNBC | 1,061 |
| mER-MH0068-T | ER | 829 |
| ER-MH0151 | ER | 808 |
| ER-AH0319 | ER | 645 |
| ER-MH0032 | ER | 590 |

---

## Klughammer (Klughammer et al. 2024)

**Reference**: Klughammer et al. 2024. Metastatic breast cancer dataset.
**Compartment used**: Malignant (annotated in original dataset as "Malignant").
**Patient column**: `sampleid` (format: HTAPP-XXX-SMP-XXXX)
**Note**: no raw counts available — only log-normalised expression matrix.
Cell cycle scores (S.Score, G2M.Score, Phase) pre-computed by Seurat.

### QC filter (≥500 cells per patient)

| Subtype | Total patients | Passing patients | Total cells | Cells after filter |
|---|---|---|---|---|
| ER | 18 | 13 | 60,191 | 59,237 |
| HER2 | 3 | 2 | 8,118 | 7,917 |
| TNBC | 5 | 4 | 16,724 | 16,336 |
| **Total** | **26** | **19** | **85,033** | **83,490** |

**Excluded patients (< 500 cells):**
- HTAPP-627-SMP-3242 (ER, n=427)
- HTAPP-586-SMP-8599 (TNBC, n=388; mito=36.5% — likely low quality)
- HTAPP-851-SMP-4351 (ER, n=353)
- HTAPP-364-SMP-1321 (HER2, n=201)
- HTAPP-414-SMP-1681 (ER, n=74)
- HTAPP-382-SMP-1441 (ER, n=62)
- HTAPP-330-SMP-1082 (ER, n=38)

**Flagged patients (passing but notable QC):**
- HTAPP-262-SMP-602 (ER, n=5,897): mean mito = 19.3% — borderline
- HTAPP-982-SMP-7629 (ER, n=6,017): mean mito = 16.6%
- HTAPP-649-SMP-3422 (ER, n=6,598): mean mito = 17.1%
- HTAPP-983-SMP-7659 (HER2, n=616): only 2 HER2 patients pass — underpowered subtype

**⚠ HER2 subtype is underpowered in Klughammer**: only 2 patients pass the
500-cell threshold. Subtype-specific results for HER2 should be interpreted
with caution; only pooled (Global) analysis is reliable.

### Per-patient cell counts (passing patients only)

| Patient | Subtype | N cells |
|---|---|---|
| HTAPP-917-SMP-4531 | TNBC | 10,760 |
| HTAPP-611-SMP-3062 | ER | 9,495 |
| HTAPP-759-SMP-8359 | ER | 7,757 |
| HTAPP-662-SMP-3602 | ER | 7,604 |
| HTAPP-313-SMP-932 | HER2 | 7,301 |
| HTAPP-649-SMP-3422 | ER | 6,598 |
| HTAPP-982-SMP-7629 | ER | 6,017 |
| HTAPP-262-SMP-602 | ER | 5,897 |
| HTAPP-285-SMP-751 | ER | 3,675 |
| HTAPP-735-SMP-3841 | ER | 3,570 |
| HTAPP-853-SMP-4381 | ER | 2,972 |
| HTAPP-812-SMP-8239 | TNBC | 2,372 |
| HTAPP-423-SMP-1741 | ER | 2,148 |
| HTAPP-908-SMP-4412 | TNBC | 2,073 |
| HTAPP-745-SMP-3961 | ER | 1,320 |
| HTAPP-783-SMP-4081 | ER | 1,152 |
| HTAPP-425-SMP-1771 | TNBC | 1,131 |
| HTAPP-321-SMP-1021 | ER | 1,032 |
| HTAPP-983-SMP-7659 | HER2 | 616 |

---

## Combined summary

| | Visvader | Klughammer |
|---|---|---|
| Tissue | Primary breast tumour | Metastatic biopsy |
| Compartment | Epithelial | Malignant |
| Patients (total) | 34 | 26 |
| Patients (≥500 cells) | 31 | 19 |
| Cells (after QC) | 121,632 | 83,490 |
| ER patients (passing) | 20 | 13 |
| HER2 patients (passing) | 5 | 2 ⚠️ |
| TNBC patients (passing) | 6 | 4 |
| Raw counts available | Yes | No |
| CC scores pre-computed | No (computed here) | Yes (Seurat) |
| CytoTRACE2 available | Yes | Yes |
