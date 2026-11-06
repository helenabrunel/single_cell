# LCOR -> T-cell phenotype (tissue residency / exhaustion) -- clean rebuild

**Question:** does LCOR (malignant cells) associate with CXCL16 (malignant)
and, downstream, with a CXCR6+/tissue-resident-memory (TRM), less-exhausted
CD8 T-cell phenotype? Rebuilt from scratch (separate from the LCOR-CXCL16
project) to test this cleanly in Visvader and Klughammer, with consistent
methodology across both: NO integrated/batch-corrected data used for any
expression value (Harmony/scVI/DRVI embeddings in these objects are used
only for the authors' own compartment/cell-type calls, never for
quantification -- every LCOR/CXCL16/CXCR6/signature value here is computed
from raw UMI counts, renormalized to log1p(CP10K) in this project).

---

## Pipeline

### `01_signatures/` -- the 3 external, lab-derived gene signatures
- `LCOR_OE_malignant` (50 genes): top genes by DESeq2 Wald statistic,
  LCOR-OE vs control in MDA-MB-231 (`results_MDA_MB231.xlsx`). CXCL16 is
  the single most induced gene in that dataset (log2FC=5.75); LCOR and
  CXCL16 excluded from the signature itself.
- `Tcell_UP24H` (117 genes) / `Tcell_UP96H` (91 genes): genes induced in T
  cells by LCOR at 24h/96h (`FIRMAS Tcells induced by LCOR.xlsx`). UP96H
  includes **CXCR6 and ITGAE (CD103)** -- the hypothesis's own markers.
  UP24H is activation/proliferation-heavy (MYC, cell cycle, IEGs) -- a
  secondary, less specific readout.

### `02_malignant_scores/` -- LCOR (raw), LCOR_OE, CXCL16 (raw), per patient
Malignant compartment only (`compartment_epithelial.h5ad` / `klughammer_
Malignant.h5ad`). Pseudobulk = sum(raw)/sum(total_counts), log1p(1e4x).
LCOR_OE = `sc.tl.score_genes` on each dataset's own log1p(CP10K) matrix,
rebuilt from raw counts for Klughammer (not the shipped Seurat-LogNormalize
`.X`). MIN 20 malignant cells/patient.
- Visvader: 34 patients. Klughammer: 26 patients.

### `03_tcell_scores/` -- T-cell phenotype, per patient
T-cell gate (coarse): Visvader = CD8_T+CD4_T+Treg (marker-score argmax
already in the object); Klughammer = `author_cell_type=="T"`. CellTypist
(`Immune_All_Low`, `majority_voting=False`) then gives a NAMED per-cell
state, from which CD8/CD4/Treg lineage and TRM status are derived (label
map in the script docstring) -- CellTypist is the primary lineage source
because it is far less sensitive to single-gene dropout than a 2-3-marker
threshold.

**A second, independent marker-based lineage/TRM call is also computed**
(CD8A/CD8B-vs-CD4 for lineage; ITGAE/CD103+ for TRM -- not CXCR6, to avoid
circularity with the CXCR6 hypothesis test) as a secondary check.
**Concordance between the two methods is low: 43% (Visvader) / 52%
(Klughammer)** for cells classified CD8/CD4 by either method -- expected
given single-cell dropout on 2-3 genes, not a sign CellTypist is wrong, but
a real limitation: conclusions about CD8/CD4/TRM status in this data are
sensitive to which definition is used, and single-method results should be
read cautiously.

Within CD8 (CellTypist): CXCR6 pseudobulk, UP24H/UP96H signature scores,
TRM_fraction (Trm + Tem/Trm cytotoxic labels), Exhaustion_score (PDCD1,
LAG3, HAVCR2, TIGIT, CTLA4, TOX, ENTPD1). Within CD4: Exhaustion_score only
(TRM and the lab's CD8-centric signatures aren't meaningful for CD4).

### `04_biased_analysis/` -- the hypothesis-driven test
LCOR (raw, LCOR_OE) vs CXCL16 (malignant) / CXCR6 (CD8) / UP24H (CD8) /
UP96H (CD8), same patient. 16 figures (2 predictors x 4 outcomes x 2
datasets), Bassez-style scatter (subtype-colored, linear fit + 95% CI).

**Results:**
| | Visvader | Klughammer |
|---|---|---|
| LCOR_OE vs CXCL16 | **R=0.85, p=8e-10** (n=32) | R=0.26, p=0.22, ns (n=24) |
| LCOR_raw vs CXCL16 | R=-0.29, p=0.11, ns | R=-0.31, p=0.14, ns |
| LCOR_OE vs CXCR6 (CD8) | R=-0.09, ns | R=-0.07, ns |
| LCOR_raw vs CXCR6 (CD8) | R=-0.23, ns | R=-0.13, ns |
| LCOR_raw vs UP24H (CD8) | **R=-0.61, p=0.007** | R=-0.01, ns |
| LCOR_raw vs UP96H (CD8) | **R=-0.57, p=0.013** | **R=-0.53, p=0.014** |
| LCOR_OE vs UP24H (CD8) | R=0.42, p=0.082, ns | R=-0.19, ns |
| LCOR_OE vs UP96H (CD8) | R=0.13, ns | R=0.14, ns |

The LCOR_OE-vs-CXCL16 result is the strongest, most significant finding in
either phase of this project (and now at full n=32, not just the
CD8-filtered subset) -- but it does not replicate in Klughammer.
**LCOR_raw vs UP96H is the one result that DOES replicate in both
datasets** -- significant, same (negative) direction -- but that direction
is opposite to the hypothesis (more LCOR -> less of the lab's own
LCOR-induced T-cell signature). Unexplained; flagged for discussion.

### `05_unbiased_analysis/` -- tissue-residency / exhaustion, broader
LCOR (raw, LCOR_OE) vs TRM_fraction (CD8, CellTypist), Exhaustion_score
(CD8 and CD4, CellTypist), plus TRM_fraction (CD8, marker/ITGAE-based) as
a secondary check. 16 figures.

**Results (CellTypist, primary):**
| | Visvader | Klughammer |
|---|---|---|
| LCOR_OE vs TRM (CD8) | R=0.42, p=0.086, ns | R=-0.12, ns |
| LCOR_OE vs exhaustion (CD8) | R=-0.44, p=0.069, ns | R=0.07, ns |
| LCOR_raw vs exhaustion (CD8) | R=-0.26, ns | R=-0.31, ns |
| LCOR vs exhaustion (CD4, either) | all ns | all ns |

**Secondary (marker/ITGAE-based TRM):**
| | Visvader | Klughammer |
|---|---|---|
| LCOR_raw vs TRM (CD8, marker) | **R=-0.71, p=0.0019** | R=-0.06, ns |
| LCOR_OE vs TRM (CD8, marker) | R=0.39, p=0.13, ns | R=-0.25, ns |

Visvader's CellTypist-based numbers point (non-significantly) in the
hypothesized direction for LCOR_OE (more LCOR_OE -> more TRM, less
exhaustion) -- the closest this project gets to supporting the hypothesis,
but n=18 and p>0.05. The marker-based secondary check in the SAME dataset
gives a significant result for a DIFFERENT predictor/outcome pair
(LCOR_raw, not LCOR_OE) in the OPPOSITE direction -- illustrating directly
why the low CellTypist/marker concordance matters: the conclusion is not
stable across equally defensible operational definitions. Nothing here
replicates in Klughammer. CD4 shows no signal by any definition, in either
dataset, consistent with this being CD8-specific if it is anything at all.

---

## Overall read

Across both the hypothesis-driven and the broader exploratory analysis,
and two independent atlases: **one strong, specific result (LCOR_OE vs
CXCL16 in Visvader) does not replicate; the one result that DOES replicate
across datasets (LCOR_raw vs UP96H) runs opposite to the hypothesis; and
the TRM/exhaustion phenotype readouts are sensitive to which CD8
definition is used, with no result surviving both datasets AND both
definitions.** This is not a methods failure -- the pipeline is
consistently built (no integrated data, CellTypist + marker cross-check,
subtype-colored, same signatures both datasets) -- it is an honest null/
mixed result for the mechanism as tested in these two public atlases.
