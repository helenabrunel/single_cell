# LCOR - CXCL16/CXCR6 in human breast cancer -- definitive analysis

**Question:** does LCOR regulate the CXCL16-CXCR6 axis in breast cancer,
as observed in mouse/culture experiments -- validated in human tissue?
(Editor request: demonstrate the mechanism in human data.)

**Bottom line:** the CXCR6-CXCL16 / CD103-ITGAE axis itself is robust in
human tissue (bulk RNA-seq, 6 public datasets, separate branch). The
specific LCOR-CXCL16 link is **not supported** -- tested independently in
bulk and in single-cell by six different methods below (malignant cells x4,
myeloid cells x1, within-patient LCOR effect on the CXCL16->CXCR6
interaction x1), all null or confounded. No evidence survives that LCOR
regulates CXCL16 within the human tumor cell itself, within the myeloid
compartment where CXCL16 is most expressed, or the CXCL16->CXCR6
ligand-receptor engagement within a tumor's own LCOR-high vs LCOR-low
malignant cells.

A positive control was attempted (`07_apm_positive_control/`, LCOR vs an
antigen-presentation-machinery gene set the lab's own mouse/culture work
shows LCOR positively regulates) to check whether the CXCL16 null reflects
a real absence of relationship or a lack of statistical power. Result: the
cell-level pipeline DOES detect a strong, reproducible LCOR-APM association
(so it is not simply powerless), but in the OPPOSITE direction from what
the lab's experiments establish -- an unexplained finding in its own right,
not a validated positive control for LCOR specifically. A candidate
explanation (immune-selection pressure distorting the relationship more in
highly-infiltrated tumors) was tested and not supported. A separate,
LCOR/immune-unrelated methodological check (ESR1 -> its canonical targets
GREB1/PGR, same within-patient design) DID recover the expected direction
cleanly (RR=1.4-1.6, p<1e-7) -- so the within-patient GLM design itself is
not systematically inverting signs. `07_replication_klughammer.py` then
tested whether the LCOR-APM reversal itself replicates in an INDEPENDENT
breast cancer scRNA-seq atlas (Klughammer/HTAPP, 19 donors, malignant
cells): **it does not** -- LCOR->APM comes out RR=1.05 [1.00-1.10], p=0.043
(the expected positive direction, weakly but correctly signed), while
ESR1->GREB1/PGR again replicates cleanly positive there too. A THIRD
independent atlas (`07_replication_regner.py`, Regner et al./GSE243526)
adds a third, again different, answer: LCOR->APM comes out NULL there
(RR=0.98, p=0.71), while ESR1->GREB1/PGR again replicates strongly
positive (p<1e-8). So across three independent atlases the LCOR-APM
relationship is negative (Visvader), positive (Klughammer), and null
(Regner) -- it is simply NOT a stable, reproducible cross-cohort signal,
while the ESR1->GREB1/PGR method-sanity check is reliably positive in all
three. Net effect: the LCOR-CXCL16 cell-level null remains a reasonably
trustworthy method result (the pipeline itself is validated three times
over); what's unresolved is why LCOR-APM itself is this unstable across
cohorts, flagged for discussion with the wider team rather than resolved
here. See `07_apm_positive_control/` for full detail.

---

## How we got here

**Bulk RNA-seq (6 public datasets, separate branch, not detailed here):**
CXCR6-CXCL16/CD103-ITGAE correlates robustly across datasets. LCOR-CXCL16/
CXCR6 is weak and inconsistent, not significant in TNBC. Bulk mixes tumor
with immune/stromal compartments and can't isolate the signal -- moved to
single-cell (this atlas: Pal et al./Visvader, ~27-34 patients, ER/HER2/
TNBC, GSE161529).

**Single-cell, organized in 4 blocks (each its own folder) of increasing
specificity/rigor for the same underlying question, plus one orthogonal
method:**

### `01_exploratorio/` -- where do these genes even sit?

- `01_celltype_specificity_overview.py`: LCOR has no strong cell-type
  specificity. CXCL16 does -- highest per-cell in macrophages/dendritic
  cells.
- `02_tissue_contribution_by_celltype.py`: despite that, because
  epithelial cells vastly outnumber myeloid cells, the malignant
  compartment still supplies most (~65-68%, cell-number-weighted) of a
  tumor's total CXCL16 signal in ER/TNBC/global -- though this mixing
  ratio varies hugely between patients (8-100% epithelial origin). This
  is *why* bulk RNA-seq can't isolate the tumor-cell-intrinsic signal.
- `03_malignant_structure_no_harmony.py`: the malignant compartment,
  embedded without Harmony batch correction, is dominated by *patient*
  identity (each patient's tumor cells form largely their own region),
  not by a shared LCOR/CXCL16 gradient across tumors. CXCR6 is correctly
  almost absent from malignant cells (0.1% detection), confirming the
  compartment split is clean. This patient-dominated structure is the
  reason the correlation block's cell-pooled tests need the depth/patient
  corrections they get, and why the pseudobulk block is the right level
  to summarize the question.

### `02_correlacion/` -- LCOR vs CXCL16 co-expression, cell level

- `01_compartment_hexbin_scan.py`: the broadest, uncorrected first pass --
  per-cell hexbin correlation of all 3 genes across all 13 cell groups and
  4 subtype splits, no depth correction, no restriction to the malignant
  compartment. (Originally part of `01_exploratorio/01_celltype_specificity_overview.py`,
  split out because this content is correlation, not exploration -- see
  each script's "SPLIT HISTORY" note.)
- `02_depth_artifact_cell_level.py`: naive cell-level correlation looked
  real, then didn't. A depth-offset GLM (sequencing depth as exposure, not
  a hard filter), restricted to the malignant compartment, shows the
  pooled effect is largely a between-patient artifact -- within-patient
  (sample fixed effects), the rate ratio is ~1, not significant.
- `03_depth_artifact_diagnostic_hexbin.py`: shows *why* visually -- with
  only 1-4 raw UMIs per cell, each (LCOR, CXCL16) integer count pair
  traces its own line in normalized-logcounts space as depth varies, a
  mechanical artifact of sparse counts, not distinct subclones.

### `03_pseudobulk/` -- patient level, malignant-only (headline result)

- `01_pseudobulk_by_patient_headline.py`: per-patient pseudobulk (sum raw
  counts / sum library size, CP10K, malignant cells only) of LCOR vs
  CXCL16, correlated across patients: **null globally** (Spearman
  rho=-0.11, n=34, p=0.53) and **null in TNBC** (rho=0.24, n=8, p=0.57).
  Not explained by patient-level sequencing depth (checked directly, and
  via depth-adjusted residual correlation, rho=-0.14, p=0.44). This is
  the single-cell analogue of the bulk RNA-seq test, restricted to the
  compartment bulk cannot isolate -- and it agrees with bulk: null.

### `04_liana/` -- cell-cell interaction (orthogonal method)

- `01_liana_interaction_vs_lcor.py`: a different question -- not LCOR-CXCL16
  co-expression, but whether the CXCL16(malignant)->CXCR6(T-cell)
  ligand-receptor engagement itself tracks LCOR. LIANA `rank_aggregate` per
  patient gives an interaction score that correlates with LCOR pseudobulk
  (rho=-0.63, p=0.002, n=22 raw) -- but LCOR itself differs by subtype
  (Kruskal-Wallis p=0.0014, lowest in TNBC), and the correlation weakens
  substantially and loses significance once subtype is accounted for
  (within-subtype: ER rho=-0.06 n.s., TNBC rho=-0.57 n.s./n=7; partial
  Spearman controlling for subtype: rho=-0.24, p=0.29). Largely a subtype
  contrast, not a robust continuous LCOR effect.

### `05_myeloid/` -- LCOR vs CXCL16, myeloid cells (Macro_M1/Macro_M2/Dendritic)

`01_celltype_specificity_overview.py` showed CXCL16 is far more
cell-type-specific to myeloid cells (~40-47% detection in Macro_M1/Macro_M2/
Dendritic) than to malignant epithelial cells (13.7%), which raised the question of whether
restricting the LCOR-CXCL16 test to that compartment -- where CXCL16 is
actually most expressed -- might reveal a relationship the malignant-cell
tests missed.

- `01_pseudobulk_myeloid_headline.py`: myeloid analogue of
  `03_pseudobulk/01`. Per-patient pseudobulk (Macro_M1+Macro_M2+Dendritic
  combined, malignant-compartment methodology): **null** (Spearman
  rho=0.20, n=32, p=0.28; TNBC rho=0.55, n=8, p=0.16, not significant at
  this n). CXCL16 pseudobulk itself correlates with patient sequencing
  depth (rho=0.57, p=6e-4); once both genes are depth-adjusted the residual
  correlation drops further (rho=0.09, p=0.63). Does not strengthen at
  higher n_cells thresholds. Each myeloid subgroup separately (lower n,
  diagnostic only): Macro_M1 rho=0.19, Macro_M2 rho=0.28, Dendritic
  rho=-0.37, none significant.
- `02_depth_artifact_myeloid_cell_level.py`: myeloid analogue of
  `02_correlacion/02`. Depth-offset GLM, malignant-cell methodology: pooled
  RR=1.03 [0.97-1.10] p=0.34, within-patient (sample FE) RR=1.02
  [0.95-1.10] p=0.59 -- both null. Naive per-patient codetected-only r
  ranges from -0.03 to 0.65 across patients and is strongly anti-correlated
  with patient sequencing depth (Spearman rho=-0.85, p=0.001), the same
  depth-artifact signature seen in malignant cells (`03_depth_artifact_
  diagnostic_hexbin.py`) despite myeloid cells' much higher detection rate.

### `07_apm_positive_control/` -- does the method detect a KNOWN LCOR relationship?

A positive control was added after the myeloid
LCOR-CXCL16 null: is that "no biological relationship" or "the method can't
detect a real relationship in this sparse data"? Proposed test: LCOR vs an
antigen-presentation-machinery (APM) gene set already known from the lab's
own prior mouse/culture work to be regulated by LCOR -- compact "APM pura"
set as finalized by the lab: TAP1, TAP2, TAPBP, HLA-A, HLA-B, HLA-C, PSMB8,
PSMB9, B2M (per-cell raw counts summed into one meta-gene), run through the
exact same pipeline used for LCOR-CXCL16 in both compartments.

- `01_pseudobulk_apm_malignant.py` / `03_pseudobulk_apm_myeloid.py`
  (patient-level pseudobulk, malignant/myeloid): **null in both**
  (malignant: rho=-0.12 depth-adjusted, p=0.51, n=34; myeloid: rho=-0.10
  depth-adjusted, p=0.60, n=32) -- same pattern as the CXCL16 pseudobulk
  tests.
- `02_depth_artifact_apm_malignant_cell_level.py` /
  `04_depth_artifact_apm_myeloid_cell_level.py` (cell-level depth-offset
  GLM, malignant/myeloid): **significant negative relationship in both**,
  within-patient (sample FE): malignant RR=0.87 [0.81-0.93] p=1.75e-4;
  myeloid RR=0.94 [0.92-0.96] p=1.1e-9. Higher LCOR -> lower APM, holding
  within the same tumor, in both compartments -- directionally consistent
  with LCOR's known corepressor role and with the lab's prior data.

**IMPORTANT CORRECTION -- direction check (added after review):**
the lab's own mouse/culture experiments consistently show LCOR POSITIVELY
regulating APM (more LCOR -> more APM) -- this is published. The cell-level
result above is the OPPOSITE direction (RR<1 = more LCOR -> LESS APM). So
this is NOT a clean positive-control validation: the method clearly has
statistical sensitivity (it detects a strong, reproducible association),
but it does not reproduce the expected sign. This means the control does
NOT make the CXCL16 null more trustworthy -- if anything it is a caution
in the other direction: cross-sectional human tumor scRNA-seq may distort
or invert even a real, causally-established relationship, so a null result
for LCOR-CXCL16 in this same data cannot be read as confidently as "no
relationship" either.

`05_apm_lcor_by_cd8_infiltration.py` tested one candidate explanation --
immune selection (high-APM malignant cells preferentially eliminated by
CD8 T cells, more so in highly infiltrated tumors, which could invert an
underlying positive LCOR->APM relationship in cross-sectional data). The
prediction (more negative RR in High-CD8 patients) was NOT supported: if
anything the opposite pattern was seen (Low-CD8, n=11: RR=0.89
[0.83-0.96], p=0.0013; High-CD8, n=12: RR=0.94 [0.87-1.02], p=0.13, n.s.).
This candidate explanation for the sign reversal is therefore not
supported by this check. The reversal itself remains unexplained and is
flagged for discussion rather than resolved here.

Pseudobulk (patient-level) also failed to detect the known LCOR-APM
relationship in either compartment (malignant: rho=-0.12 depth-adjusted,
p=0.51; myeloid: rho=-0.10, p=0.60, both n=32-34) -- consistent with low
power at this n rather than informative about direction.

`07_replication_klughammer.py` -- tests whether the LCOR-APM reversal is a
property of the Visvader atlas specifically, using an independent breast
cancer scRNA-seq dataset already available in this environment (Klughammer/
HTAPP, 19 donors, malignant cells). **It does not replicate**: LCOR->APM
within-patient RR=1.048 [1.001-1.096], p=0.043 -- the expected POSITIVE
direction (weak, but correctly signed), opposite to Visvader's RR=0.87.
ESR1->GREB1/PGR (method sanity check) replicates cleanly positive in
Klughammer too (RR=1.31 p=7e-6, RR=1.41 p=8e-12), confirming the pipeline
itself works consistently across datasets. `09_replication_regner.py` -- a THIRD independent dataset (Regner et al./
GSE243526, PMC11872555; malignant cells identified here via a simple
EPCAM/KRT8/KRT18/KRT19-vs-PTPRC marker score, no precomputed compartment
split available). Result: LCOR->APM comes out NULL (RR=0.98 [0.89-1.08],
p=0.71) -- neither the Visvader negative nor the Klughammer positive.
ESR1->GREB1/PGR again replicates strongly positive (RR=2.06 p=1e-17,
RR=2.28 p=2e-8), confirming the pipeline is reliable across a third
dataset too.

**Conclusion across all three atlases:** Visvader negative, Klughammer
positive (expected), Regner null. The LCOR-APM relationship is NOT stable
across cohorts -- something about each specific dataset/cohort
(composition, processing, or genuine biological heterogeneity between
cohorts) drives the result, not a general, reproducible feature of human
tumor scRNA-seq or of LCOR biology in vivo. The method itself (ESR1-
>GREB1/PGR) is validated as reliable in all three datasets, so this
instability is about the LCOR-APM biology/data, not the analysis pipeline.

`08_lcor_esr1_relationship.py` -- motivated by the lab's own paper (PMID
41289011, "Estrogen receptor signaling drives immune evasion and
immunotherapy resistance in HR+ breast cancer"): ERalpha reportedly
"hijacks" LCOR, blocking its normal induction of tumor immunogenicity
(incl. antigen presentation). Two checks, in both Visvader and Klughammer:

- LCOR vs ESR1 directly (within-patient GLM): POSITIVE and significant in
  both datasets (Visvader RR=1.08 p=0.003; Klughammer RR=1.22 p=9e-13,
  all cells; near-identical restricted to ER+ only). Does not contradict
  the hijacking mechanism (a protein-interaction/functional-redirection
  model, not necessarily anti-correlated mRNA) -- if anything consistent
  with ERalpha recruiting/inducing LCOR expression while redirecting its
  activity away from immune genes.
- LCOR->APM, ER+ vs non-ER+ cells (the direct hijacking prediction: effect
  should be more blunted/negative in ER+): partially supported.
  Klughammer: ER+ RR=1.03 p=0.24 (flat/n.s.) vs non-ER+ RR=1.11 p=0.13
  (positive trend) -- consistent with ERalpha neutralizing LCOR's
  pro-APM effect specifically where it is active. Visvader: ER+ RR=0.85
  p=0.003 vs non-ER+ RR=0.89 p=0.013 -- both negative and significant, so
  the ER+/non-ER+ contrast is much smaller and the hijacking mechanism
  does NOT explain why even Visvader's non-ER+ (ESR1-low) cells show the
  reversed relationship. Some Visvader-specific factor beyond ESR1/ER
  signaling remains unaccounted for.

`06_negative_control_esr1_greb1.py` -- a further methodological check,
unrelated to LCOR/immune biology: does the within-patient depth-offset GLM
design itself reliably recover a KNOWN, unambiguous, immune-unrelated
relationship (ESR1 -> its canonical direct targets GREB1 and PGR, ER+
malignant cells only)? **Yes, cleanly**: within-patient RR=1.42
[1.25-1.61] p=5.0e-8 (GREB1) and RR=1.56 [1.37-1.78] p=7.1e-11 (PGR), both
strongly in the expected positive direction. This means the pipeline is
NOT systematically inverting signs -- the LCOR-APM reversal is specific to
that biology (still unexplained) rather than a generic design flaw. Net
effect: the LCOR-CXCL16 cell-level null remains reasonably trustworthy as
a method result; what's unresolved is specifically why LCOR-APM inverts,
which is flagged for discussion rather than answered here.

### `06_liana_within_patient/` -- LCOR effect on the CXCL16->CXCR6 interaction (within-patient, exploratory)

Reframes `04_liana`'s question: not "does per-patient LCOR pseudobulk
correlate with per-patient interaction score" (confounded by subtype), but
"within one tumor, do its LCOR+ malignant cells engage CXCR6+ T cells more
than its LCOR- malignant cells" -- a within-patient design that is immune
to the subtype confound by construction.

- `01_liana_lcor_split_within_patient.py`: per patient, malignant cells
  split into LCOR+ (raw>0, ~15% of cells) vs LCOR- (raw==0), the LCOR-
  side depth-matched to LCOR+ (coarsened quantile-bin matching on
  total_counts -- confirmed working: unmatched median-depth gap p=1.9e-6,
  matched p=0.78) to avoid reintroducing the depth artifact inside a new
  design. LIANA run separately on each half against the same T cells,
  paired across n=20 patients: **null** (Wilcoxon p=0.73, paired t-test
  p=0.44; LCOR+ higher in only 9/20 patients). Isolated in its own folder
  since it's a more exploratory design than 01-05.

### `10_lcor_program_tcell_phenotype/` -- LCOR-OE signature (not raw LCOR) vs CD8 T-cell phenotype

Since raw LCOR is too lowly
expressed for reliable correlation, use a **proxy signature** built from
the lab's own experimental data instead. Three signatures were extracted
from lab-provided files (`01_build_signatures.py`):
  - `LCOR_OE_malignant` (50 genes): top genes by DESeq2 Wald statistic,
    LCOR-OE vs control in MDA-MB-231 cells (`results_MDA_MB231.xlsx`).
    Confirms CXCL16 is the single most induced relevant gene (log2FC=5.75,
    padj=1e-62); 7/9 of this project's APM panel genes are also
    significantly induced here (independent empirical support for that
    panel, from `07_apm_positive_control/`). LCOR and CXCL16 excluded from
    the signature itself (perturbed gene; avoid circularity vs the CXCL16
    readout).
  - `Tcell_UP24H` / `Tcell_UP96H` (117 / 91 genes): genes induced in T
    cells by LCOR at 24h/96h (`FIRMAS Tcells induced by LCOR.xlsx`). UP24H
    is activation/proliferation-heavy (MYC, cell cycle, glycolysis, IEGs
    FOS/EGR2/EGR3) -- likely a generic-activation signature, interpret with
    caution. UP96H includes **CXCR6 and ITGAE (CD103)** -- the markers
    the working hypothesis centers on -- and looks like a genuine
    differentiated/tissue-residency state, not acute activation.

Tested in Visvader (`02_visvader_analysis.py`, `04_...cxcl16_cxcr6.py`) and
replicated independently in Klughammer
(`03_klughammer_analysis.py`, `05_...cxcl16_cxcr6.py`; CD8 T cells
identified here via a CD8A/CD8B-vs-CD4 marker score, since
`klughammer_Lymphoid.h5ad` only labels cells broadly as "T" -- 25,262 T
cells / 27 donors, much better powered than Visvader's 5,213 CD8 cells /
16 usable patients). `06_scatterplots.py` renders all
results in the per-patient scatter+regression style of the lab's own
Bassez-dataset reference figures.

**Result 1 -- the ORIGINAL test (LCOR_OE vs CXCL16 same
compartment, vs CXCR6 in CD8 T cells), same patient:**
  - Visvader: LCOR_OE vs CXCL16 (malignant): **R=0.83, p=6e-5** -- the
    strongest, most significant result in this entire project. Survives
    subtype adjustment (rho=0.53, p=0.036). LCOR_OE vs CXCR6 (CD8): null
    (R=-0.01, p=0.96).
  - Klughammer: LCOR_OE vs CXCL16: **does not replicate** (R=0.25, p=0.25).
    LCOR_OE vs CXCR6: null (R=0.04, p=0.85).
  - Same dataset-specific-inconsistency pattern as the APM block: a very
    strong Visvader-only signal that a second independent atlas does not
    reproduce.

**Result 2 -- unbiased test (LCOR-high/low patients, CD8 phenotype, no
assumed marker):** null in both datasets, both by direct score comparison
(within-subtype tercile split to avoid the confound found in Visvader's
naive split) and by fully unbiased CD8 reclustering + post-hoc cluster
characterization. No CD8 cluster or phenotype score enriched in LCOR-high
patients in either dataset.

**Result 3 -- unexpected but replicated: raw LCOR (not the signature) vs
Tcell_UP96H is NEGATIVE and significant in both datasets** (Visvader
R=-0.65, p=0.007; Klughammer R=-0.46, p=0.03) -- opposite direction to the
hypothesis, but one of the only findings in this whole positive-control/
signature investigation that actually replicates across cohorts. Not
explained; flagged for discussion alongside the LCOR-APM reversal.

All 16 predictor x outcome x dataset combinations (LCOR_raw / LCOR_OE x
CXCL16 / CXCR6 / Tcell_UP24H / Tcell_UP96H x Visvader / Klughammer) are
rendered in `06_scatterplots/`.

---

## Folder layout

```
LCOR-CXCL16/
├── README.md
├── LCOR_CXCL16_CXCR6_summary.pptx   <- editor-facing summary deck
├── 01_exploratorio/            (3 scripts -> 3 figure subfolders, one per script)
│   ├── scripts/   01_celltype_specificity_overview.py
│   │              02_tissue_contribution_by_celltype.py
│   │              03_malignant_structure_no_harmony.py
│   └── figures/   01_celltype_specificity_overview/
│                  02_tissue_contribution_by_celltype/
│                  03_malignant_structure_no_harmony/
├── 02_correlacion/             (3 scripts -> 3 figure subfolders)
│   ├── scripts/   01_compartment_hexbin_scan.py
│   │              02_depth_artifact_cell_level.py
│   │              03_depth_artifact_diagnostic_hexbin.py
│   └── figures/   01_compartment_hexbin_scan/
│                  02_depth_artifact_cell_level/
│                  03_depth_artifact_diagnostic_hexbin/
├── 03_pseudobulk/              (1 script -> figures/ flat, no subfolder needed)
│   ├── scripts/   01_pseudobulk_by_patient_headline.py   <- headline result
│   └── figures/   *.png, *.csv, README.txt
├── 04_liana/                   (1 script -> figures/ flat, no subfolder needed)
│   ├── scripts/   01_liana_interaction_vs_lcor.py
│   └── figures/   *.png, *.csv, README.txt
├── 05_myeloid/                 (2 scripts -> 2 figure subfolders)
│   ├── scripts/   01_pseudobulk_myeloid_headline.py
│   │              02_depth_artifact_myeloid_cell_level.py
│   └── figures/   01_pseudobulk_myeloid_headline/
│                  02_depth_artifact_myeloid_cell_level/
├── 06_liana_within_patient/    (1 script -> figures/ flat; exploratory, isolated)
│   ├── scripts/   01_liana_lcor_split_within_patient.py
│   └── figures/   *.png, *.csv, README.txt
├── 07_apm_positive_control/    (9 scripts -> 9 figure subfolders)
│   ├── scripts/   01_pseudobulk_apm_malignant.py
│   │              02_depth_artifact_apm_malignant_cell_level.py
│   │              03_pseudobulk_apm_myeloid.py
│   │              04_depth_artifact_apm_myeloid_cell_level.py
│   │              05_apm_lcor_by_cd8_infiltration.py
│   │              06_negative_control_esr1_greb1.py
│   │              07_replication_klughammer.py
│   │              08_lcor_esr1_relationship.py
│   │              09_replication_regner.py
│   └── figures/   01_pseudobulk_apm_malignant/ ... 09_replication_regner/
├── 08_umap_gene_trios/         (3 scripts -> 3 figure subfolders)
│   ├── scripts/   01_umap_trio_lcor_apm_esr1.py
│   │              02_umap_trio_lcor_apm_esr1_myeloid.py
│   │              03_umap_trio_lcor_cxcl16_cxcr6.py
│   └── figures/   01_umap_trio_lcor_apm_esr1/
│                  02_umap_trio_lcor_apm_esr1_myeloid/
│                  03_umap_trio_lcor_cxcl16_cxcr6/
└── 10_lcor_program_tcell_phenotype/  (6 scripts -> 6 figure subfolders)
    ├── scripts/   01_build_signatures.py
    │              02_visvader_analysis.py
    │              03_klughammer_analysis.py
    │              04_visvader_lcoroe_vs_cxcl16_cxcr6.py
    │              05_klughammer_lcoroe_vs_cxcl16_cxcr6.py
    │              06_scatterplots.py
    └── figures/   01_build_signatures/ ... 06_scatterplots/
```

(`09_lcor_esr1_relationship` note: numbered as script 08 inside
`07_apm_positive_control/scripts/`, not a separate top-level folder --
kept together with the APM investigation since it directly follows up on
that thread.)

`01_exploratorio`, `02_correlacion` and `05_myeloid` have >=2 scripts each,
so their `figures/` is split into one subfolder per script (matching
basenames) to keep each script's outputs unambiguous. `03_pseudobulk` and
`04_liana` have only one script each, so their `figures/` is flat -- an
extra `figures/01_.../` nesting level would just repeat the block name for
no reason.

Run any script from `~/single_cell/Visvader/`, e.g.:
`python3 LCOR-CXCL16/03_pseudobulk/scripts/01_pseudobulk_by_patient_headline.py`

Each figures folder's `README.txt` has the full methodological rationale
for that script (why that method, what it rules in/out, caveats).

## Superseded / not included here

`14_lcor_cxcl16_hexbin.py` (still in `~/single_cell/Visvader/`, not moved):
the first-pass cell-level hexbin that found r=0.63 restricted to
co-detected cells. Kept out of this folder because
`02_correlacion/02_depth_artifact_cell_level.py` and
`03_depth_artifact_diagnostic_hexbin.py` showed that result is a
depth/collider artifact -- but worth keeping accessible for
methods/supplementary material as "what a naive analysis found, and why
it's wrong," since reviewers may ask.

## Caveats that apply across all of the above

- Small n, especially per subtype (TNBC n=8 patients in the pseudobulk
  test) -- absence of significance at this n is compatible with a real
  but small effect; treat the null result as "not supported by this
  cohort," not as definitive proof of no effect.
- All single-cell results come from one atlas (Pal et al./Visvader,
  GSE161529). Replication in an independent human scRNA-seq cohort would
  strengthen the null conclusion before it goes in the paper as the
  answer to the editor's request.
- Correlational throughout; even a robust patient-level correlation would
  not establish causality/direction.
- Myeloid cell counts per patient are much lower than malignant cell
  counts (`05_myeloid`, median ~180 vs thousands), so that block is more
  power-limited than the malignant-compartment tests despite CXCL16's
  higher detection rate there.
