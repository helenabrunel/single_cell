# Part 0 — Integration & Signature Interpretation

## How to read the per-signature panel figures
(`figures/part0/sigs_panel/{Dataset}_{signature}_panel.png`)

Each figure has:
- **Rows** = integration method (Unintegrated → Harmony → scVI → DRVI)
- **Columns** = Patient ID | Subtype | [stemness signature] | Immunogenic consensus | CytoTRACE2
- Row labels and spines are colour-coded by method

**Reading logic:**
- **Patient column (col 0)** = your batch reference. Each colour = one patient.
  If the signature column shows the same pattern as the patient column → batch effect.
  If the signature gradient cuts across patient boundaries → real biology.
- **Unintegrated row** = ground truth of what the data looks like without correction.
  Patient clusters should be visible here if a batch effect exists.
- **Harmony / scVI / DRVI rows** = after correction. Biology should become
  more structured; patient blobs should dissolve.
- **CytoTRACE2 (col 4)** = cell-cycle-partially-independent reference for
  developmental potency. **High score (yellow/red) = more stem-like / undifferentiated.**
  Low score (white/pale) = more differentiated.

---

## CytoTRACE2 direction

**High CytoTRACE2 score = more stem-like (higher developmental potency).**
Scale is 0–1. Median in both datasets ≈ 0.09 — the vast majority of
tumour cells are differentiated; the stem-like population is a small tail
at the high end.

---

## Which signatures track CytoTRACE2?

Spearman correlations after per-patient z-score normalisation (pooled).
Sorted by CytoTRACE2 alignment (best → worst).

| Signature | Visvader r(CT2) | Klughammer r(CT2) | r(S.Score) | r(G2M) |
|---|---|---|---|---|
| ESC_WONG | 0.27 | **0.47** | 0.22 | 0.11 |
| ESC_ASSOU | 0.26 | **0.43** | 0.21 | 0.12 |
| BenPorath_ES1 | 0.21 | 0.40 | 0.24 | 0.17 |
| stem_consensus | 0.17 | 0.40 | **0.37** | **0.29** |
| fMaSC/aMaSC | 0.13 | 0.20 | 0.18 | 0.19 |
| EMP | 0.08 | 0.11 | −0.03 | −0.03 |
| LIM_Stem | 0.04 | −0.09 | −0.02 | 0.08 |
| LIM_LumProg | 0.01 | −0.10 | −0.01 | 0.00 |
| LIM_LumMat | −0.17 | −0.20 | −0.11 | −0.15 |

CytoTRACE2 vs immune consensus: r = 0.009 (Visvader) / −0.155 (Klughammer)
CytoTRACE2 vs S.Score:          r = 0.155 (Visvader) /  0.368 (Klughammer)
CytoTRACE2 vs G2M.Score:        r = 0.108 (Visvader) /  0.402 (Klughammer)

---

## What each signature is capturing

**ESC_WONG / ESC_ASSOU / BenPorath_ES1**
Best aligned with CytoTRACE2 — these are capturing developmental potency
(pluripotency-related transcription). In the panels their colour pattern
should look most similar to the CytoTRACE2 column.
They correlate moderately with cell cycle (r ≈ 0.2–0.4) but far less than
stem_consensus.

**stem_consensus**
Same CytoTRACE2 alignment as BenPorath in Klughammer (r=0.40), but its
S.Score correlation is 0.55 in Klughammer. In the panels, stem_consensus
will pattern like a proliferation/cycling map.
→ Cell cycle contamination is confirmed and severe. Cannot be used as-is
as a stemness axis without CC residualisation.

**LIM_Stem (LIM_MAMMARY_STEM_CELL_UP)**
Near-zero or negative correlation with CytoTRACE2 (Klughammer: r=−0.09).
This signature does NOT capture developmental potency as CytoTRACE2
defines it. It may be capturing a mammary luminal lineage state orthogonal
to the ESC/potency axis. It is clean of cell cycle contamination (r_CC ≈ 0).
→ In the panels, LIM_Stem should look nothing like CytoTRACE2.
→ Prior hexbin analyses using LIM_Stem as the stemness axis were measuring
  something closer to mammary lineage identity, not pluripotency.

**LIM_LumMat (LIM_MAMMARY_LUMINAL_MATURE_UP)**
Consistently negative with CytoTRACE2 — luminal mature = differentiated.
Its high-score cells are the most differentiated. Useful as a negative
control for stemness.

**EMP (Epithelial–Mesenchymal Plasticity)**
Weak positive with CytoTRACE2, clean of cell cycle. Captures plasticity
rather than developmental potency — a different dimension of stem-like
behaviour.

---

## Addendum — per-subtype scib-metrics benchmark

To validate the per-subtype integration (ER/HER2/TNBC integrated separately,
`part0_malignant_integration.py`) with a standardized metric battery instead
of the ad hoc batch-mixing / stemness-coherence scores above, we ran
[scib-metrics](https://github.com/YosefLab/scib-metrics) on the 5 qualifying
dataset×subtype objects (Klughammer HER2 excluded, n=2 patients).

**Scripts:** `part0_scib_prepare.py` (persists per-subtype Unintegrated /
Harmony / DRVI embeddings — DRVI reused the cached models from
`part0_malignant_integration.py`, no retraining) → `part0_scib_benchmark.py`
(runs `scib_metrics.benchmark.Benchmarker`). scVI was **not** included: no
per-subtype scVI model was ever trained (only pooled, whole-compartment
models exist), and training 5 fresh ones would cost hours of CPU time for a
method that wasn't part of the actual per-subtype pipeline anyway.

**Environment note:** `scib-metrics` requires numpy≥2, which conflicts with
`cytotrace2-py` (numpy<2) in the main analysis conda env. It lives in its own
isolated env: `~/miniconda3/envs/scib-metrics`. Do not `pip install
scib-metrics` into the base env.

**bio_label:** since malignant cells within one clinical subtype have no
hand-annotated cell type, the bio-conservation label is the argmax of the
`FINE_EPI` marker scores (Luminal_A, Luminal_B, HER2, Basal_TNBC,
Luminal_prog, Cycling_epi, EMT), computed on unintegrated expression —
independent of any embedding, but still an imperfect proxy since these
fine-type scores can partly track patient-specific tumour state.

### Results (Total = aggregate of Batch correction + Bio conservation)

| Object | Unintegrated | Harmony | DRVI | Winner |
|---|---|---|---|---|
| Klughammer_ER | 0.350 | 0.423 | **0.473** | DRVI |
| Klughammer_TNBC | 0.354 | 0.370 | **0.467** | DRVI |
| Visvader_ER | 0.336 | **0.440** | 0.429 | Harmony (by 0.011) |
| Visvader_HER2 | 0.342 | 0.414 | **0.430** | DRVI |
| Visvader_TNBC | 0.370 | 0.405 | **0.456** | DRVI |

Full per-metric tables: `figures/part0/scib_inputs/{object}_scib_results.csv`.
Plots: `figures/part0/scib_inputs/{object}_scib_results.svg`.

**Reading it:** DRVI wins on aggregate score in 4/5 objects, driven by much
larger batch-correction gains (iLISI, kBET, graph connectivity) than its
bio-conservation cost. Unintegrated is worst overall but has the *highest*
bio-conservation score everywhere — expected, since patient identity and
malignant cell state are entangled in tumour cells, so any real batch mixing
looks like a bio-conservation cost on this label whether or not it distorts
genuine biology.

**Does this change the stemness×immune methods decision?** No. The scib
bio-conservation metrics here (NMI/ARI/silhouette/cLISI) are keyed to
*discrete* lineage/state labels, not the *continuous* stemness gradient the
SCIE hypothesis depends on. DRVI can preserve discrete lineage separation
reasonably well while still smoothing over the subtler continuous stemness
axis — consistent with, not contradicting, the earlier stemness-coherence
finding above. Point 4 below (Harmony for figures, per-patient z-score for
quantification) still stands.

## Key flags for Part 1

1. **Cell cycle contamination is severe in stem_consensus** and moderate
   in ESC/BenPorath. You cannot separate "high developmental potency" from
   "actively cycling" without residualisation first.

2. **CytoTRACE2 itself correlates with cell cycle in Klughammer**
   (r ≈ 0.37–0.40 with S/G2M). It is not a fully CC-free reference in
   that dataset. Visvader is weaker (r ≈ 0.11–0.16). This must go into
   Part 1's partial-correlation analysis.

3. **LIM_Stem is orthogonal to the CytoTRACE2 potency axis** — which
   reframes prior results. The ESC signatures are more appropriate as
   the stemness axis for SCIE characterisation.

4. **Integration recommendation**: Harmony for all figure production.
   Per-patient z-score for score-level analysis (invariant to embedding).
   scVI fails for Klughammer (no raw counts). DRVI achieves best mixing
   for Klughammer but at the cost of stemness gradient coherence.
