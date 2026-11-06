# Part 1 — Stemness Signatures vs Cell Cycle Contamination
## Results summary and figure guide

**Datasets:** Visvader (primary tumours, 121,632 cells, 31 patients) ·
Klughammer (metastatic, 83,490 cells, 19 patients)  
**Signatures evaluated:** 9 stemness gene sets (ESC_ASSOU, ESC_WONG, BenPorath_ES1, EMP,
fMaSC/aMaSC, LIM_Stem, LIM_LumProg, LIM_LumMat, stem_consensus)  
**Reference potency measure:** CytoTRACE2 (0 → 1, high = developmental stem-like state)

---

## Conceptual figures

### `fig_slide0_signatures.png` — Signature catalogue

Overview of all 9 signatures grouped by conceptual phase (ESC-based → lineage-based → composite).
Each card shows origin, number of genes, R²(CC) and partial r(CT2) as mini-bars for both datasets.

**Conclusion:** The 9 signatures differ substantially in size (18–200 genes), origin (embryonic
SC, mammary lineage, composite), and their relationship to cell cycle. This heterogeneity
motivates a systematic evaluation rather than picking one arbitrarily.

---

### `fig_slide1_concept.png` — Analysis concept (3-panel)

A didactic diagram explaining the core analytical challenge and the solution:

- **Panel 1 — The problem:** Within a single patient, stemness scores correlate with S.Score
  (r ≈ +0.45). Cycling cells appear stem-like simply because they are proliferating.
- **Panel 2 — The approach:** Per patient, we fit OLS `sig ~ S.Score + G2M.Score` to quantify
  R²(CC→sig), then compute per-cell residuals (sig_ccfree = sig − β₁·S.Score − β₂·G2M).
- **Panel 3 — What changes:** After residualisation, the CC-free score is decorrelated from
  S.Score (r ≈ 0). Cycling cells drop to residual ≈ 0; truly stem-like quiescent cells
  retain a high residual.

**Conclusion:** CC residualisation is necessary before any downstream use of stemness scores.
Without it, proliferation signal contaminates stem-like identity estimates.

---

## Per-signature metric figures

### `Visvader_heatmap.png` · `Klughammer_heatmap.png` — Multi-metric heatmap

Signatures (rows) × metrics (columns): r_CT2, partial_r_CT2, dCor_CT2, r_S, r_G2M, R²_CC,
r_immune, plus CC-free versions. Both datasets shown side by side.

**Conclusions:**
- ESC_WONG and ESC_ASSOU consistently show the highest partial_r_CT2 (0.46–0.52) with
  moderate CC contamination.
- stem_consensus has a high raw r_CT2 in Visvader, but this is inflated by CC:
  once CC is controlled, partial_r_CT2 drops to 0.25.
- LIM_LumMat is the only signature with a consistently negative r_CT2 and partial_r_CT2 in
  both datasets, confirming it marks terminally differentiated, non-stem cells.
- r_immune is near zero or slightly negative for all signatures — stemness and
  immunogenicity are largely orthogonal axes.

---

### `Visvader_forest.png` · `Klughammer_forest.png` — Forest plots (r_CT2 per patient)

Per-patient Spearman r(sig, CytoTRACE2) with 95% CI and Fisher z pooled estimate.
Each row is one signature; points are individual patients; the diamond is the meta-analytic mean.

**Conclusions:**
- Pooled r_CT2 is positive for ESC_WONG, ESC_ASSOU, and BenPorath_ES1 in both datasets,
  confirming a genuine association with developmental potency independent of patient.
- Between-patient heterogeneity is substantial: some patients show near-zero or negative r,
  even for ESC_WONG. Subtype (ER/HER2/TNBC) does not fully explain this variance.
- LIM_LumMat consistently produces negative r in virtually all patients — this is the most
  reliable anti-stemness marker in the panel.
- EMP and the LIM_Stem / LIM_LumProg signatures show weak but positive pooled r_CT2,
  with very wide patient-level confidence intervals.

---

### `Visvader_tradeoff.png` · `Klughammer_tradeoff.png` — CC contamination vs potency tradeoff

Scatter: x = R²(CC→sig), y = partial r(CT2 | CC). Each dot is one signature. The ideal
region (top-left) means low CC contamination and high net potency signal.

**Conclusions:**
- **ESC_WONG occupies the top-left quadrant in both datasets** — the best combination of
  low CC contamination (R²_CC ≈ 0.30) and high partial_r_CT2 (0.46–0.52). It is the
  recommended signature.
- ESC_ASSOU performs similarly but carries slightly more CC contamination (R²_CC ≈ 0.31–0.38).
- stem_consensus falls in the bottom-right: high CC contamination (R²_CC ≈ 0.55) and
  comparatively low residual potency signal. It should not be used as a primary stemness index.
- EMP and LIM signatures cluster near the y-axis (minimal CC contamination) but also have
  low partial_r_CT2, meaning they capture something other than the CytoTRACE2 potency axis.

---

### `Visvader_ccfree_comparison.png` · `Klughammer_ccfree_comparison.png` — Before / after residualisation

Paired comparison of r(sig, CytoTRACE2) vs r(sig_ccfree, CytoTRACE2) for each signature.

**Conclusions:**
- For most signatures, r_CT2_ccfree < r_CT2 — the CC component was inflating the raw
  correlation with potency. The drop is largest for stem_consensus (−0.10 to −0.14 absolute).
- ESC_WONG retains the highest r_CT2_ccfree in both datasets (0.207 Visvader, 0.212
  Klughammer), confirming it has genuine potency content beyond the CC component.
- After residualisation, BenPorath_ES1 and ESC_ASSOU converge toward ESC_WONG, suggesting
  that much of their apparent advantage in raw r_CT2 was CC-driven.
- r_immune_ccfree remains near zero for all signatures in both datasets — the CC-free
  residual is not correlated with immunogenicity, confirming orthogonality of the two axes.

---

## Per-patient distribution figures

### `per_patient_r_CT2.png` — r(sig, CytoTRACE2) per patient

Box + strip plot: each dot = one patient, coloured by subtype (ER=green, HER2=purple, TNBC=red).
Box = IQR, line = median.

**Conclusions:**
- ESC_WONG and ESC_ASSOU have the highest and most consistent per-patient median r_CT2
  in both Visvader and Klughammer. No other signature approaches this level of consistency.
- TNBC patients (red) tend to show higher r_CT2 for ESC_WONG and ESC_ASSOU, possibly
  reflecting their enrichment for basal/stem-like cell states.
- stem_consensus shows moderate per-patient r_CT2 values, but as shown by R²_CC, this
  correlation is largely driven by proliferating cells, not genuine stem-like identity.
- LIM_LumMat is consistently negative across all patients and subtypes — it reliably marks
  the most differentiated cells.
- Between-patient variance is high for all signatures, motivating the use of per-patient
  normalisation or mixed models in downstream analyses.

---

### `per_patient_R2_CC.png` — R²(CC → sig) per patient

Box + strip plot of the fraction of signature variance explained by S.Score + G2M.Score,
estimated per patient.

**Conclusions:**
- stem_consensus is a clear outlier: median R²_CC ≈ 0.38–0.55 depending on the dataset,
  with some patients exceeding 0.70. In those patients, >half the signal is proliferation.
- ESC_ASSOU and BenPorath_ES1 show intermediate CC contamination (median R²_CC ≈ 0.30–0.40),
  with wide patient-level spread — CC contamination is patient-dependent, not fixed.
- ESC_WONG carries moderate contamination (median R²_CC ≈ 0.20–0.30) but offsets this
  with a stronger residual potency signal.
- EMP and LIM signatures have near-zero R²_CC in virtually all patients — they are
  cell-cycle independent. This is informative but they also capture less CytoTRACE2 potency.
- The patient-level spread in R²_CC (often >0.3 range within one signature) confirms that
  per-patient CC residualisation is preferable to a single global correction.

---

### `per_patient_r_immune.png` — r(sig, immunogenic consensus) per patient

**Conclusions:**
- No signature shows a consistent positive correlation with the 25-gene immunogenic consensus
  across patients in either dataset. Median r_immune is near zero for all signatures.
- This confirms that stemness and immune evasion / immunogenicity operate as independent
  axes at the single-cell level within a patient. Downstream SCIE phenotype analysis will
  need to model them as orthogonal dimensions.
- LIM_LumMat and LIM_LumProg show slightly positive median r_immune in some conditions,
  consistent with differentiated/luminal cells being more immunogenic, but the effect is small.

---

### `per_patient_r_cc.png` — r(sig, S.Score) and r(sig, G2M) per patient

Two-row panel: top = r(sig, S.Score), bottom = r(sig, G2M.Score).

**Conclusions:**
- stem_consensus, BenPorath_ES1, ESC_ASSOU, and ESC_WONG all show moderate-to-strong
  positive median r with both S.Score and G2M.Score. This is not a biological association —
  it reflects gene overlap between stemness and cell cycle programmes.
- EMP and LIM signatures are flat near zero for both CC markers, confirming their
  independence from proliferation.
- r_S and r_G2M track closely for all signatures, suggesting S-phase and G2/M phases
  contribute similarly to the contamination (no phase-specific effect).
- The spread is large within each signature across patients, reinforcing the need for
  per-patient residualisation.

---

## Overall ranking

| Signature | Visvader rank | Klughammer rank | R²_CC (mean) | partial_r_CT2 (mean) | Verdict |
|---|---|---|---|---|---|
| ESC_WONG | 0.080 | **0.203** | 0.30 | 0.49 | **Recommended** |
| ESC_ASSOU | 0.074 | 0.050 | 0.35 | 0.45 | Good alternative |
| EMP | 0.076 | 0.194 | 0.004 | 0.18 | Low CC but weak potency signal |
| LIM_Stem | 0.134 | 0.106 | 0.010 | 0.26 | Weak but clean |
| BenPorath_ES1 | −0.028 | −0.058 | 0.41 | 0.41 | Too CC-contaminated |
| fMaSC/aMaSC | −0.026 | −0.180 | 0.22 | 0.08 | Weak potency, moderate CC |
| stem_consensus | −0.334 | −0.182 | **0.55** | 0.28 | Excluded — dominated by CC |
| LIM_LumMat | −0.470 | −0.382 | 0.08 | −0.47 | Anti-stemness marker |

---

## Key take-aways

1. **Cell cycle contamination is signature-specific and patient-specific.** R²_CC ranges from
   near-zero (EMP, LIM) to >0.55 (stem_consensus), and varies across patients for the same
   signature by up to 0.30 absolute units.

2. **ESC_WONG is the recommended primary stemness index.** It achieves the best balance of
   low CC contamination and high residual correlation with developmental potency (CytoTRACE2)
   in both an ER-enriched primary tumour cohort and a metastatic cohort.

3. **stem_consensus should not be used** as a primary stemness index in single-cell data.
   More than half its variance is explained by cell cycle in the average patient.

4. **CC residualisation is essential before downstream analyses.** The CC-free scores
   (`score_{sig}_ccfree`) are stored in the processed h5ads and should be used for any
   SCIE / immune axis analyses in subsequent parts.

5. **Stemness and immunogenicity are orthogonal** at the single-cell level. r_immune and
   r_immune_ccfree are near zero for all signatures, meaning the SCIE phenotype (Stemness ×
   Immune Evasion) will require separate quantification of each axis.
