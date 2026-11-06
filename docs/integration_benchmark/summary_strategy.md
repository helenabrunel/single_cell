# Part 0 — Summary Figure Strategy & Narrative

## Which UMAP to use for a summary

**Use Klughammer** — it has the most dramatic batch effect (batch mixing
score 0.005 unintegrated vs 0.352 Harmony) so the contrast is unambiguous.
Visvader is cleaner and would understate the problem.

### Best single summary figure: 2 rows × 4 columns

|  | Patient | ESC_ASSOU | Immunogenic consensus | CytoTRACE2 |
|---|---|---|---|---|
| **Unintegrated** | batch visible | signal inside patient blobs | — | — |
| **Harmony** | mixed | signal crosses patients | preserved | preserved |

Use **ESC_ASSOU** (or ESC_WONG) as the stemness column — these are the
signatures most aligned with CytoTRACE2 (r = 0.43–0.47 in Klughammer)
and least contaminated by cell cycle. Better than LIM_Stem for a summary
because it makes the biological argument clearer.

**Source file:** `figures/part0/sigs_panel/Klughammer_ESC_ASSOU_panel.png`
→ crop to rows 1–2 (Unintegrated + Harmony), drop scVI and DRVI for the
main figure. scVI and DRVI go to supplement alongside the batch mixing
boxplot (`Klughammer_metrics_summary.png`).

---

## How to explain integration vs per-patient analysis

The key conceptual point is that they answer **different questions**:

**Integration (Harmony)** corrects the *embedding* — it moves cells in
PCA space so that patients overlap in the UMAP. This is for
**visualization only**. It lets you show all patients together in one plot.

**Per-patient z-score** corrects the *scores* — it removes each patient's
mean and variance from the stemness/immune values before pooling them.
This is for **quantification**.

### The argument for keeping per-patient analysis as the primary strategy

Stemness and immune scores are derived from the expression matrix, not
the embedding. The per-patient Spearman r(stemness × immune) — the key
signal we are quantifying — is identical across all four integration
methods (median −0.099 in Klughammer regardless of method), because it
never touches the UMAP coordinates. What varies between methods is how
well patients overlap visually. Integration is therefore useful for
showing the data, but the biological signal we measure is invariant to it.

### One-sentence methods summary

> "Integration (Harmony) was used for visualization; all quantitative
> analyses were performed on per-patient z-score normalized scores to
> remove inter-patient mean/variance shifts while preserving
> within-patient biological variation."

---

## Addendum — scib-metrics validation of per-subtype integration

The pooled batch-mixing / stemness-coherence metrics above are ad hoc. As a
more rigorous check specifically on the **per-subtype** integration (ER /
HER2 / TNBC integrated separately — the design used throughout this
project), we benchmarked Unintegrated / Harmony / DRVI with the standardized
[scib-metrics](https://github.com/YosefLab/scib-metrics) battery (iLISI,
kBET, graph connectivity, NMI/ARI, silhouette, cLISI) on the 5 qualifying
dataset×subtype objects. See `INTERPRETATION.md` addendum for the full
table and method notes (scripts: `part0_scib_prepare.py` +
`part0_scib_benchmark.py`; isolated `scib-metrics` conda env).

**Headline result:** DRVI has the best aggregate (batch + bio) score in 4/5
subtype objects; Harmony wins only for Visvader ER, by a margin of 0.011.
Unintegrated is worst overall in every object.

**Why this doesn't change the methods summary sentence below:** the scib
bio-conservation label is a discrete lineage/state proxy (FINE_EPI marker
argmax), not the continuous stemness gradient central to the SCIE
hypothesis. So this result validates that per-subtype batch correction is
doing its job (patients mix, discrete lineage identity is largely retained)
— it doesn't argue for using DRVI's embedding in place of per-patient
z-scored expression for the actual stemness×immune quantification. Harmony
remains the pick for main-figure UMAPs (simpler, linear, easier to explain
to reviewers); DRVI's marginally better scib score is noted as supplementary
validation, not a reason to switch the primary visualization.

## Main vs supplement allocation

| Figure | Where |
|---|---|
| `Klughammer_ESC_ASSOU_panel.png` rows 1–2 (Unintegrated + Harmony) | **Main** |
| `Klughammer_metrics_summary.png` (batch mixing + stemness coherence, all 4 methods) | Supplement |
| `Visvader_ESC_ASSOU_panel.png` rows 1–2 | Supplement (replication) |
| `qc_Visvader.csv` + `qc_Klughammer.csv` | Supplement table |
| `scib_inputs/{object}_scib_results.svg` (5 objects) + `scib_summary_all_objects.csv` | Supplement (per-subtype integration validation) |
