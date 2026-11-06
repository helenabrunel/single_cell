# Decision log

Why the main analytical decisions were made. Details and full results are in the linked notes and in each script's header.

## Stemness and cell cycle

**Evaluate stemness signatures against the cell cycle.** Stemness signatures, especially ESC-like ones, correlate with cell cycle in scRNA-seq. A high score could reflect proliferation rather than potency, so each signature is compared with CytoTRACE2 and with cell-cycle scores. See `scripts/04_stemness_cell_cycle/`.

**Use more than Pearson r for the stem-immune relationship.** The pattern is not linear, so Pearson r underestimates the association. See `scripts/05_stemness_immune_axis/04_compare_stem_immune_metrics.py`.

## TIM3 and hypoxia

**Abandon the threshold-defined groups.** The first design compared TIM3-high/hypoxia-low cells with TIM3-low/hypoxia-high cells. Those groups turned out to be confounded with sequencing depth: a shallow cell gets an inflated HAVCR2 value and a deflated hypoxia score.

**Restart with a depth-corrected design.** The clean analysis splits cells into HAVCR2-positive and HAVCR2-negative by raw count and corrects for depth. See `scripts/06_tim3_hypoxia/11_clean_analysis_primary.py`.

**Test the reverse direction.** Comparing percent HAVCR2-positive across hypoxia groups checks the same relationship from the other side. Quartiles were added to the tertiles because a more extreme split might increase the difference.

**Check single-UMI positives.** Removing cells with expression close to zero tests whether weak positives drive the depth artifact.

## LCOR, CXCL16 and CXCR6

**Do not rely on per-cell correlation alone.** Uncorrected cell-level correlation can reflect sequencing depth or differences between patients. Key tests therefore use a within-patient, depth-offset GLM at cell level and pseudobulk at patient level.

**Repeat the test in myeloid cells.** CXCL16 detection is about three times higher in myeloid than in malignant cells, so the malignant-cell tests might have missed a relationship in the compartment that expresses it most.

**Add a positive control.** The null LCOR-CXCL16 result could mean no relationship, or a method unable to detect one in sparse data. LCOR and antigen presentation (APM) are linked in the lab's earlier experiments, so that pair was run through the identical pipeline.

**Do not treat that control as validating the null.** At cell level LCOR and APM came out negatively associated, the opposite of the experimental direction. A known relationship that appears inverted cannot add confidence to the CXCL16 null.

**Test immune selection as an explanation.** If CD8 T cells preferentially removed high-APM cells, the negative association should be stronger in highly infiltrated tumors. It was not, so this explanation is not supported.

**Check the method with an unrelated known relationship.** ESR1 and its direct targets GREB1 and PGR are well established and unrelated to LCOR or immunity. The design recovered them in the expected direction in all three cohorts, so it does not systematically invert signals.

**Replicate in independent cohorts.** The LCOR-APM direction differed between cohorts (negative, positive and null) while the ESR1 control held. It is therefore not a reproducible signal in human tumor single-cell data, and it depends on something specific to each cohort.

**Examine the ESR1 relationship.** The lab's paper (PMID 41289011) proposes that ERα sequesters LCOR, which predicts a weaker LCOR-APM link in ER+ cells. This fits one cohort but does not fully explain the other, because non-ER+ cells were also negative there.

## LCOR program and T-cell phenotype

**Use a signature instead of raw LCOR.** LCOR is too lowly expressed for reliable correlation, so a proxy signature (LCOR_OE) was built from the lab's experimental data, independently of the single-cell data.

**Treat the T-cell signatures differently.** Tcell_UP24H mostly reflects generic activation and proliferation, so it needs caution. Tcell_UP96H includes CXCR6 and ITGAE (CD103) and looks like a differentiated, tissue-resident state.

**Add an unbiased test.** Comparing LCOR-high and LCOR-low patients without assuming a marker avoids depending on the CXCL16/CXCR6 hypothesis.

**Replace Leiden reclustering with CellTypist.** Reference labels come from outside this data, so they cannot pick up a patient-batch signal the way de novo clustering can.

**Use non-integrated data for patient-level scores.** See `docs/lcor_tcell_scores.md`.

## Gamma-delta T cells

**Calibrate on a dataset with TCR data.** A rule for calling gamma-delta cells from expression alone needs ground truth, which TCR sequencing provides. The rule is then applied where no TCR data exists.

**Test ambient RNA correction.** decontX checks whether background RNA explains TRDC and TRGC detection.

**Check NK-labelled cells.** CD3E is detected in many NK-labelled cells, so label accuracy, T-like subclusters and NK+T doublets were examined. See `docs/gamma_delta_expression_only_call.md`.

## Open questions

- What explains the inverted LCOR-APM direction in one cohort: protocol, platform or patient composition?
- Why is raw LCOR negatively associated with Tcell_UP96H in two cohorts, the opposite direction to the hypothesis?
