# TIM3 (HAVCR2) x Hypoxia — Visvader, epithelial compartment

Final results. Nothing here is a new analysis — these are the
already-validated outputs from the depth-corrected design, gathered into one place
per signature.

## Dataset

Epithelial compartment, 122,101 cells (ER 74,287 / HER2 19,178 / TNBC 28,636).

## Important caveat — read this before the figures

HAVCR2 (TIM3) is a checkpoint gene normally expressed by immune cells, not epithelial
cells. In this compartment, cells we call "HAVCR2+" are overwhelmingly single-UMI
detections with essentially no dynamic range: 445 cells are HAVCR2+ (0.36%), of which
425 (95.5%) have exactly 1 UMI, and the compartment-wide ceiling is 3 UMIs. For
comparison, the immune compartment shows real dynamic range for the same gene (11.1%
positive, max count 13) — consistent with genuine expression there. The epithelial
signal looks like ambient RNA contamination from neighboring immune cells rather than
true epithelial TIM3 expression. (The same pattern was checked independently in the
Klughammer dataset and replicates: 98.4% single-UMI there too.)

**Practical implication**: treat the results below as a *detection-level* association
(does hypoxia make a single ambient/spillover HAVCR2 read more or less likely to turn
up in a cell) rather than a *dose-response* one. The magnitude-correlation analysis
(does HAVCR2 expression level scale with hypoxia, among positive cells) is included for
completeness but is the weakest and least reliable piece of evidence here — see below.

## Methodology

Two directions, both depth-corrected because HAVCR2 detection and the hypoxia score are
each independently correlated with per-cell sequencing depth:

1. **HAVCR2 status -> hypoxia score** (`violin_HAVCR2_vs_hypoxia.png`): HAVCR2+ (raw
   count > 0) vs HAVCR2- (raw count = 0), naive Mann-Whitney + depth-adjusted OLS
   (`hypoxia_z ~ HAVCR2_status + log(total_counts)`). Y-axis clipped to the pooled
   1st-99th percentile for readability (display only — stats/data unaffected).
2. **Hypoxia group -> % HAVCR2+** (`barplot_pct_HAVCR2_by_hypoxia_group.png`): cells
   split into hypoxia tertiles (top/bottom third, middle dropped), % HAVCR2+ per group,
   naive Fisher's exact test + depth-adjusted logistic regression
   (`HAVCR2_positive ~ hypoxia_group + log(total_counts)`).

Also included per signature (supporting evidence, not headline):
- `scatter_magnitude_naive.png` / `scatter_magnitude_depth_corrected.png`: within
  HAVCR2+ cells only, does HAVCR2 magnitude scale with hypoxia score, naively vs
  depth-corrected (both axes). This is where the original spurious "two-population"
  pattern lived (r=-0.28 to -0.41 naive, depending on signature) and it collapses
  toward null once depth is corrected in most signatures — i.e. no reliable
  dose-response.
- `umap_overview.png`: subtype / hypoxia score / HAVCR2+ cells in UMAP space.

`summary_all_signatures_pooled.csv` pulls the "All subtypes" pooled statistics from
every signature into one table for quick scanning.

## Headline result (pooled, depth-adjusted)

| signature | HAVCR2+ vs HAVCR2- hypoxia, depth-adj p | hypoxia-high vs -low, % HAVCR2+, depth-adj p (OR) |
|---|---|---|
| HALLMARK_HYPOXIA | 2.6e-4 | 4.7e-4 (OR=1.6) |
| GO_0001666 | 4.4e-14 | 5.4e-9 (OR=2.2) |

Both signatures agree on direction in both test directions: hypoxic epithelial cells
are modestly but significantly more likely to carry a detectable HAVCR2 read. This
replicates independently in the Klughammer Malignant compartment (see that dataset's
own results folder). TNBC is the least consistent subtype (weak/null in several
signatures) — worth flagging rather than over-interpreting.

The remaining 5 signatures (GO/Hallmark variants with fewer genes) are weaker or mixed
— see `summary_all_signatures_pooled.csv` and the per-signature folders.

## Quartile sensitivity check

We also tried splitting hypoxia-high/low on quartiles
(top/bottom 25%) instead of tertiles (top/bottom 33%), to see if a more extreme split
sharpens the %HAVCR2+ difference: `barplot_pct_HAVCR2_by_hypoxia_group_quartiles.png`
/ `stats_proportions_by_hypoxia_group_quartiles.csv`. Result is mixed, not a clear
improvement — pooled depth-adjusted OR goes from 1.60 (tertiles) to 1.75 (quartiles)
for HALLMARK_HYPOXIA, a small gain, but HER2/TNBC (already the smaller subgroups)
get noisier with fewer cells rather than more reliable. We're keeping the tertile
version as the primary result and including quartiles here as a sensitivity check.

## Folder structure

```
results_Visvader_TIM3_hypoxia/
  README.md                                  <- this file
  summary_all_signatures_pooled.csv
  <signature_name>/
    violin_HAVCR2_vs_hypoxia.png
    barplot_pct_HAVCR2_by_hypoxia_group.png
    barplot_pct_HAVCR2_by_hypoxia_group_quartiles.png   <- sensitivity check
    scatter_magnitude_naive.png
    scatter_magnitude_depth_corrected.png
    umap_overview.png
    stats_violin_HAVCR2_vs_hypoxia.csv
    stats_correlation_magnitude.csv
    stats_proportions_by_hypoxia_group.csv
    stats_proportions_by_hypoxia_group_quartiles.csv    <- sensitivity check
```

Signatures included: HALLMARK_HYPOXIA, GO_0001666 (primary two), plus
GAVISH_3CA_MALIGNANT_METAPROGRAM_6_HYPOXIA, GOBP_REGULATION_OF_CELLULAR_RESPONSE_TO_HYPOXIA,
GOBP_HIF1A_SIGNALING_PATHWAY, GOBP_NEG_REG_HYPOXIA_INDUCED_APOPTOTIC_SIGNALING,
GOBP_POS_REG_CELLULAR_RESPONSE_TO_HYPOXIA (supporting/consistency check).
