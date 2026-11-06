# Part 0 — Technical Methods Summary

## Script
`part0_malignant_integration.py`
Run from: `~/single_cell/`

---

## 1. Data loading

### Visvader (epithelial compartment)
- Source: `Visvader/compartment_epithelial.h5ad`
- Already subset to epithelial/malignant cells by script `03_split_integrate.py`
  (winner-takes-all compartment scoring + PTPRC exclusion)
- Raw counts available in `layers["counts"]`; log-normalised in `layers["logcounts"]`

### Klughammer (malignant compartment)
- Source: `Klughammer/klughammer_prepared.h5ad`, filtered to `compartments == "Malignant"`
- No raw counts available — only log-normalised X
- Cell cycle scores (S.Score, G2M.Score, Phase) were pre-computed by Seurat
  and are already in `obs`

---

## 2. QC and patient filtering

Per-patient cell counts were computed and a minimum threshold of **500 cells per patient**
was applied. Patients below threshold are excluded from all downstream steps.

QC metrics reported per patient: n_cells, mean_genes_per_cell, mean_pct_mitochondrial_reads.

Outputs: `figures/part0/qc_Visvader.csv` and `figures/part0/qc_Klughammer.csv`

---

## 3. Normalisation

### Visvader
Expression matrix set to log-normalised counts (`layers["logcounts"]`) for all scoring steps.
`logcounts` = `normalize_total(target_sum=10,000)` followed by `log1p`, computed in
script `02_qc_filter_normalize.py`. Raw counts (`layers["counts"]`) are retained and
used as input to scVI (negative binomial likelihood).

### Klughammer
Only log-normalised X available (no raw counts). Used directly for all steps.
scVI run with `gene_likelihood="normal"` as a consequence.

---

## 4. Signature scoring

All scores computed with `scanpy.tl.score_genes` (Seurat-style average difference
between gene set and random control genes) on log-normalised expression.

### Stemness signatures (9 total, from `signatures_confirmed.json`)
| Key | Full name | N genes (Vis/Klug) |
|---|---|---|
| ESC_ASSOU | Embryonic stem cell (Assou) | 645 / 637 |
| ESC_WONG | Embryonic stem cell (Wong) | 289 / 285 |
| BENPORATH_ES1 | Ben-Porath ES module 1 | 318 / 314 |
| EMP | Epithelial–mesenchymal plasticity | 13 / 14 |
| fMaSC_over_aMaSC | Foetal vs adult mammary stem cell | 1701 / 1675 |
| LIM_MAMMARY_STEM_CELL_UP | LIM mammary stem cell | 465 / 478 |
| LIM_MAMMARY_LUMINAL_PROGENITOR_UP | LIM luminal progenitor | 58 / 57 |
| LIM_MAMMARY_LUMINAL_MATURE_UP | LIM luminal mature | 108 / 109 |
| stem_consensus | Consensus stemness | 55 / 55 |

### Immunogenic consensus signature (25 genes, both datasets)
Hand-curated from KEGG Antigen Presentation Machinery (APM) + Hallmarks IFN-α/γ:
B2M, HLA-A/B/C, HLA-DMA/DMB/DOA/DOB/DPA1/DPB1/DQA1/DQA2/DQB1/DRA/DRB1,
CD74, TAP1, TAP2, TAPBP, CIITA, PSME1, PSME2, IFI30, CD8A, CD8B.
Stored as `obs["score_immune"]`.

### Cell cycle
- **Visvader**: computed with `scanpy.tl.score_genes_cell_cycle` using the
  standard Seurat S-phase (43 genes) and G2M-phase (54 genes) gene lists.
  Columns: `S.Score`, `G2M.Score`, `Phase`.
- **Klughammer**: pre-computed by Seurat, already in `obs`.

### CytoTRACE2
Pre-computed scores loaded from CSV files:
- `Visvader/cytotrace2_scores_Epithelial_Global.csv`
- `Klughammer/cytotrace2_scores_klughammer_Malignant_Global.csv`

Matched to cells by barcode. High score (→1) = more stem-like / undifferentiated;
low score (→0) = more differentiated.

---

## 5. Integration

### Shared preprocessing (all methods)
- **HVG selection**: 2,000 highly variable genes, Seurat flavour,
  batch-aware (`batch_key = patient`) when >1 patient present
- **Scaling**: `scanpy.pp.scale(max_value=10)` on HVG submatrix
- **PCA**: 30 components, ARPACK solver, `random_state=42`
- **Neighbours**: k=15, using the integration-specific representation

### Unintegrated (baseline)
Neighbours and UMAP built directly on PCA (`X_pca_fresh`). No batch correction.

### Harmony
`harmonypy.run_harmony` applied to PCA embedding.
Parameters: `max_iter_harmony=20`, `random_state=42`.
Neighbours and UMAP built on `X_harmony`.

### scVI
- Library: `scvi-tools` v1.4.3
- Latent dimensions: 20
- Gene likelihood: `"nb"` (Visvader, raw counts) / `"normal"` (Klughammer, log-X)
- Training: subsample of 20,000 cells; all cells projected after training
- `max_epochs=100`, `batch_size=1024`, early stopping with patience=10,
  validation check every 5 epochs
- All cells projected via `model.get_latent_representation()`
- Neighbours and UMAP built on `X_scVI`
- Models cached to `figures/part0/{dataset}_scvi_model/`

### DRVI
- Library: `drvi-py` v0.2.5
- Latent dimensions: 40 (2 × scVI latent)
- `is_count_data=True` (Visvader) / `False` (Klughammer)
- Same training parameters as scVI
- Models cached to `figures/part0/{dataset}_drvi_model/`

---

## 6. Evaluation metrics

### Batch mixing score
Fraction of k-nearest neighbours (from the connectivity graph) belonging to
a **different patient**. Computed on a random subsample of 5,000 cells.
Higher = better inter-patient mixing. Random expectation = 1 − 1/n_patients.

### Stemness gradient coherence
Pearson r between a cell's stemness score (`score_LIM_MAMMARY_STEM_CELL_UP`)
and the mean stemness score of its k-nearest neighbours in the integrated graph.
Computed on 5,000 cells. Higher = the stemness gradient is preserved in the embedding.

### Per-patient Spearman r (stem × immune)
Spearman r(stemness, immune) computed independently for each patient (≥50 cells).
**Invariant across integration methods** — computed from expression scores, not
from the embedding. Reported as median ± IQR across patients.

---

## 7. Output files

| File | Description |
|---|---|
| `visvader_malignant_processed.h5ad` | Scored + integrated Visvader object |
| `klughammer_malignant_processed.h5ad` | Scored + integrated Klughammer object |
| `figures/part0/qc_{dataset}.csv` | Per-patient QC table |
| `figures/part0/{dataset}_umap_panel.png` | 4 methods × 6 features UMAP grid |
| `figures/part0/{dataset}_allsigs_umap.png` | All 9 sigs × 4 methods (patient ref col 0) |
| `figures/part0/sigs_panel/{dataset}_{sig}_panel.png` | Per-signature full panel (9 per dataset) |
| `figures/part0/{dataset}_batch_mixing.png` | Batch mixing boxplot |
| `figures/part0/{dataset}_metrics_summary.png` | Batch mixing + stemness coherence |
| `figures/part0/{dataset}_metrics.csv` | Quantitative metrics table |
