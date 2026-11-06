# Azizi 2018 (GSE114727) — γδ T cells in breast cancer

## Data
| Platform | Patients | Tissues | Cells | Notes |
|---|---|---|---|---|
| 10x 5' GEX + TCR (αβ kit) | BC09–BC11 | tumor | CD3+ sorted, 28,473 after QC | all T cells |
| inDrop | BC01–BC08 | tumor, normal breast, blood, lymph node | CD45+ sorted, 63,022 after QC | all immune cells, shallow 3' |

Raw files: `data/raw/` (from `GSE114727_RAW.tar`). Series matrices: `data/geo/`.
Metadata: `data/metadata/` — GEO only has age/tissue; `clinical_patients.csv`
has subtype/ER/PR/HER2/grade for BC01–BC08 from Fig. S1A; BC09–BC11 only age.

## Pipeline
| Script | Output |
|---|---|
| `01_prepare_dataset.py` | `data/azizi_{10x,indrop}_raw.h5ad` (counts, QC, TCR summary in obs for 10x) |
| `02_process_and_TNK.py` | `*_all.h5ad` (Harmony/UMAP/Leiden, automatic lineage label), `*_TNK.h5ad` (T + NK-like compartment) |
| `03_gd_calling.py` | `gd_call`, `gd_subset` in `*_TNK.h5ad`; `*_gdT.h5ad` (γδ cells only, re-embedded) |
| `04_stcat.py export / run (env stcat) / import` | `stcat_*` columns, `consensus_gd` |
| `06_gd_signatures.py` | `sig_*` scores in `*_TNK.h5ad` / `*_gdT.h5ad`, figures/06_gd_signatures/, `report/data/*_signatures.csv.gz` |
| `07_constant_region_markers.py` | `report/data/{marker_specificity_tcr,constant_genes_gd_vs_ab,combined_rules}.csv` (TCR genes / combined score without TCR; report A.4) |
| `05_export_for_report.py` | `report/data/*.csv.gz` (UMAP + metadata + selected genes) |
| `report/azizi_gdT_report.Rmd` | `report/azizi_gdT_report.html` |

Render the report (pandoc lives in conda env `pandoc`):
```
cd report && RSTUDIO_PANDOC=~/miniconda3/envs/pandoc/bin Rscript -e 'rmarkdown::render("azizi_gdT_report.Rmd")'
```

Clusters and lineage labels are computed here; the GEO data has no annotations.
Raw counts are always in `layers["counts"]`.

## γδ calling — key decisions
- 10x TCR kit only amplifies α/β: TRD/TRG contigs are V-only fragments → ignored.
  TCR used as negative evidence (no productive αβ pair).
- γδ express germline TRAC/TRBC and often have non-productive TRB contigs → no
  TRDC-vs-TRAC ratio; tiers use *productive* chains.
- NK cells can transcribe germline TRDC → inDrop calls require CD3 expression.
- Vδ subset from dominant TRDV1/2/3 UMI; Vγ9Vδ2 = TRDV2 + TRGV9. Reliable on 10x
  (5'); inDrop (3') almost never detects V genes → `Vd_undetermined`.
- STCAT does not recover γδ here (see docstring of `04_stcat.py`).
