# Script index

Scripts read and write files relative to the folder they are run from, so each one must be run from the folder below. That folder also holds its input data, which is not in this repository.

`Previous path` is where the script lived before the reorganization. Comments inside the scripts may still use those old names.

| Script | Run from | Previous path |
|---|---|---|
| `scripts/01_data_preparation/01_load_10x_samples.py` | `Regner/` | `Regner/01_load_samples.py` |
| `scripts/01_data_preparation/02_add_clinical_metadata.py` | `Regner/` | `Regner/01b_add_clinical_metadata.py` |
| `scripts/01_data_preparation/03_qc_filter_mixed_tissue.py` | `Regner/` | `Regner/02_qc_filter.py` |
| `scripts/01_data_preparation/04_cohort_overview.py` | `Regner/` | `Regner/02b_cohort_overview.py` |
| `scripts/01_data_preparation/05_subtype_overview.py` | `Regner/` | `Regner/02c_subtype_overview.py` |
| `scripts/01_data_preparation/06_convert_h5ad_to_seurat.R` | `Regner/` | `Regner/04_convert_raw_to_seurat.R` |
| `scripts/01_data_preparation/07_dump_seurat_for_verification.R` | `Regner/` | `Regner/04b_dump_seurat.R` |
| `scripts/01_data_preparation/08_verify_seurat_equivalence.py` | `Regner/` | `Regner/04b_verify_seurat_equivalence.py` |
| `scripts/01_data_preparation/01_filter_primary_tumors.py` | `Visvader/` | `Visvader/01_filter_tumor.py` |
| `scripts/01_data_preparation/02_qc_filter_tumor_only.py` | `Visvader/` | `Visvader/02_qc_filter_normalize.py` |
| `scripts/01_data_preparation/03_split_and_integrate_compartments.py` | `Visvader/` | `Visvader/03_split_integrate.py` |
| `scripts/01_data_preparation/04_reintegrate_filtered_er_epithelium.py` | `Visvader/` | `Visvader/23_reintegrate_ER_filtered.py` |
| `scripts/01_data_preparation/00_h5seurat_to_rds.R` | `Klughammer/` | `Klughammer/00_h5seurat_to_rds.R` |
| `scripts/01_data_preparation/01_prepare_metastatic_cohort.py` | `Klughammer/` | `Klughammer/01_prepare_dataset.py` |
| `scripts/01_data_preparation/02_convert_residual_ensembl_ids.py` | `repository root` | `fix_klughammer_gene_names.py` |
| `scripts/01_data_preparation/seurat_transfer/01_export_reference.py` | `any folder` | `seurat_transfer/01_export_reference.py` |
| `scripts/01_data_preparation/seurat_transfer/02_import_counts.R` | `any folder` | `seurat_transfer/02_import_counts.R` |
| `scripts/01_data_preparation/seurat_transfer/03_build_seurat.R` | `any folder` | `seurat_transfer/03_build_seurat.R` |
| `scripts/02_integration_benchmark/01_compare_integration_methods.py` | `repository root` | `part0_integration_comparison.py` |
| `scripts/02_integration_benchmark/02_integrate_malignant_by_subtype.py` | `repository root` | `part0_malignant_integration.py` |
| `scripts/02_integration_benchmark/03_plot_drvi_reconstruction_scores.py` | `repository root` | `part0_drvi_reconstruction_scores.py` |
| `scripts/02_integration_benchmark/04_scib_prepare.py` | `repository root` | `part0_scib_prepare.py` |
| `scripts/02_integration_benchmark/05_scib_benchmark.py` | `repository root` | `part0_scib_benchmark.py` |
| `scripts/02_integration_benchmark/06_replot_main_panel.py` | `repository root` | `part0_replot_main_panel.py` |
| `scripts/02_integration_benchmark/07_replot_all_signatures.py` | `repository root` | `part0_replot_allsigs.py` |
| `scripts/02_integration_benchmark/08_replot_per_signature.py` | `repository root` | `part0_replot_persig.py` |
| `scripts/02_integration_benchmark/09_replot_per_signature_panel.py` | `repository root` | `part0_replot_persig_panel.py` |
| `scripts/03_potency_scoring/01_cytotrace2_scoring.py` | `Visvader/` | `Visvader/04_cytotrace2_scoring.py` |
| `scripts/03_potency_scoring/02_cytotrace2_scoring_without_raw_counts.py` | `Klughammer/` | `Klughammer/02_cytotrace.py` |
| `scripts/03_potency_scoring/03_export_for_landscent.py` | `Visvader/` | `Visvader/10a_export_for_landscent.py` |
| `scripts/03_potency_scoring/04_landscent_scoring.R` | `Visvader/` | `Visvader/10_landscent_scoring.R` |
| `scripts/03_potency_scoring/05_landscent_cytotrace_umap.py` | `Visvader/` | `Visvader/11_landscent_cytotrace_umap.py` |
| `scripts/03_potency_scoring/06_landscent_stemconsensus_umap.py` | `Visvader/` | `Visvader/12_landscent_stemconsensus_umap.py` |
| `scripts/03_potency_scoring/07_all_signatures_bubble.py` | `Visvader/` | `Visvader/13_allsigs_bubble.py` |
| `scripts/03_potency_scoring/08_cytotrace2_potency_umap.py` | `Visvader/` | `Visvader/22f_cytotrace2_potency_umap.py` |
| `scripts/04_stemness_cell_cycle/01_evaluate_signatures_vs_cell_cycle.py` | `repository root` | `part1_stemness_vs_cellcycle.py` |
| `scripts/04_stemness_cell_cycle/02_plot_hexbin_cc_free.py` | `repository root` | `part1_plot_hexbin_ccfree.py` |
| `scripts/04_stemness_cell_cycle/03_plot_partial_r_bars.py` | `repository root` | `part1_plot_partial_r_bars.py` |
| `scripts/04_stemness_cell_cycle/04_plot_per_patient.py` | `repository root` | `part1_plot_per_patient.py` |
| `scripts/04_stemness_cell_cycle/05_plot_standalone_results.py` | `repository root` | `part1_plot_standalone_results.py` |
| `scripts/04_stemness_cell_cycle/06_subtype_outlier_analysis.py` | `repository root` | `part1_subtype_outlier_analysis.py` |
| `scripts/04_stemness_cell_cycle/07_subtype_summary.py` | `repository root` | `part1_subtype_summary.py` |
| `scripts/05_stemness_immune_axis/01_state_scoring_hexbin_tumor_compartments.py` | `Visvader/` | `Visvader/05_hexbin_correlation.py` |
| `scripts/05_stemness_immune_axis/02_state_scoring_hexbin_metastatic_compartments.py` | `Klughammer/` | `Klughammer/03_analysis.py` |
| `scripts/05_stemness_immune_axis/03_stem_immune_cell_cycle_corrected.py` | `Visvader/` | `Visvader/08_stem_cc_regressed.py` |
| `scripts/05_stemness_immune_axis/04_compare_stem_immune_metrics.py` | `Visvader/` | `Visvader/09_stem_immune_metrics.py` |
| `scripts/05_stemness_immune_axis/05_signature_immune_correlation.py` | `repository root` | `part1b_immune_correlation.py` |
| `scripts/05_stemness_immune_axis/06_subpopulation_metrics.py` | `repository root` | `part1c_subpopulation_metrics.py` |
| `scripts/05_stemness_immune_axis/07_signature_selection.py` | `repository root` | `part2_01_signature_selection.py` |
| `scripts/05_stemness_immune_axis/08_patient_hexbin.py` | `repository root` | `part2_02_hexbin.py` |
| `scripts/05_stemness_immune_axis/09_phenotype_classification.py` | `repository root` | `part2_03_phenotypes.py` |
| `scripts/05_stemness_immune_axis/10_axis_signatures_umap.py` | `Visvader/` | `Visvader/22_axis_signatures_umap.py` |
| `scripts/05_stemness_immune_axis/11_axis_signature_states.py` | `Visvader/` | `Visvader/22b_axis_signatures_states.py` |
| `scripts/05_stemness_immune_axis/12_axis_signature_labels.py` | `Visvader/` | `Visvader/22c_axis_signatures_labels.py` |
| `scripts/05_stemness_immune_axis/13_axis_phenotype_umaps.py` | `Visvader/` | `Visvader/22d_axis_phenotype_umaps.py` |
| `scripts/05_stemness_immune_axis/14_stem_signatures_umap.py` | `Visvader/` | `Visvader/22e_stem_signatures_umap.py` |
| `scripts/06_tim3_hypoxia/01_hypoxia_tim3_overview.py` | `Visvader/` | `Visvader/07_hipoxia.py` |
| `scripts/06_tim3_hypoxia/02_hypoxia_tim3_overview_positive_cells.py` | `Visvader/` | `Visvader/07_hipoxia_tim3.py` |
| `scripts/06_tim3_hypoxia/03_hypoxia_tim3_extended.py` | `Visvader/` | `Visvader/07b_hipoxia_tim3_extended.py` |
| `scripts/06_tim3_hypoxia/04_aucell_pathways_threshold_groups.py` | `Visvader/` | `Visvader/07d_tim3_hipoxia_epthelial.py` |
| `scripts/06_tim3_hypoxia/05_dea_gsea_threshold_groups.py` | `Visvader/` | `Visvader/07e_DEA_GSEA.py` |
| `scripts/06_tim3_hypoxia/06_depth_adjusted_reanalysis.py` | `Visvader/` | `Visvader/07f_TIM3_hypoxia_depth_adjusted.py` |
| `scripts/06_tim3_hypoxia/07_dea_gsea_filtered_genes.py` | `Visvader/` | `Visvader/07g_DEA_GSEA_filtered.py` |
| `scripts/06_tim3_hypoxia/08_patient_pseudobulk.py` | `Visvader/` | `Visvader/07h_pseudobulk_TIM3_hypoxia.py` |
| `scripts/06_tim3_hypoxia/09_genomewide_depth_regression.py` | `Visvader/` | `Visvader/07i_genomewide_depth_regression.py` |
| `scripts/06_tim3_hypoxia/10_naive_vs_depth_corrected_plots.py` | `Visvader/` | `Visvader/07j_TIM3_hypoxia_depth_checks.py` |
| `scripts/06_tim3_hypoxia/11_clean_analysis_primary.py` | `Visvader/` | `Visvader/08_TIM3_hypoxia_clean.py` |
| `scripts/06_tim3_hypoxia/11_clean_analysis_metastatic.py` | `Klughammer/` | `Klughammer/04_TIM3_hypoxia_clean.py` |
| `scripts/06_tim3_hypoxia/12_confident_positive_primary.py` | `Visvader/` | `Visvader/10_TIM3_hypoxia_confident_positive.py` |
| `scripts/06_tim3_hypoxia/13_clipped_violins_primary.py` | `Visvader/` | `Visvader/11_TIM3_hypoxia_clipped_violins.py` |
| `scripts/06_tim3_hypoxia/13_clipped_violins_metastatic.py` | `Klughammer/` | `Klughammer/05_TIM3_hypoxia_clipped_violins.py` |
| `scripts/06_tim3_hypoxia/14_reverse_comparison_primary.py` | `Visvader/` | `Visvader/12_TIM3_hypoxia_reverse_comparison.py` |
| `scripts/06_tim3_hypoxia/14_reverse_comparison_metastatic.py` | `Klughammer/` | `Klughammer/06_TIM3_hypoxia_reverse_comparison.py` |
| `scripts/06_tim3_hypoxia/15_reverse_quartiles_primary.py` | `Visvader/` | `Visvader/13_TIM3_hypoxia_reverse_comparison_quartiles.py` |
| `scripts/06_tim3_hypoxia/15_reverse_quartiles_metastatic.py` | `Klughammer/` | `Klughammer/07_TIM3_hypoxia_reverse_comparison_quartiles.py` |
| `scripts/06_tim3_hypoxia/16_collect_final_results.py` | `repository root` | `build_results.py` (root) |
| `scripts/07_ros_tme/ros_tme_analysis.py` | `Visvader/` | `Visvader/ROS_tme.py` |
| `scripts/07_ros_tme/ros_tme_analysis_archived.py` | `Visvader/` | `Visvader/GO0072593_old/ROS_TME_analysis.py` |
| `scripts/08_gene_expression_by_tissue/01_lcor_cxcl16_hexbin.py` | `Visvader/` | `Visvader/14_lcor_cxcl16_hexbin.py` |
| `scripts/08_gene_expression_by_tissue/02_lcor_epithelial_lineage.py` | `Visvader/` | `Visvader/20_lcor_epithelial_lineage_normal.py` |
| `scripts/08_gene_expression_by_tissue/03_lcor_lineage_three_categories.py` | `Visvader/` | `Visvader/20b_lcor_epithelial_lineage_3cat.py` |
| `scripts/08_gene_expression_by_tissue/04_lcor_malignant_only.py` | `Visvader/` | `Visvader/20c_lcor_malignant_only.py` |
| `scripts/08_gene_expression_by_tissue/05_malignant_umap_by_subtype.py` | `Visvader/` | `Visvader/20c_umap_by_subtype.py` |
| `scripts/08_gene_expression_by_tissue/06_replot_large_text.py` | `Visvader/` | `Visvader/20d_replot_large_text.py` |
| `scripts/08_gene_expression_by_tissue/07_recompute_umap_no_harmony.py` | `Visvader/` | `Visvader/20e_recompute_umap_no_harmony.py` |
| `scripts/08_gene_expression_by_tissue/08_bach1_ppard_klf5_normal_vs_tumor.py` | `Visvader/` | `Visvader/21_bach1_ppard_klf5_normal_vs_tumor.py` |
| `scripts/08_gene_expression_by_tissue/09_bach1_ppard_klf5_with_cell_lines.py` | `Regner/` | `Regner/03_bach1_ppard_klf5_normal_vs_tumor.py` |
| `scripts/09_lcor_cxcl16/01_exploratory/01_celltype_specificity_overview.py` | `Visvader/` | `Visvader/LCOR-CXCL16/01_exploratorio/scripts/01_celltype_specificity_overview.py` |
| `scripts/09_lcor_cxcl16/01_exploratory/02_tissue_contribution_by_celltype.py` | `Visvader/` | `Visvader/LCOR-CXCL16/01_exploratorio/scripts/02_tissue_contribution_by_celltype.py` |
| `scripts/09_lcor_cxcl16/01_exploratory/03_malignant_structure_no_harmony.py` | `Visvader/` | `Visvader/LCOR-CXCL16/01_exploratorio/scripts/03_malignant_structure_no_harmony.py` |
| `scripts/09_lcor_cxcl16/02_correlation/01_compartment_hexbin_scan.py` | `Visvader/` | `Visvader/LCOR-CXCL16/02_correlacion/scripts/01_compartment_hexbin_scan.py` |
| `scripts/09_lcor_cxcl16/02_correlation/02_depth_artifact_cell_level.py` | `Visvader/` | `Visvader/LCOR-CXCL16/02_correlacion/scripts/02_depth_artifact_cell_level.py` |
| `scripts/09_lcor_cxcl16/02_correlation/03_depth_artifact_diagnostic_hexbin.py` | `Visvader/` | `Visvader/LCOR-CXCL16/02_correlacion/scripts/03_depth_artifact_diagnostic_hexbin.py` |
| `scripts/09_lcor_cxcl16/03_pseudobulk/01_pseudobulk_by_patient_headline.py` | `Visvader/` | `Visvader/LCOR-CXCL16/03_pseudobulk/scripts/01_pseudobulk_by_patient_headline.py` |
| `scripts/09_lcor_cxcl16/04_liana/01_liana_interaction_vs_lcor.py` | `Visvader/` | `Visvader/LCOR-CXCL16/04_liana/scripts/01_liana_interaction_vs_lcor.py` |
| `scripts/09_lcor_cxcl16/05_myeloid/01_pseudobulk_myeloid_headline.py` | `Visvader/` | `Visvader/LCOR-CXCL16/05_myeloid/scripts/01_pseudobulk_myeloid_headline.py` |
| `scripts/09_lcor_cxcl16/05_myeloid/02_depth_artifact_myeloid_cell_level.py` | `Visvader/` | `Visvader/LCOR-CXCL16/05_myeloid/scripts/02_depth_artifact_myeloid_cell_level.py` |
| `scripts/09_lcor_cxcl16/06_liana_within_patient/01_liana_lcor_split_within_patient.py` | `Visvader/` | `Visvader/LCOR-CXCL16/06_liana_within_patient/scripts/01_liana_lcor_split_within_patient.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/01_pseudobulk_apm_malignant.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/01_pseudobulk_apm_malignant.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/02_depth_artifact_apm_malignant_cell_level.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/02_depth_artifact_apm_malignant_cell_level.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/03_pseudobulk_apm_myeloid.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/03_pseudobulk_apm_myeloid.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/04_depth_artifact_apm_myeloid_cell_level.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/04_depth_artifact_apm_myeloid_cell_level.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/05_apm_lcor_by_cd8_infiltration.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/05_apm_lcor_by_cd8_infiltration.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/06_method_control_esr1_greb1.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/06_negative_control_esr1_greb1.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/07_replication_metastatic_cohort.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/07_replication_klughammer.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/08_lcor_esr1_relationship.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/08_lcor_esr1_relationship.py` |
| `scripts/09_lcor_cxcl16/07_apm_positive_control/09_replication_primary_cohort.py` | `Visvader/` | `Visvader/LCOR-CXCL16/07_apm_positive_control/scripts/09_replication_regner.py` |
| `scripts/09_lcor_cxcl16/08_umap_gene_trios/01_umap_trio_lcor_apm_esr1.py` | `Visvader/` | `Visvader/LCOR-CXCL16/08_umap_gene_trios/scripts/01_umap_trio_lcor_apm_esr1.py` |
| `scripts/09_lcor_cxcl16/08_umap_gene_trios/02_umap_trio_lcor_apm_esr1_myeloid.py` | `Visvader/` | `Visvader/LCOR-CXCL16/08_umap_gene_trios/scripts/02_umap_trio_lcor_apm_esr1_myeloid.py` |
| `scripts/09_lcor_cxcl16/08_umap_gene_trios/03_umap_trio_lcor_cxcl16_cxcr6.py` | `Visvader/` | `Visvader/LCOR-CXCL16/08_umap_gene_trios/scripts/03_umap_trio_lcor_cxcl16_cxcr6.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/01_build_signatures.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/01_build_signatures.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/02_tcell_phenotype_test_primary.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/02_visvader_analysis.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/03_tcell_phenotype_test_metastatic.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/03_klughammer_analysis.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/04_lcor_oe_vs_cxcl16_cxcr6_primary.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/04_visvader_lcoroe_vs_cxcl16_cxcr6.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/05_lcor_oe_vs_cxcl16_cxcr6_metastatic.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/05_klughammer_lcoroe_vs_cxcl16_cxcr6.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/06_scatterplots.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/06_scatterplots.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/07_composition_plot.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/07_composition_plot.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/08_celltypist_annotation.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/08_celltypist_annotation.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/09_celltypist_annotation_all_t_cells.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/09_celltypist_annotation_allT.py` |
| `scripts/09_lcor_cxcl16/10_tcell_program/10_cd8_vs_cd4_lineage.py` | `Visvader/` | `Visvader/LCOR-CXCL16/10_lcor_program_tcell_phenotype/scripts/10_cd8_vs_cd4_lineage.py` |
| `scripts/10_lcor_tcell_scores/01_build_signatures.py` | `LCOR_Tcells/` | `LCOR_Tcells/01_signatures/scripts/01_build_signatures.py` |
| `scripts/10_lcor_tcell_scores/02_malignant_scores.py` | `LCOR_Tcells/` | `LCOR_Tcells/02_malignant_scores/scripts/01_malignant_scores.py` |
| `scripts/10_lcor_tcell_scores/03_tcell_scores.py` | `LCOR_Tcells/` | `LCOR_Tcells/03_tcell_scores/scripts/01_tcell_scores.py` |
| `scripts/10_lcor_tcell_scores/04_hypothesis_scatterplots.py` | `LCOR_Tcells/` | `LCOR_Tcells/04_biased_analysis/scripts/01_biased_scatterplots.py` |
| `scripts/10_lcor_tcell_scores/05_unbiased_scatterplots.py` | `LCOR_Tcells/` | `LCOR_Tcells/05_unbiased_analysis/scripts/01_unbiased_scatterplots.py` |
| `scripts/10_lcor_tcell_scores/06_celltypist_composition.py` | `LCOR_Tcells/` | `LCOR_Tcells/05_unbiased_analysis/scripts/02_composition_plot.py` |
| `scripts/10_lcor_tcell_scores/07_celltypist_multiscatter.py` | `LCOR_Tcells/` | `LCOR_Tcells/05_unbiased_analysis/scripts/03_multiscatter_categories.py` |
| `scripts/10_lcor_tcell_scores/08_zheng_states_composition.py` | `LCOR_Tcells/` | `LCOR_Tcells/05_unbiased_analysis/scripts/04_abe6474_composition.py` |
| `scripts/10_lcor_tcell_scores/09_zheng_states_multiscatter.py` | `LCOR_Tcells/` | `LCOR_Tcells/05_unbiased_analysis/scripts/05_abe6474_multiscatter.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/01_build_platform_objects.py` | `Azizi/` | `Azizi/01_prepare_dataset.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/02_process_and_extract_tnk.py` | `Azizi/` | `Azizi/02_process_and_TNK.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/03_gd_calling_with_tcr.py` | `Azizi/` | `Azizi/03_gd_calling.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/04_stcat_annotation.py` | `Azizi/` | `Azizi/04_stcat.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/05_export_for_report.py` | `Azizi/` | `Azizi/05_export_for_report.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/06_gd_signatures.py` | `Azizi/` | `Azizi/06_gd_signatures.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/07_constant_region_markers.py` | `Azizi/` | `Azizi/07_constant_region_markers.py` |
| `scripts/11_gamma_delta_tcells/tcr_calibration/gd_tcr_calibration_report.Rmd` | `Azizi/` | `Azizi/report/azizi_gdT_report.Rmd` |
| `scripts/11_gamma_delta_tcells/expression_only_call/01_extract_tnk.py` | `Klughammer/gdT/` | `Klughammer/gdT/01_extract_TNK.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/02_process_tnk.py` | `Klughammer/gdT/` | `Klughammer/gdT/02_process_TNK.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/03_gd_calling_expression_only.py` | `Klughammer/gdT/` | `Klughammer/gdT/03_gd_calling.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/04a_export_for_decontx.py` | `Klughammer/gdT/` | `Klughammer/gdT/04a_export_for_decontx.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/04b_decontx.R` | `Klughammer/gdT/` | `Klughammer/gdT/04b_decontx.R` |
| `scripts/11_gamma_delta_tcells/expression_only_call/04c_background_test.py` | `Klughammer/gdT/` | `Klughammer/gdT/04c_background_test.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/05_export_for_report.py` | `Klughammer/gdT/` | `Klughammer/gdT/05_export_for_report.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/06_combined_rule_test.py` | `Klughammer/gdT/` | `Klughammer/gdT/06_combined_rule_test.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/07_cd3e_by_celltype.py` | `Klughammer/gdT/` | `Klughammer/gdT/07_cd3e_by_celltype.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/08_nk_subclusters.py` | `Klughammer/gdT/` | `Klughammer/gdT/08_nk_subclusters.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/09_doublets.py` | `Klughammer/gdT/` | `Klughammer/gdT/09_doublets.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/10_combined_score_call.py` | `Klughammer/gdT/` | `Klughammer/gdT/10_combined_score_call.py` |
| `scripts/11_gamma_delta_tcells/expression_only_call/gd_expression_feasibility_report.Rmd` | `Klughammer/gdT/` | `Klughammer/gdT/report/klughammer_gdT_feasibility.Rmd` |
| `scripts/12_patient_malignant_programs/00_h5seurat_to_rds.R` | `Klughammer/KlughammeR/` | `Klughammer/KlughammeR/00_h5seurat_to_rds.R` |
| `scripts/12_patient_malignant_programs/01_split_by_patient_qc_metastatic.R` | `Klughammer/KlughammeR/` | `Klughammer/KlughammeR/01_split_raw_by_patient_malignant_QC.R` |
| `scripts/12_patient_malignant_programs/01_split_by_patient_qc_primary.R` | `Regner/` | `Regner/05_split_raw_by_patient_malignant_QC.R` |
| `scripts/12_patient_malignant_programs/02_sctransform_clustering_metastatic.R` | `Klughammer/KlughammeR/` | `Klughammer/KlughammeR/02_SCTransform_clustering_per_patient.R` |
| `scripts/12_patient_malignant_programs/02_sctransform_clustering_primary.R` | `Regner/` | `Regner/06_SCTransform_clustering_per_patient.R` |
| `scripts/12_patient_malignant_programs/03_cnmf_metastatic.R` | `Klughammer/KlughammeR/` | `Klughammer/KlughammeR/03_cNMF_per_patient.R` |
| `scripts/12_patient_malignant_programs/03_cnmf_primary.R` | `Regner/` | `Regner/cNMF_per_patient.R` |
| `scripts/13_presentation/01_concept_slide.py` | `repository root` | `part1_draw_concept_slide.py` |
| `scripts/13_presentation/02_signatures_slide.py` | `repository root` | `part1_draw_signatures.py` |
| `scripts/13_presentation/03_analysis_and_results_slides.py` | `repository root` | `part1_draw_slides.py` |
| `scripts/13_presentation/04_analysis_and_results_slides_v2.py` | `repository root` | `part1_draw_slides_v2.py` |
| `scripts/13_presentation/05_result_slides_per_cohort.py` | `repository root` | `part1_draw_slides_results.py` |
| `scripts/13_presentation/06_lab_meeting_figures.py` | `repository root` | `presentation_figures.py` |
| `scripts/13_presentation/07_lab_meeting_umaps.py` | `repository root` | `presentation_umaps.py` |
