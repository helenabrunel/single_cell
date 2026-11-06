"""
01_liana_lcor_split_within_patient.py
======================================
EXPLORATORY / ISOLATED FROM THE MAIN RESULT CHAIN. See README.md at the
project root for why this lives in its own numbered block instead of
extending 04_liana.

Within-patient test of whether LCOR level modulates the CXCL16(Malignant)
-> CXCR6(T_cell) ligand-receptor interaction (LIANA), instead of correlating
one interaction score per patient against that patient's LCOR pseudobulk.

WHY THIS SCRIPT EXISTS
-----------------------
04_liana/01_liana_interaction_vs_lcor.py correlated a single per-patient
LIANA score against per-patient LCOR pseudobulk (rho=-0.63) but that
correlation collapsed once subtype was accounted for -- subtype drives both
LCOR level and interaction strength, so the pooled correlation was largely a
subtype contrast, not evidence that LCOR itself affects the interaction.

The actual question of interest is causal-shaped: "within one tumor, do the
LCOR-higher malignant cells engage CXCR6+ T cells more than the
LCOR-lower malignant cells of the SAME tumor?" A within-patient design
answers this directly and is immune to the subtype confound by
construction -- each patient is compared only to itself, so nothing that
varies between patients (subtype, batch, overall depth) can produce a
spurious effect.

METHOD
------
Per patient (cell type definitions identical to 04_liana: Malignant =
compartment=='Epithelial', T_cell = compartment=='Immune' & CD3+):

1. Split that patient's malignant cells into LCOR+ (raw count > 0) and
   LCOR- (raw count == 0). LCOR is detected in only ~15% of malignant
   cells, so a detected/undetected split is the natural binary grouping
   here (an arbitrary median split would be degenerate at this detection
   rate -- the median is 0).

2. DEPTH-MATCHING (the reason this needs its own script, not just a
   groupby): LCOR detection itself depends on sequencing depth (established
   repeatedly in 02_correlacion/03_pseudobulk) -- LCOR+ cells are
   systematically deeper-sequenced than LCOR- cells. Left uncorrected, that
   would make the LCOR+ half detect MORE of everything (including CXCL16)
   simply because it is deeper, reproducing the exact depth artifact this
   project has spent most of its effort ruling out, just relocated into a
   within-patient design instead of a between-patient one.
   Fix: coarsened matching on total_counts. All of a patient's malignant
   cells (LCOR+ and LCOR- together) are cut into N_DEPTH_BINS quantile
   bins of total_counts; within each bin, a LCOR- comparison set is drawn
   (without replacement where possible) to match the number of LCOR+ cells
   in that same bin. The result is a "LCOR-low" cell set with a total_counts
   distribution matched to the "LCOR-high" (LCOR+) cell set, so any
   remaining difference between the two halves is not attributable to depth
   by design. Reported directly: median depth of LCOR+ vs the UNMATCHED
   LCOR- pool (expected to differ) vs the MATCHED LCOR- set (expected not
   to, if matching worked).

3. LIANA rank_aggregate (single-pair CXCL16-CXCR6, same settings as
   04_liana: expr_prop=0.02, min_cells=5, n_perms=1000) run TWICE per
   patient: once on (LCOR+ malignant + all of that patient's T cells), once
   on (depth-matched LCOR- malignant + the SAME T cells). Using the same
   T-cell pool in both keeps the receptor side identical between the two
   runs -- only the malignant/ligand side changes.

4. Paired Wilcoxon signed-rank test of lr_means (LCOR+ half vs matched
   LCOR- half) across patients -- the direct test of "does the LCOR+ half
   of a tumor show a stronger CXCL16->CXCR6 signal than the LCOR- half of
   THE SAME tumor," free of the between-patient subtype confound that
   affected 04_liana's pooled correlation by construction.

Output: LCOR-CXCL16/06_liana_within_patient/figures/
    persample_split_interaction.csv
    persample_skipped.csv
    fig_paired_lr_means.png     (LCOR+ vs matched LCOR-, per patient, paired)
    fig_depth_matching_check.png (median depth: LCOR+ / unmatched LCOR- / matched LCOR-)
    README.txt

Run from: ~/single_cell/Visvader/
  python3 LCOR-CXCL16/06_liana_within_patient/scripts/01_liana_lcor_split_within_patient.py
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import liana as li
import os, warnings
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

H5AD_FILE   = "GSE161529_tumor_compartments.h5ad"
OUTDIR      = "LCOR-CXCL16/06_liana_within_patient/figures"
MIN_TCELL   = 20
MIN_HALF    = 20     # min LCOR+ malignant cells (and matched LCOR- cells) per patient
N_DEPTH_BINS = 10
N_PERMS     = 1000
EXPR_PROP   = 0.02
LIGAND, RECEPTOR = "CXCL16", "CXCR6"
LCOR_GENE   = "LCOR"
T_MARKERS   = ["CD3D", "CD3E", "CD3G"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
RNG = np.random.default_rng(0)
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(f" 01 -- {LIGAND}(Malignant) -> {RECEPTOR}(T_cell) -- within-patient LCOR+/LCOR- split (depth-matched)")
print("="*65)

print(f"\n  Loading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
print(f"  Cells: {adata.n_obs:,}")

needed = [LIGAND, RECEPTOR, LCOR_GENE] + T_MARKERS
missing = [g for g in needed if g not in adata.var_names]
if missing:
    raise SystemExit(f"  Gene(s) not found: {missing}")

tidx = [adata.var_names.get_loc(g) for g in T_MARKERS]
tcounts = adata.layers["counts"][:, tidx]
tcounts = np.asarray(tcounts.todense()) if issparse(tcounts) else np.asarray(tcounts)
is_tcell = (adata.obs["compartment"] == "Immune").values & (tcounts > 0).any(axis=1)
is_malig = (adata.obs["compartment"] == "Epithelial").values
adata.obs["celltype"] = np.where(is_malig, "Malignant", np.where(is_tcell, "T_cell", "Other"))
print(f"  Malignant cells: {is_malig.sum():,}   T cells (CD3+): {is_tcell.sum():,}")

lcor_idx = adata.var_names.get_loc(LCOR_GENE)
lcor_raw = adata.layers["counts"][:, lcor_idx]
lcor_raw = np.asarray(lcor_raw.todense()).ravel() if issparse(lcor_raw) else np.asarray(lcor_raw).ravel()
adata.obs["LCOR_raw"] = lcor_raw
adata.obs["total_counts_"] = adata.obs["total_counts"].values.astype(float)

ligand_idx = adata.var_names.get_loc(LIGAND)
receptor_idx = adata.var_names.get_loc(RECEPTOR)
ligand_raw = adata.layers["counts"][:, ligand_idx]
receptor_raw = adata.layers["counts"][:, receptor_idx]
ligand_raw = np.asarray(ligand_raw.todense()).ravel() if issparse(ligand_raw) else np.asarray(ligand_raw).ravel()
receptor_raw = np.asarray(receptor_raw.todense()).ravel() if issparse(receptor_raw) else np.asarray(receptor_raw).ravel()
adata.obs["_ligand_pos"] = ligand_raw > 0
adata.obs["_receptor_pos"] = receptor_raw > 0


def stratified_depth_match(pos_depth, neg_depth, neg_positions, n_bins, rng):
    """Coarsened matching: bin ALL cells (pos+neg) into n_bins quantile bins
    of total_counts; within each bin, draw as many neg cells as there are
    pos cells (without replacement if the bin has enough, with replacement
    -- flagged -- if it doesn't). Returns (matched_neg_positions, n_replacement)."""
    all_depth = np.concatenate([pos_depth, neg_depth])
    try:
        bins = pd.qcut(all_depth, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(len(all_depth), dtype=int)
    pos_bins = bins[:len(pos_depth)]
    neg_bins = bins[len(pos_depth):]
    matched, n_replacement = [], 0
    for b in np.unique(pos_bins):
        n_needed = int((pos_bins == b).sum())
        pool = neg_positions[neg_bins == b]
        if len(pool) == 0:
            continue  # no depth-matched candidates in this bin; those pos cells go unmatched
        replace = len(pool) < n_needed
        if replace:
            n_replacement += n_needed
        chosen = rng.choice(pool, size=n_needed, replace=replace)
        matched.extend(chosen.tolist())
    return np.array(matched, dtype=int), n_replacement


def run_liana_pair(sub_adata, label):
    """Run rank_aggregate on a 2-group (Malignant/T_cell) AnnData subset,
    restricted to CXCL16-CXCR6. Returns (lr_means, cellphone_pvals, reason)
    where reason is None on success or a skip reason string."""
    malig_mask = (sub_adata.obs["celltype"] == "Malignant").values
    tcell_mask = (sub_adata.obs["celltype"] == "T_cell").values
    ligand_prop = sub_adata.obs["_ligand_pos"].values[malig_mask].mean()
    receptor_prop = sub_adata.obs["_receptor_pos"].values[tcell_mask].mean()
    if ligand_prop < EXPR_PROP or receptor_prop < EXPR_PROP:
        return None, None, (f"[{label}] insufficient detection: {LIGAND} in {100*ligand_prop:.1f}% of "
                             f"Malignant, {RECEPTOR} in {100*receptor_prop:.1f}% of T_cell")
    try:
        li.mt.rank_aggregate(sub_adata, groupby="celltype", resource_name="consensus",
                              interactions=[(LIGAND, RECEPTOR)], expr_prop=EXPR_PROP,
                              min_cells=5, use_raw=False, layer="logcounts",
                              n_perms=N_PERMS, verbose=False)
    except Exception as e:
        return None, None, f"[{label}] LIANA error: {e}"
    res = sub_adata.uns["liana_res"]
    hit = res[(res["source"] == "Malignant") & (res["target"] == "T_cell")]
    if hit.empty:
        return None, None, f"[{label}] interaction filtered out by LIANA's own expr_prop check"
    hit = hit.iloc[0]
    return hit["lr_means"], hit["cellphone_pvals"], None


# ============================================================================
# Per-patient candidate selection + split + depth-matching + LIANA x2
# ============================================================================
cell_counts = adata.obs.groupby(["sample", "celltype"]).size().unstack(fill_value=0)
samples = adata.obs["sample"].unique().tolist()

rows, skipped = [], []
for s in samples:
    subtype = adata.obs.loc[adata.obs["sample"] == s, "subtype"].iloc[0]
    n_tcell = int(cell_counts.get("T_cell", pd.Series(dtype=int)).get(s, 0))
    if n_tcell < MIN_TCELL:
        skipped.append({"sample": s, "reason": f"n_tcell={n_tcell} < MIN_TCELL={MIN_TCELL}"})
        continue

    sample_mask = (adata.obs["sample"] == s).values
    malig_mask_all = sample_mask & is_malig
    pos_mask = malig_mask_all & (adata.obs["LCOR_raw"].values > 0)
    neg_mask = malig_mask_all & (adata.obs["LCOR_raw"].values == 0)
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
    if n_pos < MIN_HALF or n_neg < MIN_HALF:
        skipped.append({"sample": s, "reason": f"n_LCOR+={n_pos}, n_LCOR-={n_neg} (need >={MIN_HALF} each)"})
        continue

    pos_positions = np.where(pos_mask)[0]
    neg_positions = np.where(neg_mask)[0]
    pos_depth = adata.obs["total_counts_"].values[pos_positions]
    neg_depth = adata.obs["total_counts_"].values[neg_positions]

    matched_neg_positions, n_repl = stratified_depth_match(pos_depth, neg_depth, neg_positions, N_DEPTH_BINS, RNG)
    if len(matched_neg_positions) < MIN_HALF:
        skipped.append({"sample": s, "reason": f"depth-matching yielded only {len(matched_neg_positions)} "
                                                 f"LCOR- cells (need >={MIN_HALF})"})
        continue
    matched_neg_depth = adata.obs["total_counts_"].values[matched_neg_positions]

    tcell_positions = np.where(sample_mask & is_tcell)[0]

    pos_sub = adata[np.concatenate([pos_positions, tcell_positions])].copy()
    neg_sub = adata[np.concatenate([matched_neg_positions, tcell_positions])].copy()

    lr_means_pos, pval_pos, reason_pos = run_liana_pair(pos_sub, f"{s}/LCOR+")
    lr_means_neg, pval_neg, reason_neg = run_liana_pair(neg_sub, f"{s}/LCOR-matched")
    if reason_pos or reason_neg:
        for r in (reason_pos, reason_neg):
            if r:
                skipped.append({"sample": s, "reason": r})
        continue

    rows.append({
        "sample": s, "subtype": subtype,
        "n_pos": int(n_pos), "n_neg_unmatched": int(n_neg), "n_neg_matched": len(matched_neg_positions),
        "n_replacement_matches": n_repl, "n_tcell": n_tcell,
        "median_depth_pos": float(np.median(pos_depth)),
        "median_depth_neg_unmatched": float(np.median(neg_depth)),
        "median_depth_neg_matched": float(np.median(matched_neg_depth)),
        "lr_means_LCORpos": lr_means_pos, "lr_means_LCORneg_matched": lr_means_neg,
        "cellphone_pvals_LCORpos": pval_pos, "cellphone_pvals_LCORneg_matched": pval_neg,
    })
    print(f"  {s} ({subtype}): n_pos={n_pos}, n_neg_matched={len(matched_neg_positions)} "
          f"(depth pos={np.median(pos_depth):,.0f} vs matched neg={np.median(matched_neg_depth):,.0f} "
          f"vs unmatched neg={np.median(neg_depth):,.0f})  "
          f"lr_means: LCOR+={lr_means_pos:.3f}  LCOR-(matched)={lr_means_neg:.3f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUTDIR}/persample_split_interaction.csv", index=False)
print(f"\n  Saved: {OUTDIR}/persample_split_interaction.csv  ({len(df)} patients with both halves tested)")

df_skipped = pd.DataFrame(skipped)
df_skipped.to_csv(f"{OUTDIR}/persample_skipped.csv", index=False)
print(f"  Saved: {OUTDIR}/persample_skipped.csv  ({len(df_skipped)} patients excluded, with reason)")

if len(df) < 4:
    print(f"\n  Only {len(df)} patients survived -- too few for a meaningful paired test. Stopping after saving tables.")
    raise SystemExit(0)

# ============================================================================
# DEPTH-MATCHING SANITY CHECK
# ============================================================================
print("\n" + "="*65)
print(" Depth-matching sanity check")
print("="*65)
w_unmatched, p_unmatched = stats.wilcoxon(df["median_depth_pos"], df["median_depth_neg_unmatched"])
w_matched, p_matched = stats.wilcoxon(df["median_depth_pos"], df["median_depth_neg_matched"])
print(f"  Paired Wilcoxon, median depth LCOR+ vs UNMATCHED LCOR-:  p={p_unmatched:.3g} "
      f"(median diff={np.median(df['median_depth_pos']-df['median_depth_neg_unmatched']):+.0f})")
print(f"  Paired Wilcoxon, median depth LCOR+ vs MATCHED LCOR-:    p={p_matched:.3g} "
      f"(median diff={np.median(df['median_depth_pos']-df['median_depth_neg_matched']):+.0f})")
print("  (Matching worked as intended if the first p is small/diff large and the second is not.)")

# ============================================================================
# PRIMARY TEST -- paired lr_means, LCOR+ vs depth-matched LCOR-
# ============================================================================
print("\n" + "="*65)
print(" Primary test: paired lr_means, LCOR+ vs depth-matched LCOR- (same patient)")
print("="*65)
w_stat, w_p = stats.wilcoxon(df["lr_means_LCORpos"], df["lr_means_LCORneg_matched"])
t_stat, t_p = stats.ttest_rel(df["lr_means_LCORpos"], df["lr_means_LCORneg_matched"])
delta = df["lr_means_LCORpos"] - df["lr_means_LCORneg_matched"]
n_higher = int((delta > 0).sum())
print(f"  n={len(df)} patients.  LCOR+ half higher in {n_higher}/{len(df)}.")
print(f"  Paired Wilcoxon signed-rank: stat={w_stat:.3f}, p={w_p:.3g}")
print(f"  Paired t-test (cross-check): t={t_stat:.3f}, p={t_p:.3g}")
print(f"  Mean delta (LCOR+ - LCOR-matched) lr_means: {delta.mean():+.4f}  (median {delta.median():+.4f})")

print("\n  By subtype (descriptive only -- paired design already controls for between-patient")
print("  confounds like subtype by construction; not re-tested as a confound here):")
for st in ["ER", "HER2", "TNBC"]:
    d_st = df[df["subtype"] == st]
    if len(d_st) == 0:
        continue
    print(f"    {st} (n={len(d_st)}): mean delta={d_st['lr_means_LCORpos'].sub(d_st['lr_means_LCORneg_matched']).mean():+.4f}, "
          f"higher in {(d_st['lr_means_LCORpos']>d_st['lr_means_LCORneg_matched']).sum()}/{len(d_st)}")

# ============================================================================
# FIGURE 1 -- paired lr_means, LCOR+ vs matched LCOR-
# ============================================================================
print("\n  Building figures...")
fig, ax = plt.subplots(figsize=(6, 5.5))
for _, row in df.iterrows():
    color = SUBTYPE_COLORS.get(row["subtype"], "gray")
    ax.plot([0, 1], [row["lr_means_LCORneg_matched"], row["lr_means_LCORpos"]],
            color=color, alpha=0.6, lw=1.3, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.4)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"LCOR- (matched)\nn={len(df)}", f"LCOR+\nn={len(df)}"], fontsize=9.5)
ax.set_ylabel(f"{LIGAND}->{RECEPTOR} LIANA lr_means", fontsize=9.5)
ax.set_title(f"Within-patient, depth-matched: LCOR+ vs LCOR- malignant cells\n"
             f"Paired Wilcoxon p={w_p:.3g}, paired t-test p={t_p:.3g} (n={len(df)} patients)\n"
             f"LCOR+ higher in {n_higher}/{len(df)} patients",
             fontsize=10, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=st) for st, c in SUBTYPE_COLORS.items() if st in df["subtype"].values],
          fontsize=8, title="Subtype", loc="upper left")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_paired_lr_means.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig_paired_lr_means.png")

# ============================================================================
# FIGURE 2 -- depth-matching sanity check
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 5.5))
for _, row in df.iterrows():
    color = SUBTYPE_COLORS.get(row["subtype"], "gray")
    ax.plot([0, 1, 2],
            [row["median_depth_pos"], row["median_depth_neg_unmatched"], row["median_depth_neg_matched"]],
            color=color, alpha=0.5, lw=1.1, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.4)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["LCOR+\n(this patient's\nmalignant cells)", "LCOR-\n(unmatched)", "LCOR-\n(depth-matched)"], fontsize=9)
ax.set_yscale("log")
ax.set_ylabel("Median total_counts (per-patient half)", fontsize=9.5)
ax.set_title(f"Depth-matching check\nLCOR+ vs unmatched LCOR-: Wilcoxon p={p_unmatched:.3g}  |  "
             f"LCOR+ vs matched LCOR-: Wilcoxon p={p_matched:.3g}",
             fontsize=10, fontweight="bold")
ax.legend(handles=[Patch(color=c, label=st) for st, c in SUBTYPE_COLORS.items() if st in df["subtype"].values],
          fontsize=8, title="Subtype", loc="best")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_depth_matching_check.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig_depth_matching_check.png")

# ============================================================================
# README
# ============================================================================
readme = [
    f"01 -- {LIGAND}(Malignant) -> {RECEPTOR}(T_cell) -- within-patient LCOR+/LCOR- split, depth-matched",
    "="*70,
    "",
    "STATUS: exploratory, isolated in its own folder -- see project README.md",
    "",
    "WHY THIS SCRIPT EXISTS",
    "-"*70,
    "04_liana correlated one LIANA score per patient against that patient's",
    "LCOR pseudobulk (rho=-0.63) but the correlation collapsed once subtype",
    "was accounted for -- subtype drives both variables, so it was largely a",
    "subtype contrast, not an LCOR effect. This script asks the more direct,",
    "causal-shaped question instead: within ONE tumor, do its LCOR-higher",
    "malignant cells engage CXCR6+ T cells more than its LCOR-lower",
    "malignant cells? A within-patient design is immune to the subtype",
    "confound BY CONSTRUCTION (each patient is only compared to itself).",
    "",
    "METHOD",
    "-"*70,
    "  1) Per patient, malignant cells split into LCOR+ (raw>0) vs LCOR-",
    "     (raw==0) -- LCOR is detected in only ~15% of malignant cells, so",
    "     detected/undetected is the natural binary split (a median split",
    "     would be degenerate: the median is 0).",
    "  2) DEPTH-MATCHING: LCOR detection itself depends on sequencing depth",
    "     (established in 02_correlacion/03_pseudobulk), so LCOR+ cells are",
    "     expected to be systematically deeper than LCOR- cells. Left",
    "     uncorrected this would make the 'LCOR+' half detect more of",
    "     EVERYTHING (including CXCL16) purely from depth -- the same",
    "     artifact this project spent most of its effort ruling out,",
    "     relocated into a within-patient design. Fixed by coarsened",
    f"     matching: all malignant cells cut into {N_DEPTH_BINS} quantile bins of",
    "     total_counts, then a LCOR- comparison set drawn per bin to match",
    "     the number of LCOR+ cells in that bin (without replacement where",
    "     possible).",
    f"  3) LIANA rank_aggregate (single-pair {LIGAND}-{RECEPTOR}, same settings as",
    f"     04_liana: expr_prop={EXPR_PROP}, min_cells=5, n_perms={N_PERMS}) run twice",
    "     per patient: once on (LCOR+ malignant + that patient's T cells),",
    "     once on (depth-matched LCOR- malignant + the SAME T cells).",
    "  4) Paired Wilcoxon signed-rank test of lr_means, LCOR+ half vs",
    "     matched LCOR- half, across patients.",
    "",
    f"  MIN_TCELL={MIN_TCELL}, MIN_HALF={MIN_HALF} (min. LCOR+ AND matched-LCOR- cells/patient).",
    f"  Patients tested: {len(df)}  |  Patients skipped: {len(df_skipped)} (see persample_skipped.csv)",
    "",
    "DEPTH-MATCHING SANITY CHECK",
    "-"*70,
    f"  Paired Wilcoxon, median depth LCOR+ vs UNMATCHED LCOR-: p={p_unmatched:.3g} "
    f"(median diff={np.median(df['median_depth_pos']-df['median_depth_neg_unmatched']):+.0f})",
    f"  Paired Wilcoxon, median depth LCOR+ vs MATCHED LCOR-:   p={p_matched:.3g} "
    f"(median diff={np.median(df['median_depth_pos']-df['median_depth_neg_matched']):+.0f})",
    "  Matching is doing its job if the first comparison is a large/significant",
    "  gap and the second is much smaller/non-significant -- see",
    "  fig_depth_matching_check.png.",
    "",
    "PRIMARY RESULT",
    "-"*70,
    f"  n={len(df)} patients. LCOR+ half showed higher lr_means than the matched",
    f"  LCOR- half in {n_higher}/{len(df)} patients.",
    f"  Paired Wilcoxon signed-rank: stat={w_stat:.3f}, p={w_p:.3g}",
    f"  Paired t-test (cross-check): t={t_stat:.3f}, p={t_p:.3g}",
    f"  Mean delta (LCOR+ minus matched LCOR-): {delta.mean():+.4f} (median {delta.median():+.4f})",
    "",
    "OUTPUT FILES",
    "-"*70,
    "  persample_split_interaction.csv  -- per-patient n's, depths, lr_means/pvals, both halves",
    "  persample_skipped.csv            -- excluded patients + reason",
    "  fig_paired_lr_means.png          -- paired LCOR+ vs matched LCOR- lr_means, per patient",
    "  fig_depth_matching_check.png     -- confirms the depth confound is removed by matching",
    "",
    "CAVEATS",
    "-"*70,
    "  - Coarsened (quantile-bin) matching, not exact 1:1 nearest-neighbor --",
    "    residual within-bin depth differences can remain; see the matched",
    "    Wilcoxon p-value above as the direct check of how well it worked.",
    "  - Splitting on raw detection (LCOR+/-) is a coarse binarization of a",
    "    continuous (if sparse) biological quantity; some LCOR- cells surely",
    "    express LCOR below detection.",
    "  - n is limited by requiring MIN_HALF cells on both sides after",
    "    matching -- lower-depth or more homogeneous-subtype patients drop",
    "    out (see persample_skipped.csv).",
    "  - Still correlational/observational, not a perturbation -- cannot by",
    "    itself establish that LCOR causes any interaction-strength",
    "    difference, only whether the association exists within tumors once",
    "    depth is controlled for.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"\n  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Patients tested: {len(df)}   Patients skipped: {len(df_skipped)}
  Depth-matching check: unmatched p={p_unmatched:.3g}  ->  matched p={p_matched:.3g}
  Paired Wilcoxon (LCOR+ vs matched LCOR- lr_means): stat={w_stat:.3f}, p={w_p:.3g}
  Paired t-test:                                     t={t_stat:.3f}, p={t_p:.3g}
  LCOR+ higher in {n_higher}/{len(df)} patients
  Output: {OUTDIR}/
============================================================
""")
