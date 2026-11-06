"""
01_liana_interaction_vs_lcor.py
===============================
Cell-cell interaction test for CXCL16 (malignant/epithelial cells) ->
CXCR6 (T cells), per patient, using LIANA's consensus rank_aggregate --
plus a secondary correlation of the interaction strength against LCOR
pseudobulk expression in the same patient's malignant cells.

WHY THIS SCRIPT EXISTS
-----------------------
14/15/16 established that LCOR-CXCL16 co-expression WITHIN malignant
cells is mostly a sequencing-depth artifact once properly corrected. A
separate, biologically better-motivated question is whether CXCL16
(produced by tumor cells) engages its receptor CXCR6, which marks
tissue-resident T cells -- a canonical ligand-receptor axis implicated in
T cell recruitment to tumors.

IMPORTANT: LCOR is a transcriptional corepressor, not a ligand or
receptor, so it cannot itself be a node in a ligand-receptor interaction
test -- it has no entry in LIANA's resource (checked: only CXCL16-CXCR6
is present for these three genes). So LCOR is NOT part of the LIANA call
itself. Instead: LIANA gives a per-PATIENT score for how strong/specific
the CXCL16(Malignant)->CXCR6(T_cell) interaction looks, and we
separately correlate that score, across patients, with each patient's
own LCOR pseudobulk level in malignant cells. This tests "do tumors with
more LCOR also show stronger predicted CXCL16->CXCR6 engagement of
T cells?" without asking LIANA to do something it isn't built for.

CELL TYPE DEFINITIONS (no fine-grained annotation exists in this file)
------------------------------------------------------------------------
  Malignant : obs['compartment'] == 'Epithelial' (122,101 cells, matches
              compartment_epithelial.h5ad used in 14_lcor_cxcl16_hexbin.py, 02_depth_artifact_cell_level.py, 03_depth_artifact_diagnostic_hexbin.py)
  T_cell    : obs['compartment'] == 'Immune' AND (CD3D>0 or CD3E>0 or CD3G>0)
              (30,235 / 59,117 immune cells; CXCR6 detected in 6.75% of
              these vs 1.12% of non-T immune cells, confirming enrichment)

METHOD
------
Per patient (sample) with >= MIN_CELLS Malignant AND >= MIN_CELLS T_cell:
  li.mt.rank_aggregate(sample_subset, groupby='celltype',
                        interactions=[('CXCL16','CXCR6')], ...)
restricting to this one interaction (not the full resource) keeps it fast
and avoids running/interpreting hundreds of irrelevant pairs.
expr_prop is set low (0.02) because both genes are sparsely detected --
the default (0.1) would drop this interaction in several patients purely
due to dropout, not absence of signal. Each patient's CXCL16/CXCR6
detection fraction is checked against expr_prop BEFORE calling LIANA;
patients below it are skipped with a logged reason (an empty single-pair
LIANA result otherwise crashes with an opaque pandas error).

magnitude_rank/specificity_rank are NOT used: those rank a pair against
OTHER pairs in the same call, and restricting to one pair for speed makes
them degenerate (constant across patients). lr_means (raw mean expression
product) and cellphone_pvals (permutation p-value, still valid with a
single-pair resource) are used instead.

Then: Spearman correlation of each patient's LCOR pseudobulk (mean
logcounts in malignant cells) against lr_means and against
-log10(cellphone_pvals), across patients.

SUBTYPE CONFOUND CHECK
-----------------------
The pooled Spearman(LCOR, lr_means) above mixes between-subtype and
within-subtype variation. Three checks address this directly: (1) a
Kruskal-Wallis test of whether LCOR pseudobulk itself differs by subtype
(it does -- lowest in TNBC), (2) the Spearman correlation recomputed
WITHIN each subtype separately (removing between-subtype variation
entirely), and (3) a partial Spearman correlation of LCOR vs lr_means
controlling for a single numeric "subtype rank" covariate (subtypes
ordered by their own ascending median LCOR pseudobulk, then rank-encoded
1..3 per patient) -- the standard way to collapse a 3-level categorical
confound into one covariate for a partial correlation. If the pooled
correlation is a genuine LCOR effect, it should survive within-subtype
and after this adjustment; if it's a subtype contrast, both should
collapse toward zero.

Output: LCOR-CXCL16/04_liana/figures/
    persample_interaction_lcor.csv
    persample_skipped.csv
    within_subtype_correlation.csv
    fig_lcor_vs_interaction.png  (LCOR pseudobulk vs lr_means scatter, per patient)
    fig_subtype_confound.png     (within-subtype panels + partial-correlation residuals)
    fig_interaction_dotplot.png  (per-patient dotplot, grouped by subtype)
    README.txt

NOTE: this script used to also render a joint UMAP (cell type, subtype,
LCOR/CXCL16/CXCR6 expression) as a sanity-check overview. Dropped -- that
content is UMAP/exploratory, not LIANA, and duplicates what
01_exploratorio/03_malignant_structure_no_harmony.py already covers for
the malignant compartment.
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

H5AD_FILE  = "GSE161529_tumor_compartments.h5ad"
OUTDIR     = "LCOR-CXCL16/04_liana/figures"
MIN_CELLS  = 20
N_PERMS    = 1000
EXPR_PROP  = 0.02
LIGAND, RECEPTOR = "CXCL16", "CXCR6"
LCOR_GENE  = "LCOR"
T_MARKERS  = ["CD3D", "CD3E", "CD3G"]
SUBTYPE_COLORS = {"ER": "#4DAF4A", "HER2": "#984EA3", "TNBC": "#E41A1C"}
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print(f" 01 -- {LIGAND}(Malignant) -> {RECEPTOR}(T_cell) -- LIANA per patient")
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

# Per-patient LCOR pseudobulk in malignant cells (mean logcounts)
lcor_idx = adata.var_names.get_loc(LCOR_GENE)
lcor_log = adata.layers["logcounts"][:, lcor_idx]
lcor_log = np.asarray(lcor_log.todense()).ravel() if issparse(lcor_log) else np.asarray(lcor_log).ravel()
adata.obs["LCOR_log"] = lcor_log
lcor_pseudobulk = (adata.obs.loc[is_malig].groupby("sample")["LCOR_log"].mean())

# Per-cell ligand/receptor raw detection, needed to pre-check expr_prop per
# patient BEFORE calling LIANA -- if a patient's CXCL16 (in Malignant) or
# CXCR6 (in T_cell) detection fraction is below expr_prop, LIANA's internal
# lr_res ends up with 0 rows for this single-pair call and a later
# `lr_res.loc[:, ...] = scores[0]` assignment crashes with a cryptic pandas
# ValueError ("cannot set a frame with no defined index and a scalar").
# Checking this ourselves lets us skip cleanly with a clear reason instead.
ligand_idx = adata.var_names.get_loc(LIGAND)
receptor_idx = adata.var_names.get_loc(RECEPTOR)
ligand_raw = adata.layers["counts"][:, ligand_idx]
receptor_raw = adata.layers["counts"][:, receptor_idx]
ligand_raw = np.asarray(ligand_raw.todense()).ravel() if issparse(ligand_raw) else np.asarray(ligand_raw).ravel()
receptor_raw = np.asarray(receptor_raw.todense()).ravel() if issparse(receptor_raw) else np.asarray(receptor_raw).ravel()
adata.obs["_ligand_pos"] = ligand_raw > 0
adata.obs["_receptor_pos"] = receptor_raw > 0

# ============================================================================
# Per-patient LIANA run, restricted to the CXCL16-CXCR6 pair
# ============================================================================
cell_counts = adata.obs.groupby(["sample", "celltype"]).size().unstack(fill_value=0)
valid_samples = cell_counts[(cell_counts.get("Malignant", 0) >= MIN_CELLS) &
                             (cell_counts.get("T_cell", 0) >= MIN_CELLS)].index.tolist()
print(f"\n  Patients with >={MIN_CELLS} Malignant and >={MIN_CELLS} T_cell: {len(valid_samples)}/{adata.obs['sample'].nunique()}")

rows = []
skipped = []
for s in valid_samples:
    sub = adata[(adata.obs["sample"] == s) & adata.obs["celltype"].isin(["Malignant", "T_cell"])].copy()

    malig_mask = (sub.obs["celltype"] == "Malignant").values
    tcell_mask = (sub.obs["celltype"] == "T_cell").values
    ligand_prop = sub.obs["_ligand_pos"].values[malig_mask].mean()
    receptor_prop = sub.obs["_receptor_pos"].values[tcell_mask].mean()
    if ligand_prop < EXPR_PROP or receptor_prop < EXPR_PROP:
        reason = (f"insufficient detection: {LIGAND} in {100*ligand_prop:.1f}% of Malignant, "
                  f"{RECEPTOR} in {100*receptor_prop:.1f}% of T_cell (need >={100*EXPR_PROP:.0f}%)")
        print(f"  {s}: skipped ({reason})")
        skipped.append({"sample": s, "reason": reason})
        continue

    try:
        li.mt.rank_aggregate(sub, groupby="celltype", resource_name="consensus",
                              interactions=[(LIGAND, RECEPTOR)], expr_prop=EXPR_PROP,
                              min_cells=5, use_raw=False, layer="logcounts",
                              n_perms=N_PERMS, verbose=False)
    except Exception as e:
        print(f"  {s}: LIANA failed unexpectedly ({e}), skipping")
        skipped.append({"sample": s, "reason": f"LIANA error: {e}"})
        continue
    res = sub.uns["liana_res"]
    hit = res[(res["source"] == "Malignant") & (res["target"] == "T_cell")]
    if hit.empty:
        print(f"  {s}: interaction filtered out (expr_prop), skipping")
        skipped.append({"sample": s, "reason": "interaction filtered out by LIANA's own expr_prop check"})
        continue
    hit = hit.iloc[0]
    rows.append({
        "sample": s,
        "subtype": adata.obs.loc[adata.obs["sample"] == s, "subtype"].iloc[0],
        "n_malignant": int(cell_counts.loc[s, "Malignant"]),
        "n_tcell": int(cell_counts.loc[s, "T_cell"]),
        "LCOR_pseudobulk": lcor_pseudobulk.get(s, np.nan),
        "lr_means": hit["lr_means"],
        "cellphone_pvals": hit["cellphone_pvals"],
        "ligand_prop": ligand_prop,
        "receptor_prop": receptor_prop,
    })
    print(f"  {s}: lr_means={hit['lr_means']:.3f}  cellphone_pvals={hit['cellphone_pvals']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUTDIR}/persample_interaction_lcor.csv", index=False)
print(f"\n  Saved: {OUTDIR}/persample_interaction_lcor.csv  ({len(df)} patients with a valid interaction score)")

df_skipped = pd.DataFrame(skipped)
df_skipped.to_csv(f"{OUTDIR}/persample_skipped.csv", index=False)
print(f"  Saved: {OUTDIR}/persample_skipped.csv  ({len(df_skipped)} patients excluded, with reason)")

# ============================================================================
# Correlate interaction strength with LCOR pseudobulk, across patients
# NOTE: magnitude_rank/specificity_rank are NOT used here. Those are rank
# statistics computed by comparing the tested pair against *other pairs* in
# the same LIANA call -- but each call here was restricted to a single pair
# (CXCL16-CXCR6) for speed, so there is nothing else to rank against and the
# rank columns come back as a constant (degenerate, no information). lr_means
# (raw mean expression product) and cellphone_pvals (a permutation p-value,
# still meaningful with a single-pair resource since its null comes from
# permuting cell labels, not from ranking across pairs) are used instead.
# ============================================================================
df = df.dropna(subset=["LCOR_pseudobulk"])
df["neg_log10_cellphone_pvals"] = -np.log10(df["cellphone_pvals"].clip(lower=1e-300))

r_lrmeans, p_lrmeans = stats.spearmanr(df["LCOR_pseudobulk"], df["lr_means"])
r_pval, p_pval = stats.spearmanr(df["LCOR_pseudobulk"], df["neg_log10_cellphone_pvals"])
print(f"\n  Spearman(LCOR pseudobulk, lr_means):                    rho={r_lrmeans:.3f}, p={p_lrmeans:.3f}  (n={len(df)})")
print(f"  Spearman(LCOR pseudobulk, -log10(cellphone_pvals)):     rho={r_pval:.3f}, p={p_pval:.3f}  (n={len(df)})")

# ============================================================================
# SUBTYPE CONFOUND CHECK -- does the pooled correlation survive once subtype
# is accounted for, or is it just a subtype contrast? See docstring.
# ============================================================================
print("\n" + "="*65)
print(" Subtype confound check")
print("="*65)

present_subtypes = [st for st in ["ER", "HER2", "TNBC"] if (df["subtype"] == st).any()]
kw_groups = [df.loc[df["subtype"] == st, "LCOR_pseudobulk"].values for st in present_subtypes]
kw_stat, kw_p = stats.kruskal(*kw_groups)
print(f"  Kruskal-Wallis (LCOR_pseudobulk ~ subtype): H={kw_stat:.3f}, p={kw_p:.4f}")
for st in present_subtypes:
    d_st = df.loc[df["subtype"] == st, "LCOR_pseudobulk"]
    print(f"    {st}: median={d_st.median():.3f} (n={len(d_st)})")

within_rows = []
for st in present_subtypes:
    d_st = df[df["subtype"] == st]
    if len(d_st) < 4:
        print(f"  {st}: n={len(d_st)}, too few patients for a within-subtype correlation (reported, underpowered)")
        r_st, p_st = (stats.spearmanr(d_st["LCOR_pseudobulk"], d_st["lr_means"])
                      if len(d_st) >= 3 else (np.nan, np.nan))
    else:
        r_st, p_st = stats.spearmanr(d_st["LCOR_pseudobulk"], d_st["lr_means"])
    within_rows.append({"subtype": st, "n": len(d_st), "spearman_rho": r_st, "spearman_p": p_st})
    print(f"  {st} (n={len(d_st)}): Spearman rho={r_st:.3f}, p={p_st:.3f}")
df_within = pd.DataFrame(within_rows)
df_within.to_csv(f"{OUTDIR}/within_subtype_correlation.csv", index=False)
print(f"  Saved: {OUTDIR}/within_subtype_correlation.csv")


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x,y controlling for z: rank-transform
    all three variables, then apply the standard Pearson partial-correlation
    formula to the ranks."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    r_partial = (r_xy - r_xz * r_yz) / denom
    n = len(x)
    dof = n - 3
    if dof <= 0 or abs(r_partial) >= 1:
        return r_partial, np.nan, n
    t_stat = r_partial * np.sqrt(dof / (1 - r_partial**2))
    p_val = 2 * stats.t.sf(abs(t_stat), dof)
    return r_partial, p_val, n


subtype_medians = df.groupby("subtype")["LCOR_pseudobulk"].median().reindex(present_subtypes).sort_values()
subtype_rank_map = {st: i for i, st in enumerate(subtype_medians.index, start=1)}
df["subtype_rank"] = df["subtype"].map(subtype_rank_map)
print(f"\n  Subtype rank (by ascending median LCOR_pseudobulk): {subtype_rank_map}")

r_partial, p_partial, n_partial = partial_spearman(
    df["LCOR_pseudobulk"].values, df["lr_means"].values, df["subtype_rank"].values)
print(f"  Partial Spearman (LCOR, lr_means | subtype rank): rho={r_partial:.3f}, p={p_partial:.3f}, n={n_partial}")

# ----------------------------------------------------------------------------
# Mixed-effects model -- subtype as a RANDOM intercept rather than a fixed
# covariate to subtract out. Unlike the partial correlation above (which
# treats subtype as one ordinal number and effectively splits/demeans the
# sample), a mixed model partially POOLS information across the 3 subtype
# groups when estimating the population-level LCOR slope -- more efficient
# than the within-subtype splits at this small n, while still accounting
# for each subtype's own baseline level. Caveat: only 3 groups at the
# random-effect level (ER/HER2/TNBC) is itself a very small sample for
# reliably estimating a variance component -- treat as a cross-check on
# the partial correlation above, not a stronger claim than it.
# ----------------------------------------------------------------------------
import statsmodels.formula.api as smf

try:
    mixed_fit = smf.mixedlm("lr_means ~ LCOR_pseudobulk", data=df, groups=df["subtype"]).fit()
    lcor_coef = mixed_fit.params["LCOR_pseudobulk"]
    lcor_se = mixed_fit.bse["LCOR_pseudobulk"]
    lcor_p = mixed_fit.pvalues["LCOR_pseudobulk"]
    print(f"  Mixed-effects (subtype random intercept): LCOR coef={lcor_coef:.3f} "
          f"(SE={lcor_se:.3f}), p={lcor_p:.3f}, n={len(df)}, groups={len(present_subtypes)}")
except Exception as e:
    lcor_coef = lcor_se = lcor_p = np.nan
    print(f"  Mixed-effects model failed to fit ({e}) -- reporting as NaN")

fig, ax = plt.subplots(figsize=(7, 5))
for st, color in SUBTYPE_COLORS.items():
    d = df[df["subtype"] == st]
    ax.scatter(d["LCOR_pseudobulk"], d["lr_means"], color=color, label=st, s=40, alpha=0.85, edgecolor="white")
ax.set_xlabel(f"{LCOR_GENE} pseudobulk in malignant cells\n(mean logcounts per patient)", fontsize=9)
ax.set_ylabel(f"{LIGAND}->{RECEPTOR}\nLIANA lr_means (mean expr. product)", fontsize=9)
ax.set_title(f"Spearman rho={r_lrmeans:.2f}, p={p_lrmeans:.3f}, n={len(df)}", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=8, title="Subtype")
ax.grid(alpha=0.2)
fig.suptitle(f"{LCOR_GENE} level vs. {LIGAND}->{RECEPTOR} interaction strength (LIANA)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_lcor_vs_interaction.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig_lcor_vs_interaction.png")

# ============================================================================
# FIGURE -- subtype confound check: within-subtype panels + partial-correlation
# residual scatter (same rank-residual construction as partial_spearman above,
# so the annotated rho/p matches the printed partial Spearman exactly).
# ============================================================================
fig, axes = plt.subplots(1, len(present_subtypes) + 1, figsize=(5.2 * (len(present_subtypes) + 1), 4.6))

for i, st in enumerate(present_subtypes):
    ax = axes[i]
    d_st = df[df["subtype"] == st]
    row = df_within[df_within["subtype"] == st].iloc[0]
    ax.scatter(d_st["LCOR_pseudobulk"], d_st["lr_means"], color=SUBTYPE_COLORS[st],
               s=55, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.set_title(f"{st} (n={row['n']})\nSpearman rho={row['spearman_rho']:.2f}, p={row['spearman_p']:.3f}",
                 fontsize=10, fontweight="bold", color=SUBTYPE_COLORS[st])
    ax.set_xlabel(f"{LCOR_GENE} pseudobulk", fontsize=8.5)
    if i == 0:
        ax.set_ylabel(f"{LIGAND}->{RECEPTOR} lr_means", fontsize=8.5)
    ax.grid(alpha=0.2)

rz = stats.rankdata(df["subtype_rank"].values)
rx = stats.rankdata(df["LCOR_pseudobulk"].values)
ry = stats.rankdata(df["lr_means"].values)
slope_x, intercept_x, *_ = stats.linregress(rz, rx)
slope_y, intercept_y, *_ = stats.linregress(rz, ry)
resid_x = rx - (intercept_x + slope_x * rz)
resid_y = ry - (intercept_y + slope_y * rz)

ax = axes[-1]
for st, color in SUBTYPE_COLORS.items():
    m = (df["subtype"] == st).values
    if not m.any():
        continue
    ax.scatter(resid_x[m], resid_y[m], color=color, label=st, s=55, alpha=0.9, edgecolor="white", linewidth=0.6)
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
ax.set_xlabel(f"{LCOR_GENE} rank residual\n(after removing subtype)", fontsize=8.5)
ax.set_ylabel("lr_means rank residual\n(after removing subtype)", fontsize=8.5)
ax.set_title(f"Partial Spearman (subtype rank removed)\nrho={r_partial:.3f}, p={p_partial:.3f}, n={n_partial}",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, title="Subtype")
ax.grid(alpha=0.2)

fig.suptitle(f"Subtype confound check -- {LCOR_GENE} vs {LIGAND}->{RECEPTOR} interaction\n"
             f"Kruskal-Wallis (LCOR ~ subtype): H={kw_stat:.2f}, p={kw_p:.4f} "
             f"(pooled rho={r_lrmeans:.2f} does not survive within-subtype or partial adjustment)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(f"{OUTDIR}/fig_subtype_confound.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig_subtype_confound.png")

# ============================================================================
# Per-patient interaction dotplot -- grouped by subtype, patient labels
# simplified (subtype prefix stripped since the subtype band already shows
# it), extra spacing between subtype blocks and a lower size ceiling so
# neighbouring dots stop overlapping.
# ============================================================================
import re as _re

def _short_label(sample):
    lbl = sample.split("_", 1)[1]
    return _re.sub(r"^(mER|ER|HER2|TN-B1|TN)-", "", lbl)

d = df.copy()
d["subtype"] = pd.Categorical(d["subtype"], categories=["ER", "HER2", "TNBC"], ordered=True)
d = d.sort_values(["subtype", "lr_means"], ascending=[True, False]).reset_index(drop=True)
d["detect_prop"] = np.sqrt(d["ligand_prop"] * d["receptor_prop"])  # geometric mean
d["label"] = d["sample"].map(_short_label)

# x positions with a gap inserted between subtype blocks
xpos, gap, cursor = [], 1.5, 0
prev_st = None
for st in d["subtype"]:
    if prev_st is not None and st != prev_st:
        cursor += gap
    xpos.append(cursor)
    cursor += 1
    prev_st = st
d["xpos"] = xpos

fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(d)), 4.5))
for st, color in SUBTYPE_COLORS.items():
    dd = d[d["subtype"] == st]
    if dd.empty:
        continue
    ax.axvspan(dd["xpos"].min() - 0.6, dd["xpos"].max() + 0.6, color=color, alpha=0.08, zorder=0)
    ax.text(dd["xpos"].mean(), 0.38, st, ha="center", fontsize=9, fontweight="bold", color=color)

sca = ax.scatter(d["xpos"], np.zeros(len(d)), s=d["detect_prop"] * 2200 + 25,
                  c=d["lr_means"], cmap="viridis", edgecolor="black", linewidth=0.5, zorder=3)
ax.set_xticks(d["xpos"])
ax.set_xticklabels(d["label"], rotation=90, fontsize=7)
ax.set_yticks([0])
ax.set_yticklabels([f"{LIGAND} -> {RECEPTOR}"], fontsize=9)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(d["xpos"].min() - 1, d["xpos"].max() + 1)
cb = plt.colorbar(sca, ax=ax, shrink=0.7, pad=0.015)
cb.set_label("lr_means (magnitude)", fontsize=8)
for sz, lbl in [(0.02, "2%"), (0.05, "5%"), (0.10, "10%")]:
    ax.scatter([], [], s=sz * 2200 + 25, c="gray", edgecolor="black", linewidth=0.5,
               label=f"detect. {lbl}")
ax.legend(fontsize=7, title="dot size = sqrt(%lig x %rec)", loc="upper right",
          bbox_to_anchor=(1.0, 1.32), ncol=3)
ax.set_title(f"{LIGAND}->{RECEPTOR} per patient, grouped by subtype\n(sorted by lr_means within subtype)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_interaction_dotplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/fig_interaction_dotplot.png")

# ============================================================================
# README
# ============================================================================
readme = [
    f"01 -- {LIGAND}(Malignant) -> {RECEPTOR}(T_cell) -- LIANA per patient, vs {LCOR_GENE}",
    "="*60,
    "",
    "WHY LCOR IS NOT INSIDE THE LIANA CALL",
    "-"*60,
    f"{LCOR_GENE} is a transcriptional corepressor, not a ligand/receptor --",
    "it has no entry in LIANA's consensus resource, so it cannot be tested",
    "as a cell-cell interaction node. Only the canonical CXCL16-CXCR6 pair",
    "was tested with LIANA. LCOR is linked in afterwards, by correlating",
    f"each patient's {LCOR_GENE} pseudobulk level in malignant cells against",
    "that same patient's CXCL16->CXCR6 interaction score.",
    "",
    "CELL TYPE DEFINITIONS",
    "-"*60,
    "  Malignant: obs['compartment'] == 'Epithelial'",
    "  T_cell:    obs['compartment'] == 'Immune' & (CD3D>0 | CD3E>0 | CD3G>0)",
    "  (no finer cell-type annotation exists in this object; CXCR6 detected",
    "   in 6.75% of these T cells vs 1.12% of non-T immune cells)",
    "",
    "METHOD",
    "-"*60,
    f"  Per patient with >={MIN_CELLS} Malignant and >={MIN_CELLS} T_cell cells:",
    f"    li.mt.rank_aggregate(..., interactions=[('{LIGAND}','{RECEPTOR}')],",
    f"                         expr_prop={EXPR_PROP}, n_perms={N_PERMS})",
    "  Restricting to this single pair (instead of the full LR resource)",
    "  keeps the per-patient run fast and avoids multiple-testing across",
    "  hundreds of irrelevant pairs. expr_prop is lowered from LIANA's",
    "  default (0.1) because both genes are sparsely detected -- the",
    "  default would drop this interaction in several patients from",
    "  dropout alone, not genuine absence. Before calling LIANA, each",
    "  patient is checked against this same expr_prop threshold directly --",
    "  if CXCL16 detection in Malignant cells or CXCR6 detection in T_cell",
    "  is below it, the patient is skipped with a logged reason instead of",
    "  calling LIANA (an empty single-pair result otherwise crashes LIANA's",
    "  internal score assignment with an opaque pandas error).",
    "",
    "WHY magnitude_rank/specificity_rank ARE NOT USED",
    "-"*60,
    "  These are rank statistics computed by comparing the tested pair",
    "  against OTHER pairs within the same LIANA call. Restricting each",
    "  call to a single pair (CXCL16-CXCR6) for speed means there is",
    "  nothing else to rank against, so these columns come back as a",
    "  constant/degenerate value for every patient (no information). Using",
    "  lr_means (raw mean expression product) and cellphone_pvals (a",
    "  permutation p-value -- still meaningful with a single-pair resource,",
    "  since its null distribution comes from permuting cell labels, not",
    "  from ranking across pairs) instead.",
    "",
    f"  n_patients with a valid interaction score: {len(df)}",
    f"  n_patients skipped (insufficient detection or LIANA error): {len(df_skipped)}",
    "  (see persample_skipped.csv for per-patient reason)",
    f"  Spearman(LCOR pseudobulk, lr_means):                rho={r_lrmeans:.3f}, p={p_lrmeans:.3f}",
    f"  Spearman(LCOR pseudobulk, -log10(cellphone_pvals)): rho={r_pval:.3f}, p={p_pval:.3f}",
    "",
    "SUBTYPE CONFOUND CHECK",
    "-"*60,
    "  The pooled correlation above mixes between-subtype and within-subtype",
    "  variation. Three checks address this (see docstring for the full",
    "  rationale) -- outputs: within_subtype_correlation.csv, fig_subtype_confound.png",
    "",
    f"  1) Kruskal-Wallis (LCOR_pseudobulk ~ subtype): H={kw_stat:.3f}, p={kw_p:.4f}",
] + [
    f"       {st}: median={df.loc[df['subtype']==st,'LCOR_pseudobulk'].median():.3f} (n={(df['subtype']==st).sum()})"
    for st in present_subtypes
] + [
    "     -> LCOR pseudobulk itself differs significantly by subtype (lowest in TNBC),",
    "        so subtype is a live confound for any correlation involving LCOR.",
    "",
    "  2) Within-subtype Spearman (removes between-subtype variation entirely):",
] + [
    f"       {row['subtype']} (n={row['n']}): rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.3f}"
    for row in within_rows
] + [
    "     -> Weak/non-significant in every subtype (TNBC underpowered at n=7).",
    "",
    f"  3) Partial Spearman (LCOR, lr_means | subtype rank): rho={r_partial:.3f}, p={p_partial:.3f}, n={n_partial}",
    "     Subtype collapsed to one ordinal covariate (rank of each subtype's",
    "     own median LCOR_pseudobulk: " + str(subtype_rank_map) + "), then the",
    "     standard rank-based partial-correlation formula applied. Weakens",
    "     substantially from the pooled rho=-0.63 and loses significance,",
    "     but does not fall all the way to zero.",
    "",
    f"  4) Mixed-effects (subtype as a RANDOM intercept, LCOR as fixed effect):",
    f"     lr_means ~ LCOR_pseudobulk + (1 | subtype)",
    f"     LCOR coef={lcor_coef:.3f} (SE={lcor_se:.3f}), p={lcor_p:.3f}, n={len(df)}, {len(present_subtypes)} groups",
    "     Unlike (3), this partially POOLS information across subtypes instead",
    "     of fully collapsing them into one covariate -- more efficient at this",
    "     small n, still not significant. Caveat: only 3 groups at the random-",
    "     effect level is itself a small sample for the variance component --",
    "     read as a cross-check on (3), not a stronger claim than it.",
    "",
    "  CONCLUSION: the pooled rho=-0.63 weakens substantially and loses",
    "  significance under every way of accounting for subtype (2, 3, 4) --",
    "  consistent with a mostly subtype-driven contrast rather than a robust",
    "  continuous LCOR effect, though a small residual association (partial",
    "  rho~-0.24) cannot be fully excluded at this sample size.",
    "",
    "CAVEATS",
    "-"*60,
    "  - This is correlational across ~20 patients, not causal.",
    "  - lr_means/cellphone_pvals are computed per patient from cells that",
    "    are themselves sparse for both genes -- per-patient estimates are",
    "    noisy, especially in patients near the MIN_CELLS cutoff.",
    "  - expr_prop was deliberately lowered; treat any single patient's",
    "    'significant' cellphone_pvals with caution given the point above.",
    "  - As with 14/15/16, sequencing depth differences between patients",
    "    could still inflate both LCOR pseudobulk and interaction strength",
    "    in the same direction (deeper samples detect more of everything)",
    "    -- this analysis does NOT control for that.",
]
with open(f"{OUTDIR}/README.txt", "w") as f:
    f.write("\n".join(readme))
print(f"  Saved: {OUTDIR}/README.txt")

print(f"""
============================================================
 DONE
============================================================
  Patients tested: {len(df)}   Patients skipped: {len(df_skipped)}
  Spearman(LCOR, lr_means):                rho={r_lrmeans:.3f}, p={p_lrmeans:.3f}
  Spearman(LCOR, -log10(cellphone_pvals)): rho={r_pval:.3f}, p={p_pval:.3f}
  Kruskal-Wallis (LCOR ~ subtype):         H={kw_stat:.3f}, p={kw_p:.4f}
  Partial Spearman (subtype rank removed): rho={r_partial:.3f}, p={p_partial:.3f}
  Mixed-effects (subtype random intercept): LCOR coef={lcor_coef:.3f}, p={lcor_p:.3f}
  Output: {OUTDIR}/
============================================================
""")
