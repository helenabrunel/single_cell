"""
Part 2 – Step 1  Stemness signature selection (rank-stability approach).

Logic
-----
For each patient, a single percentile P defines symmetric thresholds on
BOTH the stem axis and the immune axis:

    stem_high   = top P%     (>= (100-P)th percentile of that patient)
    immune_low  = bottom P%  (<=      P th percentile of that patient)

SCIE fraction = fraction of cells that are simultaneously stem_high
                AND immune_low.

The same P is tested at 6 values (5 – 30 %).  At each (dataset × P)
condition the 9 signatures are ranked 1–9 by mean SCIE fraction across
patients.  The signature with the most stable top rank is selected.

Outputs
-------
figures/part2/01_rank_stability.png
figures/part2/01_patient_variability.png
figures/part2/01_selected_signature.txt
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
os.makedirs("figures/part2", exist_ok=True)
OUT = "figures/part2"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── configuration ──────────────────────────────────────────────────────────────
DATASETS = {
    "Visvader"  : {"file": "visvader_malignant_ccfree.h5ad",  "batch": "sample"},
    "Klughammer": {"file": "klughammer_malignant_ccfree.h5ad", "batch": "sampleid"},
}
SIGNATURES = {
    "ESC_ASSOU"  : "score_ESC_ASSOU",
    "ESC_WONG"   : "score_ESC_WONG",
    "BenPorath"  : "score_BENPORATH_ES1",
    "EMP"        : "score_EMP",
    "fMaSC"      : "score_fMaSC_over_aMaSC",
    "LIM_Stem"   : "score_LIM_MAMMARY_STEM_CELL_UP",
    "LIM_LumProg": "score_LIM_MAMMARY_LUMINAL_PROGENITOR_UP",
    "LIM_LumMat" : "score_LIM_MAMMARY_LUMINAL_MATURE_UP",
    "stem_cons"  : "score_stem_consensus",
}
IMMUNE_COL = "score_immune"
P_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MIN_CELLS  = 200

# Set to None to use the algorithmic winner
FORCED_SELECTION = "LIM_Stem"

# ── colours ────────────────────────────────────────────────────────────────────
BG   = "#FFFFFF"
DARK = "#1F2937"
GREY = "#6B7280"
BORDER = "#E5E7EB"
C_SCIE = "#26C6DA"
DS_ACCENTS     = {"Visvader": "#2563EB", "Klughammer": "#7C3AED"}
SUBTYPE_COLORS = {"ER": "#2563EB", "HER2": "#059669", "TNBC": "#DC2626"}

# ── compute SCIE fraction per patient × signature × P ─────────────────────────
rows = []
for ds_name, cfg in DATASETS.items():
    print(f"Loading {ds_name} ...")
    adata = sc.read_h5ad(cfg["file"])
    obs   = adata.obs.copy(); del adata
    print(f"  {len(obs):,} cells, {obs[cfg['batch']].nunique()} patients")

    for pat in obs[cfg["batch"]].unique():
        g = obs[obs[cfg["batch"]] == pat]
        if len(g) < MIN_CELLS:
            continue
        subtype = str(g["subtype"].iloc[0])
        imm = g[IMMUNE_COL].values.astype(float)

        for sig_label, col in SIGNATURES.items():
            if col not in g.columns:
                continue
            stem = g[col].values.astype(float)
            valid = np.isfinite(stem) & np.isfinite(imm)
            if valid.sum() < MIN_CELLS:
                continue
            s, i = stem[valid], imm[valid]

            for p in P_VALUES:
                thr_s = np.percentile(s, (1 - p) * 100)   # top P% stem
                thr_i = np.percentile(i,       p  * 100)  # bottom P% immune
                scie  = ((s >= thr_s) & (i <= thr_i)).mean() * 100
                rows.append({
                    "dataset": ds_name, "patient": pat, "subtype": subtype,
                    "signature": sig_label, "P": p, "scie_pct": scie,
                })

df = pd.DataFrame(rows)
print(f"\n  {len(df):,} rows computed")

# ── rank at each (dataset × P) ─────────────────────────────────────────────────
agg = (df.groupby(["dataset", "signature", "P"])["scie_pct"]
         .agg(mean="mean", sem="sem").reset_index())
agg["rank"] = (agg.groupby(["dataset", "P"])["mean"]
                  .rank(ascending=False, method="min").astype(int))

mean_rank = agg.groupby("signature")["rank"].mean().sort_values()
sig_order = mean_rank.index.tolist()

# ── selection ──────────────────────────────────────────────────────────────────
selected = FORCED_SELECTION if FORCED_SELECTION else sig_order[0]
tag      = "manual override" if FORCED_SELECTION else "algorithmic"
print(f"\n  Selected ({tag}): {selected}")
with open(os.path.join(OUT, "01_selected_signature.txt"), "w") as f:
    f.write(selected)

# ── Figure 1: rank stability heatmap ──────────────────────────────────────────
ds_list   = list(DATASETS.keys())
n_s, n_p  = len(sig_order), len(P_VALUES)
n_cols    = len(ds_list) * n_p

rank_mat = np.zeros((n_s, n_cols))
mean_mat = np.zeros((n_s, n_cols))
for di, ds in enumerate(ds_list):
    for pi, p in enumerate(P_VALUES):
        ci  = di * n_p + pi
        sub = agg[(agg["dataset"] == ds) & (agg["P"] == p)]
        for ri, sig in enumerate(sig_order):
            r = sub.loc[sub["signature"] == sig, "rank"].values
            m = sub.loc[sub["signature"] == sig, "mean"].values
            rank_mat[ri, ci] = r[0] if len(r) else np.nan
            mean_mat[ri, ci] = m[0] if len(m) else np.nan

cmap_r = mcolors.LinearSegmentedColormap.from_list("rk", ["#F5F3FF", "#312E81"], N=256)

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)

for ri in range(n_s):
    for ci in range(n_cols):
        rk = rank_mat[ri, ci]
        ax.add_patch(plt.Rectangle(
            (ci, n_s - 1 - ri), 1, 1,
            facecolor=cmap_r((n_s - rk) / (n_s - 1)),
            edgecolor="white", lw=1.5))
        tc = "white" if rk <= 4 else DARK
        ax.text(ci + 0.5, n_s - ri - 0.38, f"{int(rk)}",
                ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
        ax.text(ci + 0.5, n_s - ri - 0.72, f"{mean_mat[ri, ci]:.1f}%",
                ha="center", va="center", fontsize=7.5, color=tc)

ax.set_xticks([c + 0.5 for c in range(n_cols)])
ax.set_xticklabels([f"P={int(p*100)}%" for p in P_VALUES] * len(ds_list), fontsize=8.5)
ax.set_yticks([n_s - ri - 0.5 for ri in range(n_s)])
ax.set_yticklabels(sig_order, fontsize=10)

for di, ds in enumerate(ds_list):
    x0 = di * n_p
    ax.annotate("", xy=(x0 + n_p, n_s + 0.55), xytext=(x0, n_s + 0.55),
                arrowprops=dict(arrowstyle="-", color=DS_ACCENTS[ds], lw=2))
    ax.text(x0 + n_p / 2, n_s + 0.80, ds, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=DS_ACCENTS[ds])
    if di < len(ds_list) - 1:
        ax.axvline(x0 + n_p, color=DARK, lw=2)

sel_row = n_s - 1 - sig_order.index(selected)
ax.add_patch(plt.Rectangle((0, sel_row), n_cols, 1,
             facecolor="none", edgecolor=C_SCIE, lw=3, zorder=5))
ax.text(n_cols + 0.1, sel_row + 0.5, "← selected",
        va="center", fontsize=9, color=C_SCIE, fontweight="bold")

ax.set_xlim(0, n_cols); ax.set_ylim(0, n_s + 1.2)
ax.set_title(
    "Signature rank stability  –  rank 1 = highest SCIE fraction at that (dataset × P)\n"
    "Single symmetric P applied to both stem and immune axes  |  thresholds are per-patient",
    fontsize=10.5, fontweight="bold", color=DARK, pad=10)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(left=False, bottom=False)

sm  = plt.cm.ScalarMappable(cmap=cmap_r, norm=mcolors.Normalize(0, 1))
cb  = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.08)
cb.set_ticks([0, 1]); cb.set_ticklabels(["rank 9\n(worst)", "rank 1\n(best)"])
cb.ax.tick_params(labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "01_rank_stability.png"), dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  01_rank_stability.png")

# ── Figure 2: patient-level SCIE fraction for top 3 signatures ────────────────
top3  = sig_order[:3]
P_REP = 0.15

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"Patient-level SCIE fraction – top 3 signatures  (P = {int(P_REP*100)}%)\n"
    "Box = distribution across patients  |  dots coloured by subtype",
    fontsize=11, fontweight="bold", color=DARK, y=1.03)

for ax, ds in zip(axes, ds_list):
    sub = df[(df["dataset"] == ds) & (df["P"] == P_REP)]
    for xi, sig in enumerate(top3):
        vals    = sub.loc[sub["signature"] == sig, "scie_pct"].values
        pat_sub = (sub[sub["signature"] == sig][["patient", "subtype", "scie_pct"]]
                   .drop_duplicates("patient"))
        col = C_SCIE if sig == selected else GREY
        ax.boxplot(vals, positions=[xi], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.28),
                   medianprops=dict(color=DARK, lw=2),
                   whiskerprops=dict(color=GREY), capprops=dict(color=GREY),
                   flierprops=dict(marker="o", color=GREY, alpha=0.3, ms=4))
        rng = np.random.default_rng(xi)
        for _, row in pat_sub.iterrows():
            jit = rng.uniform(-0.14, 0.14)
            ax.scatter(xi + jit, row["scie_pct"],
                       color=SUBTYPE_COLORS.get(row["subtype"], GREY),
                       s=26, alpha=0.80, zorder=4, edgecolors="white", lw=0.5)
    ax.set_xticks(range(len(top3)))
    ax.set_xticklabels([f"{s}\n(mean rank {mean_rank[s]:.1f})" for s in top3], fontsize=9)
    ax.set_ylabel("SCIE fraction (%)", fontsize=10)
    ax.set_title(ds, fontsize=11, fontweight="bold", color=DS_ACCENTS[ds])
    ax.set_facecolor("#F9FAFB")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.grid(axis="y", color=BORDER, lw=0.7)

handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=SUBTYPE_COLORS[s], markersize=8, label=s)
           for s in ["ER", "HER2", "TNBC"]]
fig.legend(handles=handles, title="Subtype", fontsize=9,
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08),
           framealpha=0.9, edgecolor=BORDER)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "01_patient_variability.png"), dpi=160,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  01_patient_variability.png")

print("\n" + "=" * 55)
print("Mean rank across all 12 conditions:")
for i, sig in enumerate(sig_order, 1):
    mark = "  ← SELECTED" if sig == selected else ""
    print(f"  {i:2d}. {sig:14s}  mean rank = {mean_rank[sig]:.2f}{mark}")
