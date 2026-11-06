"""
fix_klughammer_gene_names.py
=============================
Converts the 4740 residual ENSG-formatted var_names in
Klughammer/klughammer_Malignant.h5ad to HGNC gene symbols using mygene.

Rules applied in order:
  1. Query mygene for all ENSG IDs in one batched call.
  2. For each ENSG ID: use the returned gene symbol if it is non-empty,
     unique after mapping, and not already present in var_names.
  3. Otherwise: keep the original ENSG ID (safe fallback).

The updated h5ad is written back in place (original renamed to .h5ad.bak).
Identical result if re-run (idempotent).

Run from: ~/single_cell/
  /home/hbrunel/miniconda3/bin/python3 fix_klughammer_gene_names.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import mygene
import os, shutil

H5AD = "Klughammer/klughammer_Malignant.h5ad"
BAK  = H5AD + ".bak"

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {H5AD} …")
adata = sc.read_h5ad(H5AD)

var_names = adata.var_names.tolist()
ensg_mask = [vn.startswith("ENSG") for vn in var_names]
ensg_ids  = [vn for vn, m in zip(var_names, ensg_mask) if m]

print(f"Total genes : {len(var_names)}")
print(f"ENSG IDs    : {len(ensg_ids)}")

if len(ensg_ids) == 0:
    print("No ENSG IDs found — nothing to do.")
    exit(0)

# ── Query mygene ──────────────────────────────────────────────────────────────
print(f"Querying mygene for {len(ensg_ids)} IDs …")
mg = mygene.MyGeneInfo()
results = mg.querymany(
    ensg_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    returnall=True,
    verbose=False,
)

# Build mapping: ENSG → symbol
ensg_to_sym: dict[str, str] = {}
for hit in results.get("out", []):
    query = hit.get("query", "")
    sym   = hit.get("symbol", "")
    if query and sym and not hit.get("notfound", False):
        # Keep first hit if query appears twice
        if query not in ensg_to_sym:
            ensg_to_sym[query] = sym

mapped = sum(1 for e in ensg_ids if e in ensg_to_sym)
print(f"Mapped      : {mapped} / {len(ensg_ids)}")

# ── Build new var_names (unique, no collision with existing symbols) ───────────
existing_symbols = set(vn for vn, m in zip(var_names, ensg_mask) if not m)
new_var_names = var_names.copy()
seen: set[str] = set(existing_symbols)

renamed = 0
for i, (vn, is_ensg) in enumerate(zip(var_names, ensg_mask)):
    if not is_ensg:
        seen.add(vn)
        continue
    sym = ensg_to_sym.get(vn, "")
    if sym and sym not in seen:
        new_var_names[i] = sym
        seen.add(sym)
        renamed += 1
    else:
        # Keep ENSG ID — either unmapped or would collide
        seen.add(vn)

print(f"Renamed     : {renamed}")
print(f"Kept as ENSG: {len(ensg_ids) - renamed}")

# Verify uniqueness
assert len(set(new_var_names)) == len(new_var_names), \
    "BUG: duplicate var_names after renaming!"

# ── Check no signature genes were affected ────────────────────────────────────
import json
with open("Klughammer/signatures_confirmed.json") as f:
    sigs = json.load(f)
sig_genes = set(g for gl in sigs.values() for g in gl)
immune = {
    "B2M","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB",
    "HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DRA",
    "HLA-DRB1","CD74","TAP1","TAP2","TAPBP","CIITA","PSME1","PSME2","IFI30",
    "CD8A","CD8B",
}
all_target_genes = sig_genes | immune
# Verify all are still present in the new names
new_set = set(new_var_names)
missing = all_target_genes - new_set
if missing:
    print(f"WARNING: {len(missing)} signature/immune genes missing after rename: {missing}")
else:
    print("All signature + immune genes retained — OK")

# ── Apply and save ────────────────────────────────────────────────────────────
if new_var_names == var_names:
    print("No changes needed — file already has gene symbols.")
else:
    print(f"\nBacking up original → {BAK}")
    shutil.copy2(H5AD, BAK)

    adata.var_names = pd.Index(new_var_names)
    # Also update var['feature_name'] and var['symbol'] if present
    if "feature_name" in adata.var.columns:
        adata.var["feature_name"] = new_var_names
    if "symbol" in adata.var.columns:
        adata.var["symbol"] = new_var_names

    print(f"Saving updated h5ad → {H5AD}")
    adata.write_h5ad(H5AD)
    print("Done.")

# ── Quick sanity check ────────────────────────────────────────────────────────
remaining_ensg = sum(1 for vn in adata.var_names if vn.startswith("ENSG"))
print(f"\nRemaining ENSG-format names: {remaining_ensg}")
print(f"Sample new names (first ENSG positions): "
      f"{[new_var_names[i] for i, m in enumerate(ensg_mask) if m][:10]}")
