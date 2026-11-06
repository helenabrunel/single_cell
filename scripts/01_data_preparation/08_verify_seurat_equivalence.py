"""
04b_verify_seurat_equivalence.py
================================
Step 2/2 of the equivalence check. Compares the original gse243526_raw.h5ad
(read by anndata, an implementation independent of the R converter) against
the dump of the reloaded Seurat object written by 04b_dump_seurat.R.

Checks are exact (no tolerances): matrix structure + values, cell/gene names
and order, every obs/var column (values, NA positions, categories, category
order, ordered flag, type class), and that nothing else exists in the h5ad.

  python3 04b_verify_seurat_equivalence.py
"""
import sys
import h5py
import numpy as np
import pandas as pd
import anndata as ad

H5AD = "gse243526_raw.h5ad"
DUMP = "gse243526_raw_seurat_dump.h5"
SEURAT_EXTRA_OBS = {"orig.ident", "nCount_RNA", "nFeature_RNA"}
SEURAT_EXTRA_VAR = {"original_var_name"}

fails = []
def check(name, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        fails.append(name)

def s(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x], dtype=object)

a = ad.read_h5ad(H5AD)
d = h5py.File(DUMP, "r")

print("\n== Structure: nothing in the h5ad left unconverted")
check("layers empty", len(a.layers) == 0)
check("obsm empty", len(a.obsm) == 0)
check("obsp empty", len(a.obsp) == 0)
check("varm empty", len(a.varm) == 0)
check("varp empty", len(a.varp) == 0)
check("uns empty", len(a.uns) == 0)
check("raw is None", a.raw is None)

print("\n== Names and order")
cells = s(d["cells"][:]); genes = s(d["genes"][:])
check("cell names identical & same order", np.array_equal(cells, a.obs_names.values.astype(object)))
orig_genes = a.var_names.values.astype(object)
check("gene names identical except '_'->'-'",
      np.array_equal(genes, np.array([g.replace("_", "-") for g in orig_genes], dtype=object)))
seurat_var_orig = s(d["var/original_var_name/values"][:])
check("var original_var_name == h5ad var_names", np.array_equal(seurat_var_orig, orig_genes))
print(f"     ({int((genes != orig_genes).sum())} genes renamed '_'->'-', original kept)")

print("\n== Count matrix (h5ad X, CSR cells x genes  vs  Seurat counts, CSC genes x cells)")
X = a.X
check("dims", tuple(d["counts_dim"][:]) == (a.n_vars, a.n_obs))
check("X is canonical CSR (sorted, no dups)", X.has_canonical_format)
check("indptr identical", np.array_equal(X.indptr.astype(np.int64), d["counts_p"][:].astype(np.int64)))
check("indices identical", np.array_equal(X.indices.astype(np.int64), d["counts_i"][:].astype(np.int64)))
x_r = d["counts_x"][:]
check("all Seurat values are integers", bool(np.all(x_r == np.round(x_r))))
check("values identical (int64 == float64, exact)", np.array_equal(X.data.astype(np.float64), x_r)
      and np.array_equal(x_r.astype(np.int64), X.data.astype(np.int64)))
check("nnz identical", X.nnz == x_r.size)
check("total sum identical", int(X.data.sum()) == int(x_r.sum()))
del x_r

def cmp_df(df, grp, extra, label):
    print(f"\n== {label}")
    cols = list(s(d[f"{grp}/__columns__"][:]))
    check(f"{label}: all h5ad columns present", set(df.columns) <= set(cols))
    check(f"{label}: only expected extra columns {sorted(extra)}", set(cols) - set(df.columns) == extra)
    check(f"{label}: h5ad column order preserved",
          [c for c in cols if c in df.columns] == list(df.columns))
    for c in df.columns:
        src = df[c]; g = d[f"{grp}/{c}"]; kind = s(g["kind"][:])[0]
        if isinstance(src.dtype, pd.CategoricalDtype):
            ok = (kind == "factor"
                  and np.array_equal(s(g["levels"][:]), src.cat.categories.astype(str).values.astype(object))
                  and np.array_equal(g["codes"][:].astype(np.int64), src.cat.codes.values.astype(np.int64))
                  and bool(g["ordered"][:][0]) == bool(src.cat.ordered))
            check(f"{c}: categorical (levels, order, codes, NA={int(src.isna().sum())})", ok)
        elif src.dtype == bool:
            check(f"{c}: bool", kind == "logical" and np.array_equal(g["values"][:].astype(bool), src.values))
        elif src.dtype.kind in "iu":
            v = g["values"][:]
            check(f"{c}: {src.dtype} -> R {kind}", kind == "integer" and np.array_equal(v.astype(np.int64), src.values.astype(np.int64)))
        elif src.dtype.kind == "f":
            v = g["values"][:]
            check(f"{c}: {src.dtype} -> R {kind} (bit-exact)", kind == "numeric"
                  and np.array_equal(v, src.values.astype(np.float64), equal_nan=True))
        else:
            check(f"{c}: string", kind == "character" and np.array_equal(s(g["values"][:]), src.astype(str).values.astype(object)))

cmp_df(a.obs, "obs", SEURAT_EXTRA_OBS, "Cell metadata (obs -> meta.data)")
cmp_df(a.var, "var", SEURAT_EXTRA_VAR, "Gene metadata (var -> RNA meta.features)")

print("\n" + "=" * 60)
if fails:
    print(f" FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print(" ALL CHECKS PASSED: Seurat object is exactly equivalent to the h5ad")
print("=" * 60)
