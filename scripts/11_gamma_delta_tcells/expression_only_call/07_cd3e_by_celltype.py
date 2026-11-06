"""
07_cd3e_by_celltype.py
======================
Where does CD3E in author-NK cells come from? (Section 4.1 of the report:
34.7 % of author-NK cells express CD3E.)

CD3E is compared with other T-cell transcripts (CD3D, CD3G, TRAC, TRBC2, CD8B)
in every author cell type of the full atlas (raw counts). If CD3E in a cell type
comes from T cells (ambient RNA, doublets, mislabelled T cells), it should come
with CD3D in the same proportion as in T cells (CD3E/CD3D detection ratio ≈ T).

Input:  ../data/dataset.h5ad
Output: report/data/cd3e_by_celltype.csv, report/data/cd3e_nk_noT.csv

Run from: ~/single_cell/Klughammer/gdT/
  python3 07_cd3e_by_celltype.py > logs/07_cd3e_by_celltype.log
"""

import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse

SRC = "../data/dataset.h5ad"
OUT = "report/data"
GENES = ["CD3E", "CD3D", "CD3G", "TRAC", "TRBC2", "CD247", "KLRF1", "CD8B"]
OTHER_T = ["CD3D", "CD3G", "TRAC", "TRBC2", "CD8B"]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    a = ad.read_h5ad(SRC, backed="r")
    sym = a.var.loc[a.raw.var.index, "feature_name"].astype(str).values
    idx = [int(np.where(sym == g)[0][0]) for g in GENES]
    X = np.vstack([sparse.csr_matrix(a.raw.X[s:s + 20000])[:, idx].toarray()
                   for s in range(0, a.n_obs, 20000)])
    X = pd.DataFrame(X, columns=GENES, index=a.obs_names)
    ct = a.obs["author_cell_type"].astype(str).values

    det = (X > 0).groupby(ct).mean().mul(100).round(1)
    det.insert(0, "n", pd.Series(ct).value_counts())
    det["CD3E/CD3D"] = (det["CD3E"] / det["CD3D"]).round(2)
    det.index.name = "cell_type"
    print(det.to_string())
    det.to_csv(f"{OUT}/cd3e_by_celltype.csv")

    noT = (X[OTHER_T] == 0).all(axis=1).values
    rows = []
    for t in ["NK", "Macrophage", "Fibroblast", "Endothelial", "B"]:
        m = (ct == t) & noT
        rows.append({"cell_type": t, "n without other T genes": int(m.sum()),
                     "% CD3E+": round(100 * (X["CD3E"].values[m] > 0).mean(), 1)})
    nt = pd.DataFrame(rows)
    print("\nCells with none of " + ", ".join(OTHER_T) + ":\n" + nt.to_string(index=False))
    nt.to_csv(f"{OUT}/cd3e_nk_noT.csv", index=False)

    nk = ct == "NK"
    exp = det.loc["T", "CD3E/CD3D"] * det.loc["NK", "CD3D"]
    print(f"\nNK: CD3E expected if it followed CD3D as in T cells = {exp:.1f} %, observed {det.loc['NK', 'CD3E']} %")
    print(f"NK CD3E+ cells lacking all other T genes: {100 * noT[nk & (X['CD3E'].values > 0)].mean():.1f} %")
