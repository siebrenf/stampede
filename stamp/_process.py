import numpy as np
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
from natsort import natsorted


def binarize(adata, verbose=True):
    """
    Binarize the values in adata.X.
    """
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    elif verbose:
        print("counts layer already set")

    if "binary" not in adata.layers:
        X = adata.layers["counts"].copy()
        X.data = np.ones_like(X.data, dtype=np.float32)  # set all nonzero entries to 1
        X.eliminate_zeros()  # just to be sure
        adata.layers["binary"] = X.copy()
    elif verbose:
        print("binary layer already set")

    adata.X = adata.layers["binary"].copy()
    if verbose:
        print("binary layer set as adata.X")


def knn_count_smoothing(
        adata,
        layer_added=None,
        obms_key=None,
        neighbours_key=None,
):
    """
    Compute a per-cell neighborhood average of the binary feature matrix.
    """
    layer = "binary"
    if layer not in adata.layers:
        raise KeyError(
            f"{layer=} not found in adata.layers. "
            "Please run st.pp.binarize first!"
        )

    if obms_key is None:
        obms_key = "X_svd"
    if obms_key != "X" and obms_key not in adata.obsm:
        raise KeyError(
            f"{obms_key=} not found in adata.obsm. "
            "Please run st.pp.dim_red (or a similar function) first!"
        )
    if neighbours_key is None:
        neighbours_key = "neighbors_svd"
    connectivities = f"{neighbours_key}_connectivities"
    if connectivities not in adata.obsp:
        sc.pp.neighbors(
            adata,
            use_rep=obms_key,
            key_added=neighbours_key,
        )

    # create connectivity map
    knn = adata.obsp[connectivities].copy()
    knn.data = np.ones_like(knn.data)
    knn.setdiag(1)
    # print(type(knn), knn.shape, knn.dtype)

    # number of neighbours per cell
    deg = np.asarray(knn.sum(axis=1))
    # print(type(deg), deg.shape, deg.dtype)

    # normalize
    knn = knn.multiply(1 / deg)
    # print(type(knn), knn.shape, knn.dtype)

    X = adata.layers[layer]
    # print(type(X), X.shape, X.dtype)

    # apply to all genes
    data = knn.dot(X)
    # print(type(data), data.shape, data.dtype)

    # sanity checks
    assert data.shape == X.shape
    assert data.dtype == np.float32
    assert isinstance(data, sp.csr_matrix)

    if layer_added is None:
        layer_added = "KNN_binary_mean"
    adata.layers[layer_added] = data


def pseudobulk(
        adata,
        ctype,
        ctype_col,
        sample_col,
        layer=None,
):
    """
    Generate a pseudobulk table (genes x samples) for all
    samples in sample_col and the specified ctype in ctype_col.
    """
    if ctype not in adata.obs[ctype_col].unique():
        raise ValueError(f"{ctype=} not found in adata.obs['{ctype_col}']")
    sample2counts = {}
    adata_sub = adata[adata.obs[ctype_col] == ctype]
    for sample in natsorted(adata_sub.obs[sample_col].unique()):
        obj = adata_sub[adata_sub.obs[sample_col] == sample]
        X = obj.layers[layer] if layer else obj.X
        sample2counts[sample] = X.sum(axis=0).A1

    pseudobulk_df = pd.DataFrame(sample2counts, index=adata_sub.var_names)
    return pseudobulk_df
