from __future__ import annotations

from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from natsort import natsorted


def binarize(adata: ad.anndata, verbose: bool = True):
    """
    Binarize the values in adata.X

    Args:
        adata: adata object
        verbose: provide written feedback (default: True)

    Returns:
        None: updates adata.layers and adata.X
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
    adata: ad.anndata,
    layer_added=None,
    neighbors_use_rep=None,
    neighbors_key_added=None,
    neighbors_kwargs=None,
    verbose=True,
):
    """
    For each cell, replace its gene vector with the average of its KNN neighborhood.

    Runs sc.pp.neighbors if it has not run.
    See https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html

    Args:
        adata: adata object
        layer_added: key in adata.layers for function output (default: "KNN_binary_mean")
        neighbors_use_rep: See sc.pp.neighbors for details
        neighbors_key_added: See sc.pp.neighbors for details
        neighbors_kwargs: kwargs passed to sc.pp.neighbors
        verbose: provide written feedback (default: True)

    Returns:
        None: updates adata.layers and adata.X
    """
    layer = "binary"
    if layer not in adata.layers:
        raise KeyError(
            f"{layer=} not found in adata.layers. Please run st.pp.binarize first!"
        )

    if neighbors_use_rep is None:
        neighbors_use_rep = "X_svd"
    if neighbors_use_rep != "X" and neighbors_use_rep not in adata.obsm:
        raise KeyError(
            f"{neighbors_use_rep=} not found in adata.obsm. "
            "Please run st.pp.dim_red (or a similar function) first!"
        )

    if neighbors_key_added is None:
        neighbors_key_added = "neighbors_svd"
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    connectivities = f"{neighbors_key_added}_connectivities"
    if connectivities not in adata.obsp:
        if verbose:
            print("running sc.pp.neighbors")
        sc.pp.neighbors(
            adata,
            use_rep=neighbors_use_rep,
            key_added=neighbors_key_added,
            **neighbors_kwargs,
        )

    if layer_added is None:
        layer_added = f"KNN_{layer}_mean"
    if layer_added not in adata.layers:
        # KNN neighborhood connectivity map
        knn = adata.obsp[connectivities].copy()
        knn.data = np.ones_like(knn.data)
        knn.setdiag(1)  # include self

        # number of neighbors per cell + itself
        deg = np.asarray(knn.sum(axis=1))

        # row-normalized connectivity map
        knn = knn.multiply(1 / deg)  # coo_matrix

        # average gene presence across its neighborhood
        X = adata.layers[layer]
        data = knn.dot(X)

        # sanity checks
        assert data.shape == X.shape
        assert data.dtype == np.float32
        assert isinstance(data, sp.csr_matrix)

        adata.layers[layer_added] = data
    elif verbose:
        print(f"{layer_added} layer already set")

    adata.X = adata.layers[layer_added].copy()
    if verbose:
        print(f"{layer_added} layer set as adata.X")


def pseudobulk(
    adata: ad.anndata,
    cluster: str,
    cluster_column: str,
    sample_column: str,
    samples: Iterable = None,
    layer: str = None,
):
    """
    Generate a pseudobulk table (genes x samples) for all samples in the sample_column
    and the specified cluster in the cluster_column.

    Args:
        adata: adata object
        cluster: name of the cluster in cluster_column to aggregate to pseudobulk
        cluster_column: column in adata.obs
        sample_column: column in adata.obs
        samples: samples in the sample columns to use (default: all)
        layer: layer to aggregate (default: "counts")

    Returns:
        pd.DataFrame
    """
    if cluster not in adata.obs[cluster_column].unique():
        raise ValueError(f"{cluster=} not found in adata.obs['{cluster_column}']")
    if layer is None:
        layer = "counts"

    sample2counts = {}
    adata_sub = adata[adata.obs[cluster_column] == cluster]
    if samples is None:
        samples = natsorted(adata_sub.obs[sample_column].unique())
    for sample in samples:
        X = adata_sub[adata_sub.obs[sample_column] == sample].layers[layer]
        sample2counts[sample] = X.sum(axis=0).A1

    pseudobulk_df = pd.DataFrame(data=sample2counts, index=adata_sub.var_names)
    return pseudobulk_df
