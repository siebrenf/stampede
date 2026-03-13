import numpy as np


def binarize(adata, verbose=True):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    elif verbose:
        print("counts layer already set")

    if "binary" not in adata.layers:
        X = adata.layers["counts"].copy()
        X.data = np.ones_like(X.data)  # set all nonzero entries to 1
        X.eliminate_zeros()  # just to be sure
        adata.layers["binary"] = X.copy()
    elif verbose:
        print("binary layer already set")

    if verbose:
        print("binary layer set as adata.X")
    adata.X = adata.layers["binary"].copy()
