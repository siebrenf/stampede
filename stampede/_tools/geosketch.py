from __future__ import annotations

import anndata as ad
import matplotlib.figure
import scanpy as sc


def sketch(
    adata: ad.AnnData,
    n: int = None,
    frac: float = 0.05,
    use_rep: str = "X_svd",
    obs_column: str = "subset",
    return_subset: bool = False,
) -> ad.AnnData | None:
    """
    Subset the cells in adata using GeoSketch.

    Args:
        adata: adata object
        n: the number of cells to keep. If None, `frac` will be used instead.
        frac: the fraction of cells to keep. Only used if `n` is None.
        use_rep: use the indicated representation.
        obs_column: add this column to adata.obs with boolean values if the cell is kept.
        return_subset: if True, return a subset adata object.

    Returns:
         The subset anndata object (if specified)
    """
    # optional dependency
    from geosketch import gs  # noqa

    if n is None:
        n = round(len(adata) * frac)
    sketch_index = gs(adata.obsm[use_rep], n, replace=False)
    adata.obs[obs_column] = adata.obs.index.isin(adata.obs.iloc[sketch_index].index)

    if return_subset:
        return adata[adata.obs[obs_column], :].copy()
    else:
        return None


def plot_sketch(
    adata: ad.AnnData,
    obs_column: str = "subset",
    use_rep: str = "X_svd",
    plot_kwargs: dict = None,
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    """
    Scatterplot highlighting the cells that were sampled.
    Requires the full adata object.

    Args:
        adata: adata object
        obs_column: column in adata.obs with boolean values if the cell is kept
        use_rep: use the indicated representation
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    adata.uns[f"{obs_column}_colors"] = [
        "#dadafe",  # light blue (False/discarded)
        "#000000",  # black (True/selected)
    ]
    fig = sc.pl.embedding(
        adata, color=obs_column, basis=use_rep, return_fig=True, **plot_kwargs
    )
    axs = fig.axes
    return fig, axs
