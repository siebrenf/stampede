from __future__ import annotations

import itertools
import os
from collections.abc import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from natsort import natsort_keygen, natsorted


def slide_qc(adata: ad.AnnData, slides: dict, data_dir: str = None) -> None:
    """
    Use the fov_positions file to create a dataframe with metadata columns per slide
    and fov, and store this in adata.uns["fov_metadata"].
    Additional adds columns to adata.obs reflecting the distance from the cell to the
    camera's FOV edge.

    Args:
        adata: adata object generated using the slides dict
        slides: a dictionary with the slide number as keys, and a dictionary as values.
          The value dict must contain keys "exprmat" and "metadata", with should map to
          matching respective files
        data_dir: optional filepath prefix (default: "")

    Returns:
        Nothing, updates adata.uns and adata.obs
    """
    if data_dir is None:
        data_dir = ""
    # add a composite column to adata.obs if it does not exist
    if "slide-fov" not in adata.obs.columns:
        adata.obs["slide-fov"] = (
            adata.obs["slide"].astype(str) + "-" + adata.obs["fov"].astype(str)
        )

    # get the coordinates of each fov from the fov_positions file
    fovsxy = []
    for slide, files in slides.items():
        fovdata = pd.read_csv(
            os.path.join(data_dir, files["fov_positions"]),
            usecols=["FOV", "x_global_px", "y_global_px"],
            dtype=int,
        )
        for _, row in fovdata.iterrows():
            fov = row["FOV"]
            idx = f"{slide}-{fov}"
            x = row["x_global_px"]
            y = row["y_global_px"]
            fovsxy.append([idx, slide, fov, x, y])
    fov_df = pd.DataFrame(
        data=fovsxy, columns=["slide-fov", "slide", "fov", "x", "y"]
    ).set_index("slide-fov")

    # Add additional metadata columns
    fov_df["nCounts"] = adata.obs.groupby("slide-fov", observed=True)[
        "nCount_RNA"
    ].sum()
    fov_df = pd.merge(
        left=fov_df,
        right=adata.obs["slide-fov"].value_counts().rename("nCell"),
        on="slide-fov",
    )
    fov_df["meanCountsPerCell"] = fov_df["nCounts"] / fov_df["nCell"]
    fov_df["nCount_negprobes"] = adata.obs.groupby("slide-fov", observed=True)[
        "nCount_negprobes"
    ].sum()
    fov_df["mean_NegProbe-CountsPerCell"] = fov_df["nCount_negprobes"] / fov_df["nCell"]
    fov_df["nCount_falsecode"] = adata.obs.groupby("slide-fov", observed=False)[
        "nCount_falsecode"
    ].sum()
    fov_df["mean_FalseCode-CountsPerCell"] = (
        fov_df["nCount_falsecode"] / fov_df["nCell"]
    )
    fov_df["meanCellSize"] = (
        adata.obs.groupby("slide-fov", observed=False)["Area.um2"].sum()
        / fov_df["nCell"]
    )
    slidefov2passfail = (
        adata.obs.groupby("slide-fov", observed=False)["qcFlagsFOV"].first().to_dict()
    )
    fov_df = fov_df.reset_index()
    fov_df["Failed_AtoMX_QC"] = (
        fov_df["slide-fov"]
        .replace(slidefov2passfail)
        .replace({"Pass": "0", "Fail": "1"})
        .astype(int)
    )
    adata.uns["fov_metadata"] = fov_df

    # determine the dimensions of the camera's FOV
    dim_x_px, dim_y_px = _fov_dimensions(fov_df)
    adata.uns["fov_dims_px"] = {"x": dim_x_px, "y": dim_y_px}

    # compute the distance of each cell to the nearest edge along the x- and y-axis
    if "dist2edge_px" not in adata.obs.columns:
        adata.obs["x2edge_px"] = np.minimum(
            adata.obs["CenterX_local_px"], dim_x_px - adata.obs["CenterX_local_px"]
        )
        adata.obs["y2edge_px"] = np.minimum(
            adata.obs["CenterY_local_px"], dim_y_px - adata.obs["CenterY_local_px"]
        )
        adata.obs["dist2edge_px"] = np.minimum(
            adata.obs["x2edge_px"], adata.obs["y2edge_px"]
        )


def gene_qc(
    adata: ad.AnnData,
    signal2noise_threshold: float | Iterable = None,
    mult: int | float = 1,
    overwrite: bool = False,
) -> None:
    r"""
    Add QC parameters to adata.var.

    About the Signal-to-noise filter:
        Approach from https://doi.org/10.1038/s41467-025-64990-y
        Wang et al. "Systematic benchmarking of imaging spatial
        transcriptomics platforms in FFPE tissues" Nat Com, 2025.

        Calculate the mean expression and standard deviation of the negative control
        probes.
        Remove genes with average expression < mean + mult\* x STD of ctrl probes.

        \*the paper used mult=2

    Args:
        adata: an adata object
        signal2noise_threshold: manually specify the threshold.
         If None, use the filter specified above.
        mult: if signal2noise_threshold is None, mult is used in the signal2noise
         threshold computation specified above.
        overwrite: overwrite existing qc columns (default: False)

    Returns:
        Nothing, updates adata.var
    """
    if "is_negctrl" not in adata.var.columns or overwrite:
        adata.var["is_negctrl"] = adata.var_names.str.startswith("Negative")
    if "is_sysctrl" not in adata.var.columns or overwrite:
        adata.var["is_sysctrl"] = adata.var_names.str.startswith("System")
    if "nCell" not in adata.var.columns or overwrite:
        # number of nonzero cells per gene
        adata.var["nCell"] = (adata.X > 0).sum(axis=0).A1
        adata.var["pctCell"] = 100 * adata.var["nCell"] / adata.n_obs
    if "nTranscript" not in adata.var.columns or overwrite:
        adata.var["nTranscript"] = np.array(adata.X.sum(axis=0)).ravel()
    if "meanTranscript" not in adata.var.columns or overwrite:
        adata.var["meanTranscript"] = adata.var["nTranscript"] / adata.n_obs
    if "signal2noise" not in adata.var.columns or overwrite:
        if signal2noise_threshold is None:
            negctrls = adata.var.loc[adata.var["is_negctrl"], "meanTranscript"]
            signal2noise_threshold = negctrls.mean() + mult * negctrls.std()
        adata.var["signal2noise"] = adata.var["meanTranscript"] / signal2noise_threshold


def gene_qc_postfilter(adata: ad.AnnData) -> None:
    """
    Compute metadata after filtering

    Args:
        adata: an adata object

    Returns:
        Nothing, updates adata.var
    """
    adata.var["nCell_postfilter"] = adata.X.count_nonzero(axis=0)
    adata.var["pctCell_postfilter"] = (
        100 * adata.var["nCell_postfilter"] / adata.n_obs
    ).round(2)

    if adata.var["nCell"].eq(adata.var["nCell_postfilter"]).all():
        raise ValueError("adata was not filtered")


def cell_qc_postfilter(adata: ad.AnnData) -> None:
    """
    Compute metadata after filtering

    Args:
        adata: an adata object

    Returns:
        Nothing, updates adata.obs
    """
    adata.obs["nFeature_RNA_postfilter"] = adata.X.count_nonzero(axis=1)
    adata.obs["nCount_RNA_postfilter"] = adata.X.sum(axis=1)

    if adata.obs["nCount_RNA"].eq(adata.obs["nCount_RNA_postfilter"]).all():
        raise ValueError("adata was not filtered")


def _fov_dimensions(fov_df):
    dx = []
    dy = []
    for slide in sorted(fov_df["slide"].unique()):
        last_x = None
        last_y = None
        df = fov_df[fov_df["slide"] == slide].sort_values("fov")
        for fov, row in df.iterrows():
            x, y = row["x"], row["y"]
            if last_x is None:
                pass
            else:
                # compute dx if the fov only changed along the x-axis
                # compute dy if the fov changed along the y-axis
                if y == last_y:
                    _dx = abs(x - last_x)
                    dx.append(_dx)
                else:
                    _dy = abs(y - last_y)
                    dy.append(_dy)
            last_x = x
            last_y = y
    # median distance between 2 FOVs in pixels
    x_px = sorted(dx)[len(dx) // 2]
    y_px = sorted(dy)[len(dy) // 2]
    return x_px, y_px


def plot_slide_qc(
    adata: ad.AnnData,
    columns: str | Iterable = None,
    figsize: tuple = None,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the values from one or QC columns in adata.uns["fov_metadata"]
    (added by `slide_qc_data()`).
    Specify columns to limit the number of plots.

    Args:
        adata: an adata object
        columns: columns in adata.uns["fov_metadata"] to plot (default: all)
        figsize: tuple of figure, will be multiplied by the number of plots
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if figsize is None:
        figsize = (5, 4)
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    fov_df = adata.uns["fov_metadata"]
    required_cols = ["slide", "x", "y"]
    for col in required_cols:
        if col not in fov_df.columns:
            raise ValueError(f"column={col} is required in adata.obs")
    if columns is None:
        columns = fov_df.columns.to_list()
    elif isinstance(columns, str):
        columns = [columns]
    fov_df = fov_df[[c for c in columns if c not in required_cols] + required_cols]

    slides = natsorted(fov_df["slide"].unique())
    blacklist = {"slide-fov", "slide", "fov", "x", "y"}
    qc_columns = [c for c in fov_df.columns if c not in blacklist]
    nrows = len(slides)
    ncols = len(qc_columns)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
        layout="tight",
        **subplot_kwargs,
    )
    fig.supylabel("y", x=1.0, ha="right", rotation=0)
    fig.supxlabel("x")
    for i_col, column in enumerate(qc_columns):
        # normalize the colormap for all slides at once
        norm = Normalize(vmin=fov_df[column].min(), vmax=fov_df[column].max())
        for i_row, slide in enumerate(slides):
            ax = axs[i_row, i_col]

            # main plot
            sns.scatterplot(
                data=fov_df[fov_df["slide"] == slide],
                x="x",
                y="y",
                hue=column,
                palette="coolwarm",
                hue_norm=norm,
                legend=False,
                ax=ax,
                **plot_kwargs,
            )

            # tweak all ticks, borders & labels
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i_row == nrows - 1:
                ax.spines["bottom"].set_visible(False)
            if i_col == 0:
                ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            if i_col == 0:
                ax.set_ylabel(f"slide {slide}")

            # only plot the colorbar once per column
            if i_row == 0:
                sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(
                    sm,
                    location="top",
                    orientation="horizontal",
                    shrink=0.4,
                    pad=0.1,
                    aspect=30,
                    ax=ax,
                )
                # colorbar label doubles as the figure title
                cbar.set_label(column)

    # return the plot elements for manual post-processing
    return fig, axs


# def plot_fov_edge_distances(
#     adata,
#     figsize=(12, 6),
#     subplot_kwargs=None,
#     # plot_kwargs=None,
# ):
#     bins = 50
#     cmap = "Blues"
#     color1 = "dodgerblue"
#     color2 = "limegreen"
#     if subplot_kwargs is None:
#         subplot_kwargs = {}
#     # if plot_kwargs is None:
#     #     plot_kwargs = {}
#
#     fig, axs = plt.subplots(
#         ncols=3,
#         gridspec_kw={"width_ratios": [1, 1, 0.05]},
#         constrained_layout=True,
#         figsize=figsize,
#         **subplot_kwargs,
#     )
#     x, y = adata.uns["fov_dims_px"].values()
#     fig.suptitle(
#         "Distributions of cell distances to the edge of the camera's FOV "
#         + f"({x}x{y} pixels)"
#     )
#
#     # 1D histplot
#     sns.histplot(
#         np.log1p(adata.obs["dist2edge_px"]),
#         bins=bins,
#         stat="percent",
#         color=color1,
#         ax=axs[0],
#         alpha=0.5,
#     )
#     axs[0].set_title("Minimum distance to the FOV edge")
#     axs[0].set_xlabel("log1p(pixels)", color=color1)
#     axs[0].set_ylabel("% of cells")
#     axs01 = axs[0].twiny()
#     sns.histplot(
#         adata.obs["dist2edge_px"],
#         bins=bins,
#         stat="percent",
#         color=color2,
#         ax=axs01,
#         alpha=0.5,
#     )
#     axs01.set_xlabel("pixels", color=color2)
#     axs[0].set_zorder(1)
#     axs01.set_zorder(0)
#     axs[0].patch.set_alpha(0)  # background transparency
#
#     # 2D histplot
#     sns.histplot(
#         x=adata.obs["x2edge_px"],
#         y=adata.obs["y2edge_px"],
#         cmap=cmap,
#         cbar=False,
#         ax=axs[1],
#     )
#     axs[1].set_title("2D distance to FOV edge")
#     axs[1].set_xlabel("Pixels along the x-axis")
#     axs[1].set_ylabel("Pixels along the y-axis")
#
#     # colormap
#     norm = Normalize(
#         vmin=0,
#         vmax=max(adata.obs["x2edge_px"].max(), adata.obs["y2edge_px"].max()),
#     )
#     ColorbarBase(axs[2], cmap=cmap, norm=norm)
#     axs[2].set_ylabel("Number of cells")
#
#     return fig, axs


def plot_2d_correlations(
    adata: ad.AnnData,
    xcolumn: str,
    ycolumn: str,
    log1p_xcolumn: bool = False,
    log1p_ycolumn: bool = False,
    color_xcolumn: ColorType = None,
    color_ycolumn: ColorType = None,
    cmap_2d: ColorType = None,
    bins_1d: str | int = 50,
    bins_2d: str | int = None,
    stat: str = None,
    figsize: tuple = (8, 7),
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the distributions and 2D correlation between two columns in adata.obs.

    Args:
        adata: an adata object
        xcolumn: columns in adata.obs to plot on the x-axis
        ycolumn: columns in adata.obs to plot on the y-axis
        log1p_xcolumn: normalize the xcolumn? (default: False)
        log1p_ycolumn: normalize the ycolumn? (default: False)
        color_xcolumn: color of the xcolumn plot
        color_ycolumn: color of the ycolumn plot
        cmap_2d: colormap of the 2d correlation plot (default: "Blues")
        bins_1d: number of bins on the 1-dimensional histogram plots
        bins_2d: number of bins on the 2-dimensional histogram plot
        stat: which statistic to plot, see sns.histplot for more details
         (default: "percent")
        figsize: figure size
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if cmap_2d is None:
        cmap_2d = "Blues"
    if bins_1d is None:
        bins_1d = "auto"
    if bins_2d is None:
        bins_2d = "auto"
    if stat is None:
        stat = "percent"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        gridspec_kw={
            "width_ratios": [1, 1, 0.05],
            "height_ratios": [1, 1],
            # "wspace": 0,
            # "hspace": 0,
        },
        constrained_layout=True,
        figsize=figsize,
        **subplot_kwargs,
    )

    y = adata.obs[ycolumn]
    ylabel = ycolumn
    if log1p_ycolumn:
        y = np.log1p(y)
        ylabel = f"log1p({ycolumn})"
    sns.histplot(
        y=y,
        bins=bins_1d,
        stat=stat,
        color=color_ycolumn,
        ax=axs[0, 0],
        alpha=0.5,
        **plot_kwargs,
    )
    axs[0, 0].invert_xaxis()
    axs[0, 0].yaxis.tick_right()
    axs[0, 0].xaxis.tick_top()
    axs[0, 0].xaxis.set_label_position("top")
    # axs[0, 0].yaxis.set_label_position("right")
    axs[0, 0].set_ylabel(ylabel)

    x = adata.obs[xcolumn]
    xlabel = xcolumn
    if log1p_xcolumn:
        x = np.log1p(x)
        xlabel = f"log1p({xcolumn})"
    sns.histplot(
        x=x,
        bins=bins_1d,
        stat=stat,
        color=color_xcolumn,
        ax=axs[1, 1],
        alpha=0.5,
        **plot_kwargs,
    )
    axs[1, 1].invert_yaxis()
    axs[1, 1].xaxis.tick_top()
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].yaxis.set_label_position("right")
    # axs[1, 1].set_xlabel(None)
    axs[1, 1].set_xlabel(xlabel)

    sns.histplot(
        x=x,
        y=y,
        bins=bins_2d,
        stat=stat,
        cmap=cmap_2d,
        cbar=False,
        ax=axs[0, 1],
        **plot_kwargs,
    )
    axs[0, 1].set_xlabel(None)
    axs[0, 1].set_ylabel(None)
    # axs[0, 1].set_xlabel(xlabel)
    # axs[0, 1].xaxis.set_label_position("top")
    # axs[0, 1].set_ylabel(ylabel)
    # axs[0, 1].yaxis.set_label_position("right")
    axs[0, 1].get_xaxis().set_ticklabels([])
    axs[0, 1].get_yaxis().set_ticklabels([])
    # axs[0, 1].get_xaxis().set_visible(False)
    # axs[0, 1].get_yaxis().set_visible(False)

    # colormap
    norm = Normalize(vmin=min(y.min(), x.min()), vmax=max(y.max(), x.max()))
    ColorbarBase(axs[0, 2], cmap=cmap_2d, norm=norm)
    cbar_label = "Number of cells"
    if log1p_xcolumn and log1p_ycolumn:
        cbar_label = "log1p(cells)"
    axs[0, 2].set_ylabel(cbar_label)

    # hide empty subfigs
    axs[1, 0].set_axis_off()
    axs[1, 2].set_axis_off()

    return fig, axs


def plot_avg_per_pixel(
    adata: ad.AnnData,
    column: str,
    fill_cell_area: bool = False,
    log1p: bool = False,
    cmap: ColorType = None,
    background_color: ColorType = None,
    figsize: tuple = (20, 15),
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the average values of the given column over all FOVs.
    Color's the cell's center pixel, unless fill_cell_area is set to True (slow).

    Args:
        adata: an adata object
        column: a column in adata.obs with numeric values
        fill_cell_area: distribute the column value over all pixels covered by the cell,
         assuming square cells (default: False)
        log1p: normalize the final values per pixel?
        cmap: colormap (default: "gist_rainbow")
        background_color: color for pixels with 0 values (default: "black")
        figsize: figure size
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if cmap is None:
        cmap = "gist_rainbow"
    if background_color is None:
        background_color = "black"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # create a 2D array with the average values
    x_max = adata.uns["fov_dims_px"]["x"]
    y_max = adata.uns["fov_dims_px"]["y"]
    grid = np.full((y_max, x_max), 0.0)
    if not fill_cell_area:
        # Group and average per coordinate
        df = (
            adata.obs[[column, "CenterX_local_px", "CenterY_local_px"]]
            .groupby(["CenterX_local_px", "CenterY_local_px"], observed=True)
            .mean()
            .reset_index()
        )
        grid[df["CenterY_local_px"], df["CenterX_local_px"]] = df[column]
    else:
        grid_n = grid.copy()
        for _, row in adata.obs.iterrows():
            # divide the column value over the area of the cell
            val = row[column] / (row["Width"] * row["Height"])  # row['Area']

            # add the normalized value to all pixels
            half_w = row["Width"] // 2
            half_h = row["Height"] // 2
            x_start = max(row["CenterX_local_px"] - half_w, 0)
            x_end = min(row["CenterX_local_px"] + half_w + 1, x_max)
            y_start = max(row["CenterY_local_px"] - half_h, 0)
            y_end = min(row["CenterY_local_px"] + half_h + 1, y_max)
            grid[y_start:y_end, x_start:x_end] += val

            # track the number of cells covering each pixel
            grid_n[y_start:y_end, x_start:x_end] += 1

        # average the value over the number of cells
        np.clip(grid_n, min=1, max=None, out=grid_n)
        grid = grid / grid_n
    if log1p:
        grid = np.log1p(grid)

    # assign the background_color
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color=background_color)
    masked_grid = np.ma.masked_where(grid == 0.0, grid)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        gridspec_kw={
            "width_ratios": [2, 1],
        },
        figsize=figsize,
        **subplot_kwargs,
    )
    n = len(adata.obs["slide-fov"].unique())
    axs[0].set_title(f"Average {column} per pixel over {n} FOVs")
    vmin = np.nanmin(grid[grid > 0])  # first real, nonzero value
    vmax = np.nanmax(grid)
    im = axs[0].imshow(
        masked_grid,
        cmap=cmap,
        interpolation="none",
        # aspect='equal',
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs,
    )
    axs[0].set_box_aspect(1)
    axs[0].xaxis.set_ticks(np.arange(0, x_max, (x_max // 2000) * 100))
    axs[0].yaxis.set_ticks(np.arange(0, y_max, (y_max // 2000) * 100))
    axs[0].xaxis.tick_top()
    axs[0].xaxis.set_tick_params(rotation=45)
    axs[0].xaxis.set_label_position("top")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    label = f"{'log1p(' if log1p else ''}mean({column}){')' if log1p else ''}/pixel"
    fig.colorbar(
        im,
        cmap=cmap,
        norm=norm,
        ax=axs[0],
        location="bottom",
        # orientation='horizontal',
        fraction=0.025,
        pad=0.005,
        label=label,
    )

    # Lineplot
    axs[1].set_title(f"Sum of values per row/column")
    x_sum = np.sum(grid, axis=0)
    y_sum = np.sum(grid, axis=1)
    max_sum = max(max(y_sum), max(x_sum))
    axs[1].set_xlim(-x_max * 0.02, x_max * 1.02)
    axs[1].set_ylim(-max_sum * 0.02, max_sum * 1.02)
    axs[1].plot(x_sum, label="x")
    axs[1].plot(y_sum, label="y")
    axs[1].set_box_aspect(1)
    axs[1].set_ylabel("sum(values)")
    axs[1].set_xlabel("axis coordinate")
    axs[1].legend()

    return fig, axs


def plot_violin(
    adata: ad.AnnData,
    columns: str | list,
    inner: str = None,
    fill: bool = False,
    cut: int = 0,
    log_scale: tuple[bool, bool] = (False, True),
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Violin plots for one or more columns in adata.obs.

    Wraps seaborn's violinplot.
    See https://seaborn.pydata.org/generated/seaborn.violinplot.html

    Args:
        adata: an adata object
        columns: one or more column in adata.obs
        inner: See sns.violinplot for more details.
        fill: See sns.violinplot for more details.
        cut: See sns.violinplot for more details.
        log_scale: See sns.violinplot for more details.
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if isinstance(columns, str):
        columns = [columns]
    if inner is None:
        inner = "quart"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    ncols = len(columns)
    fig, axs = plt.subplots(nrows=1, ncols=ncols, squeeze=False, **subplot_kwargs)
    color_cycle = itertools.cycle(plt.get_cmap("tab10").colors)  # noqa
    for i, column in enumerate(columns):
        color = next(color_cycle)
        sns.violinplot(
            y=adata.obs[column],
            color=color,
            inner=inner,
            fill=fill,
            cut=cut,
            log_scale=log_scale,
            ax=axs[0, i],
            **plot_kwargs,
        )
        # sns.stripplot(
        #     y=adata.obs[key],
        #     color=color,
        #     alpha=0.005,
        #     log_scale=log_scale,
        #     ax=axs[i],
        #     **plot_kwargs,
        # )
        axs[0, i].spines["top"].set_visible(False)
        axs[0, i].spines["right"].set_visible(False)
        axs[0, i].spines["bottom"].set_visible(False)
        axs[0, i].spines["left"].set_visible(True)
        axs[0, i].set_title(column)
    return fig, axs


def plot_ncell_per_condition(
    adata: ad.AnnData,
    columns: str | list,
    offset_between_conditions: int | list = 1,
    palette: ColorType = None,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the number of cells per condition in a column in adata.obs.

    Args:
        adata: an adata object
        columns: one or more columns in adata.obs to visualize, in order of significance
        offset_between_conditions: distance between different conditions
         Can be a single value, or a list of offset values for each column
         (length=len(columns)-1)
        palette: color palette (default: "terrain")
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.set_xticks and ax.set_yticks

    Returns:
        matplotlib figure and array of axes
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(offset_between_conditions, Iterable):
        if len(offset_between_conditions) != len(columns) - 1:
            raise IndexError(
                f"{offset_between_conditions=} must be a value, "
                "or a list of values of len(columns)-1"
            )
    if palette is None:
        palette = "terrain"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    # sure a separator that is unlikely to be used already
    endash = "–"
    ncell = (
        adata.obs[columns]
        .astype(str)
        .agg(endash.join, axis=1)
        .value_counts()
        .fillna(0)
        .sort_index(key=natsort_keygen())
    )
    x = []
    y = []
    labels = []
    i = 0
    last = None
    for idx, val in ncell.items():
        y.append(val)
        labels.append(idx)
        idx_sep = idx.split(endash)
        if not isinstance(offset_between_conditions, Iterable):
            if last is None:
                last = idx_sep[0]
            # add the offset if the first column changes
            if idx_sep[0] != last:
                i += offset_between_conditions
                last = idx_sep[0]
        else:
            if last is None:
                last = idx_sep
            # add the j-th offset if the j-th column changes
            for j, offset in enumerate(offset_between_conditions):
                if idx_sep[j] != last[j]:
                    i += offset
                    last = idx_sep
                    break
        x.append(i)
        i += 1

    fig, ax = plt.subplots(**subplot_kwargs)
    sns.barplot(
        x=x,
        y=y,
        edgecolor=".9",
        native_scale=True,  # use x & y values as-is
        palette=palette,
        hue=x,
        legend=False,
        ax=ax,
        **plot_kwargs,
    )
    ax.set_xticks(
        ticks=x,
        labels=labels,
        rotation=45,
        ha="right",
        **text_kwargs,
    )
    if text_kwargs:
        ax.set_yticks(
            ticks=ax.get_yticks(),
            labels=ax.get_yticklabels(),
            **text_kwargs,
        )
    ax.set_xlim(min(x) - 2, max(x) + 2)
    ax.set_title(f"Number of cells per {endash.join(columns)}", **text_kwargs)
    ax.set_xlabel(None)
    ax.set_ylabel("Number of cells", **text_kwargs)
    return fig, ax


def plot_value_distribution(
    adata: ad.AnnData,
    layer: str = None,
    min_quantile: float = 0.0,
    max_quantile: float = 0.95,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the number of occurrences of values in the dataset.

    Args:
        adata: an adata object.
        layer: the layer the values are drawn from (default: X)
        min_quantile: lowest quantile of values to plot (default: 0.00)
        max_quantile: highest quantile of values to plot (default: 0.95)
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if layer is None:
        array = adata.X
    else:
        array = adata.layers[layer]
    if isinstance(array, sp.csr_matrix):
        array = array.toarray()
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    values, counts = np.unique(array, return_counts=True, sorted=True)
    log_counts = np.log10(counts)
    x_min = np.quantile(values, min_quantile)
    x_max = np.quantile(values, max_quantile)
    x_pad = (x_max - x_min) * 0.02

    fig, ax = plt.subplots(**subplot_kwargs)
    x_prev = None
    for x, y in zip(values, log_counts):
        if x < x_min:
            continue
        if x > x_max:
            break
        if x_prev is None:
            width = values[1] - values[0]
        else:
            width = x - x_prev
        x_prev = x
        ax.bar(x=x, height=y, width=width, **plot_kwargs)
    ax.set_title("Distributions of raw values in the dataset")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_xlabel("value")
    ax.set_ylabel("log10(occurrences)")

    return fig, ax


def plot_column_distribution(
    adata: ad.AnnData,
    column: str,
    axis: int = None,
    min_quantile: float = 0.0,
    max_quantile: float = 0.95,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the distribution of values for a column present in either adata.obs or
    adata.var.

    Args:
        adata: an adata object.
        column: a column in either adata.obs or adata.var
        axis: specify if the column name is present in both obs (0) and var (1).
        min_quantile: lowest quantile of values to plot (default: 0.00)
        max_quantile: highest quantile of values to plot (default: 0.95)
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if axis is None:
        if column in adata.obs.columns:
            axis = 0
        elif column in adata.var.columns:
            axis = 1
        else:
            raise IndexError(f"{column=} not found in adata.obs or var")
    if axis == 0:
        series = adata.obs[column]
        unit = "cells"
    elif axis == 1:
        series = adata.var[column]
        unit = "genes"
    else:
        raise IndexError("Axis must be None, 0 or 1")
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    x_min = series.quantile(min_quantile)
    x_max = series.quantile(max_quantile)
    x_pad = (x_max - x_min) * 0.02

    fig, ax = plt.subplots(**subplot_kwargs)
    sns.histplot(series, ax=ax, **plot_kwargs)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_title(f"Distribution of {column} over all {unit}")

    return fig, ax
