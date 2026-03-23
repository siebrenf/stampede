import itertools
import math
import os
from collections.abc import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from natsort import natsort_keygen, natsorted


def slide_qc_data(adata: ad.anndata, slides: dict, data_dir: str = None):
    """
    Use the fov_positions file to create a dataframe with metadata columns per slide
    and fov, and store this in adata.uns["fov_metadata"].
    Additional adds columns to adata.obs reflecting the distance from the cell to the
    camera's FOV edge.

    Args:
        adata: the adata object generated using slides
        slides: a dict with files per slide
        data_dir: optional filepath prefix

    Returns:
        fov_df: a dataframe with metadata columns per slide and fov
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
    fov_df["nCounts"] = adata.obs.groupby("slide-fov")["nCount_RNA"].sum()
    fov_df = pd.merge(
        left=fov_df,
        right=adata.obs["slide-fov"].value_counts().rename("nCell"),
        on="slide-fov",
    )
    fov_df["meanCountsPerCell"] = fov_df["nCounts"] / fov_df["nCell"]
    fov_df["nCount_negprobes"] = adata.obs.groupby("slide-fov")[
        "nCount_negprobes"
    ].sum()
    fov_df["mean_NegProbe-CountsPerCell"] = fov_df["nCount_negprobes"] / fov_df["nCell"]
    fov_df["nCount_falsecode"] = adata.obs.groupby("slide-fov")[
        "nCount_falsecode"
    ].sum()
    fov_df["mean_FalseCode-CountsPerCell"] = (
        fov_df["nCount_falsecode"] / fov_df["nCell"]
    )
    fov_df["meanCellSize"] = (
        adata.obs.groupby("slide-fov")["Area.um2"].sum() / fov_df["nCell"]
    )
    slidefov2passfail = adata.obs.groupby("slide-fov")["qcFlagsFOV"].first().to_dict()
    fov_df = fov_df.reset_index()
    fov_df["Failed_AtoMX_QC"] = (
        fov_df["slide-fov"].replace(slidefov2passfail).replace({"Pass": 0, "Fail": 1})
    )
    adata.uns["fov_metadata"] = fov_df

    # determine the dimensions of the camera's FOV
    dim_x_px, dim_y_px = fov_dimensions(fov_df)
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


def fov_dimensions(fov_df):
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


def slide_qc_plots(adata, columns=None):
    """
    Plot the values from each QC column in fov_df on the slide layout.
    Manually add columns to the dataframe for additional plots.
    """
    fov_df = adata.uns["fov_metadata"]
    if columns is not None:
        fov_df = fov_df[columns]
    for col in ["slide", "x", "y"]:
        if col not in fov_df.columns:
            raise ValueError(f"column={col} is required")

    fig_axs_list = []
    # total number of slides
    ncols = len(fov_df["slide"].unique())
    for col in fov_df.columns:
        # don't plot the metadata columns
        if col in ["slide-fov", "slide", "fov", "x", "y"]:
            continue

        # normalize the colormap for all slides at once
        qc_param = col
        norm = Normalize(vmin=fov_df[qc_param].min(), vmax=fov_df[qc_param].max())

        fig, axs = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True)
        if not isinstance(axs, Iterable):
            axs = [axs]
        # fig.suptitle(qc_param, y=0.875)
        fig.supxlabel("x", y=0.075)
        for i, s in enumerate(natsorted(fov_df["slide"].unique())):
            axs[i].set_title(f"slide {s}")
            fov_df_slide = fov_df[fov_df["slide"] == s]
            sns.scatterplot(
                data=fov_df_slide,
                x="x",
                y="y",
                hue=qc_param,
                palette="coolwarm",
                hue_norm=norm,
                legend=False,
                ax=axs[i],
            )
            # hide all ticks and borders
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["left"].set_visible(False)
            axs[i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            axs[i].set_xlabel(None)
            if i == 0:
                # only plot the colorbar once
                sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(
                    sm,
                    location="top",
                    orientation="horizontal",
                    shrink=0.4,
                    pad=0.1,
                    aspect=30,
                    ax=axs,
                )
                # colorbar label doubles as the figure suptitle
                cbar.set_label(qc_param)
            else:
                # only plot the y label for the first plot
                axs[i].set_ylabel(None)
        fig_axs_list.append((fig, axs))

    # return the plot elements for manual post-processing
    return fig_axs_list


def plot_fov_edge_distances(
    adata,
    figsize=(12, 6),
    subplot_kwargs=None,
    # plot_kwargs=None,
):
    bins = 50
    cmap = "Blues"
    color1 = "dodgerblue"
    color2 = "limegreen"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    # if plot_kwargs is None:
    #     plot_kwargs = {}

    fig, axs = plt.subplots(
        ncols=3,
        gridspec_kw={"width_ratios": [1, 1, 0.05]},
        constrained_layout=True,
        figsize=figsize,
        **subplot_kwargs,
    )
    x, y = adata.uns["fov_dims_px"].values()
    fig.suptitle(
        "Distributions of cell distances to the edge of the camera's FOV "
        + f"({x}x{y} pixels)"
    )

    # 1D histplot
    sns.histplot(
        np.log1p(adata.obs["dist2edge_px"]),
        bins=bins,
        stat="percent",
        color=color1,
        ax=axs[0],
        alpha=0.5,
    )
    axs[0].set_title("Minimum distance to the FOV edge")
    axs[0].set_xlabel("log1p(pixels)", color=color1)
    axs[0].set_ylabel("% of cells")
    axs01 = axs[0].twiny()
    sns.histplot(
        adata.obs["dist2edge_px"],
        bins=bins,
        stat="percent",
        color=color2,
        ax=axs01,
        alpha=0.5,
    )
    axs01.set_xlabel("pixels", color=color2)
    axs[0].set_zorder(1)
    axs01.set_zorder(0)
    axs[0].patch.set_alpha(0)  # background transparency

    # 2D histplot
    sns.histplot(
        x=adata.obs["x2edge_px"],
        y=adata.obs["y2edge_px"],
        cmap=cmap,
        cbar=False,
        ax=axs[1],
    )
    axs[1].set_title("2D distance to FOV edge")
    axs[1].set_xlabel("Pixels along the x-axis")
    axs[1].set_ylabel("Pixels along the y-axis")

    # colormap
    norm = Normalize(
        vmin=0,
        vmax=max(adata.obs["x2edge_px"].max(), adata.obs["y2edge_px"].max()),
    )
    ColorbarBase(axs[2], cmap=cmap, norm=norm)
    axs[2].set_ylabel("Number of cells")

    return fig, axs


def plot_correlations(
        adata,
        xcolumn,
        ycolumn,
        log1p_xcolumn=False,
        log1p_ycolumn=False,
        color_xcolumn=None,
        color_ycolumn=None,
        cmap_2d=None,
        bins_1d=50,
        bins_2d=None,
        stat=None,
        figsize=(8, 7),
        subplot_kwargs=None,
        plot_kwargs=None,
):
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
            'width_ratios': [1, 1, 0.05],
            'height_ratios': [1, 1],
            #             'wspace': 0,
            #             'hspace': 0,
        },
        constrained_layout=True,
        figsize=figsize,
        **subplot_kwargs,
    )
    # fig.suptitle(f"Distributions of cell distances to the edge of the camera's FOV ({x_px}x{y_px} pixels)")

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
    )
    axs[1, 1].invert_yaxis()
    axs[1, 1].xaxis.tick_top()
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].yaxis.set_label_position("right")
    #     axs[1, 1].set_xlabel(None)
    axs[1, 1].set_xlabel(xlabel)

    sns.histplot(
        x=x,
        y=y,
        bins=bins_2d,
        stat=stat,
        cmap=cmap_2d,
        cbar=False,
        ax=axs[0, 1],
    )
    axs[0, 1].set_xlabel(None)
    axs[0, 1].set_ylabel(None)
    #     axs[0, 1].set_xlabel(xlabel)
    #     axs[0, 1].xaxis.set_label_position("top")
    #     axs[0, 1].set_ylabel(ylabel)
    #     axs[0, 1].yaxis.set_label_position("right")
    axs[0, 1].get_xaxis().set_ticklabels([])
    axs[0, 1].get_yaxis().set_ticklabels([])
    #     axs[0, 1].get_xaxis().set_visible(False)
    #     axs[0, 1].get_yaxis().set_visible(False)

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
        adata,
        column,
        fill_cell_area=False,
        log1p=False,
        cmap=None,
        background_color=None,
        figsize=(20, 15),
        subplot_kwargs=None,
        plot_kwargs=None,
):
    """
    Plot the average values of the given column over all FOVs.
    Color's the cell's center pixel, unless fill_cell_area is set to True (slow).

    Args:
        adata: an adata object
        column: a numeric column in adata.obs
        fill_cell_area: distribute the column value over all pixels covered by the cell, assuming square cells (default: False)
        log1p: normalize the final values per pixel using np.log1p
        cmap: colormap (default: "gist_rainbow")
        background_color: color for pixels with 0 values (default: "black")
        figsize: figure size
        subplot_kwargs: kwargs passed to plt.subplot
        plot_kwargs: kwargs passed to the main plotting function
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
    x_max = adata.uns['fov_dims_px']["x"]
    y_max = adata.uns['fov_dims_px']["y"]
    grid = np.full((y_max, x_max), 0.0)
    if fill_cell_area is False:
        # Group and average per coordinate
        df = (
            adata.obs[[column, 'CenterX_local_px', 'CenterY_local_px']]
            .groupby(['CenterX_local_px', 'CenterY_local_px'])
            .mean()
            .reset_index()
        )
        grid[df['CenterY_local_px'], df['CenterX_local_px']] = df[column]
    else:
        grid_n = grid.copy()
        for _, row in adata.obs.iterrows():
            # divide the column value over the area of the cell
            val = row[column] / (row['Width'] * row['Height'])  # row['Area']

            # add the normalized value to all pixels
            half_w = row['Width'] // 2
            half_h = row['Height'] // 2
            x_start = max(row['CenterX_local_px'] - half_w, 0)
            x_end = min(row['CenterX_local_px'] + half_w + 1, x_max)
            y_start = max(row['CenterY_local_px'] - half_h, 0)
            y_end = min(row['CenterY_local_px'] + half_h + 1, y_max)
            grid[y_start:y_end, x_start:x_end] += val

            # track the number of cells covering each pixel
            grid_n[y_start:y_end, x_start:x_end] += 1

        # average the value over the number of cells
        np.clip(grid_n, min=1, out=grid_n)
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
            'width_ratios': [2, 1],
        },
        figsize=figsize,
        **subplot_kwargs,
    )
    n = len(adata.obs["slide-fov"].unique())
    axs[0].set_title(f'Average {column} per pixel over {n} FOVs')
    vmin = np.nanmin(grid[grid > 0])  # first real, nonzero value
    vmax = np.nanmax(grid)
    im = axs[0].imshow(
        masked_grid,
        cmap=cmap,
        interpolation='none',
        # aspect='equal',
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs
    )
    axs[0].set_box_aspect(1)
    axs[0].xaxis.set_ticks(np.arange(0, x_max, (x_max // 2000) * 100))
    axs[0].yaxis.set_ticks(np.arange(0, y_max, (y_max // 2000) * 100))
    axs[0].xaxis.tick_top()
    axs[0].xaxis.set_tick_params(rotation=45)
    axs[0].xaxis.set_label_position("top")
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

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
    axs[1].set_title(f'Sum of values per row/column')
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


def violin(
    adata: ad.anndata,
    keys: str | list,
    inner=None,
    fill=False,
    cut=0,
    log_scale=(False, True),
    subplot_kwargs=None,
    plot_kwargs=None,
):
    if isinstance(keys, str):
        keys = [keys]
    if inner is None:
        inner = "quart"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    ncols = len(keys)
    fig, axs = plt.subplots(nrows=1, ncols=ncols, **subplot_kwargs)
    color_cycle = itertools.cycle(plt.get_cmap("tab10").colors)
    if not isinstance(axs, Iterable):
        axs = [axs]
    for i, key in enumerate(keys):
        color = next(color_cycle)
        sns.violinplot(
            y=adata.obs[key],
            color=color,
            inner=inner,
            fill=fill,
            cut=cut,
            log_scale=log_scale,
            ax=axs[i],
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
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["left"].set_visible(True)
        axs[i].set_title(key)
    return fig, axs


def gene_qc(adata):
    """
    Add QC parameters to adata.var
    """
    if "is_negctrl" not in adata.var:
        adata.var["is_negctrl"] = adata.var_names.str.startswith("Negative")
    if "is_sysctrl" not in adata.var:
        adata.var["is_sysctrl"] = adata.var_names.str.startswith("System")
    if "nCell" not in adata.var:
        # number of nonzero cells per gene
        adata.var["nCell"] = (adata.X > 0).sum(axis=0).A1
        adata.var["pctCell"] = 100 * adata.var["nCell"] / adata.n_obs
    if "nTranscript" not in adata.var:
        adata.var["nTranscript"] = np.array(adata.X.sum(axis=0)).ravel()
    if "meanTranscript" not in adata.var:
        adata.var["meanTranscript"] = adata.var["nTranscript"] / adata.n_obs


def plot_ncell_per_condition(
    adata,
    columns: str | list,
    offset_between_conditions: int | list = 1,
    palette=None,
    subplot_kwargs=None,
    barplot_kwargs=None,
    text_kwargs=None,
):
    """
    Plot the number of cells per condition.

    Args:
        adata: an adata object
        columns: one or more columns in adata.obs to visualize, in order of significance.
        offset_between_conditions: distance between different conditions. Can be a single value, or a list of offset values for each column (length=len(columns)-1)
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(offset_between_conditions, Iterable):
        if len(offset_between_conditions) != len(columns) - 1:
            raise IndexError(
                f"{offset_between_conditions=} must be a value, or a list of values of len(columns)-1"
            )
    if palette is None:
        palette = "terrain"
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if barplot_kwargs is None:
        barplot_kwargs = {}
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
        **barplot_kwargs,
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
    adata,
    max_1s: int = 10,
    subplot_kwargs=None,
    barplot_kwargs=None,
):
    """
    Plot the number of occurences of each value in the dataset.

    Args:
        adata: an adata object.
        max_1s: the maximum number of consecutive values with 10 or fewer occurences.
    """
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if barplot_kwargs is None:
        barplot_kwargs = {}

    values, counts = np.unique(adata.X.toarray(), return_counts=True)
    fig, ax = plt.subplots(**subplot_kwargs)
    n = 0
    x_max = max(values)
    for x, y in zip(values, np.log10(counts)):
        ax.bar(x=x, height=y, **barplot_kwargs)
        # stop plotting if the y-value stays at or below 1
        if math.ceil(y) <= 1:
            if n == max_1s:
                x_max = x
                break
            n += 1
        else:
            n = 0
    ax.set_title("Distributions of raw values in the dataset")
    ax.set_xlim(-2, x_max + 2)
    ax.set_xlabel("value")
    ax.set_ylabel("log10(occurences)")
    return fig, ax


def plot_distribution(
    adata,
    column,
    axis=0,
    min_quantile=0,
    max_quantile=0.95,
    xpad=1,
    subplot_kwargs=None,
    plot_kwargs=None,
):
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if axis == 0:
        series = adata.obs[column]
        axis = "cell"
    elif axis == 1:
        series = adata.var[column]
        axis = "gene"
    else:
        raise IndexError("Axis must be 0 or 1")
    fig, ax = plt.subplots(**subplot_kwargs)
    sns.histplot(series, ax=ax, **plot_kwargs)
    ax.set_xlim(
        series.quantile(min_quantile) - xpad,
        series.quantile(max_quantile) + xpad,
    )
    ax.set_title(f"Distribution of {column} per {axis}")
    return fig, ax
