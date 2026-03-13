import itertools
import math
import os
from collections.abc import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from natsort import natsort_keygen, natsorted


def slide_qc_data(adata: ad.anndata, slides: dict, data_dir: str = None):
    """
    Create a dataframe with metadata columns per slide and fov

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

    return fov_df


def slide_qc_plots(fov_df):
    """
    Plot the values from each QC column in fov_df on the slide layout.
    Manually add columns to the dataframe for additional plots.
    """
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
