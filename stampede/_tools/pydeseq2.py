from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.inference import Inference


def pydeseq2(
    adata: ad.AnnData,
    design: str,
    contrast: list,
    inference: Inference = None,
    n_cpus: int = 16,
    return_objects: bool = False,
    dds_kwargs: dict = None,
    ds_kwargs: dict = None,
) -> tuple[DeseqDataSet, DeseqStats, pd.DataFrame] | pd.DataFrame:
    """
    Wrapper around pyDEseq2 for adata objects.

    See https://pydeseq2.readthedocs.io/en/latest/auto_examples/plot_minimal_pydeseq2_pipeline.html

    Args:
        adata: adata object
        design: a formula in the format 'x + z' or '~x+z'.
         Each factor must be a column in adata.obs
        contrast:  a list of three strings in the following format:
         ['variable_of_interest', 'tested_level', 'ref_level']
        inference: pyDESeq2 inference class instance
        n_cpus: number of threads to use
        return_objects: return the DeseqDataSet, DeseqStats and the results_df.
         If False, only return the results_df
        dds_kwargs: kwargs passed to DeseqDataSet
        ds_kwargs: kwargs passed to DeseqStats

    Returns:
        pydeseq2 output
    """
    # optional dependency
    from pydeseq2.dds import DeseqDataSet  # noqa
    from pydeseq2.default_inference import DefaultInference  # noqa
    from pydeseq2.ds import DeseqStats  # noqa

    if inference is None:
        inference = DefaultInference(n_cpus=n_cpus)
    if dds_kwargs is None:
        dds_kwargs = {}
    if ds_kwargs is None:
        ds_kwargs = {}
    dds = DeseqDataSet(
        adata=adata,
        design=design,
        inference=inference,
        **dds_kwargs,
    )
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=contrast,
        n_cpus=n_cpus,
        **ds_kwargs,
    )
    ds.summary()

    if return_objects:
        return dds, ds, ds.results_df
    else:
        return ds.results_df


def plot_pydeseq2_volcano(
    df: pd.DataFrame,
    symbol_column: str = "index",
    log2fc_column: str = "log2FoldChange",
    pvalue_column: str = "padj",
    baseMean_column: str = "baseMean",
    pval_thresh: float = 0.05,
    log2fc_thresh: float = 0.75,
    to_label: int | list = 5,
    colors: list = None,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[Figure, Axes]:
    """
    Generate a volcano plot from a pyDESeq2 results dataframe.

    Adapted from https://github.com/mousepixels/sanbomics/blob/master/sanbomics/plots.py

    Args:
        df: a pyDESeq2 results dataframe
        symbol_column: column name of gene IDs to use
        log2fc_column: column name of log2 Fold-Change values
        pvalue_column: column name of the adjusted p values to be converted to -log10 p-values
        baseMean_column: column name of base mean values for each gene
        pval_thresh: threshold pvalue_column for points to be significant
        log2fc_thresh: threshold for the absolute value of the log2 fold change to be
         considered significant
        to_label: If an int is passed, that number of top down and up genes will be labeled.
            If a list of gene Ids is passed, only those will be labeled
        colors: order and colors to use
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.text

    Returns:
        matplotlib figure and axis object
    """
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    df = df.copy().reset_index(drop=False).dropna()
    if df[pvalue_column].min() == 0:
        df[pvalue_column][df[pvalue_column] == 0] = 1e-9

    pval_thresh = -np.log10(pval_thresh)  # convert p value threshold to nlog10
    df["nlog10"] = -np.log10(df[pvalue_column])  # make nlog10 column
    df["sorter"] = df["nlog10"] * df[log2fc_column]  # make a column to pick top genes
    df["logBaseMean"] = np.log1p(df[baseMean_column])  # size the dots by basemean

    # make label list of top x genes up and down, or based on list input
    if isinstance(to_label, int):
        label_df = pd.concat(
            (df.sort_values("sorter")[-to_label:], df.sort_values("sorter")[0:to_label])
        )
    else:
        label_df = df[df[symbol_column].isin(to_label)]

    # color light grey if below thresh, color picked black
    def map_color(a):
        zymbol, log2FoldChange, nlog10 = a
        if zymbol in label_df[symbol_column].tolist():
            return "picked"

        if abs(log2FoldChange) < log2fc_thresh or nlog10 < pval_thresh:
            return "not DE"
        return "DE"

    df["color"] = df[[symbol_column, log2fc_column, "nlog10"]].apply(map_color, axis=1)
    hues = ["DE", "not DE", "picked"][: len(df.color.unique())]  # order of colors
    if colors is None:
        colors = [
            "dimgrey",
            "lightgrey",
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]
    colors = colors[: len(df.color.unique())]

    fig, ax = plt.subplots(**subplot_kwargs)
    ax = sns.scatterplot(
        data=df,
        x=log2fc_column,
        y="nlog10",
        hue="color",
        hue_order=hues,
        palette=colors,
        size="logBaseMean",
        **plot_kwargs,
    )

    # make labels
    texts = []
    for i in range(len(label_df)):
        txt = ax.text(
            x=label_df.iloc[i][log2fc_column],
            y=label_df.iloc[i].nlog10,
            s=label_df.iloc[i][symbol_column],
            **text_kwargs,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
        texts.append(txt)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", zorder=5))

    # plot the p-value threshold (horizontally) and log2FC thresholds (vertically)
    ax.axhline(pval_thresh, zorder=0, c="k", lw=2, ls="--")
    ax.axvline(log2fc_thresh, zorder=0, c="k", lw=2, ls="--")
    ax.axvline(log2fc_thresh * -1, zorder=0, c="k", lw=2, ls="--")

    ax.set_xlabel("$log_{2}$ fold change")
    ax.set_ylabel(f"-$log_{{10}}$ adjusted p-value")
    ax.legend()

    return fig, ax
