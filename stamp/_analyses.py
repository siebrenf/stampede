from __future__ import annotations

import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from adjustText import adjust_text
from matplotlib import patheffects
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.inference import Inference
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


def pydeseq2(
    adata: ad.anndata,
    design: str,
    contrast: list,
    inference: Inference = None,
    n_cpus: int = 16,
    return_objects: bool = False,
    dds_kwargs: dict = None,
    ds_kwargs: dict = None,
):
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
        (pydeseq2 DeseqDataSet, DeseqStats and) pd.DataFrame
    """
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


def plot_volcano(
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
):
    """
    Generate a volcano plot from a pyDESeq2 results dataframe.

    Adapted from https://github.com/mousepixels/sanbomics/blob/master/sanbomics/plots.py

    Args:
        df: a pyDESeq2 results dataframe
        symbol_column: column name of gene IDs to use
        log2fc_column: column name of log2 Fold-Change values
        pvalue_column: column name of the adjusted p values to be converted to -log10 P values
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
        fig, ax: matplotlib figure and axis object
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


def paired_binomial_glm(
    df,
    condition_of_interest=None,
    reference_condition=None,
    gene_col="Gene",
    donor_col="Donor",
    condition_col="Condition",
    detection_col="Detection_rate",
    total_col="N_cells",
):
    """
    Runs paired donor-level binomial GLM:
        prop_detected ~ condition + donor

    Parameters
    ----------
    df : pd.DataFrame
        Must contain exactly two conditions.

    condition_of_interest : str
        The condition to compare (e.g., "treated").

    reference_condition : str
        The baseline condition (e.g., "control").

    Returns
    -------
    results_df : pd.DataFrame
        Per-gene results including:
        beta, odds_ratio, pval, padj
    """

    if condition_of_interest is None or reference_condition is None:
        raise ValueError("Must specify condition_of_interest and reference_condition.")
    unique_conditions = df[condition_col].unique()
    if len(unique_conditions) != 2:
        raise ValueError("Data must contain exactly 2 conditions.")
    if reference_condition not in unique_conditions:
        raise ValueError("reference_condition not found in dataframe.")
    if condition_of_interest not in unique_conditions:
        raise ValueError("condition_of_interest not found in dataframe.")

    # Relevel condition so reference is baseline
    df[condition_col] = pd.Categorical(
        df[condition_col],
        categories=[reference_condition, condition_of_interest],
        ordered=True,
    )
    # Ensure donor categorical
    df[donor_col] = df[donor_col].astype(str).astype("category")

    def fit_one_gene(gene_df):

        perfect_sep = False

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", PerfectSeparationWarning)

                model = smf.glm(
                    # formula=f"{detection_col} ~ {condition_col} + {donor_col}", #### << Donor and condition are perfectly colinear, doesn't apply in this data..
                    formula=f"{detection_col} ~ {condition_col}",
                    data=gene_df,
                    family=sm.families.Binomial(),
                    var_weights=gene_df[total_col],
                )
                result = model.fit()

                for warn in w:
                    if issubclass(warn.category, PerfectSeparationWarning):
                        perfect_sep = True
                        break

            # Coefficient name will be condition_col[T.condition_of_interest]
            coef_name = f"{condition_col}[T.{condition_of_interest}]"

            beta = result.params.get(coef_name, np.nan)
            se = result.bse.get(coef_name, np.nan)
            pval = result.pvalues.get(coef_name, np.nan)
            odds_ratio = np.exp(beta) if pd.notnull(beta) else np.nan

            return pd.Series(
                {
                    "beta": beta,
                    "se": se,
                    "odds_ratio": odds_ratio,
                    "pval": pval,
                    "Perfect_separation": perfect_sep,
                }
            )

        except Exception as exc:
            return pd.Series(
                {
                    "beta": np.nan,
                    "se": np.nan,
                    "odds_ratio": np.nan,
                    "pval": np.nan,
                    "Perfect_separation": np.nan,
                    "Error": str(exc),
                }
            )

    # Run across genes
    results = (
        df.groupby(gene_col, group_keys=False, observed=False)
        .apply(fit_one_gene, include_groups=False)
        .reset_index()
    )

    n_perfect_sep = results["Perfect_separation"].sum()

    # Drop failed fits
    results = results.dropna(subset=["pval"])

    if len(results) > 0:
        results["padj"] = multipletests(results["pval"], method="fdr_bh")[1]
        results["-log10(padj)"] = -np.log10(results["padj"])
        results["log2(Odds_ratio)"] = np.log2(results["odds_ratio"])
    else:
        return None

    # Sort by adjusted p-value
    results = results.sort_values("odds_ratio")

    if n_perfect_sep > 0:
        warnings.warn(
            f"Perfect separation detected in {int(n_perfect_sep)} genes. "
            "Parameter estimates may be unstable for these genes. "
            "Check the 'Perfect_separation' column in the results.",
            RuntimeWarning,
        )

    return results


def plot_volcano2(
    name,
    dedf,
    savefig=False,
    drop_perfect_separation=True,
    p_cutoff=0.05,
    or_cutoff=1.5,
):
    rng = np.random.default_rng(42)

    # p_cutoff = 0.05 #Before -log10!
    or_cutoff = np.log2(or_cutoff)  # After log2!

    n_txt = 30  # Number of genes names to plot
    n_hior = 30  # Number of genes with highest ORs to add to the names to be plotted if not already included in the most significant ones.

    upregs = []
    downregs = []

    dir2color = {"up": "indianred", "down": "cornflowerblue", "no change": "gainsboro"}

    if drop_perfect_separation:
        dedf = dedf.loc[~dedf["Perfect_separation"]]
    else:
        max_or = dedf.loc[~dedf["Perfect_separation"], "odds_ratio"].max()
        min_or = dedf.loc[~dedf["Perfect_separation"], "odds_ratio"].min()
        dedf.loc[
            dedf["Perfect_separation"] & (dedf["odds_ratio"] > 1), "odds_ratio"
        ] = (1.1 * max_or)
        dedf.loc[
            dedf["Perfect_separation"] & (dedf["odds_ratio"] < 1), "odds_ratio"
        ] = (0.9 * min_or)

    dedf = dedf.sort_values(by="padj", ascending=True)
    dedf["padj"] = np.clip(dedf["padj"], 1e-300, None)
    dedf["-log(Padj)"] = -np.log10(dedf["padj"])
    dedf["log2(odds_ratio)"] = np.log2(dedf["odds_ratio"])

    up = list(
        dedf["Gene"].loc[
            (dedf["padj"] <= p_cutoff) & (dedf["log2(odds_ratio)"] >= or_cutoff)
        ]
    )
    down = list(
        dedf["Gene"].loc[
            (dedf["padj"] <= p_cutoff) & (dedf["log2(odds_ratio)"] <= -or_cutoff)
        ]
    )
    not_sg = [gene for gene in dedf["Gene"] if not ((gene in up) or (gene in down))]
    upregs.append(up)
    downregs.append(down)

    if len(up) > n_txt:
        top_up = up[:n_txt]
    else:
        top_up = up
    top_up += [
        gene
        for gene in list(
            dedf.loc[
                (dedf["padj"] <= p_cutoff) & (dedf["log2(odds_ratio)"] >= or_cutoff)
            ]
            .sort_values(by="log2(odds_ratio)", ascending=False)["Gene"]
            .head(n_hior)
        )
        if gene not in top_up
    ]

    if len(down) > n_txt:
        top_down = down[:n_txt]
    else:
        top_down = down
    top_down += [
        gene
        for gene in list(
            dedf.loc[
                (dedf["padj"] <= p_cutoff) & (dedf["log2(odds_ratio)"] <= or_cutoff)
            ]
            .sort_values(by="log2(odds_ratio)", ascending=True)["Gene"]
            .head(n_hior)
        )
        if gene not in top_down
    ]

    symbol2dir = {}
    for gene in up:
        symbol2dir[gene] = "up"
    for gene in down:
        symbol2dir[gene] = "down"
    for gene in not_sg:
        symbol2dir[gene] = "no change"
    dedf["Change"] = dedf["Gene"].map(symbol2dir)
    with plt.rc_context(
        {
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
        }
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.scatterplot(
            dedf,
            x="log2(odds_ratio)",
            y="-log(Padj)",
            hue="Change",
            palette=dir2color,
            s=15,
            ax=ax,
        )
        ax.axhline(-np.log10(p_cutoff), color="black", linestyle="--", linewidth=0.5)
        ax.axvline(or_cutoff, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(-or_cutoff, color="black", linestyle="--", linewidth=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_legend().remove()

        ax.set_ylim(0, ax.get_ylim()[1])
        xlim_max = max(abs(lim) for lim in ax.get_xlim())
        ax.set_xlim(-xlim_max, xlim_max)
        ax.set_title(name)

        texts = []
        for gene in top_up + top_down:
            x, y = dedf.loc[dedf["Gene"] == gene][
                ["log2(odds_ratio)", "-log(Padj)"]
            ].iloc[0, 0:2]
            x += rng.uniform(-0.01, 0.01)
            y += rng.uniform(-0.01, 0.01)
            texts.append(ax.text(x=x, y=y, s=gene, fontdict={"fontsize": 8}))
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="gray"),
            force_explode=(0.3, 0.8),
            expand=[1.8, 1.2],
            max_move=(3600, 3600),
        )  # ,only_move={"text":"y+x+-"})

        plt.show()

    if savefig:
        fig.savefig(
            f"pseudobulk/results/binomial_glm/figures/{name}.pdf", bbox_inches="tight"
        )

    return None
