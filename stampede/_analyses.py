from __future__ import annotations

import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from matplotlib import patheffects


def pydeseq2(
    adata: ad.anndata,
    design: str,
    contrast: list,
    inference: "Inference" = None,  # noqa
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
):
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


def sketch(
    adata: ad.anndata,
    n: int = None,
    frac: float = 0.05,
    use_rep: str = "X_svd",
    obs_column: str = "subset",
    return_subset: bool = False,
):
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
         The subset dataframe (if specified)
    """
    # optional dependency
    from geosketch import gs  # noqa

    if n is None:
        n = round(len(adata) * frac)
    sketch_index = gs(adata.obsm[use_rep], n, replace=False)
    adata.obs[obs_column] = adata.obs.index.isin(adata.obs.iloc[sketch_index].index)

    if return_subset:
        return adata[adata.obs[obs_column], :].copy()


def plot_sketch(
    adata,
    obs_column: str = "subset",
    use_rep="X_svd",
    plot_kwargs=None,
):
    """
    Scatterplot highlighting the cells that were sampled.
    Requires the full adata object.
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


def paired_binomial_glm(
    df,
    test_condition: str,
    reference_condition: str,
    condition_column: str = "condition",
    gene_column: str = None,
    detection_column: str = "detection_rate",
    covariate_columns: str = None,
    total_column: str = None,
):
    """
    Runs paired donor-level binomial GLM:
        gene_detection_rate ~ condition + covariate(s)

    Args:
        df: pandas DataFrame
        test_condition: the condition to compare (e.g., "treated")
        reference_condition: the baseline condition (e.g., "control")
        condition_column: column with the condition
        gene_column: column with gene names (e.g. "gene"). If None, use the index.
        detection_column: column with the gene detection rates
        covariate_columns: column(s) with covariates (e.g. "donor")
        total_column: column with total cells per sample (e.g. "ncells").
         Used to give weight to each gene. Unused if None.

    Returns:
        pd.DataFrame: per-gene results including: beta, odds_ratio, pval, padj
    """
    # optional dependency
    import statsmodels.api as sm  # noqa
    import statsmodels.formula.api as smf  # noqa
    from statsmodels.stats.multitest import multipletests  # noqa
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning  # noqa

    if covariate_columns is None:
        covariate_columns = []
    elif isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]

    unique_conditions = df[condition_column].unique()
    if reference_condition not in unique_conditions:
        raise ValueError(f"{reference_condition=} not found in dataframe")
    if test_condition not in unique_conditions:
        raise ValueError(f"{test_condition=} not found in dataframe")

    df = df.copy()

    if gene_column is None:
        if df.index.name:
            gene_column = df.index.name
        else:
            gene_column = "index"
        df.reset_index(inplace=True)

    # re-level the condition column so the reference condition is the baseline
    df = df[df[condition_column].isin(unique_conditions)]
    df[condition_column] = pd.Categorical(
        df[condition_column],
        categories=[reference_condition, test_condition],
        ordered=True,
    )

    # ensure all batch and covariate columns are categorical
    for col in covariate_columns:
        df[col] = df[col].astype(str).astype("category")

    design_formula = f"{detection_column} ~ " + " + ".join(
        [condition_column] + covariate_columns
    )

    def fit_one_gene(gene_df):
        perfect_sep = False
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", PerfectSeparationWarning)

                weights = gene_df[total_column] if total_column else None
                model = smf.glm(
                    formula=design_formula,
                    data=gene_df,
                    family=sm.families.Binomial(),
                    var_weights=weights,
                )
                result = model.fit()

                for warn in w:
                    if issubclass(warn.category, PerfectSeparationWarning):
                        perfect_sep = True
                        break

            # Coefficient name will be condition_column[T.condition_of_interest]
            coef_name = f"{condition_column}[T.{test_condition}]"

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
                    "perfect_separation": perfect_sep,
                    "error": None,
                }
            )

        except Exception as exc:
            print(exc)  # TODO: replace exception and remove print
            return pd.Series(
                {
                    "beta": np.nan,
                    "se": np.nan,
                    "odds_ratio": np.nan,
                    "pval": np.nan,
                    "perfect_separation": np.nan,
                    "error": str(exc),
                }
            )

    # run across genes
    results = (
        df.groupby(gene_column, group_keys=False, observed=False)
        .apply(fit_one_gene, include_groups=False)
        .reset_index()
    )

    # drop failed fits
    results = results.dropna(subset=["pval"])
    if len(results) == 0:
        return None

    results["padj"] = multipletests(results["pval"], method="fdr_bh")[1]
    results["-log10(padj)"] = -np.log10(np.clip(results["padj"], 1e-9))
    results["log2(odds_ratio)"] = np.log2(results["odds_ratio"])
    results.sort_values("odds_ratio", inplace=True)

    n_perfect_sep = results["perfect_separation"].sum()
    if n_perfect_sep > 0:
        warnings.warn(
            f"Perfect separation detected in {int(n_perfect_sep)} genes. "
            "Parameter estimates may be unstable for these genes. "
            "Check the 'perfect_separation' column in the results.",
            RuntimeWarning,
        )

    return results


def plot_paired_binomial_glm_volcano(
    df: pd.DataFrame,
    gene_column: str = None,
    or_column: str = "odds_ratio",
    pvalue_column: str = "padj",
    separation_column: str = "perfect_separation",
    drop_perfect_separation: bool = True,
    pval_thresh: float = 0.05,
    or_thresh: float = 0.75,
    to_label: int | list = 5,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
):
    """
    Generate a volcano plot from the detection_rates results dataframe.

    Args:
        df: a pandas dataframe
        gene_column: column with gene names (e.g. "gene"). If None, use the index.
        or_column: column with odds ratios
        pvalue_column: column name of the adjusted p values to be converted to -log10 p-values
        separation_column: column with perfect separation data
        drop_perfect_separation: whether to drop the genes with perfect separations
        pval_thresh: threshold pvalue_column for points to be significant
        or_thresh: threshold for the log2 odds ratios to be considered significant
        to_label: the number of top down and up genes to be labeled
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.text

    Returns:
        fig, ax: matplotlib figure and axis object
    """
    or_thresh = abs(or_thresh)
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    df = df.copy().dropna(subset=[pvalue_column, or_column, separation_column])

    if df[pvalue_column].min() == 0:
        df[pvalue_column][df[pvalue_column] == 0] = 1e-9

    if gene_column is None:
        if df.index.name:
            gene_column = df.index.name
        else:
            gene_column = "index"
        df.reset_index(inplace=True)

    # rng = np.random.default_rng(42)
    #
    # # p_cutoff = 0.05 #Before -log10!
    # or_cutoff = np.log2(or_cutoff)  # After log2!

    # n_txt = 30  # Number of genes names to plot
    # n_hior = 30  # Number of genes with highest ORs to add to the names to be plotted if not already included in the most significant ones.
    #
    # upregs = []
    # downregs = []

    if drop_perfect_separation:
        df = df.loc[~df[separation_column]]
    # else:
    #     nps_or = df.loc[~df[separation_column], or_column]
    #     df.loc[df[separation_column] & (df[or_column] > 1), or_column] = (
    #         nps_or.max() * 1.1
    #     )
    #     df.loc[df[separation_column] & (df[or_column] < 1), or_column] = (
    #         nps_or.min() * 0.9
    #     )

    df["-log10(padj)"] = -np.log10(df[pvalue_column])
    df["log2(odds_ratio)"] = np.log2(df[or_column])
    # if len(up) > n_txt:
    #     top_up = up[:n_txt]
    # else:
    #     top_up = up
    # top_up += [
    #     gene
    #     for gene in list(
    #         df.loc[
    #             (df[pvalue_column] <= p_cutoff) & (df["log2(odds_ratio)"] >= or_cutoff)
    #         ]
    #         .sort_values(by="log2(odds_ratio)", ascending=False)["Gene"]
    #         .head(n_hior)
    #     )
    #     if gene not in top_up
    # ]

    # if len(down) > n_txt:
    #     top_down = down[:n_txt]
    # else:
    #     top_down = down
    # top_down += [
    #     gene
    #     for gene in list(
    #         df.loc[
    #             (df[pvalue_column] <= p_cutoff) & (df["log2(odds_ratio)"] <= or_cutoff)
    #         ]
    #         .sort_values(by="log2(odds_ratio)", ascending=True)["Gene"]
    #         .head(n_hior)
    #     )
    #     if gene not in top_down
    # ]

    symbol2dir = {gene: "no change" for gene in df[gene_column]}
    up = (
        df[gene_column]
        .loc[(df[pvalue_column] < pval_thresh) & (df["log2(odds_ratio)"] >= or_thresh)]
        .to_list()
    )
    for gene in up:
        symbol2dir[gene] = "up"
    down = (
        df[gene_column]
        .loc[(df[pvalue_column] < pval_thresh) & (df["log2(odds_ratio)"] <= -or_thresh)]
        .to_list()
    )
    for gene in down:
        symbol2dir[gene] = "down"
    df["change"] = df[gene_column].map(symbol2dir)

    dir2color = {
        "up": "indianred",
        "down": "cornflowerblue",
        "no change": "gainsboro",
    }

    fig, ax = plt.subplots(**subplot_kwargs)
    sns.scatterplot(
        df,
        x="log2(odds_ratio)",
        y="-log(padj)",
        hue="change",
        palette=dir2color,
        ax=ax,
        **plot_kwargs,
    )
    ax.axhline(-np.log10(pval_thresh), color="black", linestyle="--", linewidth=0.5)
    ax.axvline(or_thresh, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(-or_thresh, color="black", linestyle="--", linewidth=0.5)

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    #
    # ax.get_legend().remove()

    # ax.set_ylim(0, ax.get_ylim()[1])
    # xlim_max = max(abs(lim) for lim in ax.get_xlim())
    # ax.set_xlim(-xlim_max, xlim_max)

    # texts = []
    # for gene in top_up + top_down:
    #     x, y = df.loc[df["Gene"] == gene][
    #         ["log2(odds_ratio)", "-log(Padj)"]
    #     ].iloc[0, 0:2]
    #     x += rng.uniform(-0.01, 0.01)
    #     y += rng.uniform(-0.01, 0.01)
    #     texts.append(ax.text(x=x, y=y, s=gene, fontdict={"fontsize": 8}))
    # adjust_text(
    #     texts,
    #     arrowprops=dict(arrowstyle="-", color="gray"),
    #     force_explode=(0.3, 0.8),
    #     expand=[1.8, 1.2],
    #     max_move=(3600, 3600),
    # )  # ,only_move={"text":"y+x+-"})

    df["sorter"] = (
        df["-log10(padj)"] * df["log2(odds_ratio)"]
    )  # make a column to pick top genes
    df = df.sort_values(by="sorter", ascending=False)
    top_up = df[gene_column].head(to_label).to_list()
    top_down = df[gene_column].tail(to_label).to_list()

    texts = []
    for gene in top_up + top_down:
        x, y = df.loc[df[gene_column] == gene][["log2(odds_ratio)", "-log(padj)"]].iloc[
            0, 0:2
        ]
        txt = ax.text(
            x=x,
            y=y,
            s=gene,
            **text_kwargs,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
        texts.append(txt)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", zorder=5))

    return fig, ax
