from __future__ import annotations

import warnings

import anndata as ad
import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib import pyplot as plt


def paired_binomial_glm(
    df: pd.DataFrame,
    adata: ad.AnnData,
    samples_column: str,
    test_condition: str,
    reference_condition: str,
    condition_column: str = "condition",
    covariate_columns: str = None,
) -> pd.DataFrame | None:
    """
    Runs paired donor-level binomial GLM:
        gene_detection_rate ~ condition + covariate(s)

    Args:
        df: dataframe with detection rates per gene per sample
        adata: the adata from which the detection rates were obtained
        samples_column: the column in adata.obs from which the detection rate df
         column names were obtained
        test_condition: the condition to compare (e.g., "treated")
        reference_condition: the baseline condition (e.g., "control")
        condition_column: column with the condition
        covariate_columns: column(s) with covariates (e.g. "donor")

    Returns:
        per-gene results including beta, odds_ratio, pval, padj
    """
    # optional dependency
    import statsmodels.api as sm  # noqa
    import statsmodels.formula.api as smf  # noqa
    from statsmodels.stats.multitest import multipletests  # noqa
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning  # noqa

    df = df.copy().stack().reset_index()
    df.columns = ["gene", samples_column, "detection_rate"]

    subset = adata.obs[[samples_column, condition_column]]
    sample2condition = subset.set_index(samples_column)[condition_column].to_dict()
    df[condition_column] = df[samples_column].map(sample2condition)

    sample2ncells = adata.obs[samples_column].value_counts().to_dict()
    df["ncells"] = df[samples_column].replace(sample2ncells)

    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].astype("category")

    unique_conditions = df[condition_column].unique()
    if reference_condition not in unique_conditions:
        raise ValueError(f"{reference_condition=} not found in dataframe")
    if test_condition not in unique_conditions:
        raise ValueError(f"{test_condition=} not found in dataframe")

    # re-level the condition column so the reference condition is the baseline
    df = df[df[condition_column].isin(unique_conditions)]
    df[condition_column] = pd.Categorical(
        df[condition_column],
        categories=[reference_condition, test_condition],
        ordered=True,
    )
    df.dropna(subset=condition_column, inplace=True)

    if covariate_columns is None:
        covariate_columns = []
    elif isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]

    # add all covariate columns and
    # ensure all batch and covariate columns are categorical
    for col in covariate_columns:
        s2c = subset.set_index(samples_column)[col].to_dict()
        df[col] = df[samples_column].map(s2c).astype(str).astype("category")

    design_formula = "detection_rate ~ " + " + ".join(
        [condition_column] + covariate_columns
    )

    def fit_one_gene(gene_df):
        perfect_sep = False
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", PerfectSeparationWarning)

                model = smf.glm(
                    formula=design_formula,
                    data=gene_df,
                    family=sm.families.Binomial(),
                    var_weights=gene_df["ncells"],
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
        df.groupby("gene", group_keys=False, observed=False)
        .apply(fit_one_gene, include_groups=False)
        .reset_index()
    )

    # drop failed fits
    results.dropna(subset=["pval"], inplace=True)
    if len(results) == 0:
        return None

    results["padj"] = multipletests(results["pval"], method="fdr_bh")[1]
    results["-log10(padj)"] = -np.log10(np.clip(results["padj"], 1e-9, None))
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


def paired_binomial_glm_volcano(
    df: pd.DataFrame,
    drop_perfect_separation: bool = True,
    pval_thresh: float = 0.05,
    or_thresh: float = 0.75,
    to_label: int | list = 5,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Generate a volcano plot from the detection_rates results dataframe.

    Args:
        df: a dataframe
        drop_perfect_separation: whether to drop the genes with perfect separations
        pval_thresh: threshold pvalue_column for points to be significant
        or_thresh: threshold for the log2 odds ratios to be considered significant
        to_label: the number of top down and up genes to be labeled
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.text

    Returns:
        matplotlib figure and axis object
    """
    or_thresh = abs(or_thresh)
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    gene_column = "gene"
    or_column = "odds_ratio"
    pvalue_column = "padj"
    separation_column = "perfect_separation"

    df = df.copy().dropna(subset=[pvalue_column, or_column, separation_column])

    if df[pvalue_column].min() == 0:
        df[pvalue_column][df[pvalue_column] == 0] = 1e-9

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
        y="-log10(padj)",
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
        x, y = df.loc[df[gene_column] == gene][
            ["log2(odds_ratio)", "-log10(padj)"]
        ].iloc[0, 0:2]
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
