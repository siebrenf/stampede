from __future__ import annotations

import os
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
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


def pydeseq2(
    adata: ad.anndata, contrast, force_paired=True, control_genes=None, n_cpus=16
):
    """
    Runs pyDEseq2 on an adata object.

    Takes an adata object, with conditions to compare in adata.obs["Condition"]
    and donors in adata.obs["Donor"], and runs (pairwise) DGEA based on contrast.
    Contrast must be ["Condition", "A", "B"], with "A" and "B" string values of
    categories in adata.obs["Condition"] that will be compared.

    Args:
        adata: adata object
        contrast:
        force_paired:
        control_genes:
        n_cpus:

    Returns:
        pd.DataFrame
    """

    donors_a = (
        adata.obs["Donor"].loc[adata.obs["Condition"] == contrast[1]].unique().tolist()
    )
    donors_b = (
        adata.obs["Donor"].loc[adata.obs["Condition"] == contrast[2]].unique().tolist()
    )
    fully_paired = set(donors_a) == set(donors_b)
    if fully_paired:
        design = "~Donor + Condition"
    else:
        if force_paired:
            design = "~Donor + Condition"

            common_donors = list(set(donors_a) & set(donors_b))
            n_common_donors = len(common_donors)
            if n_common_donors < 2:
                print(
                    f"{n_common_donors} donor(s) in common between groups: "
                    "Cannot run analysis"
                )
                return None
            else:
                adata = adata[adata.obs["Donor"].isin(common_donors)].copy()
        else:
            design = "~Condition"

    dds = DeseqDataSet(
        adata=adata,
        design=design,
        inference=DefaultInference(n_cpus=n_cpus),
        control_genes=control_genes,
    )

    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    dds.refit()

    ds = DeseqStats(
        dds,
        contrast=contrast,
        alpha=0.05,
        n_cpus=n_cpus,
    )

    ds.run_wald_test()
    ds._independent_filtering()
    ds.summary()

    return ds.results_df


# source: https://github.com/mousepixels/sanbomics/blob/master/sanbomics/plots.py
def volcano(
    data,
    log2fc="log2FoldChange",
    pvalue="padj",
    symbol="symbol",
    baseMean=None,
    pval_thresh=0.05,
    log2fc_thresh=0.75,
    to_label=5,
    color_dict=None,
    shape_dict=None,
    fontsize=10,
    colors=["dimgrey", "lightgrey", "black"],
    top_right_frame=False,
    figsize=(5, 5),
    legend_pos=(1.4, 1),
    point_sizes=(15, 150),
    save=False,
    shapes=None,
    shape_order=None,
):
    """
    Make a volcano plot from a pandas dataframe of directly from a csv.

    data : pandas.DataFrame or path to csv
    log2fc : string
        column name of log2 Fold-Change values
    pvalue : string
        column name of the p values to be converted to -log10 P values
    symbol : string
        column name of gene IDs to use
    baseMean : string
        column name of base mean values for each gene. If this is passed,
        the size of the points will vary.
    pval_thresh : numeric
        threshold pvalue for points to be significant. Also controls horizontal
        line.
    log2fc_thresh : numeric
        threshold for the absolute value of the log2 fold change to be considered
        significant. Also controls vertical lines
    to_label : int or list
        If an int is passed, that number of top down and up genes will be labeled.
        If a list of gene Ids is passed, only those will be labeled.
    color_dict : dictionary
        dictionary to color dots by. Up to 11 categories with default colors.
        Pass list of genes and the category to group them by. {category : ['gene1', gene2]}
        Default colors are: ['dimgrey', 'lightgrey', 'tab:blue', 'tab:orange',
        'tab:green', 'tab:red', 'tab:purple','tab:brown', 'tab:pink',
        'tab:olive', 'tab:cyan']
    shape_dict : dictionary
        dictionary to shape dots by. Up to 6 categories. Pass list of genes as values
        and category as key. {category : ['gene1', gene2], category2 : ['gene3']}
    fontsize : int
        size of labels
    colors : list
        order and colors to use. Default ['dimgrey', 'lightgrey', 'black']
    top_right_frame : Boolean
        Show the top and right frame. True/False
    figsize : tuple
        Size of figure. (x, y)
    point_sizes : tuple
        lower and upper bounds of point sizes. If baseMean is not None.
        (lower, upper)
    save : boolean | string
        If true saves default file name. Pass string as path to output file. Will
        add a .svg/.png to string. Saves as both png and svg.
    shapes : list
        pass matplotlib marker ids to change default shapes/order
        Default shapes order is: ['o', '^', 's', 'X', '*', 'd']
    shape_order : list
        If you want to change the order of categories for your shapes. Pass
        a list of your categories.

    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy(deep=True)

    # clean and imput 0s
    df = df.dropna()
    if df[pvalue].min() == 0:
        print("0s encountered for p value, imputing 1e-323")
        print("impute your own value if you want to avoid this")
        df[pvalue][df[pvalue] == 0] = 1e-323

    pval_thresh = -np.log10(pval_thresh)  # convert p value threshold to nlog10
    df["nlog10"] = -np.log10(df[pvalue])  # make nlog10 column
    df["sorter"] = df["nlog10"] * df[log2fc]  # make a column to pick top genes

    # size the dots by basemean if a column id is passed
    if baseMean is not None:
        df["logBaseMean"] = np.log(df[baseMean])
        baseMean = "logBaseMean"
    else:
        point_sizes = None

    # color dots by {label:[list of genes]}

    # make label list of top x genes up and down, or based on list input
    if isinstance(to_label, int):
        label_df = pd.concat(
            (df.sort_values("sorter")[-to_label:], df.sort_values("sorter")[0:to_label])
        )

    else:
        label_df = df[df[symbol].isin(to_label)]

    # color light grey if below thresh, color picked black
    def map_color_simple(a):
        log2FoldChange, zymbol, nlog10 = a
        if zymbol in label_df[symbol].tolist():
            return "picked"

        if abs(log2FoldChange) < log2fc_thresh or nlog10 < pval_thresh:
            return "not DE"
        return "DE"

    if color_dict is None:
        df["color"] = df["color"] = df[[log2fc, symbol, "nlog10"]].apply(
            map_color_simple, axis=1
        )
        hues = ["DE", "not DE", "picked"][: len(df.color.unique())]  # order of colors

    # coloring if dictionary passed
    def map_color_complex(a):
        log2FoldChange, zymbol, nlog10 = a

        for k in list(color_dict):
            if zymbol in color_dict[k]:
                return k
        if abs(log2FoldChange) < log2fc_thresh or nlog10 < pval_thresh:
            return "not DE"
        return "DE"

    if color_dict is not None:
        df["color"] = df["color"] = df[[log2fc, symbol, "nlog10"]].apply(
            map_color_complex, axis=1
        )
        user_added_cats = [x for x in df.color.unique() if x not in ["DE", "not DE"]]
        hues = ["DE", "not DE"] + user_added_cats
        hues = hues[: len(df.color.unique())]  # order of colors
        if colors == ["dimgrey", "lightgrey", "black"]:
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

    # map shapes if dictionary exists
    def map_shape(zymbol):
        for k in list(shape_dict):
            if zymbol in shape_dict[k]:
                return k

        return "other"

    if shape_dict is not None:
        df["shape"] = df[symbol].map(map_shape)
        user_added_cats = [x for x in df["shape"].unique() if x != "other"]
        shape_order = ["other"] + user_added_cats
        if shapes is None:
            shapes = ["o", "^", "s", "X", "*", "d"]
        shapes = shapes[: len(df["shape"].unique())]
        shape_col = "shape"
    else:
        shape_col = None

    # build palette
    colors = colors[: len(df.color.unique())]

    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=df,
        x=log2fc,
        y="nlog10",
        hue="color",
        hue_order=hues,
        palette=colors,
        size=baseMean,
        sizes=point_sizes,
        style=shape_col,
        style_order=shape_order,
        markers=shapes,
    )

    # make labels
    texts = []
    for i in range(len(label_df)):
        txt = plt.text(
            x=label_df.iloc[i][log2fc],
            y=label_df.iloc[i].nlog10,
            s=label_df.iloc[i][symbol],
            fontsize=fontsize,
            weight="bold",
        )

        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
        texts.append(txt)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", zorder=5))

    # plot vertical and horizontal lines
    ax.axhline(pval_thresh, zorder=0, c="k", lw=2, ls="--")
    ax.axvline(log2fc_thresh, zorder=0, c="k", lw=2, ls="--")
    ax.axvline(log2fc_thresh * -1, zorder=0, c="k", lw=2, ls="--")

    # make things pretty
    for axis in ["bottom", "left", "top", "right"]:
        ax.spines[axis].set_linewidth(2)

    if not top_right_frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.tick_params(width=2)
    plt.xticks(size=11, weight="bold")
    plt.yticks(size=11, weight="bold")
    plt.xlabel("$log_{2}$ fold change", size=15)
    plt.ylabel("-$log_{10}$ FDR", size=15)

    plt.legend(loc=1, bbox_to_anchor=legend_pos, frameon=False, prop={"weight": "bold"})

    if save == True:
        files = os.listdir()
        for x in range(100):
            file_pref = "volcano_" + "%02d" % (x,)
            if len([x for x in files if x.startswith(file_pref)]) == 0:
                plt.savefig(file_pref + ".png", dpi=300, bbox_inches="tight")
                plt.savefig(file_pref + ".svg", bbox_inches="tight")
                break
    elif isinstance(save, str):
        plt.savefig(save + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save + ".svg", bbox_inches="tight")

    plt.show()


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


def plot_volcano(
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
