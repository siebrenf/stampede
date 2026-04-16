import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def dim_red(
    adata,
    n_dims=50,
    key_added=None,
    random_state=42,
):
    """
    Dimensionality reduction using Term Frequency Latent Semantic Indexing.
    """
    if key_added is None:
        key_added = "X_svd"
    prefix, uns_key = key_added.split("_", 1)
    if prefix != "X" or len(uns_key) == 0:
        raise ValueError(f"{key_added=} must start with 'X_', e.g. 'X_svd'")

    layer = "binary"
    if layer not in adata.layers:
        raise KeyError(
            f"{layer=} not found in adata.layers. Please run st.pp.binarize first!"
        )
    X = adata.layers[layer]
    if "nFeature_RNA_postfilter" not in adata.obs.columns:
        raise KeyError(
            "nFeature_RNA_postfilter not in adata.obs. "
            "Please run st.pp.cell_qc_postfilter first!"
        )
    cell_sums = adata.obs["nFeature_RNA_postfilter"].to_numpy(dtype=np.float32)
    if "nCell_postfilter" not in adata.var.columns:
        raise KeyError(
            "nCell_postfilter not in adata.var. "
            "Please run st.pp.gene_qc_postfilter first!"
        )
    gene_counts = adata.var["nCell_postfilter"].to_numpy(dtype=np.float32)
    n_cells = X.shape[0]

    # Latent Semantic Indexing:
    # Term Frequency:
    tf = X.multiply(1.0 / cell_sums[:, None])

    # Inverse Document Frequency
    idf = np.log1p(n_cells / (1.0 + gene_counts))
    X_tfidf = tf.multiply(idf)

    # Truncated Singular Value Decomposition
    svd = TruncatedSVD(n_components=n_dims + 1, random_state=random_state)

    # Drop 1st dimension, is not informative (like scATAC; this case expl var 200-fold lower than 2nd dim)
    adata.obsm[key_added] = svd.fit_transform(X_tfidf)[:, 1:]
    adata.uns[uns_key] = {
        "idf": idf,
        "explained_variance_ratio": svd.explained_variance_ratio_[1:],
    }


def plot_scree(adata, obsm_key=None):
    if obsm_key is None:
        obsm_key = "X_svd"
    uns_key = obsm_key.split("_", 1)[1]

    evr = adata.uns[uns_key]["explained_variance_ratio"]
    xs = [i + 1 for i in range(len(evr))]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.scatter(x=xs, y=evr, s=5, c="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Explained variance ratio")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Explained variance ratio")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylim(0, ax.get_ylim()[1])

    return fig, ax


def plot_dim_red(
    adata,
    color,
    obsm_key=None,
    cmap="tab10",
    ndims=6,
    subset_size=1_000,
    random_state=42,
):
    if obsm_key is None:
        obsm_key = "X_svd"
    uns_key = obsm_key.split("_", 1)[1]

    cmap = plt.get_cmap(cmap)
    if isinstance(color, str):
        color = [color]

    data = adata.obs[color].copy()
    data[[str(dim + 1) for dim in range(ndims)]] = adata.obsm[obsm_key][:, :ndims]
    evr_array = adata.uns[uns_key]["explained_variance_ratio"]

    for c in color:
        df = data
        if subset_size:
            n = subset_size * len(data[c].unique())
            if len(data) > n:
                df = data.groupby(c).sample(n=n, random_state=random_state)
        levels, categories = pd.factorize(df[c])
        colors = [cmap.colors[i] for i in levels]

        # plot the upper triangle of a cross correlation matrix of PCs
        # excluding the diagonal
        fig, axs = plt.subplots(
            nrows=ndims - 1,
            ncols=ndims - 1,
            figsize=(
                (ndims - 1) * 3,
                (ndims - 1) * 3,
            ),
            # gridspec_kw={'wspace': 0, 'hspace': 0},
            tight_layout=True,
        )
        fig.suptitle(c, fontweight="bold")
        n = fig._suptitle.get_fontsize()  # noqa
        fig._suptitle.set_fontsize(n + 10)  # noqa
        for row in range(0, ndims - 1):
            row_label = str(row + 1)
            for col in range(0, ndims - 1):
                col_label = str(col + 2)
                ax = axs[row, col]

                if row <= col:
                    # add plots only in the upper triangle
                    ax.scatter(
                        data=df,
                        x=col_label,
                        y=row_label,
                        c=colors,
                        alpha=0.1,
                        s=1,
                    )
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # add labels and edges on the straight sides of the plot
                    if row == 0:
                        evr = np.format_float_scientific(
                            evr_array[int(col_label)],
                            precision=2,
                        )
                        if col == 0:
                            # explain the label
                            ax.set_xlabel(f"Dim {col_label} (explained var: {evr})")
                        else:
                            ax.set_xlabel(f"Dim {col_label} ({evr})")
                        ax.xaxis.set_label_position("top")
                        ax.spines["top"].set_visible(True)
                    if col == ndims - 2:
                        evr = np.format_float_scientific(
                            evr_array[int(row_label)],
                            precision=2,
                        )
                        ax.set_ylabel(f"Dim {row_label} ({evr})")
                        ax.yaxis.set_label_position("right")
                        ax.spines["right"].set_visible(True)
                else:
                    # hide the lower triangle
                    ax.axis("off")

        # legend
        ax = axs[1, 0]
        handles = []
        for i, cat in enumerate(categories):
            patch = mpatches.Patch(
                color=cmap.colors[i],
                label=cat,
            )
            handles.append(patch)
        ax.legend(handles=handles, loc="center", title=c)
        ax.axis("off")

        plt.show()

    return
