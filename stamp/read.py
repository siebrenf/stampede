import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import (
    exprmat_md_columns,
    metadata_md_columns,
)


def get_X(fname, columns, chunksize=50_000, verbose=True, **kwargs):
    sparse_blocks = []
    i = 1
    for chunk in pd.read_csv(
        fname, dtype=int, usecols=columns, chunksize=chunksize, **kwargs
    ):
        # Convert to sparse matrix
        sparse_blocks.append(sp.csr_matrix(chunk.values))
        if verbose:
            print(i * chunksize, "rows parsed")
            i += 1

    # Stack all chunks vertically
    X = sp.vstack(sparse_blocks, format="csr")
    if verbose:
        print(i * chunksize, "rows parsed")
    return X


def add_sample_df_metadata(obs, sample_df, slide, columns=None):
    if columns is None:
        columns = sample_df.columns.to_list()
    if "sample" not in columns:
        columns.append("sample")
    if "slide" in columns:
        columns.remove("slide")

    sdf = sample_df.loc[sample_df["slide"] == slide]

    start = sdf["fov_start"].to_numpy()
    end = sdf["fov_end"].to_numpy()
    sample = sdf["sample"].to_numpy()
    fov = obs["fov"].to_numpy(copy=False)

    # find candidate interval
    idx = np.searchsorted(start, fov, side="right") - 1

    # check if fov actually falls inside interval
    valid = (idx >= 0) & (fov <= end[np.clip(idx, 0, None)])

    # create sample column
    result = np.empty(len(fov), dtype=object)
    result[:] = None
    result[valid] = sample[idx[valid]]
    obs["sample"] = result

    # merge sample_df to obs
    obs = obs.merge(sdf[columns], on="sample", how="left")
    return obs


def add_metadata(obs, file_md, slide, columns=None, data_dir: str = None, **kwargs):
    if data_dir is None:
        data_dir = ""
    if columns:
        for col in metadata_md_columns:
            if col not in columns:
                columns.append(col)
    md = pd.read_csv(os.path.join(data_dir, file_md), usecols=columns, **kwargs)

    # add a unique identifier
    index = "slide-fov-cell_ID"
    md[index] = f"{slide}-" + md["fov"].astype(str) + "-" + md["cell_ID"].astype(str)

    # if both files are sorted, we can concatenate blindly
    assert (md[index] == obs[index]).all(), "exprMat and metadata files are not sorted!"
    obs = pd.concat([obs, md.drop(columns=metadata_md_columns + [index])], axis=1)
    return obs


def read_cosmx(
    slides,
    sample_df: pd.DataFrame,
    adata_file: str,
    sample_df_columns: list = None,
    metadata_df_columns: list = None,
    data_dir: str = None,
    overwrite: bool = True,
    verbose: bool = True,
    **kwargs,
):
    """
    Read exprMat_file for each slide, convert the contents to sparse anndata objects,
    and concatenate the results.

    Args:
        slides: a dictionary with the slide number as keys, and a dictionary as values. The value dict must contain keys "exprmat" and "metadata", with should map to matching respective files
        sample_df: a dataframe with sample metadata to be added to adata.obs
        adata_file: filepath to write the adata object to
        sample_df_columns: list of columns in sample_df to add to adata.obs (default: all)
        metadata_df_columns: list of columns in the metadata file to add to adata.obs (default: all)
        overwrite: overwrite existing output (default: True)
        **kwargs: keyword arguments passed to pd.read_csv

    Returns:
        adata: a sparse adata object
    """
    if data_dir is None:
        data_dir = ""
    for col in ["usecols", "dtype", "chunksize"]:
        if col in kwargs:
            raise NotImplementedError(f"cannot use '{col}' in kwargs")
    if overwrite is False and os.path.exists(adata_file):
        if verbose:
            print(f"{overwrite=}. Returning preloaded data")
        return

    adatas = []
    for slide, files in slides.items():
        fname = os.path.join(data_dir, files["exprmat"])

        # get the gene columns from the exprMat_file
        columns = pd.read_csv(fname, dtype=int, nrows=0).columns.to_list()
        for col in exprmat_md_columns:
            if col not in columns:
                raise ValueError(f"column={col} not found in {files['exprmat']}")
            columns.remove(col)

        # get the cell metadata from the exprMat_file, the metadata_file and the sample_df
        obs = pd.read_csv(fname, dtype=int, usecols=exprmat_md_columns, **kwargs)
        obs["slide"] = slide
        obs["slide-fov"] = f"{slide}-" + obs["fov"].astype(str)
        index = "slide-fov-cell_ID"
        obs[index] = obs["slide-fov"] + "-" + obs["cell_ID"].astype(str)
        obs = add_metadata(
            obs, files["metadata"], slide, metadata_df_columns, data_dir, **kwargs
        )
        obs = add_sample_df_metadata(obs, sample_df, slide, sample_df_columns)
        obs.set_index(index, inplace=True)

        # convert to sparse adata object
        with warnings.catch_warnings():
            # warning: Transforming to str index.
            warnings.simplefilter("ignore", ad.ImplicitModificationWarning)
            adata = ad.AnnData(
                X=get_X(fname, columns, verbose=verbose, **kwargs),
                obs=obs,
                var=pd.DataFrame(index=columns),
            )
        # write adata object to file
        dirname = os.path.dirname(adata_file)
        basename = os.path.basename(adata_file)
        base, suffix = basename.split(".h5ad")
        adata_file_slide = os.path.join(dirname, f".{base}_slide{slide}.h5ad{suffix}")
        adata.write_h5ad(adata_file_slide)
        adatas.append(adata_file_slide)

        # free memory
        del adata, obs, columns
        if verbose:
            print(f"slide {slide}/{len(slides)} done")

    # concatenate all adatas
    if len(adatas) == 0:
        raise ValueError("No data to concatenate")
    elif len(adatas) == 1:
        os.rename(adatas[0], adata_file)
    else:
        ad.experimental.concat_on_disk(adatas, adata_file)
        for f in adatas:
            os.remove(f)
