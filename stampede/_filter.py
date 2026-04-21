from __future__ import annotations

import functools
import operator

import anndata as ad


def filter_genes(
    adata: ad.anndata,
    ncell_min: int = 0,
    ncell_max: int = float("inf"),
    ntranscript_min: int = 0,
    ntranscript_max: int = float("inf"),
    signal2noise_threshold: float = 1.0,
    filter_columns: str | list = None,
    verbose: bool = True,
):
    """
    Filter adata.var by a set of qc_params.

    Args:
        adata: adata object
        ncell_min: minimum number of cells the gene is found in.
        ncell_max: maximum number of cells the gene is found in.
        ntranscript_min: minimum number of transcripts the gene must have.
        ntranscript_max: maximum number of transcripts the gene must have.
        signal2noise_threshold: the minimum signal-to-noise ratio the gene must have.
        filter_columns: a list of additional columns to filter by.
         Columns by (convertible to) boolean, where False values are removed.
        verbose: provide written feedback (default: True)

    Returns:
        adata: the filtered adata object
    """
    if filter_columns is None:
        filter_columns = []
    elif isinstance(filter_columns, str):
        filter_columns = [filter_columns]
    adata.strings_to_categoricals()
    filter_columns = [adata.obs[col] for col in filter_columns]

    ncells_filter = adata.var["nCell"].between(ncell_min, ncell_max)
    filter_columns.append(ncells_filter)
    ntranscript_filter = adata.var["nTranscript"].between(
        ntranscript_min, ntranscript_max
    )
    filter_columns.append(ntranscript_filter)
    noise_filter = adata.var["signal2noise"] > signal2noise_threshold
    filter_columns.append(noise_filter)
    negprobe_filter = ~adata.var["is_negctrl"]
    filter_columns.append(negprobe_filter)
    falsecode_filter = ~adata.var["is_sysctrl"]
    filter_columns.append(falsecode_filter)

    # combine all filters
    total_gene_filter = functools.reduce(operator.and_, filter_columns)

    before = len(adata.var)
    adata = adata[:, total_gene_filter]
    after = len(adata.var)
    if verbose:
        print(f"{before - after:_} genes filtered out, {after:_} genes remaining.")
    return adata


def filter_cells(
    adata: ad.anndata,
    dist2edge_px_min: int = 0,
    falsecode_max: int = 5,
    negprobe_max: int = 3,
    ntranscript_min: int = 250,
    ntranscript_max: int = 1500,
    area_min: int = 25,
    area_max: int = 100,
    filter_columns: list = None,
    verbose: bool = True,
):
    """
    Filter adata.obs by a set of qc_params.

    Args:
        adata: adata object
        dist2edge_px_min:
        falsecode_max: maximum number of false codes the cell may have
        negprobe_max: maximum number of negative probes the cell may have
        ntranscript_min: minimum number of transcripts the cell must have
        ntranscript_max: maximum number of transcripts the cell must have
        area_min: minimum area (in pixels) the cell must have
        area_max: maximum area (in pixels) the cell must have
        filter_columns: a list of additional columns to filter by.
         Columns by (convertible to) boolean, where False values are removed.
        verbose: provide written feedback (default: True)

    Returns:
        adata: the filtered adata object
    """
    if filter_columns is None:
        filter_columns = []
    elif isinstance(filter_columns, str):
        filter_columns = [filter_columns]
    # else:
    #     for col in filter_columns:
    #         if adata.obs[col].dtype != bool:
    #             raise TypeError(f"filter_column '{col}' must have a boolean dtype")
    adata.strings_to_categoricals()
    filter_columns = [adata.obs[col] for col in filter_columns]

    dist2edge_filter = adata.obs["dist2edge_px"] >= dist2edge_px_min
    filter_columns.append(dist2edge_filter)
    falsecode_filter = ~(adata.obs["nCount_falsecode"] >= falsecode_max)
    filter_columns.append(falsecode_filter)
    negprobe_filter = ~(adata.obs["nCount_negprobes"] >= negprobe_max)
    filter_columns.append(negprobe_filter)
    transcript_filter = adata.obs["nCount_RNA"].between(
        ntranscript_min, ntranscript_max
    )
    filter_columns.append(transcript_filter)
    area_filter = adata.obs["Area.um2"].between(area_min, area_max)
    filter_columns.append(area_filter)
    internal_qc = adata.obs["qcCellsPassed"] & (adata.obs["qcFlagsFOV"] == "Pass")
    filter_columns.append(internal_qc)

    # combine all filters
    total_cell_filter = functools.reduce(operator.and_, filter_columns)

    before = len(adata.obs)
    adata = adata[total_cell_filter, :].copy()
    after = len(adata.obs)
    if verbose:
        print(f"{before - after:_} cells filtered out, {after:_} cells remaining.")

    return adata
