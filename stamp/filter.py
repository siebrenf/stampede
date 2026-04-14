import functools
import operator


def filter_genes(
    adata,
    ncell_min: int = 0,
    ncell_max=float("inf"),
    ntranscript_min=0,
    ntranscript_max=float("inf"),
    signal2noise_threshold=1,
    filter_columns: list = None,
    verbose=True,
):
    """
    Filter adata.var by a set of qc_params.

    Args:
        ncell_min: minimum number of cells the gene is found in.
        filter_columns: a list of additional columns to filter by. Columns by (convertible to) boolean, where False values are removed.
    """
    if filter_columns is None:
        filter_columns = []
    adata.strings_to_categoricals()

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
    adata = adata[:, total_gene_filter].copy()
    after = len(adata.var)
    if verbose:
        print(f"{before - after:_} genes filtered out, {after:_} genes remaining.")
    return adata


def filter_cells(
    adata,
    dist2edge_px_min=0,
    falsecode_max=5,
    negprobe_max=3,
    transcripts_min=250,
    transcripts_max=1500,
    area_min=25,
    area_max=100,
    filter_columns: list = None,
    verbose=True,
):
    """
    Filter adata.obs by a set of qc_params.

    Args:
        filter_columns: a list of additional columns to filter by. Columns by (convertible to) boolean, where False values are removed.
    """
    if filter_columns is None:
        filter_columns = []
    else:
        for col in filter_columns:
            if adata.obs[col].dtype != bool:
                raise TypeError(f"filter_column '{col}' must have a boolean dtype")
        filter_columns = [adata.obs[col] for col in filter_columns]
    adata.strings_to_categoricals()

    dist2edge_filter = adata.obs["dist2edge_px"] >= dist2edge_px_min
    filter_columns.append(dist2edge_filter)
    falsecode_filter = ~(adata.obs["nCount_falsecode"] >= falsecode_max)
    filter_columns.append(falsecode_filter)
    negprobe_filter = ~(adata.obs["nCount_negprobes"] >= negprobe_max)
    filter_columns.append(negprobe_filter)
    transcript_filter = adata.obs["nCount_RNA"].between(
        transcripts_min, transcripts_max
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
