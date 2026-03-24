import functools
import operator


def genes_signal_to_noise(adata, mult=1):
    """
    Filter for signal-to-noise;
    Approach from https://doi.org/10.1038/s41467-025-64990-y
    Wang et al. Systematic benchmarking of imaging spatial
    transcriptomics platforms in FFPE tissues. Nat Com, 2025

    Calculate mean expression of negative control probes
    and standard deviation of those means.
    Remove genes with average expression < mean + STD of ctrl probes.
    * Paper uses 2x STD(!)
    """
    means = adata.var.loc[adata.var["is_negctrl"], "meanTranscript"]
    mean = means.mean()
    std = means.std()
    threshold = mean + mult * std
    adata.var["above_noise"] = adata.var["meanTranscript"] > threshold


def filter_genes(
    adata, ncell_min: int = 1000, filter_columns: list = None, verbose=True
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

    ncells_filter = adata.var["nCell"] >= ncell_min
    filter_columns.append(ncells_filter)
    noise_filter = adata.var["above_noise"]
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
    max_falsecode=5,
    max_negprobe=3,
    min_transcripts=250,
    max_transcripts=1500,
    min_area=25,
    max_area=100,
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

    falsecode_filter = ~(adata.obs["nCount_falsecode"] >= max_falsecode)
    filter_columns.append(falsecode_filter)
    negprobe_filter = ~(adata.obs["nCount_negprobes"] >= max_negprobe)
    filter_columns.append(negprobe_filter)
    transcript_filter = (adata.obs["nCount_RNA"] >= min_transcripts) & (
        adata.obs["nCount_RNA"] <= max_transcripts
    )
    filter_columns.append(transcript_filter)
    area_filter = (adata.obs["Area.um2"] >= min_area) & (
        adata.obs["Area.um2"] <= max_area
    )
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
