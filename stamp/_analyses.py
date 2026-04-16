from __future__ import annotations

import anndata as ad
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

# TODO: pydeseq2 + volcano plot: /bank/experiments/2024-08-sc/03_simple_analyses.ipynb


# TODO: make universal
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
