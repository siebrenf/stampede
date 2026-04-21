"""plotting functions"""

from ._analyses import plot_paired_binomial_glm_volcano as paired_binomial_glm_volcano
from ._analyses import plot_pydeseq2_volcano as pydeseq2_volcano
from ._analyses import plot_sketch as sketch
from ._dim_red import plot_dim_red as dim_red
from ._dim_red import plot_scree as scree
from ._qc import plot_2d_correlations as correlations
from ._qc import plot_avg_per_pixel as avg_per_pixel
from ._qc import plot_column_distribution as column_distribution
from ._qc import plot_ncell_per_condition as ncell_per_condition
from ._qc import plot_slide_qc as slide_qc
from ._qc import plot_value_distribution as value_distribution
from ._qc import plot_violin as violin

__all__ = [
    "slide_qc",
    "correlations",
    "avg_per_pixel",
    "violin",
    "ncell_per_condition",
    "value_distribution",
    "column_distribution",
    "scree",
    "dim_red",
    "sketch",
    "pydeseq2_volcano",
    "paired_binomial_glm_volcano",
]
