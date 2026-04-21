# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from datetime import date

import stampede


# -- Project information -----------------------------------------------------

project = 'stampede'
author = "Niels Velthuijs & Siebren Frölich"
copyright = f"{date.today().year}, {author}"
version = stampede.__version__


# -- General configuration ---------------------------------------------------

autodoc_mock_imports = [
    "adjustText",
    "anndata",
    "matplotlib",
    "natsort",
    "numpy",
    "pandas",
    "igraph",
    "scanpy",
    "scikit-learn",
    "scipy",
    "seaborn",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # for Google-style docstrings
    # "sphinx.ext.autosummary",  # Create neat summary tables
    "m2r2",                    # recognize markdown files
]

exclude_patterns = ["README.md"]

# Configuration of sphinx.ext.autodoc
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# # Configuration of sphinx.ext.autosummary
# autosummary_generate = True

# Configuration of m2r2
source_suffix = ['.rst', '.md']


# # Napoleon settings
# # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
#
# napoleon_google_docstring = True
# napoleon_numpy_docstring = False
# napoleon_preprocess_types = True  # generate hyperlinks for parameter types
#
# napoleon_type_aliases = {
#     "DeseqDataSet": ":class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>`",
#     "Inference": ":class:`Inference <pydeseq2.inference.Inference>`",
#     "DefaultInference": ":class:`DefaultInference "
#     "<pydeseq2.default_inference.DefaultInference>`",
#     # numpy short forms
#     "ndarray": ":class:`numpy.ndarray`",
#     "np.ndarray": ":class:`numpy.ndarray`",
#     # pandas short forms
#     "pd.DataFrame": ":class:`pandas.DataFrame`",
#     "pd.Series": ":class:`pandas.Series`",
#     "pandas.Index": ":class:`pandas.Index`",
# }


# -- Options for HTML output -------------------------------------------------

# add the global ToC to the sidebar of each page
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

html_theme = "sphinx_rtd_theme"  # "alabaster"
html_last_updated_fmt = '%Y-%m-%d, %H:%M (UTC)'
