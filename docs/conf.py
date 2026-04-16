# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # so Sphinx finds your package


# -- Project information -----------------------------------------------------

project = 'stamp'  # TODO
copyright = 'Niels Velthuijs, Siebren Frölich'
author = 'Niels Velthuijs, Siebren Frölich'


# -- General configuration ---------------------------------------------------

autodoc_mock_imports = [
    "scanpy",
    "anndata",
    "pydeseq2",
    "matplotlib",
    "seaborn",
    "adjusttext",
    # "mygene",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # for Google-style docstrings
    # "sphinx.ext.autosummary",  # Create neat summary tables
    "m2r2",                    # recognize markdown files
    # "nbsphinx",                # recognize notebooks
]

exclude_patterns = ["README.md"]

# Configuration of sphinx.ext.autodoc
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# # Configuration of sphinx.ext.autosummary
# autosummary_generate = True

# Configuration of m2r2
source_suffix = ['.rst', '.md']

# Configuration of nbsphinx
nbsphinx_execute = "never"


# -- Options for HTML output -------------------------------------------------

# add the global ToC to the sidebar of each page
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

html_theme = "sphinx_rtd_theme"  # "alabaster"
html_last_updated_fmt = '%Y-%m-%d, %H:%M (UTC)'
