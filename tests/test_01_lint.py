"""
run from the command line (from the opticlust directory) with:
    pytest --disable-pytest-warnings -vvv
"""

import os
import subprocess as sp
from os.path import dirname, join

import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too late to lint!")
def test_lint():
    base = dirname(dirname(__file__))
    sp.check_output(
        "black "
        + f"{join(base, 'stampede')} {join(base, 'tests')} "
        + f"{join(base, 'tutorial.ipynb')} "
        + f"{join(base, 'docs')}",
        shell=True,
    )
    sp.check_output(
        "isort --overwrite-in-place --profile black --conda-env requirements.yaml "
        + f"{join(base, 'stampede')} {join(base, 'tests')}  {join(base, 'docs')}",
        shell=True,
    )
