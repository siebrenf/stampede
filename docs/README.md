# Building the documentation

## Setup

Create the environment (including all optional dependencies)
```bash
conda env create -n stampede -f requirements.yaml
```

Install the package
```bash
pip install -e .
```

## Build

Make sure all `.. mdinclude::` directives have been updated if the root `README.md` changed!

Make sure all `.. literalinclude::` directives have been updated if the code changed!

```bash
sphinx-build -b html docs _build --write-all --fresh-env --fail-on-warning
```

open `_build/index.html` in a browser.
