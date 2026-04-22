class Config(dict):
    """
    A dictionary with package specific settings that may be altered during runtime.
    Accessed using `import stampede as st; st.config`.

    Keys may not be added or removed, but values may be changed.
    """

    def __init__(self):
        super().__init__(
            {
                # columns found in the exprmat_file that represent metadata
                "exprmat_md_columns": ["fov", "cell_ID"],
                # columns found in the metadata_file that represents metadata
                "metadata_md_columns": ["fov", "cell_ID"],
                # columns found in the sample_file that represents metadata
                "sample_md_columns": ["sample", "slide", "fov_start", "fov_end"],
                # directory to write (temporary) adata objects to
                "adata_dir": "adatas",
            }
        )

    def __setitem__(self, key, value):
        if key not in self:
            raise IndexError(
                f"{key=} not in config. "
                f"You may only update existing keys ({sorted(self.keys())})."
            )
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v  # reuse validation

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __delitem__(self, key):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def pop(self, key):
        raise NotImplementedError

    def popitem(self):
        raise NotImplementedError


config = Config()
