"""Loading and light validation of a run configuration.

A run is described by one YAML file: where the base chains live, which base
model they came from, and which extended models to reweight onto. Keeping the
schema in one place means a configuration can be checked before any expensive
chain reading or likelihood evaluation begins.
"""

import os
from pathlib import Path

import yaml

#: Environment variable naming the root of a local chain grid.
CHAINS_ENV = "COSMOCOMPARE_CHAINS"

#: Environment variable naming the default output directory.
OUTPUT_ENV = "COSMOCOMPARE_OUTPUT"


def chains_root():
    """Return the configured root of a local grid of chains, if any.

    Returns
    -------
    pathlib.Path or None
        ``$COSMOCOMPARE_CHAINS`` as a path, or None when it is unset.
    """
    value = os.environ.get(CHAINS_ENV)
    return Path(value) if value else None


def output_root():
    """Return the directory results are written to.

    Returns
    -------
    pathlib.Path
        ``$COSMOCOMPARE_OUTPUT`` if set, otherwise ``./cosmocompare_output``.
    """
    return Path(os.environ.get(OUTPUT_ENV, "cosmocompare_output"))


def load_config(path):
    """Read and validate a run configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration.

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    KeyError
        If a required top-level section is missing.
    """
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    for section in ("global_settings", "target_models"):
        if section not in config:
            raise KeyError(f"Configuration is missing required section {section!r}.")

    settings = config["global_settings"]
    if "base_chain_path" not in settings:
        raise KeyError("global_settings must set 'base_chain_path'.")

    return config
