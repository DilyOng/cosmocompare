"""Loading and light validation of a run configuration.

A run is described by one YAML file: where the base chains live, which base
model they came from, and which extended models to reweight onto. Keeping the
schema in one place means a configuration can be checked before any expensive
chain reading or likelihood evaluation begins.
"""

import glob
import os
from pathlib import Path

import yaml

#: Environment variable naming the root of a local chain grid.
CHAINS_ENV = "COSMOCOMPARE_CHAINS"

#: Environment variable naming the *primary* grid root (searched first).
CHAINS_PRIMARY_ENV = "COSMOCOMPARE_CHAINS_PRIMARY"

#: Environment variable naming the *fallback* grid root (searched second).
CHAINS_FALLBACK_ENV = "COSMOCOMPARE_CHAINS_FALLBACK"

#: Environment variable naming the default output directory.
OUTPUT_ENV = "COSMOCOMPARE_OUTPUT"

#: Default grid roots on the DiRAC RDS. ``new_grid`` holds the corrected/newer
#: chains and is searched first; ``grid`` is the fallback. This reverses the order
#: used by the older ``tension_stats_grid`` scripts (which read ``grid`` first),
#: deliberately, so the corrected chains take precedence. Override with the
#: environment variables above for a different machine.
_RDS = "/home/dlo26/rds/rds-dirac-dp192-63QXlf5HuFo/dlo26"
DEFAULT_PRIMARY_ROOT = f"{_RDS}/new_grid"
DEFAULT_FALLBACK_ROOT = f"{_RDS}/grid"


def chains_root():
    """Return the configured root of a local grid of chains, if any.

    Returns
    -------
    pathlib.Path or None
        ``$COSMOCOMPARE_CHAINS`` as a path, or None when it is unset.
    """
    value = os.environ.get(CHAINS_ENV)
    return Path(value) if value else None


def chain_stem(root, model, dataset, method="ns"):
    """Return the anesthetic chain stem for one (model, dataset) under a root.

    The grid stores each run as PolyChord output files sharing a stem inside a
    ``<dataset>_polychord_raw`` directory, so the stem passed to
    ``anesthetic.read_chains`` is
    ``<root>/<method>/<model>/<dataset>/<dataset>_polychord_raw/<dataset>``.

    Parameters
    ----------
    root : str or pathlib.Path
        A grid root (for example ``.../new_grid``).
    model : str
        Model name (for example ``lcdm`` or ``walcdm``).
    dataset : str
        Dataset identifier.
    method : str, optional
        Sampling method directory. Defaults to ``"ns"`` (nested sampling).

    Returns
    -------
    pathlib.Path
        The chain stem (a path prefix, not an existing file on its own).
    """
    raw = f"{dataset}_polychord_raw"
    return Path(root) / method / model / dataset / raw / dataset


def _stem_exists(stem):
    """Return True when PolyChord output files for a stem are present."""
    return bool(glob.glob(f"{stem}*"))


def available_datasets(model, method="ns", primary=None, fallback=None):
    """List dataset names that have a chain for ``model`` in either grid root.

    Parameters
    ----------
    model : str
        Model directory name.
    method : str, optional
        Sampling method directory. Defaults to ``"ns"``.
    primary, fallback : str or pathlib.Path, optional
        Grid roots; default to the environment or :data:`DEFAULT_PRIMARY_ROOT` /
        :data:`DEFAULT_FALLBACK_ROOT`.

    Returns
    -------
    list of str
        Sorted dataset names for which a resolvable chain exists.
    """
    primary = primary or os.environ.get(CHAINS_PRIMARY_ENV, DEFAULT_PRIMARY_ROOT)
    fallback = fallback or os.environ.get(CHAINS_FALLBACK_ENV, DEFAULT_FALLBACK_ROOT)

    datasets = set()
    for root in (primary, fallback):
        model_dir = Path(root) / method / model
        if not model_dir.is_dir():
            continue
        for sub in model_dir.iterdir():
            if sub.is_dir() and _stem_exists(chain_stem(root, model, sub.name, method)):
                datasets.add(sub.name)
    return sorted(datasets)


def resolve_chain_stem(
    model, dataset, method="ns", primary=None, fallback=None
):
    """Resolve a (model, dataset) to a chain stem, trying new_grid then grid.

    The primary root (``new_grid`` by default) is searched first; if no files are
    found there, the fallback root (``grid``) is tried. This prefers the corrected
    chains, reversing the order of the older ``tension_stats_grid`` scripts.

    Parameters
    ----------
    model : str
        Model name.
    dataset : str
        Dataset identifier.
    method : str, optional
        Sampling method directory. Defaults to ``"ns"``.
    primary : str or pathlib.Path, optional
        Primary grid root. Defaults to ``$COSMOCOMPARE_CHAINS_PRIMARY`` or
        :data:`DEFAULT_PRIMARY_ROOT`.
    fallback : str or pathlib.Path, optional
        Fallback grid root. Defaults to ``$COSMOCOMPARE_CHAINS_FALLBACK`` or
        :data:`DEFAULT_FALLBACK_ROOT`.

    Returns
    -------
    pathlib.Path
        The chain stem in the first root that has the run's files.

    Raises
    ------
    FileNotFoundError
        If neither root holds the run.
    """
    primary = primary or os.environ.get(CHAINS_PRIMARY_ENV, DEFAULT_PRIMARY_ROOT)
    fallback = fallback or os.environ.get(CHAINS_FALLBACK_ENV, DEFAULT_FALLBACK_ROOT)

    for root in (primary, fallback):
        stem = chain_stem(root, model, dataset, method=method)
        if _stem_exists(stem):
            return stem
    raise FileNotFoundError(
        f"No chain for model {model!r}, dataset {dataset!r} (method {method!r}) "
        f"in the primary root {primary!r} or the fallback root {fallback!r}."
    )


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
