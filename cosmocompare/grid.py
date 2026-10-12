"""Enumeration of a (model, prior) grid for batch dispatch.

A survey of prior sensitivity runs the same reweighting over many models and
many prior choices. Mapping that two-dimensional grid onto a flat list of
integer job indices is what lets a scheduler array - Slurm or otherwise - launch
one task per combination without any shared state.
"""

#: Extended models considered by default.
DEFAULT_MODELS = (
    "wlcdm",
    "klcdm",
    "walcdm",
    "Alcdm",
    "mlcdm",
    "nrunlcdm",
    "rlcdm",
)

#: Prior families considered by default.
DEFAULT_PRIOR_TYPES = (
    "uniform",
    "truncated_gaussian",
    "gaussian",
    "log_uniform",
)

#: Models for which a log-uniform prior is not meaningful and is skipped.
DEFAULT_NO_LOG_UNIFORM = ("wlcdm", "klcdm", "walcdm", "nrunlcdm")


def all_jobs(
    models=DEFAULT_MODELS,
    prior_types=DEFAULT_PRIOR_TYPES,
    no_log_uniform=DEFAULT_NO_LOG_UNIFORM,
):
    """Return every valid ``(model, prior_type)`` combination.

    Parameters
    ----------
    models : sequence of str, optional
        Models to include. Defaults to :data:`DEFAULT_MODELS`.
    prior_types : sequence of str, optional
        Prior families to include. Defaults to :data:`DEFAULT_PRIOR_TYPES`.
    no_log_uniform : sequence of str, optional
        Models for which ``log_uniform`` is skipped. Defaults to
        :data:`DEFAULT_NO_LOG_UNIFORM`.

    Returns
    -------
    list of tuple
        ``(model, prior_type)`` pairs in a stable order, suitable for indexing
        by a scheduler array.
    """
    jobs = []
    for model in models:
        for prior_type in prior_types:
            if prior_type == "log_uniform" and model in no_log_uniform:
                continue
            jobs.append((model, prior_type))
    return jobs


def job_by_index(index, **kwargs):
    """Return the ``(model, prior_type)`` at a zero-based job index.

    Parameters
    ----------
    index : int
        Zero-based index into :func:`all_jobs`.
    **kwargs
        Forwarded to :func:`all_jobs` to restrict the grid.

    Returns
    -------
    tuple of str
        The ``(model, prior_type)`` pair.

    Raises
    ------
    IndexError
        If ``index`` is outside the grid.
    """
    jobs = all_jobs(**kwargs)
    if not 0 <= index < len(jobs):
        raise IndexError(f"Job index {index} out of range [0, {len(jobs) - 1}].")
    return jobs[index]


def index_of_job(model, prior_type, **kwargs):
    """Return the zero-based index of a ``(model, prior_type)`` pair.

    Parameters
    ----------
    model : str
        Model name.
    prior_type : str
        Prior family.
    **kwargs
        Forwarded to :func:`all_jobs` to restrict the grid.

    Returns
    -------
    int
        The job index.

    Raises
    ------
    ValueError
        If the pair is not in the grid.
    """
    jobs = all_jobs(**kwargs)
    try:
        return jobs.index((model, prior_type))
    except ValueError:
        raise ValueError(
            f"Combination ({model}, {prior_type}) is not in the grid."
        ) from None
