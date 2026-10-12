"""Health checks for a set of importance weights.

Importance sampling is only trustworthy while the weights stay reasonably
even. A handful of samples carrying almost all of the weight means the proposal
and the target disagree, and any estimate built from them is dominated by those
few points. The effective sample size makes that failure mode visible as a
single number.
"""

import numpy as np


def effective_sample_size(weights):
    """Return Kish's effective sample size of a weight vector.

    Parameters
    ----------
    weights : array_like
        Importance weights. Need not be normalised; non-finite entries are
        ignored.

    Returns
    -------
    float
        ``(sum w)^2 / sum(w^2)``, which lies in ``[1, N]`` for ``N`` finite
        positive weights and collapses towards 1 as the weights concentrate.
        Returns ``0.0`` when no usable weight is present.

    Notes
    -----
    This is the standard Kish approximation to the number of independent,
    equally weighted samples the set is worth.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return 0.0

    finite = weights[np.isfinite(weights)]
    if finite.size == 0:
        return 0.0

    sum_w = np.sum(finite)
    sum_w2 = np.sum(finite**2)
    if sum_w2 == 0:
        return 0.0

    return float(sum_w**2 / sum_w2)


def efficiency(weights):
    """Return the effective sample size as a fraction of the sample count.

    Parameters
    ----------
    weights : array_like
        Importance weights.

    Returns
    -------
    float
        ``effective_sample_size(weights) / len(weights)`` in ``[0, 1]``. A value
        near 1 means evenly spread weights; a small value warns that the result
        rests on few samples.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return 0.0
    return effective_sample_size(weights) / weights.size
