r"""The importance-sampling weight update, with the full prior correction.

Switching a set of samples from a base model to an extended one, or onto a
changed prior, multiplies each posterior weight by the likelihood ratio and by
the ratio of the new prior to the proposal (or original prior) the parameter
came from:

.. math::

    w_\text{new} = w_\text{old} \,
        \frac{\mathcal{L}_\text{new}}{\mathcal{L}_\text{old}} \,
        \frac{\Pi_\text{new}}{q}.

Dropping the prior-over-proposal factor is a common shortcut; it is correct only
when the proposal happens to equal the target prior. This module always carries
the term - it is zero in that case and matters in every other.

The re-weighting multiplies the posterior weights directly and does *not*
re-sort the run or recompute nested-sampling volumes: once the samples no longer
form a nested sequence under the changed measure, that geometry is invalid. The
evidence must therefore come from :mod:`cosmocompare.evidence`, never from
``logZ`` of the object returned here.
"""

import numpy as np


def log_importance_weights(log_likelihood_new, log_prior_correction=None):
    """Return the log weight update for an importance-sampling step.

    Parameters
    ----------
    log_likelihood_new : array_like
        Target-model log likelihood for each sample, already expressed relative
        to the base likelihood (that is, ``log L_new - log L_base``), or the
        absolute target log likelihood when the caller forms the ratio itself.
    log_prior_correction : array_like, optional
        Summed ``log(prior / proposal)`` for the new or changed parameters, as
        produced by :meth:`cosmocompare.models.TargetModel.log_prior_correction`
        or :func:`cosmocompare.prior.prior_log_ratio`. Treated as zero when
        omitted.

    Returns
    -------
    numpy.ndarray
        ``log_likelihood_new + log_prior_correction``, the per-sample log factor
        multiplying the base posterior weight.
    """
    log_likelihood_new = np.asarray(log_likelihood_new, dtype=float)
    if log_prior_correction is None:
        return log_likelihood_new
    return log_likelihood_new + np.asarray(log_prior_correction, dtype=float)


def apply_log_weights(samples, log_weights):
    """Return a copy of ``samples`` with weights multiplied by ``exp(log_weights)``.

    Parameters
    ----------
    samples : anesthetic samples
        Samples exposing ``get_weights``, ``set_weights`` and ``copy``.
    log_weights : array_like
        Per-sample log weight update. ``-inf`` entries (for example a sample
        outside a narrowed prior) map to zero weight.

    Returns
    -------
    anesthetic samples
        A weighted posterior whose weights are ``W_i * exp(log_weights_i)``. A
        ``NestedSamples`` input is returned as ``MCMCSamples``: no nested recompute
        is performed, so its nested ``logZ`` would be meaningless, and casting
        removes that method so it cannot be called by mistake. Take the evidence
        from :mod:`cosmocompare.evidence` instead.

    Notes
    -----
    The weights are shifted by the finite maximum of ``log_weights`` before
    exponentiating, which cancels in the normalisation and keeps the exponential
    in range.
    """
    log_weights = np.asarray(log_weights, dtype=float)
    base = np.asarray(samples.get_weights(), dtype=float)
    finite = np.isfinite(log_weights)
    scale = np.max(log_weights[finite]) if np.any(finite) else 0.0
    new = np.where(finite, base * np.exp(log_weights - scale), 0.0)

    out = samples.copy()
    out.set_weights(new, inplace=True)

    # Returning a NestedSamples would carry a nested ``logZ`` that is invalid once
    # the weights no longer describe a nested sequence. Cast it to a plain weighted
    # posterior so that trap is removed rather than merely documented. If anesthetic
    # is not importable the object cannot be a NestedSamples (they only exist via
    # anesthetic), so there is nothing to cast.
    try:
        from anesthetic import MCMCSamples, NestedSamples
    except ImportError:
        return out

    if isinstance(out, NestedSamples):
        out = MCMCSamples(out)
    return out
