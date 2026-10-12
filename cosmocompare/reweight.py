r"""The importance-sampling weight update, with the full prior correction.

Switching a set of samples from a base model to an extended one multiplies each
weight by the likelihood ratio and by the ratio of the new parameters' prior to
the proposal they were drawn from:

.. math::

    w_\\text{new} = w_\\text{old} \\,
        \\frac{\\mathcal{L}_\\text{new}}{\\mathcal{L}_\\text{old}} \\,
        \\frac{\\Pi_\\text{new}}{q}.

Dropping the prior-over-proposal factor is a common shortcut; it is correct only
when the proposal happens to equal the target prior. This module always carries
the term - it is zero in that case and matters in every other - so narrowing a
prior or sampling from an informed proposal stays unbiased.
"""

import numpy as np


def log_importance_weights(log_likelihood_new, log_prior_correction=None):
    """Return the log weight update for an importance-sampling step.

    Parameters
    ----------
    log_likelihood_new : array_like
        Target-model log likelihood for each sample.
    log_prior_correction : array_like, optional
        Summed ``log(prior / proposal)`` for the new parameters, as produced by
        :meth:`cosmocompare.models.TargetModel.log_prior_correction`. Treated as
        zero when omitted.

    Returns
    -------
    numpy.ndarray
        The quantity to hand to a sampler that replaces the base log likelihood,
        namely ``log_likelihood_new + log_prior_correction``. Combined with the
        base weights by the sampler, this realises the full weight update.
    """
    log_likelihood_new = np.asarray(log_likelihood_new, dtype=float)
    if log_prior_correction is None:
        return log_likelihood_new
    return log_likelihood_new + np.asarray(log_prior_correction, dtype=float)


def importance_sample(base_samples, log_likelihood_new, log_prior_correction=None):
    """Reweight base samples onto an extended model.

    Parameters
    ----------
    base_samples : anesthetic samples
        Base-model samples exposing anesthetic's ``importance_sample`` method.
    log_likelihood_new : array_like
        Target-model log likelihood for each base sample.
    log_prior_correction : array_like, optional
        Summed ``log(prior / proposal)`` for the new parameters. Treated as zero
        when omitted.

    Returns
    -------
    anesthetic samples
        A new, reweighted samples object. The base object is left untouched.

    Notes
    -----
    The base likelihood is *replaced* rather than multiplied: the corrected log
    likelihood already contains the full ``log(L_new / L_old) + log(prior / q)``
    update once anesthetic forms the ratio against the stored base likelihood.
    """
    corrected = log_importance_weights(log_likelihood_new, log_prior_correction)
    return base_samples.importance_sample(corrected, action="replace", inplace=False)


def attach_new_parameters(posterior, augmented_samples, new_param_names, finite_mask):
    """Copy new-parameter columns onto a reweighted posterior.

    Reweighting may drop samples whose corrected likelihood is ``-inf``, so the
    new-parameter columns have to be aligned to whatever survived.

    Parameters
    ----------
    posterior : anesthetic samples
        The reweighted samples to annotate, modified in place.
    augmented_samples : pandas.DataFrame
        The pre-reweighting samples, carrying the new-parameter columns.
    new_param_names : iterable of str
        Names of the columns to copy across.
    finite_mask : numpy.ndarray
        Boolean mask, over the original samples, of those with finite corrected
        likelihood.

    Returns
    -------
    anesthetic samples
        The same ``posterior`` object, for convenience.
    """
    n_result = len(posterior)
    n_total = len(augmented_samples)
    n_finite = int(np.sum(finite_mask))

    for name in new_param_names:
        values = augmented_samples[name].to_numpy()
        if n_result == n_total:
            posterior[name] = values
        elif n_result == n_finite:
            posterior[name] = values[finite_mask]
        else:
            posterior[name] = values[:n_result]
    return posterior
