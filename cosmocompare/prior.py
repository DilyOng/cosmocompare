r"""Re-weight an existing nested run onto changed priors for existing parameters.

The common, cheap case of model comparison is to keep the model and the data but
change the prior: narrow a range, or swap a flat prior for an informed one. The
likelihood is then unchanged, so no sampler and no likelihood re-evaluation are
needed. Each existing sample is simply re-weighted by the ratio of the new prior
to the original one at its own coordinate, and the evidence follows from the
importance-sampling identity

.. math::

    \\frac{Z_\\text{new}}{Z_\\text{old}}
        = \\frac{\\sum_i W_i\\, r_i}{\\sum_i W_i}, \\qquad
    r_i = \\prod_k \\frac{\\pi_\\text{new}^{(k)}(\\phi_i^{(k)})}
                        {\\pi_\\text{old}^{(k)}(\\phi_i^{(k)})},

with ``W_i`` the base posterior weights. For a uniform prior narrowed from width
``b-a`` to ``b'-a'`` this reduces to ``Z_new/Z_old = (b-a)/(b'-a') * f``, where
``f`` is the posterior mass that falls inside the narrower range.

This is the one route in the package that does not lean on anesthetic's nested
``logZ`` of a re-weighted run, which is not valid once the samples no longer form
a nested sequence under the changed measure.
"""

import numpy as np

from cosmocompare.evidence import importance_evidence
from cosmocompare.parameters import _Distribution
from cosmocompare.reweight import apply_log_weights


def prior_log_ratio(samples, changes):
    """Return the summed ``log(prior_new / prior_old)`` per sample.

    Parameters
    ----------
    samples : anesthetic samples
        The base run, carrying a column for each parameter in ``changes``.
    changes : dict
        Mapping ``{name: {"base": <prior cfg>, "new": <prior cfg>}}`` where each
        prior config is in the form understood by
        :class:`cosmocompare.parameters._Distribution` (for example
        ``{"type": "uniform", "min": -0.3, "max": 0.3}``). ``"base"`` is the
        prior the chain was produced under; ``"new"`` is the target prior.

    Returns
    -------
    numpy.ndarray
        Per-sample log prior ratio, ``-inf`` wherever a sample falls outside the
        new prior's support.
    """
    total = np.zeros(len(samples))
    for name, cfg in changes.items():
        values = np.asarray(samples[name], dtype=float)
        old = _Distribution(cfg["base"])
        new = _Distribution(cfg["new"])
        total = total + (new.logpdf(values) - old.logpdf(values))
    return total


def reweight_prior(samples, changes, nsamples=1000, n_bootstrap=1000, seed=None):
    """Re-weight a nested run onto changed priors and recompute the evidence.

    Parameters
    ----------
    samples : anesthetic.NestedSamples
        The base run (read with ``anesthetic.read_chains``), with a usable
        ``logZ`` and posterior weights.
    changes : dict
        Prior changes, as documented on :func:`prior_log_ratio`.
    nsamples : int, optional
        Number of ``log Z`` draws used for the base evidence and its error.
        Defaults to 1000.
    n_bootstrap : int, optional
        Number of bootstrap resamples for the ratio error. Defaults to 1000.
    seed : int, optional
        Seed for the bootstrap, for reproducibility.

    Returns
    -------
    evidence : cosmocompare.evidence.EvidenceResult
        The absolute ``log Z_new`` and the ``log(Z_new / Z_old)`` ratio, each
        with an uncertainty, plus the effective sample size. The bootstrap error
        is a reliability diagnostic, not a rigorous frequentist interval.
    posterior : anesthetic samples
        The base samples re-weighted to the new prior (weights multiplied by the
        prior ratio; samples outside the new support carry zero weight). Use it
        for parameter estimation under the new prior. Do not read ``logZ`` off
        it: the evidence is the returned ``EvidenceResult``.
    """
    log_r = prior_log_ratio(samples, changes)
    base_logL = np.asarray(samples.logL, dtype=float)
    base_weights = np.asarray(samples.get_weights(), dtype=float)

    draws = np.asarray(samples.logZ(nsamples), dtype=float)
    evidence = importance_evidence(
        base_logL,
        base_logL,
        base_weights=base_weights,
        log_prior_correction=log_r,
        base_log_evidence=float(draws.mean()),
        base_log_evidence_err=float(draws.std()),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Re-weighted posterior without touching the nested geometry: the weights are
    # multiplied by the prior ratio; samples outside the new support get zero.
    posterior = apply_log_weights(samples, log_r)

    return evidence, posterior
