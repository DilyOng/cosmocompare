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

Two conditions apply, the first enforced, the second assumed:

- The new prior's support must lie within the base prior's support. Widening a
  bounded prior reaches regions the chain never sampled and is refused.
- The prior change factorises over parameters (a sum of one-dimensional log
  ratios); correlated prior changes are not supported.
"""

import numpy as np

from cosmocompare.evidence import importance_evidence
from cosmocompare.parameters import _Distribution, check_support_contained
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
        # The new prior must not reach beyond the old one: those regions were
        # never sampled, so reweighting there would silently bias the evidence.
        check_support_contained(new, old, name)

        old_logpdf = old.logpdf(values)
        # Every existing sample must have finite density under the declared base
        # prior - it was, after all, drawn under it. A -inf here means the stated
        # base prior does not match the chain, which would corrupt the ratio
        # (-inf - -inf = NaN) silently, so it is rejected rather than trusted.
        if np.any(np.isneginf(old_logpdf)):
            raise ValueError(
                f"The declared base prior for {name!r} assigns zero probability to "
                f"some existing samples. It does not match the prior the chain was "
                f"produced under; check the 'base' bounds against the chain."
            )
        total = total + (new.logpdf(values) - old_logpdf)
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
    return _evidence_and_posterior(samples, log_r, nsamples, n_bootstrap, seed)


def _evidence_and_posterior(samples, log_r, nsamples, n_bootstrap, seed):
    """Turn a per-sample log prior ratio into an evidence and re-weighted posterior.

    Shared back end for every prior-change route: the evidence is the importance
    ratio estimator on the fixed base evidence, and the posterior is the base
    weights multiplied by the ratio (no nested recompute).

    Parameters
    ----------
    samples : anesthetic.NestedSamples
        The base run.
    log_r : numpy.ndarray
        Per-sample ``log(prior_new / prior_old)``.
    nsamples : int
        ``log Z`` draws for the base evidence and its error.
    n_bootstrap : int
        Bootstrap resamples for the ratio error.
    seed : int or None
        Seed for the bootstrap.

    Returns
    -------
    tuple
        ``(EvidenceResult, posterior)`` as documented on :func:`reweight_prior`.
    """
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
    posterior = apply_log_weights(samples, log_r)
    return evidence, posterior


def reweight_density(
    samples,
    log_prior_new,
    log_prior_old=None,
    nsamples=1000,
    n_bootstrap=1000,
    seed=None,
):
    """Re-weight onto an arbitrary, normalised joint prior given as a callable.

    This is the general route for priors that do not factorise over parameters -
    correlated priors, non-rectangular regions, or externally-supplied densities
    (for example a normalising-flow prior). The caller is responsible for two
    things: the densities being correctly normalised, and the new prior's support
    lying within the region the chain actually sampled (reweighting cannot reach
    where there are no samples - truncate and renormalise an unbounded density to
    the base prior's support first). ``reweight_region`` handles the common
    uniform-on-a-region case, including its normalisation, for you.

    Parameters
    ----------
    samples : anesthetic.NestedSamples
        The base run.
    log_prior_new : callable
        ``log_prior_new(samples) -> array`` of length ``len(samples)``, the log
        of the (normalised) target prior density at each sample.
    log_prior_old : callable, optional
        The base prior's log density. Omit when the chain's base prior is uniform
        over the sampled region: a constant then cancels in the ratio and only
        the shape of ``log_prior_new`` matters.
    nsamples, n_bootstrap, seed
        As in :func:`reweight_prior`.

    Returns
    -------
    tuple
        ``(EvidenceResult, posterior)`` as documented on :func:`reweight_prior`.
    """
    log_new = np.asarray(log_prior_new(samples), dtype=float)
    if log_prior_old is None:
        log_r = log_new
    else:
        log_r = log_new - np.asarray(log_prior_old(samples), dtype=float)
    return _evidence_and_posterior(samples, log_r, nsamples, n_bootstrap, seed)


def region_volume_fraction(inside, base_bounds, n_mc=1_000_000, rng=None):
    """Estimate the fraction of a uniform box that lies within a region.

    Parameters
    ----------
    inside : callable
        ``inside(array)`` where ``array`` has shape ``(n, d)``; returns a boolean
        array marking points within the region.
    base_bounds : sequence of tuple
        ``(low, high)`` for each of the ``d`` dimensions.
    n_mc : int, optional
        Number of Monte-Carlo draws. Defaults to 1e6.
    rng : numpy.random.Generator, optional
        Source of randomness.

    Returns
    -------
    float
        The Monte-Carlo estimate of ``V_region / V_box`` in ``[0, 1]``.
    """
    rng = np.random.default_rng() if rng is None else rng
    bounds = np.asarray(base_bounds, dtype=float)
    draws = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_mc, len(bounds)))
    return float(np.mean(inside(draws)))


def reweight_region(
    samples,
    params,
    inside,
    base_bounds,
    n_mc=1_000_000,
    rng=None,
    nsamples=1000,
    n_bootstrap=1000,
    seed=None,
):
    """Re-weight onto a uniform prior restricted to an arbitrary region.

    The new prior is uniform over the sub-region ``inside`` of the original
    uniform box ``base_bounds`` (for example the non-phantom region
    ``w0 + wa >= -1``). The region's prior volume is estimated by Monte-Carlo
    integration over the base box, giving the correct normalisation: a sample in
    the region is up-weighted by ``V_base / V_region``, one outside gets zero.

    Parameters
    ----------
    samples : anesthetic.NestedSamples
        The base run.
    params : sequence of str
        Parameter names spanning the region, in the order ``inside`` expects.
    inside : callable
        ``inside(array)`` where ``array`` has shape ``(n, len(params))``; returns
        a boolean array marking points within the region.
    base_bounds : sequence of tuple
        ``(low, high)`` for each parameter in ``params`` - the original uniform
        prior bounds.
    n_mc : int, optional
        Monte-Carlo draws for the region-volume fraction. Defaults to 1e6.
    rng : numpy.random.Generator, optional
        Source of randomness for the volume estimate.
    nsamples, n_bootstrap, seed
        As in :func:`reweight_prior`.

    Returns
    -------
    tuple
        ``(EvidenceResult, posterior)`` as documented on :func:`reweight_prior`.

    Raises
    ------
    ValueError
        If the region is empty (no Monte-Carlo draw falls inside).
    """
    frac = region_volume_fraction(inside, base_bounds, n_mc=n_mc, rng=rng)
    if frac <= 0:
        raise ValueError("Region contains no prior volume (empty region).")

    values = np.column_stack([np.asarray(samples[p], dtype=float) for p in params])
    in_region = np.asarray(inside(values), dtype=bool)
    # Uniform-on-region density / uniform-on-box density = V_base/V_region = 1/frac.
    log_r = np.where(in_region, -np.log(frac), -np.inf)

    return _evidence_and_posterior(samples, log_r, nsamples, n_bootstrap, seed)
