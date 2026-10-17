r"""Bayesian evidence for an extended model, reused from existing chains.

This is the deliverable the rest of the package exists to produce. Reweighting
turns base-model samples into target-model samples; the evidence turns them into
a verdict.

:func:`importance_evidence` is the estimator for every re-weighting case: it
forms the evidence *ratio* between the target and base models directly from the
weights and attaches a bootstrap uncertainty, needing nothing beyond the arrays.
:func:`nested_evidence` is only a thin reader of ``logZ`` from a *genuine* nested
run (for example to obtain the base ``log Z_old``); it must not be applied to a
re-weighted run, whose nested geometry no longer holds.

The evidence ratio estimator is

.. math::

    \\frac{Z_\\text{new}}{Z_\\text{old}}
        = \\frac{\\sum_i W_i\\, w_i}{\\sum_i W_i}, \\qquad
    w_i = \\frac{\\mathcal{L}_{\\text{new},i}}{\\mathcal{L}_{\\text{old},i}}
          \\frac{\\Pi_\\text{new}}{q},

with ``W_i`` the base posterior weights.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from cosmocompare.diagnostics import effective_sample_size


@dataclass
class EvidenceResult:
    """Outcome of an evidence-ratio estimate.

    Attributes
    ----------
    log_ratio : float
        ``log(Z_new / Z_old)``.
    log_ratio_err : float
        Bootstrap standard deviation of :attr:`log_ratio`.
    log_evidence : float or None
        Absolute ``log Z_new``, present only when a base ``log Z_old`` was
        supplied.
    log_evidence_err : float or None
        Uncertainty on :attr:`log_evidence`, combining the bootstrap error with
        the base evidence error in quadrature.
    effective_sample_size : float
        Kish effective sample size of the combined weights, a guide to how far
        the estimate can be trusted.
    bootstrap_degenerate_fraction : float or None
        Fraction of bootstrap resamples that contained no finite-weight sample
        and so returned a degenerate (``-inf``) ratio. A non-negligible value
        means the sample is too starved for the bootstrap error to be trusted;
        it rises in the tight-prior regime and should be read alongside
        :attr:`effective_sample_size`.
    """

    log_ratio: float
    log_ratio_err: float
    log_evidence: float = None
    log_evidence_err: float = None
    effective_sample_size: float = None
    bootstrap_degenerate_fraction: float = None


def _log_ratio(log_base_weights, log_sample_weights):
    """Return ``log(sum W w / sum W)`` stably.

    Parameters
    ----------
    log_base_weights : numpy.ndarray
        Log base posterior weights ``log W_i``.
    log_sample_weights : numpy.ndarray
        Log importance weights ``log w_i``.

    Returns
    -------
    float
        The log evidence ratio.
    """
    return float(
        logsumexp(log_base_weights + log_sample_weights) - logsumexp(log_base_weights)
    )


def importance_evidence(
    log_likelihood_new,
    log_likelihood_base,
    base_weights=None,
    log_prior_correction=None,
    base_log_evidence=None,
    base_log_evidence_err=0.0,
    n_bootstrap=1000,
    seed=None,
):
    """Estimate the target-model evidence ratio with a bootstrap error.

    Parameters
    ----------
    log_likelihood_new : array_like
        Target-model log likelihood per base sample.
    log_likelihood_base : array_like
        Base-model log likelihood per base sample (the value the samples were
        generated under).
    base_weights : array_like, optional
        Base posterior weights ``W_i``; need not be normalised. Uniform weights
        are assumed when omitted.
    log_prior_correction : array_like, optional
        Summed ``log(prior / proposal)`` for the new parameters. Treated as zero
        when omitted.
    base_log_evidence : float, optional
        Base-model ``log Z_old``. When given, the absolute target evidence is
        returned as well.
    base_log_evidence_err : float, optional
        Uncertainty on ``base_log_evidence``, combined in quadrature with the
        bootstrap error. Defaults to 0.
    n_bootstrap : int, optional
        Number of bootstrap resamples. Defaults to 1000.
    seed : int, optional
        Seed for the bootstrap resampling, for reproducibility.

    Returns
    -------
    EvidenceResult
        The evidence ratio, its bootstrap error, the absolute evidence when
        available, and the effective sample size.
    """
    log_likelihood_new = np.asarray(log_likelihood_new, dtype=float)
    log_likelihood_base = np.asarray(log_likelihood_base, dtype=float)

    log_w = log_likelihood_new - log_likelihood_base
    if log_prior_correction is not None:
        log_w = log_w + np.asarray(log_prior_correction, dtype=float)

    n = log_w.size
    if base_weights is None:
        log_base = np.zeros(n)
    else:
        base_weights = np.asarray(base_weights, dtype=float)
        with np.errstate(divide="ignore"):
            log_base = np.log(base_weights)

    log_ratio = _log_ratio(log_base, log_w)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = _log_ratio(log_base[idx], log_w[idx])
    # Under a hard-narrowed prior only a few samples carry finite weight, so some
    # bootstrap resamples contain none of them and the ratio collapses to -inf.
    # A plain np.std over those is inf/NaN and would silently poison every error
    # bar in a sweep, so the spread is taken over the finite replicates only and
    # the fraction that degenerated is reported separately as a reliability flag.
    finite_boots = boots[np.isfinite(boots)]
    degenerate_fraction = float(1.0 - finite_boots.size / n_bootstrap)
    if np.isfinite(log_ratio) and finite_boots.size >= 2:
        log_ratio_err = float(np.std(finite_boots))
    else:
        log_ratio_err = float("nan")

    # Effective sample size of the combined (base x importance) weights, in the
    # linear domain. Shifting by the maximum keeps the exponential in range.
    max_log = np.max(log_base + log_w)
    if np.isfinite(max_log):
        ess = effective_sample_size(np.exp(log_base + log_w - max_log))
    else:
        ess = 0.0

    result = EvidenceResult(
        log_ratio=log_ratio,
        log_ratio_err=log_ratio_err,
        effective_sample_size=ess,
        bootstrap_degenerate_fraction=degenerate_fraction,
    )
    if base_log_evidence is not None:
        result.log_evidence = base_log_evidence + log_ratio
        result.log_evidence_err = float(np.hypot(log_ratio_err, base_log_evidence_err))
    return result


def nested_evidence(samples, nsamples=1000):
    """Read ``log Z`` (with error) off a genuine nested-sampling run.

    .. warning::

       Use only on a real nested run - for instance to obtain the base
       ``log Z_old``. It is **not** valid on a re-weighted run of any kind
       (added parameter, changed prior, or even a like-for-like likelihood swap):
       anesthetic re-sorts and recomputes the prior-volume geometry from the base
       run's birth contours, which no longer describe the changed measure. For
       any re-weighting, use :func:`importance_evidence` (or
       :func:`cosmocompare.prior.reweight_prior`).

    Parameters
    ----------
    samples : anesthetic.NestedSamples
        A genuine nested-sampling run.
    nsamples : int, optional
        Number of posterior draws of ``log Z`` used to estimate the
        nested-sampling uncertainty. Defaults to 1000.

    Returns
    -------
    tuple of float
        ``(log_evidence, log_evidence_err)``, the mean and standard deviation of
        the ``log Z`` draws.
    """
    draws = np.asarray(samples.logZ(nsamples), dtype=float)
    return float(draws.mean()), float(draws.std())


def log_bayes_factor(evidence_a, evidence_b):
    """Return ``log(Z_a / Z_b)`` and its error for two models.

    Parameters
    ----------
    evidence_a, evidence_b : EvidenceResult or tuple of float
        Either :class:`EvidenceResult` instances (absolute evidence required) or
        ``(log_evidence, log_evidence_err)`` tuples.

    Returns
    -------
    tuple of float
        ``(log_bayes_factor, error)``, the error being the two evidence errors
        added in quadrature.
    """

    def _unpack(e):
        if isinstance(e, EvidenceResult):
            if e.log_evidence is None:
                raise ValueError(
                    "EvidenceResult has no absolute log_evidence; supply "
                    "base_log_evidence to importance_evidence first."
                )
            return e.log_evidence, e.log_evidence_err or 0.0
        return float(e[0]), float(e[1])

    logz_a, err_a = _unpack(evidence_a)
    logz_b, err_b = _unpack(evidence_b)
    return logz_a - logz_b, float(np.hypot(err_a, err_b))
