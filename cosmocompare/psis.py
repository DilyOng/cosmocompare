r"""Pareto-smoothed importance sampling (PSIS) reliability diagnostic.

The Kish effective sample size (:func:`cosmocompare.diagnostics.effective_sample_size`)
measures weight degeneracy, but it is blind to a heavy *tail*: a few samples with
runaway importance weight can give an estimator with infinite variance while the
Kish number still looks acceptable. Pareto-smoothed importance sampling (Vehtari,
Simpson, Gelman, Yao & Gabry 2024, *JMLR* 25(72)) fits a generalised Pareto
distribution to the upper tail of the importance weights and reports its shape
parameter :math:`\hat k`. The rule of thumb is

- :math:`\hat k < 0.5`   : the variance is finite, the estimate is reliable;
- :math:`0.5 \le \hat k < 0.7` : usable but with higher variance;
- :math:`\hat k \ge 0.7`  : the estimate is not to be trusted.

For reweighting an existing run onto a changed prior, the quantity whose tail
governs the estimator's risk is the *combined* weight ``W_i * (pi_new/pi_old)_i``
(the base posterior weight times the per-sample prior ratio). :math:`\hat k` is
computed on that, complementing the Kish number rather than replacing it. The
generalised-Pareto fit uses the Zhang & Stephens (2009) empirical-Bayes
estimator, the same method used by ``loo``/``arviz``.

.. note::

   For a *hard-truncated* prior the importance weights are not heavy-tailed, they
   are a spike of equal in-region weights and a mass of exact zeros, so there is
   no Pareto tail to fit and :math:`\hat k` is not meaningful. Use the Kish ESS
   and the support-containment guarantee for hard cuts; use :math:`\hat k` for
   smooth reweightings (Gaussian shrink, power-scaling, theory-density priors).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PSISResult:
    """Outcome of a PSIS reliability check.

    Attributes
    ----------
    pareto_k : float
        Fitted generalised-Pareto shape parameter of the weight tail. ``nan`` if
        the tail is too small to fit.
    ess : float
        Effective sample size from the Pareto-smoothed, normalised weights.
    n_tail : int
        Number of samples used to fit the tail.
    reliable : bool
        ``True`` when ``pareto_k < 0.7``.
    """

    pareto_k: float
    ess: float
    n_tail: int
    reliable: bool


def _gpdfit(exceedances):
    """Fit a generalised Pareto distribution to positive exceedances.

    Uses the Zhang & Stephens (2009) empirical-Bayes estimator, as implemented in
    ``arviz``/``loo``. ``exceedances`` must be sorted ascending and strictly
    positive (tail values minus the threshold).

    Parameters
    ----------
    exceedances : numpy.ndarray
        Sorted positive tail exceedances.

    Returns
    -------
    tuple of float
        ``(k, sigma)``, the fitted shape and scale.
    """
    prior_bs = 3.0
    prior_k = 10.0
    n = exceedances.size
    m_est = 30 + int(n**0.5)

    b_ary = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * exceedances[int(n / 4 + 0.5) - 1]
    b_ary += 1.0 / exceedances[-1]

    k_ary = np.log1p(-b_ary[:, None] * exceedances).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1.0)
    weights = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    real = weights >= 10 * np.finfo(float).eps
    if not np.all(real):
        weights = weights[real]
        b_ary = b_ary[real]
    # If every candidate underflowed, the fit has no information; signal a failed
    # fit with NaN rather than dividing by a zero-sum weight vector.
    total = weights.sum()
    if weights.size == 0 or not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan")
    weights /= total

    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * exceedances).mean()
    sigma = -k_post / b_post
    # Weakly-informative prior on k pulling the small-sample estimate towards 0.5.
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return float(k_post), float(sigma)


def pareto_smooth(log_weights):
    """Pareto-smooth a set of log importance weights.

    Parameters
    ----------
    log_weights : array_like
        Log importance weights (any additive constant is irrelevant). ``-inf``
        entries (zero weight) are carried through unchanged and excluded from the
        tail fit.

    Returns
    -------
    tuple
        ``(smoothed_log_weights, pareto_k, n_tail)``. ``pareto_k`` is ``nan``
        when there are too few finite weights to fit a tail.
    """
    log_weights = np.array(log_weights, dtype=float)
    finite_idx = np.where(np.isfinite(log_weights))[0]
    n = finite_idx.size

    # Too few finite weights to define a tail; nothing to smooth.
    if n < 5:
        return log_weights, float("nan"), 0

    # Standard PSIS tail size: min(0.2 S, 3 sqrt(S)).
    n_tail = int(min(0.2 * n, 3.0 * np.sqrt(n)))
    if n_tail < 5:
        return log_weights, float("nan"), n_tail

    lw = log_weights[finite_idx]
    order = np.argsort(lw)
    tail_local = order[-n_tail:]
    cutoff = lw[order[-n_tail - 1]]  # largest value NOT in the tail (the threshold)

    tail_lw = lw[tail_local]
    # Work in the linear domain on exceedances above exp(cutoff), shifting by the
    # max to keep the exponential in range.
    shift = tail_lw.max()
    exceed = np.exp(tail_lw - shift) - np.exp(cutoff - shift)
    sorted_order = np.argsort(exceed)
    exceed_sorted = exceed[sorted_order]

    if exceed_sorted[0] <= 0 or not np.all(np.isfinite(exceed_sorted)):
        return log_weights, float("nan"), n_tail

    k, sigma = _gpdfit(exceed_sorted)
    if not (np.isfinite(k) and np.isfinite(sigma)):
        return log_weights, float("nan"), n_tail

    # Replace the tail with the expected GPD order-statistic quantiles. This is the
    # generalised-Pareto inverse CDF in the same shape convention `_gpdfit` returns
    # (identical to arviz/loo's `_gpinv`: sigma * expm1(-k log1p(-p)) / k, with the
    # k -> 0 limit handled separately), so fit and inverse use one consistent k.
    p = (np.arange(1, n_tail + 1) - 0.5) / n_tail
    if np.abs(k) < 1e-30:
        q = sigma * (-np.log1p(-p))
    else:
        q = sigma * np.expm1(-k * np.log1p(-p)) / k
    smoothed_tail = np.log(q + np.exp(cutoff - shift)) + shift
    # Never exceed the largest raw weight.
    smoothed_tail = np.minimum(smoothed_tail, tail_lw.max())

    out = log_weights.copy()
    # Map the sorted-by-quantile smoothed values back to the samples they replace,
    # preserving the tail's rank order.
    tail_global = finite_idx[tail_local]
    out[tail_global[np.argsort(tail_lw)]] = smoothed_tail
    return out, k, n_tail


def _ess_from_log_weights(log_weights):
    """Return the Kish ESS of normalised weights given in log space."""
    finite = log_weights[np.isfinite(log_weights)]
    if finite.size == 0:
        return 0.0
    w = np.exp(finite - finite.max())
    s = w.sum()
    if s == 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w**2))


def psis(log_weights):
    """Pareto-smooth log importance weights and return the reliability result.

    Parameters
    ----------
    log_weights : array_like
        Log importance weights (for a prior reweight, ``log(W_i) + log r_i``).

    Returns
    -------
    PSISResult
        The fitted tail shape, the smoothed-weight ESS, the tail size, and
        whether the estimate is reliable (``pareto_k < 0.7``).
    """
    smoothed, k, n_tail = pareto_smooth(log_weights)
    ess = _ess_from_log_weights(smoothed)
    reliable = bool(np.isfinite(k) and k < 0.7)
    return PSISResult(pareto_k=k, ess=ess, n_tail=n_tail, reliable=reliable)


def psis_diagnostic(samples, log_ratio):
    """PSIS reliability of reweighting ``samples`` by a per-sample log prior ratio.

    The combined importance weight is the base posterior weight times the prior
    ratio, ``W_i * exp(log_ratio_i)``; its tail is what PSIS examines.

    Parameters
    ----------
    samples : anesthetic samples
        The base run, exposing ``get_weights``.
    log_ratio : array_like
        Per-sample ``log(pi_new / pi_old)`` (``-inf`` outside the new support),
        as from :func:`cosmocompare.prior.prior_log_ratio`.

    Returns
    -------
    PSISResult
        The PSIS reliability result for this reweighting.
    """
    base = np.asarray(samples.get_weights(), dtype=float)
    with np.errstate(divide="ignore"):
        log_combined = np.log(base) + np.asarray(log_ratio, dtype=float)
    return psis(log_combined)
