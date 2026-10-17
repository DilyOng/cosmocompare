r"""Map how the evidence and Bayes factor move as a prior is narrowed.

Bayesian evidence depends on the prior volume through the Ockham penalty, so a
model-comparison verdict can shift as the prior bounds are drawn more tightly.
This module sweeps an existing nested run over a family of progressively narrower
priors on chosen parameters and records the evidence at each step, answering the
practical question: *how narrow must the prior be before the verdict flips?*

For comparing an extended model (for example ``w0waCDM``, which adds ``w0, wa``)
against a base model (``LCDM``), only the extended model's evidence moves when its
extra parameters' priors are narrowed - the base model has no such parameters, so
its evidence is fixed. The log Bayes factor is therefore

.. math::

    \ln B(f) = \ln Z_\text{ext}(\text{prior narrowed by } f) - \ln Z_\text{base},

computed at zero sampling cost with :func:`cosmocompare.prior.reweight_prior`.

The narrowing is *homothetic* about a fixed centre: each bound moves as
``x_f = c + f (x_0 - c)``, so ``f = 1`` recovers the original prior, ``f < 1``
stays a sub-box, and the prior volume scales as ``f^(number of parameters)``.

**The centre must be chosen independently of the data.** Centring on the
posterior and shrinking would raise the extended evidence by construction -
data-dependent prior tuning that can flip any verdict trivially, and is not a
valid evidential test. The defensible choice is a fixed, data-independent centre
(for a dark-energy comparison, the LCDM point ``w0 = -1, wa = 0``).
:func:`posterior_centres` (the posterior-weighted mean) is provided only for an
explicitly-labelled "optimistic upper bound on flip-ability", never for an
evidential claim.
"""

import numpy as np

from cosmocompare.evidence import importance_evidence, nested_evidence
from cosmocompare.prior import prior_log_ratio


def posterior_centres(samples, params):
    """Return the posterior-weighted mean of each parameter.

    Parameters
    ----------
    samples : anesthetic samples
        The run whose posterior defines the centres.
    params : iterable of str
        Parameter names.

    Returns
    -------
    dict
        ``{name: weighted_mean}``, suitable as ``centres`` for a sweep that
        narrows towards the data-preferred region.
    """
    weights = np.asarray(samples.get_weights(), dtype=float)
    total = weights.sum()
    centres = {}
    for name in params:
        values = np.asarray(samples[name], dtype=float)
        centres[name] = float(np.sum(weights * values) / total)
    return centres


def shrink_uniform(base, fraction, centre=None):
    """Return a uniform prior homothetically shrunk about a fixed centre.

    Each bound moves as ``x_f = centre + fraction * (x_0 - centre)``, so the
    result is always contained within ``base`` (provided the centre lies inside
    it) and has width ``fraction`` times the original.

    Parameters
    ----------
    base : tuple of float
        The original ``(low, high)`` bounds.
    fraction : float
        Width of the new interval as a fraction of the original, in ``(0, 1]``.
    centre : float, optional
        The fixed point the interval shrinks towards. Defaults to the midpoint of
        ``base``. Must lie within ``base``; it should be chosen independently of
        the data (see the module docstring).

    Returns
    -------
    dict
        A ``{"type": "uniform", "min": .., "max": ..}`` config whose support lies
        within ``base``.
    """
    lo, hi = float(base[0]), float(base[1])
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    if centre is None:
        centre = 0.5 * (lo + hi)
    if not lo <= centre <= hi:
        raise ValueError(f"centre {centre} must lie within base [{lo}, {hi}].")

    new_lo = centre + fraction * (lo - centre)
    new_hi = centre + fraction * (hi - centre)
    return {"type": "uniform", "min": new_lo, "max": new_hi}


def _changes(param_bases, fraction, centres):
    """Build a prior-change dict narrowing every parameter by ``fraction``.

    Parameters
    ----------
    param_bases : dict
        ``{name: (low, high)}`` original (uniform) prior bounds.
    fraction : float
        Shared width fraction for every parameter.
    centres : dict or None
        Per-parameter centres, or None for midpoints.

    Returns
    -------
    dict
        A ``changes`` mapping for :func:`cosmocompare.prior.prior_log_ratio`.
    """
    centres = centres or {}
    changes = {}
    for name, base in param_bases.items():
        changes[name] = {
            "base": {"type": "uniform", "min": float(base[0]), "max": float(base[1])},
            "new": shrink_uniform(base, fraction, centres.get(name)),
        }
    return changes


def bayes_factor_sweep(
    extended,
    base,
    param_bases,
    fractions,
    centres=None,
    nsamples=1000,
    n_bootstrap=1000,
    seed=None,
):
    """Sweep the extended-model prior and record the Bayes factor at each width.

    Parameters
    ----------
    extended : anesthetic.NestedSamples
        The extended-model run (for example ``w0waCDM``), carrying the parameters
        in ``param_bases``.
    base : anesthetic.NestedSamples
        The base-model run (for example ``LCDM``); its evidence is fixed.
    param_bases : dict
        ``{name: (low, high)}`` original prior bounds of the extended model's
        extra parameters (for example ``{"w0": (-3, 1), "wa": (-3, 2)}``).
    fractions : iterable of float
        Width fractions to test, each in ``(0, 1]`` (``1.0`` reproduces the
        original prior and should recover the published Bayes factor).
    centres : dict or str, optional
        Fixed point each parameter's prior shrinks towards. A dict
        ``{name: value}`` sets a data-independent centre (the defensible choice,
        for example the LCDM point ``{"w0": -1, "wa": 0}``); None uses the prior
        midpoints. ``"posterior"`` centres on the posterior-weighted mean, which
        is data-dependent tuning - use it only for an explicitly-labelled upper
        bound on flip-ability, never for an evidential result.
    nsamples : int, optional
        ``log Z`` draws for the base/extended evidence errors. Defaults to 1000.
    n_bootstrap : int, optional
        Bootstrap resamples per cell. Defaults to 1000.
    seed : int, optional
        Base seed; cell ``k`` uses ``seed + k`` for reproducibility.

    Returns
    -------
    list of dict
        One record per fraction, each with ``fraction``, ``bounds`` (the narrowed
        prior per parameter), ``log_evidence_ext``, ``log_bayes_factor``,
        ``log_bayes_factor_err``, and ``effective_sample_size``.

    Notes
    -----
    The extended and base runs' broad ``log Z`` are each estimated once and shared
    across the sweep: they are common offsets, not per-fraction quantities, so
    re-drawing them per cell would only inject noise into the curve. Only the
    prior-reweighting log ratio (and its bootstrap error) varies with the
    fraction. ``log_bayes_factor_err`` combines that bootstrap error with both
    runs' ``log Z`` errors in quadrature.
    """
    base_logZ, base_logZ_err = nested_evidence(base, nsamples)
    ext_logZ, ext_logZ_err = nested_evidence(extended, nsamples)
    ext_logL = np.asarray(extended.logL, dtype=float)
    ext_weights = np.asarray(extended.get_weights(), dtype=float)
    offset_err = float(np.hypot(ext_logZ_err, base_logZ_err))

    if centres == "posterior":
        centres = posterior_centres(extended, param_bases.keys())

    records = []
    for k, fraction in enumerate(fractions):
        changes = _changes(param_bases, fraction, centres)
        cell_seed = None if seed is None else seed + k
        log_r = prior_log_ratio(extended, changes)
        ratio = importance_evidence(
            ext_logL,
            ext_logL,
            base_weights=ext_weights,
            log_prior_correction=log_r,
            n_bootstrap=n_bootstrap,
            seed=cell_seed,
        )
        log_evidence_ext = ext_logZ + ratio.log_ratio
        log_b = log_evidence_ext - base_logZ
        log_b_err = float(np.hypot(ratio.log_ratio_err, offset_err))
        records.append(
            {
                "fraction": float(fraction),
                "bounds": {
                    n: (c["new"]["min"], c["new"]["max"]) for n, c in changes.items()
                },
                "log_evidence_ext": log_evidence_ext,
                "log_bayes_factor": log_b,
                "log_bayes_factor_err": log_b_err,
                "effective_sample_size": ratio.effective_sample_size,
            }
        )
    return records


def flip_fraction(records):
    """Return the width fraction at which the Bayes factor crosses zero.

    Parameters
    ----------
    records : list of dict
        Output of :func:`bayes_factor_sweep`.

    Returns
    -------
    float or None
        The ``fraction`` where ``log_bayes_factor`` reaches zero, by exact match
        or linear interpolation of a sign change. None if no crossing is found
        over the swept range (including a crossing hidden by a non-finite value).
        When several crossings exist, the one at the largest fraction (widest
        prior) is returned.
    """
    ordered = sorted(records, key=lambda r: r["fraction"], reverse=True)
    # An exact zero at a sampled width is itself the crossing.
    for r in ordered:
        if r["log_bayes_factor"] == 0:
            return float(r["fraction"])
    for a, b in zip(ordered, ordered[1:]):
        fa, fb = a["log_bayes_factor"], b["log_bayes_factor"]
        # Skip any non-finite cell, not just NaN: an infinite Bayes factor (for
        # example a prior narrowed until no sample survives) would otherwise pass
        # the sign-change test and make the linear interpolation t = fa/(fa-fb)
        # return NaN or a spurious crossing.
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if (fa < 0) != (fb < 0):  # sign change between adjacent fractions
            # Linear interpolation of fraction at log_bayes_factor == 0.
            t = fa / (fa - fb)
            return float(a["fraction"] + t * (b["fraction"] - a["fraction"]))
    return None
