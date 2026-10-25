r"""Savage--Dickey density ratio for nested-model comparison.

When the base model is the extended model with its extra parameters fixed at a
point :math:`\theta_\star` (for example :math:`\Lambda\mathrm{CDM}` is
:math:`w_0w_a\mathrm{CDM}` at :math:`(w_0, w_a) = (-1, 0)`), the Bayes factor in
favour of the simpler model is the Savage--Dickey density ratio (Dickey 1971;
generalised by Verdinelli & Wasserman 1995; Trotta 2008),

.. math::

    B_{\text{base,ext}}
        = \frac{p(\theta_\star \mid d)}{\pi(\theta_\star)},

the extended-model posterior density at :math:`\theta_\star` divided by the
prior density there. In this package's convention the log Bayes factor for the
*extended* model over the base is the reciprocal,

.. math::

    \ln B = \ln \pi(\theta_\star) - \ln p(\theta_\star \mid d),

so a posterior that piles up at :math:`\theta_\star` (data like the base model)
gives a negative :math:`\ln B`, and one that avoids it gives a positive
:math:`\ln B`. It needs only the extended chain, no reweighting and no new
sampling, which makes it an independent cross-check on the importance-sampling
evidence.

.. warning::

   The estimate hinges on the posterior *density* at a single point. When the
   data pull away from :math:`\theta_\star` (as DESI does from
   :math:`\Lambda\mathrm{CDM}`), that point sits in the posterior tail, where a
   kernel density estimate is least trustworthy and tends to over-estimate the
   density, biasing :math:`\ln B` towards the base model. This routine flags when
   :math:`\theta_\star` lies outside the requested highest-density region, and
   that flag should be believed.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SDDRResult:
    """Outcome of a Savage--Dickey density-ratio estimate.

    Attributes
    ----------
    log_bayes_factor : float
        ``log(pi(theta*) / p(theta*|d))``: the log Bayes factor for the extended
        model over the nested base model.
    log_posterior_density : float
        ``log p(theta*|d)``, the extended posterior density at the nesting point.
    log_prior_density : float
        ``log pi(theta*)``, the prior density at the nesting point.
    inside_hdr : bool
        ``True`` when the nesting point lies within the ``credible`` highest
        posterior-density region, i.e. the density estimate is being made where
        there is support.
    hdr_mass : float
        Posterior mass at density at least that of the nesting point, i.e. the
        smallest highest-density region that contains the point. It is the
        continuous version of :attr:`inside_hdr`: reliability degrades smoothly as
        it approaches 1 (the point sinks into the tail), so read this number, not
        only the binary flag.
    reliable : bool
        Alias of :attr:`inside_hdr`; ``False`` means treat the number as
        indicative only. Even when ``True``, a ``hdr_mass`` close to the threshold
        means the kernel density estimate at the point is only approximate.
    """

    log_bayes_factor: float
    log_posterior_density: float
    log_prior_density: float
    inside_hdr: bool
    hdr_mass: float
    reliable: bool


def _uniform_box_log_density(bounds):
    """Return ``log(1 / volume)`` of a uniform box prior.

    Parameters
    ----------
    bounds : sequence of tuple
        ``(low, high)`` per parameter.

    Returns
    -------
    float
        The constant log prior density inside the box.
    """
    widths = np.array([hi - lo for lo, hi in bounds], dtype=float)
    if np.any(widths <= 0):
        raise ValueError("Every prior bound must have positive width.")
    return float(-np.sum(np.log(widths)))


def savage_dickey(
    samples,
    params,
    point,
    prior_bounds=None,
    log_prior_density=None,
    bw_method=None,
    credible=0.99,
):
    r"""Estimate the Savage--Dickey log Bayes factor for a nested model.

    Parameters
    ----------
    samples : anesthetic samples
        The extended-model run, carrying the extra parameters and posterior
        weights.
    params : sequence of str
        The extra-parameter column names that the base model fixes (for example
        ``("w", "wa")``).
    point : dict or sequence
        The nesting point :math:`\theta_\star`, for example ``{"w": -1, "wa": 0}``
        or ``(-1, 0)`` in ``params`` order.
    prior_bounds : sequence of tuple, optional
        ``(low, high)`` per parameter for a uniform box prior; its density at the
        point is then ``1 / volume``. Provide this or ``log_prior_density``.
    log_prior_density : float, optional
        The log prior density at the point directly, for a non-uniform prior.
        Overrides ``prior_bounds`` if both are given.
    bw_method : optional
        Bandwidth passed to :class:`scipy.stats.gaussian_kde`.
    credible : float, optional
        Highest posterior-density mass used for the tail check. Defaults to 0.99.

    Returns
    -------
    SDDRResult
        The log Bayes factor, the two densities, and the tail-reliability flag.
    """
    from scipy.stats import gaussian_kde

    p = tuple(params)
    if isinstance(point, dict):
        pt = np.array([point[name] for name in p], dtype=float)
    else:
        pt = np.asarray(point, dtype=float)

    if log_prior_density is None:
        if prior_bounds is None:
            raise ValueError("Provide prior_bounds or log_prior_density.")
        bounds = np.asarray(prior_bounds, dtype=float)
        # The nesting point must lie inside the prior support; otherwise its prior
        # density is zero and the Savage-Dickey ratio is undefined there. Catch it
        # loudly rather than return a finite box density for a point outside the box.
        if np.any(pt < bounds[:, 0]) or np.any(pt > bounds[:, 1]):
            raise ValueError(
                "The nesting point lies outside prior_bounds; its prior density "
                "is zero and the Savage-Dickey ratio is undefined there."
            )
        log_prior_density = _uniform_box_log_density(prior_bounds)
    log_prior_density = float(log_prior_density)

    values = np.column_stack([np.asarray(samples[name], dtype=float) for name in p])
    weights = np.asarray(samples.get_weights(), dtype=float)

    kde = gaussian_kde(values.T, weights=weights, bw_method=bw_method)
    # Use log(pdf) rather than kde.logpdf: scipy's weighted gaussian_kde.logpdf
    # returns NaN (at least through 1.13.x) while the weighted pdf/evaluate path
    # is correct, so go through pdf and take the log ourselves.
    log_post_point = float(np.log(np.atleast_1d(kde.pdf(pt))[0]))

    # Tail check: the point is inside the `credible` highest-density region when
    # the posterior mass at density >= its density is below `credible`. A point in
    # the far tail (mass above it close to 1) is where the KDE is least reliable.
    log_dens_samples = np.log(kde.pdf(values.T))
    mass_above = float(
        np.sum(weights[log_dens_samples >= log_post_point]) / np.sum(weights)
    )
    inside_hdr = bool(mass_above < credible)

    log_bf = log_prior_density - log_post_point
    return SDDRResult(
        log_bayes_factor=log_bf,
        log_posterior_density=log_post_point,
        log_prior_density=log_prior_density,
        inside_hdr=inside_hdr,
        hdr_mass=mass_above,
        reliable=inside_hdr,
    )
