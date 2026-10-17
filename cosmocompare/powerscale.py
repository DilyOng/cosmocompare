r"""Power-scaling the prior as a continuous sensitivity path.

Beyond contracting a box (:func:`cosmocompare.sweep.bayes_factor_sweep`), prior
sensitivity can be probed by *deforming the prior density itself*. Power-scaling
(Kallioinen, Paananen, Bürkner & Vehtari 2024, *Statistics and Computing* 34(57);
R package ``priorsense``) raises the prior to a power, which here is written as a
geometric interpolation between a base shape and a target shape,

.. math::

    \pi_\alpha(\theta) \;\propto\; \pi_0(\theta)^{1-\alpha}\, \pi_1(\theta)^{\alpha},
    \qquad \alpha \in [0, 1],

so :math:`\alpha = 0` is the base prior and :math:`\alpha = 1` is the target. The
pure power form :math:`\pi^\alpha` is the special case :math:`\pi_0 = \text{uniform}`,
:math:`\pi_1 = \pi`. Because the density is deformed rather than its support
contracted, this is genuinely distinct from the homothetic width sweep; it is
most informative when the base prior is itself informative (a flat prior is
unchanged by its own power).

Each :math:`\pi_\alpha` is normalised over the base box (so the uniform base
cancels correctly in the evidence ratio; see
:func:`cosmocompare.dark_energy.box_normalised_logdensity`) and the evidence is
recomputed by reweighting, giving an honest :math:`\ln B(\alpha)` path at no
sampling cost.
"""

import numpy as np

from cosmocompare.dark_energy import box_normalised_logdensity
from cosmocompare.evidence import log_bayes_factor, nested_evidence
from cosmocompare.prior import reweight_density


def power_scaled_logdensity(
    base_bounds,
    log_shape_target,
    alpha,
    log_shape_base=None,
    params=("w", "wa"),
    n_mc=1_000_000,
    rng=None,
):
    r"""Return a box-normalised log prior for ``pi_0^(1-alpha) pi_1^alpha``.

    Parameters
    ----------
    base_bounds : sequence of tuple
        ``(low, high)`` per parameter: the original uniform prior bounds.
    log_shape_target : callable
        ``log_shape_target(array) -> array``, the unnormalised log target shape
        :math:`\log \pi_1`.
    alpha : float
        Interpolation power in ``[0, 1]``.
    log_shape_base : callable, optional
        The unnormalised log base shape :math:`\log \pi_0`. Defaults to a uniform
        (constant) base, giving the pure power form :math:`\pi_1^\alpha`.
    params : sequence of str, optional
        Parameter column names. Defaults to ``("w", "wa")``.
    n_mc, rng
        Passed to :func:`cosmocompare.dark_energy.box_normalised_logdensity`.

    Returns
    -------
    callable
        ``log_prior_new(samples)`` for :func:`cosmocompare.prior.reweight_density`.
    """

    def log_shape(a):
        a = np.asarray(a, dtype=float)
        target = np.asarray(log_shape_target(a), dtype=float)
        if log_shape_base is None:
            base = 0.0
        else:
            base = np.asarray(log_shape_base(a), dtype=float)
        return (1.0 - alpha) * base + alpha * target

    return box_normalised_logdensity(
        log_shape, tuple(params), base_bounds, n_mc=n_mc, rng=rng
    )


def power_scale_sweep(
    extended,
    base,
    base_bounds,
    log_shape_target,
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    log_shape_base=None,
    params=("w", "wa"),
    ess_warn=50.0,
    nsamples=1000,
    n_bootstrap=1000,
    seed=None,
    n_mc=1_000_000,
):
    r"""Record the log Bayes factor along a power-scaling path ``pi_0^(1-a) pi_1^a``.

    Parameters
    ----------
    extended : anesthetic.NestedSamples
        The extended-model run (for example ``w0waCDM``).
    base : anesthetic.NestedSamples
        The base-model run (``LCDM``); its evidence is fixed across the path.
    base_bounds : sequence of tuple
        ``(low, high)`` per parameter in ``params`` order.
    log_shape_target : callable
        Unnormalised log target prior shape :math:`\log \pi_1`.
    alphas : iterable of float, optional
        Interpolation powers to evaluate. Defaults to ``(0, 0.25, 0.5, 0.75, 1)``.
    log_shape_base : callable, optional
        Unnormalised log base prior shape; defaults to uniform (pure power form).
    params : sequence of str, optional
        Parameter column names. Defaults to ``("w", "wa")``.
    ess_warn : float, optional
        ESS floor below which a row is flagged unreliable. Defaults to 50.
    nsamples, n_bootstrap, seed, n_mc
        Evidence-draw, bootstrap, seeding and Monte-Carlo controls.

    Returns
    -------
    list of dict
        One record per ``alpha``, each with ``alpha``, ``log_evidence_ext``,
        ``log_bayes_factor``, ``log_bayes_factor_err``, ``effective_sample_size``
        and ``reliable``.
    """
    p = tuple(params)
    bounds = [tuple(float(x) for x in b) for b in base_bounds]
    base_logZ, base_logZ_err = nested_evidence(base, nsamples)

    records = []
    for k, alpha in enumerate(alphas):
        cell_seed = None if seed is None else seed + k
        prior = power_scaled_logdensity(
            bounds, log_shape_target, float(alpha),
            log_shape_base=log_shape_base, params=p, n_mc=n_mc,
        )
        evidence, _ = reweight_density(
            extended, prior, nsamples=nsamples, n_bootstrap=n_bootstrap, seed=cell_seed
        )
        log_b, log_b_err = log_bayes_factor(evidence, (base_logZ, base_logZ_err))
        ess = evidence.effective_sample_size
        records.append(
            {
                "alpha": float(alpha),
                "log_evidence_ext": evidence.log_evidence,
                "log_bayes_factor": log_b,
                "log_bayes_factor_err": log_b_err,
                "effective_sample_size": ess,
                "reliable": bool(ess is not None and ess >= ess_warn),
            }
        )
    return records
