r"""Named, physically-motivated prior narrowings for the dark-energy equation of state.

The DESI preference for evolving dark energy over :math:`\Lambda\mathrm{CDM}` is
known to depend on the prior placed on the Chevallier--Polarski--Linder (CPL)
parameters :math:`(w_0, w_a)`, where :math:`w(a) = w_0 + w_a (1 - a)`. A broad flat
box carries a large Ockham penalty against the extension (Trotta 2008,
arXiv:0803.4089); the verdict can move substantially under a narrower or
physically-motivated prior (Efstathiou 2008, arXiv:0802.3185; Linder & Miquel
2008; Toomey et al. 2025, arXiv:2509.13318).

This module turns the defensible *ways to narrow the prior* into named, cited
constructors over the existing re-weighting engine, so that the flagship output
is a **sensitivity set** (several priors side by side, each with its own evidence
and effective sample size) rather than a single Bayes factor. Narrowing only:
every prior here is a sub-measure of the original box, so re-weighting stays
valid (widening would reach unsampled regions and is refused upstream).

Two kinds of narrowing are provided:

- **Region cuts** (hard indicator priors): :func:`physicality_cut` and
  :func:`no_phantom` return ``(params, inside)`` for
  :func:`cosmocompare.prior.reweight_region`, whose Monte-Carlo volume estimate
  gives the correct normalisation.
- **Shaped densities**: :func:`gaussian_shrink`, :func:`thawing_band` and
  :func:`quintessence_prior` return a ``log_prior_new(samples)`` callable for
  :func:`cosmocompare.prior.reweight_density`. Each is normalised over the base
  box internally (see :func:`box_normalised_logdensity`), so the uniform base
  prior cancels in the evidence ratio and the caller cannot get the
  normalisation wrong.

The convenience :func:`sensitivity_panel` runs the canonical set in one call.

Default parameter names are the chain columns ``w`` (:math:`w_0`) and ``wa``.
"""

import numpy as np
from scipy.special import logsumexp

from cosmocompare.diagnostics import effective_sample_size
from cosmocompare.evidence import log_bayes_factor, nested_evidence
from cosmocompare.prior import reweight_density, reweight_region
from cosmocompare.psis import psis

#: The :math:`\Lambda\mathrm{CDM}` point in CPL space, the data-independent centre
#: for any shrink-toward-base prior.
LCDM_CENTRE = {"w": -1.0, "wa": 0.0}


# =============================================================================
# Region cuts (hard, physically-motivated indicators)
# =============================================================================

def physicality_cut(params=("w", "wa")):
    r"""Non-phantom-at-early-times cut :math:`w_0 + w_a < 0`.

    As :math:`a \to 0`, :math:`w(a) \to w_0 + w_a`; requiring this to be negative
    keeps dark energy sub-dominant at early times. This is the cleanest,
    most transparent narrowing, and the DESI posterior already lives mostly
    inside it, so the effective sample size barely drops.

    Parameters
    ----------
    params : tuple of str, optional
        The ``(w_0, w_a)`` column names. Defaults to ``("w", "wa")``.

    Returns
    -------
    tuple
        ``(params, inside)`` for :func:`cosmocompare.prior.reweight_region`.
    """
    p = tuple(params)

    def inside(a):
        a = np.asarray(a, dtype=float)
        return a[:, 0] + a[:, 1] < 0.0

    return p, inside


def no_phantom(params=("w", "wa")):
    r"""Canonical no-phantom cut :math:`w(a) \ge -1` for all :math:`a \in (0, 1]`.

    Because :math:`w(a)` is linear in :math:`a`, the condition reduces to its two
    endpoints: :math:`w_0 \ge -1` (today) and :math:`w_0 + w_a \ge -1` (early).
    This restricts to canonical (non-phantom) quintessence.

    .. warning::

        The DESI DR2 best fit sits in the phantom regime (:math:`w_0 < -1`), so
        this cut slices off the posterior peak and the effective sample size can
        collapse. Treat its evidence as exploratory and check the ESS.

    Parameters
    ----------
    params : tuple of str, optional
        The ``(w_0, w_a)`` column names. Defaults to ``("w", "wa")``.

    Returns
    -------
    tuple
        ``(params, inside)`` for :func:`cosmocompare.prior.reweight_region`.
    """
    p = tuple(params)

    def inside(a):
        a = np.asarray(a, dtype=float)
        w0, wa = a[:, 0], a[:, 1]
        return (w0 >= -1.0) & (w0 + wa >= -1.0)

    return p, inside


# =============================================================================
# Shaped densities, normalised over the base box
# =============================================================================

def box_normalised_logdensity(
    log_shape, params, base_bounds, n_mc=1_000_000, rng=None
):
    r"""Wrap an unnormalised log shape into a box-normalised log prior callable.

    Given an unnormalised non-negative shape :math:`g(\theta)` on the base box,
    the evidence-ratio re-weight needs
    :math:`\log \pi_\text{new} - \log \pi_\text{old}`. With a uniform base prior
    :math:`\pi_\text{old} = 1 / V_\text{box}` and the normalised target
    :math:`\pi_\text{new} = g / \int_\text{box} g`, the box volume cancels and the
    per-sample log ratio is :math:`\log g(\theta) - \log \mathbb{E}_\text{box}[g]`.
    The expectation over the box is estimated by Monte-Carlo integration here, so
    the returned callable is correctly normalised and can be passed straight to
    :func:`cosmocompare.prior.reweight_density` with ``log_prior_old=None``.

    Parameters
    ----------
    log_shape : callable
        ``log_shape(array)`` where ``array`` has shape ``(n, d)``; returns the
        unnormalised log density at each point. Need not be finite everywhere.
    params : sequence of str
        Parameter column names spanning the shape, in the order ``log_shape``
        expects.
    base_bounds : sequence of tuple
        ``(low, high)`` for each parameter: the original uniform prior bounds.
    n_mc : int, optional
        Monte-Carlo draws for the box normalisation. Defaults to 1e6.
    rng : numpy.random.Generator, optional
        Source of randomness for the normalisation estimate.

    Returns
    -------
    callable
        ``log_prior_new(samples) -> array``, the normalised target log density at
        each sample, ``-inf`` outside the base box.
    """
    p = tuple(params)
    bounds = np.asarray(base_bounds, dtype=float)
    rng = np.random.default_rng() if rng is None else rng

    draws = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_mc, len(bounds)))
    log_g_box = np.asarray(log_shape(draws), dtype=float)
    # log E_box[g] = logsumexp(log g over box draws) - log(n_mc).
    log_mean = logsumexp(log_g_box) - np.log(log_g_box.size)

    def log_prior_new(samples):
        values = np.column_stack(
            [np.asarray(samples[name], dtype=float) for name in p]
        )
        in_box = np.all(
            (values >= bounds[:, 0]) & (values <= bounds[:, 1]), axis=1
        )
        log_g = np.asarray(log_shape(values), dtype=float)
        return np.where(in_box, log_g - log_mean, -np.inf)

    return log_prior_new


def gaussian_shrink(
    base_bounds, sigma, centre=None, params=("w", "wa"), n_mc=1_000_000, rng=None
):
    r"""Truncated-Gaussian prior shrinking :math:`(w_0, w_a)` towards the base model.

    A Gaussian centred on the :math:`\Lambda\mathrm{CDM}` point, truncated to the
    base box. A *single* width is not a defensible primary prior - it dials the
    Ockham penalty by hand and can smuggle in the answer - so this is meant to be
    scanned over a range of ``sigma`` to produce the honest :math:`\ln B(\sigma)`
    curve (Trotta 2008, section 3.3).

    Parameters
    ----------
    base_bounds : sequence of tuple
        ``(low, high)`` per parameter: the original uniform prior bounds.
    sigma : float, sequence or dict
        Gaussian width. A scalar applies to every parameter; a sequence gives one
        per parameter in ``params`` order; a dict maps name to width.
    centre : dict, optional
        ``{name: value}`` centre. Defaults to :data:`LCDM_CENTRE` restricted to
        ``params``. Must be data-independent.
    params : tuple of str, optional
        The ``(w_0, w_a)`` column names. Defaults to ``("w", "wa")``.
    n_mc, rng
        Passed to :func:`box_normalised_logdensity`.

    Returns
    -------
    callable
        ``log_prior_new(samples)`` for :func:`cosmocompare.prior.reweight_density`.
    """
    p = tuple(params)
    centre = {**LCDM_CENTRE, **(centre or {})}
    mu = np.array([centre[name] for name in p], dtype=float)

    if np.isscalar(sigma):
        sig = np.full(len(p), float(sigma))
    elif isinstance(sigma, dict):
        sig = np.array([sigma[name] for name in p], dtype=float)
    else:
        sig = np.asarray(sigma, dtype=float)
    if np.any(sig <= 0):
        raise ValueError("sigma must be positive.")

    def log_shape(a):
        a = np.asarray(a, dtype=float)
        z = (a - mu) / sig
        return -0.5 * np.sum(z * z, axis=1)

    return box_normalised_logdensity(log_shape, p, base_bounds, n_mc=n_mc, rng=rng)


def thawing_band(
    base_bounds,
    slope=-1.58,
    width=0.1,
    params=("w", "wa"),
    n_mc=1_000_000,
    rng=None,
):
    r"""Soft thawing-quintessence band :math:`w_a \approx \text{slope}\,(1 + w_0)`.

    Thawing scalar-field models trace an approximately linear track in CPL space;
    a common calibration is :math:`w_a \approx -1.58\,(1 + w_0)`. This restricts
    to that subclass as a soft Gaussian band of standard deviation ``width`` in
    :math:`w_a` about the track, truncated to the base box. A soft band (never a
    delta line) keeps the re-weight well defined and the effective sample size
    finite.

    Parameters
    ----------
    base_bounds : sequence of tuple
        ``(low, high)`` per parameter.
    slope : float, optional
        Track slope; defaults to ``-1.58`` (thawing).
    width : float, optional
        Gaussian scatter in :math:`w_a` about the track. Defaults to 0.1.
    params : tuple of str, optional
        The ``(w_0, w_a)`` column names. Defaults to ``("w", "wa")``.
    n_mc, rng
        Passed to :func:`box_normalised_logdensity`.

    Returns
    -------
    callable
        ``log_prior_new(samples)`` for :func:`cosmocompare.prior.reweight_density`.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    p = tuple(params)

    def log_shape(a):
        a = np.asarray(a, dtype=float)
        w0, wa = a[:, 0], a[:, 1]
        resid = wa - slope * (1.0 + w0)
        return -0.5 * (resid / width) ** 2

    return box_normalised_logdensity(log_shape, p, base_bounds, n_mc=n_mc, rng=rng)


def quintessence_prior(
    w0wa_samples,
    base_bounds,
    params=("w", "wa"),
    bw_method=None,
    n_mc=1_000_000,
    rng=None,
):
    r"""Theory-induced prior from quintessence models, as a kernel density estimate.

    Flat priors on the parameters of a physical dark-energy model (for example the
    exponential or hilltop quintessence potentials of Toomey et al. 2025,
    arXiv:2509.13318) push forward to a strongly non-uniform, :math:`\Lambda`-peaked
    distribution over :math:`(w_0, w_a)`. Given a table of such model-induced
    :math:`(w_0, w_a)` points, this builds their Gaussian kernel density estimate,
    normalised over the base box, as the target prior. This is the least arbitrary
    narrowing; label the result "evidence under a quintessence prior", not "the"
    evidence, since it is model-class dependent.

    Parameters
    ----------
    w0wa_samples : array_like
        Shape ``(n_models, 2)`` table of :math:`(w_0, w_a)` points drawn from the
        physical model class (the caller generates these; it is a plugin, not a
        built-in cosmology).
    base_bounds : sequence of tuple
        ``(low, high)`` per parameter.
    params : tuple of str, optional
        The ``(w_0, w_a)`` column names. Defaults to ``("w", "wa")``.
    bw_method : optional
        Bandwidth passed to :class:`scipy.stats.gaussian_kde`.
    n_mc, rng
        Passed to :func:`box_normalised_logdensity`.

    Returns
    -------
    callable
        ``log_prior_new(samples)`` for :func:`cosmocompare.prior.reweight_density`.
    """
    from scipy.stats import gaussian_kde

    table = np.asarray(w0wa_samples, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError("w0wa_samples must have shape (n_models, 2).")
    kde = gaussian_kde(table.T, bw_method=bw_method)

    def log_shape(a):
        a = np.asarray(a, dtype=float)
        return kde.logpdf(a.T)

    return box_normalised_logdensity(
        log_shape, tuple(params), base_bounds, n_mc=n_mc, rng=rng
    )


# =============================================================================
# The flagship output: a sensitivity panel, not a single number
# =============================================================================

def _pareto_k(posterior):
    """Return the PSIS Pareto-k-hat of a reweighted posterior's weights.

    Computed on the combined weights ``W_i * (pi_new/pi_old)_i``. For a hard cut
    the surviving weights are a spike of equal values with no Pareto tail, so the
    fit returns ``nan`` (correctly: the diagnostic does not apply to hard cuts,
    which are governed instead by the Kish ESS and support containment).

    Parameters
    ----------
    posterior : anesthetic samples
        A reweighted posterior exposing ``get_weights``.

    Returns
    -------
    float
        The Pareto shape parameter, or ``nan`` when no tail can be fitted.
    """
    weights = np.asarray(posterior.get_weights(), dtype=float)
    with np.errstate(divide="ignore"):
        return psis(np.log(weights)).pareto_k


def _panel_row(name, evidence, posterior, base_logZ, base_logZ_err, ess_warn):
    """Assemble one panel record from an absolute extended evidence.

    Parameters
    ----------
    name : str
        Human-readable label for the narrowing.
    evidence : cosmocompare.evidence.EvidenceResult
        The extended-model evidence under this prior (absolute ``log_evidence``
        required).
    posterior : anesthetic samples
        The reweighted posterior, used for the Pareto-k-hat diagnostic.
    base_logZ, base_logZ_err : float
        The fixed base-model (``LCDM``) evidence and its error.
    ess_warn : float
        Effective-sample-size floor below which the row is flagged unreliable.

    Returns
    -------
    dict
        One panel record.
    """
    log_b, log_b_err = log_bayes_factor(evidence, (base_logZ, base_logZ_err))
    ess = evidence.effective_sample_size
    pareto_k = _pareto_k(posterior)
    # A row is reliable when the Kish ESS clears the floor and the Pareto k-hat is
    # finite and below 0.7. A non-finite k-hat means the tail fit failed for want
    # of surviving samples, so the row is treated as unreliable.
    reliable = bool(
        ess is not None and ess >= ess_warn and np.isfinite(pareto_k) and pareto_k < 0.7
    )
    return {
        "prior": name,
        "log_evidence_ext": evidence.log_evidence,
        "log_bayes_factor": log_b,
        "log_bayes_factor_err": log_b_err,
        "effective_sample_size": ess,
        "pareto_k": pareto_k,
        "reliable": reliable,
    }


def sensitivity_panel(
    extended,
    base,
    base_bounds,
    params=("w", "wa"),
    sigmas=(1.0, 0.5, 0.25),
    quintessence_samples=None,
    include_no_phantom=True,
    include_thawing=False,
    ess_warn=50.0,
    nsamples=1000,
    n_bootstrap=1000,
    seed=None,
    n_mc=1_000_000,
):
    r"""Run the canonical set of :math:`(w_0, w_a)` priors side by side.

    The honest output for a prior-sensitive comparison is not one Bayes factor but
    a table of how it moves under scientifically-motivated priors, each with its
    effective sample size. This runs, against the same fixed base-model evidence:
    the broad baseline, the :math:`w_0 + w_a < 0` physicality cut, a
    :math:`\Lambda\mathrm{CDM}`-centred Gaussian shrink at each width in ``sigmas``
    (the :math:`\ln B(\sigma)` curve), optionally the no-phantom cut and a thawing
    band, and optionally the quintessence-induced prior.

    Parameters
    ----------
    extended : anesthetic.NestedSamples
        The extended-model run (for example ``w0waCDM``), carrying ``params``.
    base : anesthetic.NestedSamples
        The base-model run (``LCDM``); its evidence is fixed across the panel.
    base_bounds : sequence of tuple
        ``(low, high)`` per parameter in ``params`` order: the original uniform
        prior the extended chain was produced under (for example
        ``[(-3, 1), (-3, 2)]``).
    params : tuple of str, optional
        The ``(w_0, w_a)`` column names. Defaults to ``("w", "wa")``.
    sigmas : iterable of float, optional
        Gaussian shrink widths towards ``LCDM``. Defaults to ``(1.0, 0.5, 0.25)``.
    quintessence_samples : array_like, optional
        ``(n_models, 2)`` table of model-induced :math:`(w_0, w_a)` points. When
        given, a quintessence-prior row is added (see :func:`quintessence_prior`).
    include_no_phantom : bool, optional
        Add the no-phantom cut row (flagged if its ESS is low). Defaults to True.
    include_thawing : bool, optional
        Add a thawing-band row. Defaults to False.
    ess_warn : float, optional
        ESS floor below which a row is marked ``reliable = False``. Defaults to 50.
    nsamples, n_bootstrap, seed, n_mc
        Evidence-draw, bootstrap, seeding and Monte-Carlo controls, as elsewhere.

    Returns
    -------
    list of dict
        One record per prior, each with ``prior``, ``log_evidence_ext``,
        ``log_bayes_factor``, ``log_bayes_factor_err``, ``effective_sample_size``
        and ``reliable``. The first row is the broad baseline.
    """
    p = tuple(params)
    bounds = [tuple(float(x) for x in b) for b in base_bounds]

    base_logZ, base_logZ_err = nested_evidence(base, nsamples)
    ext_logZ, ext_logZ_err = nested_evidence(extended, nsamples)

    rows = []

    # Baseline: the published broad-prior number, no re-weighting.
    baseline_ess = effective_sample_size(np.asarray(extended.get_weights(), float))
    log_b, log_b_err = log_bayes_factor(
        (ext_logZ, ext_logZ_err), (base_logZ, base_logZ_err)
    )
    rows.append(
        {
            "prior": "baseline (broad uniform)",
            "log_evidence_ext": ext_logZ,
            "log_bayes_factor": log_b,
            "log_bayes_factor_err": log_b_err,
            "effective_sample_size": baseline_ess,
            "pareto_k": float("nan"),
            "reliable": bool(baseline_ess >= ess_warn),
        }
    )

    def region_row(name, inside, k):
        cell_seed = None if seed is None else seed + k
        evidence, posterior = reweight_region(
            extended, p, inside, bounds,
            n_mc=n_mc, nsamples=nsamples, n_bootstrap=n_bootstrap, seed=cell_seed,
        )
        rows.append(
            _panel_row(name, evidence, posterior, base_logZ, base_logZ_err, ess_warn)
        )

    def density_row(name, log_prior_new, k):
        cell_seed = None if seed is None else seed + k
        evidence, posterior = reweight_density(
            extended, log_prior_new,
            nsamples=nsamples, n_bootstrap=n_bootstrap, seed=cell_seed,
        )
        rows.append(
            _panel_row(name, evidence, posterior, base_logZ, base_logZ_err, ess_warn)
        )

    k = 1
    region_row("physicality cut (w0+wa<0)", physicality_cut(p)[1], k)
    k += 1
    if include_no_phantom:
        region_row("no phantom (w0>=-1, w0+wa>=-1)", no_phantom(p)[1], k)
        k += 1

    for sigma in sigmas:
        prior = gaussian_shrink(bounds, sigma, params=p, n_mc=n_mc)
        density_row(f"LCDM Gaussian shrink (sigma={sigma:g})", prior, k)
        k += 1

    if include_thawing:
        density_row("thawing band", thawing_band(bounds, params=p, n_mc=n_mc), k)
        k += 1

    if quintessence_samples is not None:
        prior = quintessence_prior(quintessence_samples, bounds, params=p, n_mc=n_mc)
        density_row("quintessence-induced prior", prior, k)
        k += 1

    return rows
