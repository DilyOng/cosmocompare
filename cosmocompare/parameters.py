"""Prior and proposal distributions for the parameters of an extended model.

Importance sampling reuses samples drawn under one distribution to answer a
question posed under another. Each new parameter therefore carries two
distributions, not one: the *proposal* it is actually drawn from, and the
*prior* the target model assigns to it. When the two coincide the correction
term vanishes, which is the common case; keeping them separate is what lets the
same code narrow a prior or change a proposal without silently biasing the
weights.
"""

import numpy as np
from scipy.stats import norm, truncnorm, uniform

#: Prior/proposal families understood by :class:`_Distribution`.
SUPPORTED_TYPES = ("uniform", "gaussian", "truncated_gaussian", "log_uniform")


class _Distribution:
    """A one-dimensional distribution described by a configuration dictionary.

    Parameters
    ----------
    config : dict
        Mapping with a ``"type"`` key naming one of :data:`SUPPORTED_TYPES` and
        the parameters that family requires:

        - ``uniform``: ``min``, ``max``
        - ``gaussian``: ``mean``, ``std``
        - ``truncated_gaussian``: ``mean``, ``std``, ``min``, ``max``
        - ``log_uniform``: ``min``, ``max`` (both strictly positive)
    """

    def __init__(self, config):
        self.type = config["type"]
        self.params = {k: v for k, v in config.items() if k != "type"}
        if self.type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported distribution type: {self.type!r}. "
                f"Expected one of {SUPPORTED_TYPES}."
            )
        self._dist = self._build()

    def _build(self):
        """Return the backing ``scipy.stats`` object, or None for log-uniform.

        Returns
        -------
        scipy.stats.rv_frozen or None
            A frozen distribution for every family except ``log_uniform``,
            which is handled analytically and returns None.
        """
        if self.type == "uniform":
            lo, hi = self.params["min"], self.params["max"]
            return uniform(loc=lo, scale=hi - lo)
        if self.type == "gaussian":
            return norm(loc=self.params["mean"], scale=self.params["std"])
        if self.type == "truncated_gaussian":
            mean, std = self.params["mean"], self.params["std"]
            lo, hi = self.params["min"], self.params["max"]
            a, b = (lo - mean) / std, (hi - mean) / std
            return truncnorm(a, b, loc=mean, scale=std)
        # log_uniform: uniform in log-space, evaluated analytically.
        self._log_min = np.log(self.params["min"])
        self._log_max = np.log(self.params["max"])
        return None

    def sample(self, n_samples, rng=None):
        """Draw samples from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : numpy.random.Generator, optional
            Source of randomness. A fresh default generator is used if omitted.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples,)``.
        """
        rng = np.random.default_rng() if rng is None else rng
        if self.type == "log_uniform":
            return np.exp(rng.uniform(self._log_min, self._log_max, size=n_samples))
        return self._dist.rvs(size=n_samples, random_state=rng)

    def logpdf(self, values):
        """Evaluate the log probability density at ``values``.

        Parameters
        ----------
        values : array_like
            Points at which to evaluate the density.

        Returns
        -------
        numpy.ndarray
            Log density, with ``-inf`` outside the support.
        """
        values = np.asarray(values, dtype=float)
        if self.type == "log_uniform":
            with np.errstate(divide="ignore"):
                log_pdf = -np.log(values) - np.log(self._log_max - self._log_min)
            out_of_bounds = (values < np.exp(self._log_min)) | (
                values > np.exp(self._log_max)
            )
            log_pdf = np.where(out_of_bounds, -np.inf, log_pdf)
            return log_pdf
        return self._dist.logpdf(values)

    def support(self):
        """Return the ``(low, high)`` bounds of the distribution's support.

        Returns
        -------
        tuple of float
            The interval outside which the density is zero. Unbounded families
            (``gaussian``) return ``(-inf, inf)``.
        """
        if self.type == "gaussian":
            return (-np.inf, np.inf)
        if self.type == "log_uniform":
            return (float(np.exp(self._log_min)), float(np.exp(self._log_max)))
        return (float(self.params["min"]), float(self.params["max"]))


def check_support_contained(inner, outer, name=""):
    """Raise if ``inner``'s support is not contained within ``outer``'s.

    Importance sampling can only move probability mass to where the original
    samples already live. A target distribution with support outside the one the
    samples were drawn from covers regions that were never explored, which biases
    the estimate silently, so it is rejected here rather than returned wrong.

    Parameters
    ----------
    inner : _Distribution
        The distribution whose support must be covered (the target prior, or an
        added parameter's prior).
    outer : _Distribution
        The distribution that must cover it (the base prior, or the proposal).
    name : str, optional
        Parameter name, used in the error message.

    Raises
    ------
    ValueError
        If ``inner`` has support beyond ``outer``.
    """
    lo_in, hi_in = inner.support()
    lo_out, hi_out = outer.support()
    if lo_in < lo_out or hi_in > hi_out:
        raise ValueError(
            f"Target support for {name!r} ([{lo_in}, {hi_in}]) extends beyond the "
            f"base/proposal support ([{lo_out}, {hi_out}]). Importance sampling "
            f"cannot reach regions the samples never covered; widening a bounded "
            f"prior requires a new run."
        )


class Parameter:
    """A new parameter introduced by an extended model.

    The parameter is sampled from its *proposal* and scored under its *prior*.
    The log ratio of the two, :meth:`log_correction`, is the per-sample term
    that corrects importance weights when the proposal differs from the prior.

    Parameters
    ----------
    name : str
        Parameter name, matching the column it will occupy in the samples and
        the name the likelihood expects (for example ``"w"`` or ``"omegak"``).
    prior : dict
        Target-model prior configuration (see :class:`_Distribution`).
    proposal : dict, optional
        Distribution the parameter is actually drawn from. Defaults to the
        prior, in which case :meth:`log_correction` is identically zero.
    """

    def __init__(self, name, prior, proposal=None):
        self.name = name
        self.prior = _Distribution(prior)
        self.proposal = self.prior if proposal is None else _Distribution(proposal)
        self._shares_proposal = proposal is None
        if not self._shares_proposal:
            # The proposal must cover the prior, or target regions go unsampled.
            check_support_contained(self.prior, self.proposal, name)

    def sample(self, n_samples, rng=None):
        """Draw values from the proposal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : numpy.random.Generator, optional
            Source of randomness.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples,)``.
        """
        return self.proposal.sample(n_samples, rng=rng)

    def log_prior(self, values):
        """Return the target-model log prior at ``values``.

        Parameters
        ----------
        values : array_like
            Parameter values to score.

        Returns
        -------
        numpy.ndarray
            Log prior density.
        """
        return self.prior.logpdf(values)

    def log_proposal(self, values):
        """Return the proposal log density at ``values``.

        Parameters
        ----------
        values : array_like
            Parameter values to score.

        Returns
        -------
        numpy.ndarray
            Log proposal density.
        """
        return self.proposal.logpdf(values)

    def log_correction(self, values):
        """Return ``log(prior / proposal)`` at ``values``.

        This is the prior-over-proposal correction a single parameter
        contributes to the importance weight. It is zero everywhere when the
        proposal was left unset.

        Parameters
        ----------
        values : array_like
            Parameter values to score.

        Returns
        -------
        numpy.ndarray
            Log correction per sample.
        """
        values = np.asarray(values, dtype=float)
        if self._shares_proposal:
            return np.zeros_like(values)
        return self.log_prior(values) - self.log_proposal(values)

    def __repr__(self):
        """Return a concise, unambiguous representation."""
        return (
            f"Parameter(name={self.name!r}, prior={self.prior.type!r}, "
            f"proposal={self.proposal.type!r})"
        )
