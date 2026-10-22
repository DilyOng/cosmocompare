r"""Cross-model importance sampling: reweight a base run onto an extended model.

A fully nested-sampled base model (for example :math:`\Lambda\mathrm{CDM}`) is
reused to estimate the evidence of an extended model (``wCDM`` adds :math:`w`,
``kLCDM`` adds :math:`\Omega_k`) without a new nested-sampling run. Each base
posterior sample is augmented with the new parameter drawn from a proposal
:math:`q`, the extended likelihood is re-evaluated there with Cobaya, and the
evidence ratio is the importance-sampling identity

.. math::

    \frac{\mathcal{Z}_\text{ext}}{\mathcal{Z}_\text{base}}
      = \left\langle \frac{\mathcal{L}_\text{ext}(\theta, \phi)}
                          {\mathcal{L}_\text{base}(\theta)}\,
                     \frac{\pi_\text{ext}(\phi)}{q(\phi)}
         \right\rangle_{\mathcal{P}_\text{base}},

with the shared-parameter priors cancelling (they are identical for an extension).

Two points are forced by a review with Gemini 3.1 Pro and GPT-5.4:

- ``L_base`` is **re-evaluated in the LCDM limit** (:math:`w=-1`, :math:`\Omega_k=0`)
  with the *same* Cobaya calculator, not read from the chain's stored log
  likelihood, so the ratio is free of likelihood-code or version mismatch. The
  maximum stored-minus-fresh discrepancy is reported as a consistency check; a
  large value means a model-definition mismatch and the result is not to be
  trusted.
- The reliability of this up-in-dimension reweighting is governed by **support
  mismatch in the shared parameters** (the base chain only covers the LCDM slice),
  which the Kish effective sample size can miss, so the **PSIS Pareto shape**
  :math:`\hat k` is reported alongside it. :math:`\hat k \ge 0.7` means the weight
  variance is effectively infinite and the estimate is untrustworthy however large
  the Kish number looks.

The honest test is whether the importance-sampled evidence reproduces the direct
nested-sampling evidence of the extended model; believe the nested value.
"""

from dataclasses import dataclass

import numpy as np

from cosmocompare.diagnostics import effective_sample_size
from cosmocompare.evidence import importance_evidence
from cosmocompare.psis import psis


@dataclass
class ExtendResult:
    """Outcome of a cross-model importance-sampling evidence estimate.

    Attributes
    ----------
    log_bayes_factor : float
        ``log(Z_ext / Z_base)`` and so, against the base model, the log Bayes
        factor for the extended model.
    log_bayes_factor_err : float
        Bootstrap standard deviation of :attr:`log_bayes_factor`, combined with
        the base nested ``log Z`` error.
    log_evidence_ext : float
        Absolute ``log Z_ext``.
    log_evidence_ext_err : float
        Uncertainty on :attr:`log_evidence_ext`.
    effective_sample_size : float
        Kish effective sample size of the combined importance weights.
    ess_fraction : float
        ``effective_sample_size / n_samples``.
    pareto_k : float
        PSIS Pareto shape of the combined weights. ``>= 0.7`` means unreliable.
    max_weight : float
        Largest single normalised importance weight (1.0 = one sample dominates).
    base_logL_consistency : float
        Maximum ``|stored logL - fresh L_base|`` over the samples. Small confirms
        the fresh LCDM-limit evaluation matches the chain; large flags a model or
        likelihood-code mismatch.
    degenerate_fraction : float
        Fraction of bootstrap resamples that degenerated (see
        :class:`cosmocompare.evidence.EvidenceResult`).
    n_samples : int
        Number of base samples reweighted.
    reliable : bool
        ``True`` when ``pareto_k < 0.7`` and the Kish ESS clears the floor.
    """

    log_bayes_factor: float
    log_bayes_factor_err: float
    log_evidence_ext: float
    log_evidence_ext_err: float
    effective_sample_size: float
    ess_fraction: float
    pareto_k: float
    max_weight: float
    base_logL_consistency: float
    degenerate_fraction: float
    n_samples: int
    reliable: bool


def extend_evidence(
    base_weights,
    logL_base,
    logL_ext,
    log_prior_correction,
    base_log_evidence,
    base_log_evidence_err=0.0,
    stored_logL=None,
    ess_floor=100.0,
    n_bootstrap=1000,
    seed=None,
):
    r"""Combine the per-sample ingredients into an extended-model evidence.

    This is the pure estimator: the caller supplies the arrays, so it is testable
    without Cobaya. :func:`run_extension` wires the Cobaya likelihood to it.

    Parameters
    ----------
    base_weights : array_like
        Base posterior weights ``W_i`` (anesthetic nested weights; need not be
        normalised).
    logL_base : array_like
        Base log likelihood at each sample, **re-evaluated in the LCDM limit with
        the extended-model calculator** (not the stored chain value).
    logL_ext : array_like
        Extended-model log likelihood at each augmented sample.
    log_prior_correction : array_like
        Per-sample ``log(pi_ext(phi) / q(phi))`` for the new parameter(s); zero
        when the proposal is the prior.
    base_log_evidence, base_log_evidence_err : float
        The base model's nested ``log Z`` and its error.
    stored_logL : array_like, optional
        The chain's stored base log likelihood, for the consistency check against
        ``logL_base``. Omit to skip the check (``nan`` reported).
    ess_floor : float, optional
        Kish ESS below which the estimate is also flagged unreliable. Defaults to
        100.
    n_bootstrap, seed
        Bootstrap controls.

    Returns
    -------
    ExtendResult
        The evidence, diagnostics, and reliability flag.
    """
    base_weights = np.asarray(base_weights, dtype=float)
    logL_base = np.asarray(logL_base, dtype=float)
    logL_ext = np.asarray(logL_ext, dtype=float)
    log_prior_correction = np.asarray(log_prior_correction, dtype=float)
    n = base_weights.size

    evidence = importance_evidence(
        logL_ext,
        logL_base,
        base_weights=base_weights,
        log_prior_correction=log_prior_correction,
        base_log_evidence=base_log_evidence,
        base_log_evidence_err=base_log_evidence_err,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Combined importance weights W_i * (L_ext/L_base) * (pi/q), in log space.
    log_w = (logL_ext - logL_base) + log_prior_correction
    with np.errstate(divide="ignore"):
        log_combined = np.log(base_weights) + log_w
    finite = np.isfinite(log_combined)
    if np.any(finite):
        shifted = np.exp(log_combined - log_combined[finite].max())
        shifted[~finite] = 0.0
        ess = effective_sample_size(shifted)
        total = shifted.sum()
        max_weight = float(shifted.max() / total) if total > 0 else 1.0
    else:
        ess, max_weight = 0.0, 1.0

    khat = psis(log_combined).pareto_k

    if stored_logL is not None:
        stored_logL = np.asarray(stored_logL, dtype=float)
        both = np.isfinite(stored_logL) & np.isfinite(logL_base)
        consistency = (
            float(np.max(np.abs(stored_logL[both] - logL_base[both])))
            if np.any(both)
            else float("nan")
        )
    else:
        consistency = float("nan")

    reliable = bool(
        np.isfinite(khat) and khat < 0.7 and ess >= ess_floor
    )

    return ExtendResult(
        log_bayes_factor=evidence.log_ratio,
        log_bayes_factor_err=float(
            np.hypot(evidence.log_ratio_err, base_log_evidence_err)
        ),
        log_evidence_ext=evidence.log_evidence,
        log_evidence_ext_err=evidence.log_evidence_err,
        effective_sample_size=ess,
        ess_fraction=float(ess / n) if n else 0.0,
        pareto_k=khat,
        max_weight=max_weight,
        base_logL_consistency=consistency,
        degenerate_fraction=evidence.bootstrap_degenerate_fraction,
        n_samples=int(n),
        reliable=reliable,
    )


def run_extension(
    base_samples,
    target_model,
    calculator,
    lcdm_limit,
    base_log_evidence,
    base_log_evidence_err=0.0,
    rng=None,
    use_parallel=True,
    n_cores=None,
    ess_floor=100.0,
    n_bootstrap=1000,
    seed=None,
):
    r"""Reweight ``base_samples`` onto the extended model and return the evidence.

    Wires the Cobaya likelihood to :func:`extend_evidence`. The extended
    likelihood is evaluated twice per sample: once at the drawn new-parameter
    value, and once in the LCDM limit to form a self-consistent baseline.

    Parameters
    ----------
    base_samples : anesthetic.NestedSamples
        The base-model nested run, carrying ``logL`` and posterior weights.
    target_model : cosmocompare.models.TargetModel
        The extended model (its new parameters and their priors/proposals).
    calculator : cosmocompare.likelihood.LikelihoodCalculator
        The extended-model Cobaya likelihood.
    lcdm_limit : dict
        New-parameter values recovering the base model (for example
        ``{"w": -1.0}`` or ``{"omegak": 0.0}``).
    base_log_evidence, base_log_evidence_err : float
        The base model's nested ``log Z`` and error.
    rng : numpy.random.Generator, optional
        Randomness for the augmentation.
    use_parallel, n_cores
        Passed to the likelihood evaluation.
    ess_floor, n_bootstrap, seed
        Passed to :func:`extend_evidence`.

    Returns
    -------
    tuple
        ``(ExtendResult, posterior)`` where ``posterior`` is the base run
        reweighted onto the extended model (for parameter estimation; its nested
        ``logZ`` is meaningless, use the result's evidence).
    """
    from cosmocompare.reweight import apply_log_weights

    rng = np.random.default_rng() if rng is None else rng

    # Augment with the new parameter(s) drawn from their proposal.
    augmented = target_model.augment_samples(base_samples, rng=rng)
    logL_ext = calculator.log_likelihood(
        augmented, use_parallel=use_parallel, n_cores=n_cores
    )

    # Re-evaluate the base likelihood in the LCDM limit with the SAME calculator.
    base_frame = augmented.copy()
    for name, value in lcdm_limit.items():
        base_frame[name] = value
    logL_base = calculator.log_likelihood(
        base_frame, use_parallel=use_parallel, n_cores=n_cores
    )

    correction = target_model.log_prior_correction(augmented)
    base_weights = np.asarray(augmented.get_weights(), dtype=float)
    stored_logL = np.asarray(augmented.logL, dtype=float)

    result = extend_evidence(
        base_weights,
        logL_base,
        logL_ext,
        correction,
        base_log_evidence,
        base_log_evidence_err=base_log_evidence_err,
        stored_logL=stored_logL,
        ess_floor=ess_floor,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    log_w = (logL_ext - logL_base) + correction
    posterior = apply_log_weights(augmented, log_w)
    return result, posterior
