"""Tests for the cross-model importance-sampling estimator (pure, no Cobaya)."""

import numpy as np

from cosmocompare.extend import extend_evidence


def test_constant_ratio_recovers_bayes_factor():
    """A constant log-likelihood ratio delta gives ln B = delta, ESS = N."""
    n = 2000
    base_weights = np.ones(n)
    logL_base = np.linspace(-10, 0, n)
    logL_ext = logL_base + 0.5  # constant ratio -> Z_ext/Z_base = e^0.5
    r = extend_evidence(
        base_weights, logL_base, logL_ext, np.zeros(n),
        base_log_evidence=-100.0, base_log_evidence_err=0.1, n_bootstrap=100, seed=0,
    )
    assert np.isclose(r.log_bayes_factor, 0.5, atol=1e-6)
    assert np.isclose(r.log_evidence_ext, -99.5, atol=1e-6)  # base_logZ + lnB
    assert r.ess_fraction > 0.99  # equal combined weights
    assert r.n_samples == n


def test_weighted_estimator_matches_direct_sum():
    """ln B equals log(sum W exp(dL) / sum W) for arbitrary weights and ratios."""
    rng = np.random.default_rng(1)
    n = 1500
    base_weights = rng.exponential(1.0, n)
    logL_base = rng.normal(0, 2, n)
    dL = rng.normal(-0.3, 0.4, n)
    logL_ext = logL_base + dL
    expected = np.log(np.sum(base_weights * np.exp(dL)) / np.sum(base_weights))
    r = extend_evidence(base_weights, logL_base, logL_ext, np.zeros(n),
                        base_log_evidence=0.0, n_bootstrap=50, seed=0)
    assert np.isclose(r.log_bayes_factor, expected, atol=1e-9)


def test_support_mismatch_collapses_ess():
    """When one sample carries the ratio, ESS collapses and k-hat flags it."""
    n = 2000
    base_weights = np.ones(n)
    logL_base = np.zeros(n)
    logL_ext = np.zeros(n)
    logL_ext[0] = 25.0  # a single sample dominates the extended likelihood
    r = extend_evidence(base_weights, logL_base, logL_ext, np.zeros(n),
                        base_log_evidence=0.0, n_bootstrap=50, seed=0)
    assert r.effective_sample_size < 5
    assert r.max_weight > 0.9
    assert not r.reliable


def test_prior_correction_enters_the_ratio():
    """A nonzero log(pi/q) correction shifts ln B by its weighted average."""
    n = 1000
    base_weights = np.ones(n)
    logL_base = np.zeros(n)
    logL_ext = np.zeros(n)
    corr = np.full(n, -0.7)  # pi/q = e^-0.7 everywhere
    r = extend_evidence(base_weights, logL_base, logL_ext, corr,
                        base_log_evidence=0.0, n_bootstrap=50, seed=0)
    assert np.isclose(r.log_bayes_factor, -0.7, atol=1e-6)


def test_rejected_samples_carry_zero_weight():
    """-inf extended likelihoods drop out without poisoning the estimate."""
    n = 1000
    base_weights = np.ones(n)
    logL_base = np.zeros(n)
    logL_ext = np.zeros(n)
    logL_ext[::2] = -np.inf  # half the augmented points are unphysical
    r = extend_evidence(base_weights, logL_base, logL_ext, np.zeros(n),
                        base_log_evidence=0.0, n_bootstrap=50, seed=0)
    # Z_ext/Z_base = (n/2 surviving) / n = 1/2 -> ln B = -log 2.
    assert np.isclose(r.log_bayes_factor, -np.log(2), atol=1e-6)


def test_consistency_check():
    """base_logL_consistency reports the max stored-vs-fresh discrepancy."""
    n = 500
    base_weights = np.ones(n)
    logL_base = np.linspace(-5, 0, n)
    r = extend_evidence(base_weights, logL_base, logL_base + 0.2, np.zeros(n),
                        base_log_evidence=0.0, stored_logL=logL_base + 0.003,
                        n_bootstrap=20, seed=0)
    assert np.isclose(r.base_logL_consistency, 0.003, atol=1e-9)
