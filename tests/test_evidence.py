"""Tests for the evidence-ratio estimator."""

import numpy as np

from cosmocompare.evidence import EvidenceResult, importance_evidence, log_bayes_factor


def test_constant_likelihood_shift_gives_that_shift():
    """A uniform log-likelihood shift of c gives a log evidence ratio of c."""
    n = 500
    base = np.zeros(n)
    new = np.full(n, 1.5)
    result = importance_evidence(new, base, n_bootstrap=100, seed=0)
    assert np.isclose(result.log_ratio, 1.5)
    # No spread in the weights, so the bootstrap error is essentially zero.
    assert result.log_ratio_err < 1e-9


def test_identical_models_give_zero_ratio():
    """Reweighting onto the same likelihood leaves the evidence unchanged."""
    rng = np.random.default_rng(1)
    logL = rng.normal(size=1000)
    result = importance_evidence(logL, logL, n_bootstrap=100, seed=1)
    assert np.isclose(result.log_ratio, 0.0)


def test_bootstrap_error_is_positive_when_weights_vary():
    """Varying weights produce a non-zero bootstrap uncertainty."""
    rng = np.random.default_rng(2)
    base = np.zeros(2000)
    new = rng.normal(scale=2.0, size=2000)
    result = importance_evidence(new, base, n_bootstrap=200, seed=2)
    assert result.log_ratio_err > 0.0
    assert 0.0 < result.effective_sample_size <= 2000


def test_absolute_evidence_combines_base_error():
    """Absolute evidence shifts by the ratio and inflates the error."""
    base = np.zeros(100)
    new = np.full(100, 0.7)
    result = importance_evidence(
        new,
        base,
        base_log_evidence=-10.0,
        base_log_evidence_err=0.3,
        n_bootstrap=50,
        seed=3,
    )
    assert np.isclose(result.log_evidence, -10.0 + result.log_ratio)
    assert result.log_evidence_err >= 0.3


def test_log_bayes_factor_from_tuples():
    """The Bayes factor subtracts evidences and adds errors in quadrature."""
    lbf, err = log_bayes_factor((-5.0, 0.3), (-7.0, 0.4))
    assert np.isclose(lbf, 2.0)
    assert np.isclose(err, np.hypot(0.3, 0.4))


def test_log_bayes_factor_from_results():
    """EvidenceResult instances feed straight into the Bayes factor."""
    a = EvidenceResult(0.0, 0.0, log_evidence=-5.0, log_evidence_err=0.1)
    b = EvidenceResult(0.0, 0.0, log_evidence=-6.0, log_evidence_err=0.2)
    lbf, err = log_bayes_factor(a, b)
    assert np.isclose(lbf, 1.0)
    assert np.isclose(err, np.hypot(0.1, 0.2))
