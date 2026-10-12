"""Tests for prior and proposal distributions."""

import numpy as np
import pytest

from cosmocompare.parameters import Parameter, _Distribution


def test_uniform_sample_within_bounds():
    """Uniform samples fall inside the configured interval."""
    dist = _Distribution({"type": "uniform", "min": -2.0, "max": 0.0})
    samples = dist.sample(1000, rng=np.random.default_rng(0))
    assert samples.min() >= -2.0
    assert samples.max() <= 0.0


def test_log_uniform_logpdf_out_of_bounds_is_neg_inf():
    """Log-uniform density is -inf outside its support."""
    dist = _Distribution({"type": "log_uniform", "min": 1e-3, "max": 1e-1})
    values = np.array([1e-4, 1e-2, 1.0])
    log_pdf = dist.logpdf(values)
    assert np.isneginf(log_pdf[0])
    assert np.isfinite(log_pdf[1])
    assert np.isneginf(log_pdf[2])


def test_unsupported_type_raises():
    """An unknown distribution family is rejected at construction."""
    with pytest.raises(ValueError):
        _Distribution({"type": "cauchy", "x0": 0.0})


def test_correction_is_zero_without_proposal():
    """A parameter with no explicit proposal contributes no correction."""
    param = Parameter("w", prior={"type": "uniform", "min": -3.0, "max": 1.0})
    values = param.sample(100, rng=np.random.default_rng(1))
    assert np.allclose(param.log_correction(values), 0.0)


def test_correction_matches_prior_minus_proposal():
    """The correction equals log prior minus log proposal when they differ."""
    param = Parameter(
        "w",
        prior={"type": "uniform", "min": -3.0, "max": 1.0},
        proposal={"type": "gaussian", "mean": -1.0, "std": 0.3},
    )
    values = np.array([-1.0, -0.5, -2.0])
    expected = param.log_prior(values) - param.log_proposal(values)
    assert np.allclose(param.log_correction(values), expected)


def test_truncated_gaussian_respects_bounds():
    """Truncated-Gaussian samples stay inside the truncation bounds."""
    dist = _Distribution(
        {"type": "truncated_gaussian", "mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0}
    )
    samples = dist.sample(1000, rng=np.random.default_rng(2))
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
