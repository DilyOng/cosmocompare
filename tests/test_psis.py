"""Tests for the PSIS Pareto-k-hat reliability diagnostic."""

import numpy as np

from cosmocompare.psis import pareto_smooth, psis


def test_light_tail_has_small_k():
    """Exponential-tailed weights give a low, finite Pareto k-hat."""
    rng = np.random.default_rng(0)
    weights = rng.exponential(1.0, size=5000)
    result = psis(np.log(weights))
    assert np.isfinite(result.pareto_k)
    assert result.pareto_k < 0.7
    assert result.reliable


def test_heavy_tail_has_larger_k_than_light():
    """A Pareto (heavy) tail yields a larger k-hat than an exponential tail."""
    rng = np.random.default_rng(1)
    light = rng.exponential(1.0, size=5000)
    # Pareto with shape ~0.8: u^{-1/alpha} tail.
    heavy = rng.uniform(size=5000) ** (-1.0 / 0.8)
    k_light = psis(np.log(light)).pareto_k
    k_heavy = psis(np.log(heavy)).pareto_k
    assert k_heavy > k_light


def test_hard_cut_weights_return_nan_k():
    """A spike of equal weights plus zeros has no tail to fit; k-hat is nan."""
    log_w = np.full(2000, -np.inf)
    log_w[:6] = 0.0  # six equal surviving weights, rest zero
    result = psis(log_w)
    assert np.isnan(result.pareto_k)
    assert not result.reliable


def test_pareto_smooth_preserves_length_and_zeros():
    """Smoothing returns one weight per input and leaves -inf entries alone."""
    rng = np.random.default_rng(2)
    log_w = np.log(rng.exponential(1.0, size=1000))
    log_w[::50] = -np.inf
    smoothed, k, n_tail = pareto_smooth(log_w)
    assert smoothed.shape == log_w.shape
    assert np.all(np.isneginf(smoothed[::50]))
    assert n_tail > 0 and np.isfinite(k)


def test_smoothing_caps_at_raw_maximum():
    """No smoothed weight exceeds the largest raw weight."""
    rng = np.random.default_rng(3)
    log_w = np.log(rng.uniform(size=3000) ** (-1.0 / 0.7))
    smoothed, _, _ = pareto_smooth(log_w)
    assert np.nanmax(smoothed) <= np.max(log_w) + 1e-9
