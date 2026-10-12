"""Tests for the weight diagnostics."""

import numpy as np

from cosmocompare.diagnostics import effective_sample_size, efficiency


def test_equal_weights_give_full_ess():
    """Equal weights have an effective sample size equal to the count."""
    weights = np.ones(100)
    assert effective_sample_size(weights) == 100
    assert efficiency(weights) == 1.0


def test_concentrated_weights_collapse_ess():
    """One dominant weight drives the effective sample size towards 1."""
    weights = np.zeros(100)
    weights[0] = 1.0
    assert effective_sample_size(weights) == 1.0


def test_non_finite_weights_ignored():
    """Non-finite weights are dropped rather than poisoning the estimate."""
    weights = np.array([1.0, 1.0, np.inf, np.nan])
    assert effective_sample_size(weights) == 2.0


def test_empty_weights_return_zero():
    """An empty weight vector yields zero, not an error."""
    assert effective_sample_size(np.array([])) == 0.0
    assert efficiency(np.array([])) == 0.0
