"""Tests for the weight update."""

import numpy as np

from cosmocompare.reweight import apply_log_weights, log_importance_weights


class _FakeSamples:
    """Minimal weighted-samples stand-in: get/set weights and copy."""

    def __init__(self, weights):
        self._weights = np.asarray(weights, dtype=float)

    def get_weights(self):
        """Return the current weights."""
        return self._weights

    def set_weights(self, weights, inplace=False):
        """Set the weights (always in place for this stub)."""
        self._weights = np.asarray(weights, dtype=float)

    def copy(self):
        """Return an independent copy."""
        return _FakeSamples(self._weights.copy())


def test_log_weights_without_correction_passthrough():
    """With no correction the corrected likelihood is the input unchanged."""
    logL = np.array([1.0, 2.0, 3.0])
    assert np.allclose(log_importance_weights(logL), logL)


def test_log_weights_add_correction():
    """The correction is added term by term."""
    logL = np.array([1.0, 2.0])
    corr = np.array([0.5, -0.5])
    assert np.allclose(log_importance_weights(logL, corr), [1.5, 1.5])


def test_apply_log_weights_multiplies_and_preserves_ratios():
    """Weights are multiplied by exp(log_weights) up to overall scale."""
    base = _FakeSamples([1.0, 1.0, 2.0])
    out = apply_log_weights(base, np.log([2.0, 1.0, 1.0]))
    w = out.get_weights()
    # Ratios follow W_i * exp(log_w_i) = [2, 1, 2] up to a constant factor.
    assert np.allclose(w / w[1], [2.0, 1.0, 2.0])


def test_apply_log_weights_zeroes_out_of_support():
    """A -inf log weight (outside a narrowed prior) gives zero weight."""
    base = _FakeSamples([1.0, 1.0, 1.0])
    out = apply_log_weights(base, np.array([0.0, -np.inf, 0.0]))
    assert out.get_weights()[1] == 0.0


def test_apply_log_weights_does_not_mutate_input():
    """The base object's weights are left untouched."""
    base = _FakeSamples([1.0, 1.0])
    apply_log_weights(base, np.log([5.0, 1.0]))
    assert np.allclose(base.get_weights(), [1.0, 1.0])
