"""Tests for the weight update."""

import numpy as np

from cosmocompare.reweight import importance_sample, log_importance_weights


class _FakeSamples:
    """Minimal stand-in exposing anesthetic's importance_sample signature."""

    def __init__(self, n):
        self.n = n
        self.captured = {}

    def importance_sample(self, logL, action, inplace):
        """Record the arguments and return a small dict-backed posterior."""
        self.captured = {"logL": np.asarray(logL), "action": action, "inplace": inplace}
        return {}


def test_log_weights_without_correction_passthrough():
    """With no correction the corrected likelihood is the input unchanged."""
    logL = np.array([1.0, 2.0, 3.0])
    assert np.allclose(log_importance_weights(logL), logL)


def test_log_weights_add_correction():
    """The correction is added term by term."""
    logL = np.array([1.0, 2.0])
    corr = np.array([0.5, -0.5])
    assert np.allclose(log_importance_weights(logL, corr), [1.5, 1.5])


def test_importance_sample_replaces_with_corrected_likelihood():
    """The sampler is called with the corrected likelihood and action=replace."""
    base = _FakeSamples(3)
    logL = np.array([1.0, 2.0, 3.0])
    corr = np.array([0.1, 0.2, 0.3])
    importance_sample(base, logL, corr)
    assert base.captured["action"] == "replace"
    assert base.captured["inplace"] is False
    assert np.allclose(base.captured["logL"], logL + corr)
