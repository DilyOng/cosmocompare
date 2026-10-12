"""Tests for the weight update and column alignment."""

import numpy as np
import pandas as pd

from cosmocompare.reweight import (
    attach_new_parameters,
    importance_sample,
    log_importance_weights,
)


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


def test_attach_parameters_all_retained():
    """When nothing is dropped, every value is copied straight across."""
    augmented = pd.DataFrame({"w": [0.1, 0.2, 0.3]})
    posterior = {}
    posterior = _DictPosterior(3)
    attach_new_parameters(posterior, augmented, ["w"], np.array([True, True, True]))
    assert np.allclose(posterior["w"], [0.1, 0.2, 0.3])


def test_attach_parameters_finite_subset():
    """When the sampler drops -inf rows, the finite values are selected."""
    augmented = pd.DataFrame({"w": [0.1, 0.2, 0.3]})
    posterior = _DictPosterior(2)
    mask = np.array([True, False, True])
    attach_new_parameters(posterior, augmented, ["w"], mask)
    assert np.allclose(posterior["w"], [0.1, 0.3])


class _DictPosterior:
    """A length-aware dict supporting column assignment, like a frame."""

    def __init__(self, n):
        self.n = n
        self._data = {}

    def __len__(self):
        return self.n

    def __setitem__(self, key, value):
        self._data[key] = np.asarray(value)

    def __getitem__(self, key):
        return self._data[key]
