"""Tests for the Savage-Dickey density ratio.

Dependency-light: a fake sample set stands in for anesthetic so no chains are read.
"""

import numpy as np
import pytest

from cosmocompare.sddr import _uniform_box_log_density, savage_dickey


class FakeSamples:
    """Minimal stand-in for an anesthetic sample set for the SDDR KDE."""

    def __init__(self, data, weights):
        self._data = data
        self._w = weights

    def __getitem__(self, key):
        return self._data[key]

    def get_weights(self):
        return self._w


def _gaussian_samples(mean, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(mean[0], 0.3, size=n)
    wa = rng.normal(mean[1], 0.5, size=n)
    return FakeSamples({"w": w, "wa": wa}, np.ones(n))


def test_uniform_box_log_density():
    """Uniform box density is 1 / volume."""
    val = _uniform_box_log_density([(-3.0, 1.0), (-3.0, 2.0)])  # volume 4*5 = 20
    assert np.isclose(val, -np.log(20.0))


def test_posterior_piled_at_point_favours_base():
    """When the posterior sits on the nesting point, ln B is negative (favours base)."""
    pytest.importorskip("scipy")
    samples = _gaussian_samples((-1.0, 0.0))
    res = savage_dickey(
        samples, ("w", "wa"), {"w": -1.0, "wa": 0.0},
        prior_bounds=[(-3.0, 1.0), (-3.0, 2.0)],
    )
    assert res.log_bayes_factor < 0
    assert res.inside_hdr  # the point is in the bulk


def test_posterior_far_from_point_favours_extended_and_flags_tail():
    """When the posterior avoids the point, ln B is positive and the tail is flagged."""
    pytest.importorskip("scipy")
    samples = _gaussian_samples((-0.5, -1.5))  # far from (-1, 0)
    res = savage_dickey(
        samples, ("w", "wa"), {"w": -1.0, "wa": 0.0},
        prior_bounds=[(-3.0, 1.0), (-3.0, 2.0)],
    )
    assert res.log_bayes_factor > 0
    assert not res.inside_hdr  # (-1, 0) is in the tail here


def test_requires_a_prior_specification():
    """Omitting both prior_bounds and log_prior_density raises."""
    samples = _gaussian_samples((-1.0, 0.0))
    with pytest.raises(ValueError):
        savage_dickey(samples, ("w", "wa"), {"w": -1.0, "wa": 0.0})


def test_point_outside_bounds_raises():
    """A nesting point outside the prior bounds is rejected (zero prior density)."""
    samples = _gaussian_samples((-1.0, 0.0))
    with pytest.raises(ValueError):
        savage_dickey(
            samples, ("w", "wa"), {"w": 5.0, "wa": 0.0},
            prior_bounds=[(-3.0, 1.0), (-3.0, 2.0)],
        )


def test_hdr_mass_reported_and_consistent():
    """hdr_mass is in [0, 1] and tracks inside_hdr."""
    pytest.importorskip("scipy")
    samples = _gaussian_samples((-1.0, 0.0))
    res = savage_dickey(
        samples, ("w", "wa"), {"w": -1.0, "wa": 0.0},
        prior_bounds=[(-3.0, 1.0), (-3.0, 2.0)],
    )
    assert 0.0 <= res.hdr_mass <= 1.0
    assert res.inside_hdr == (res.hdr_mass < 0.99)
