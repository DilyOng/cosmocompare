"""Tests for power-scaling prior deformation (dependency-light part)."""

import os

import numpy as np
import pytest

from cosmocompare.powerscale import power_scaled_logdensity

BOUNDS = [(-3.0, 1.0), (-3.0, 2.0)]


def _peaked_target(centre=(-1.0, 0.0), sigma=0.3):
    mu = np.asarray(centre, dtype=float)

    def log_shape(a):
        a = np.asarray(a, dtype=float)
        return -0.5 * np.sum(((a - mu) / sigma) ** 2, axis=1)

    return log_shape


def test_alpha_zero_is_flat():
    """At alpha=0 the power-scaled prior is the (flat) base: ratio ~ 0 everywhere."""
    prior = power_scaled_logdensity(BOUNDS, _peaked_target(), alpha=0.0, n_mc=100_000)
    samples = {"w": np.array([-1.0, -2.5, 0.5]), "wa": np.array([0.0, 1.0, -1.0])}
    lr = prior(samples)
    assert np.allclose(lr, 0.0, atol=1e-6)


def test_higher_alpha_concentrates_at_target_peak():
    """Density at the target peak grows with alpha; density far away shrinks."""
    peak = {"w": np.array([-1.0]), "wa": np.array([0.0])}
    far = {"w": np.array([0.5]), "wa": np.array([1.5])}
    vals_peak, vals_far = [], []
    for alpha in (0.25, 0.5, 1.0):
        prior = power_scaled_logdensity(
            BOUNDS,
            _peaked_target(),
            alpha=alpha,
            n_mc=200_000,
            rng=np.random.default_rng(0),
        )
        vals_peak.append(float(prior(peak)[0]))
        vals_far.append(float(prior(far)[0]))
    # Monotonic increase at the peak, decrease in the tail.
    assert vals_peak[0] < vals_peak[1] < vals_peak[2]
    assert vals_far[0] > vals_far[1] > vals_far[2]


def test_outside_box_is_minus_inf():
    """Points outside the base box get zero prior (-inf log density)."""
    prior = power_scaled_logdensity(BOUNDS, _peaked_target(), alpha=1.0, n_mc=50_000)
    out = {"w": np.array([2.0]), "wa": np.array([0.0])}  # w=2 outside [-3, 1]
    assert np.isneginf(prior(out)[0])


# --- integration: real chains, env-gated ---
_EXT = os.environ.get("COSMOCOMPARE_TEST_EXT_CHAIN")
_BASE = os.environ.get("COSMOCOMPARE_TEST_BASE_CHAIN")


@pytest.mark.skipif(
    not (_EXT and _BASE),
    reason="set COSMOCOMPARE_TEST_EXT_CHAIN and _BASE_CHAIN to chain roots",
)
def test_power_scale_sweep_on_real_chains():
    """The power-scaling sweep runs and returns an ln B per alpha."""
    pytest.importorskip("anesthetic")
    from anesthetic import read_chains

    from cosmocompare.powerscale import power_scale_sweep

    ext = read_chains(_EXT)
    base = read_chains(_BASE)
    rows = power_scale_sweep(
        ext,
        base,
        BOUNDS,
        _peaked_target(),
        alphas=(0.0, 0.5, 1.0),
        n_bootstrap=50,
        seed=0,
        n_mc=100_000,
    )
    assert len(rows) == 3
    assert all("log_bayes_factor" in r and "alpha" in r for r in rows)
