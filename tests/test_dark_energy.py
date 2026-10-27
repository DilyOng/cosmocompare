"""Tests for the named dark-energy prior narrowings (dependency-light part).

The region predicates and the shaped densities are pure numpy/scipy, so they are
tested directly on arrays and on a dict standing in for a samples table. The
full :func:`sensitivity_panel` needs real nested chains and is gated on env vars,
like the sweep integration test.
"""

import os

import numpy as np
import pytest

from cosmocompare.dark_energy import (
    box_normalised_logdensity,
    gaussian_shrink,
    no_phantom,
    physicality_cut,
    thawing_band,
)

BOUNDS = [(-3.0, 1.0), (-3.0, 2.0)]  # (w0, wa) DESI-style box


# --- region cuts -----------------------------------------------------------


def test_physicality_cut_keeps_sum_negative():
    """w0+wa<0 is kept, >=0 is rejected."""
    _, inside = physicality_cut()
    a = np.array([[-1.0, 0.0], [-0.5, 1.0], [0.5, -0.7], [0.5, 0.5]])
    assert list(inside(a)) == [True, False, True, False]


def test_no_phantom_requires_both_endpoints():
    """w(a)>=-1 everywhere reduces to w0>=-1 and w0+wa>=-1."""
    _, inside = no_phantom()
    a = np.array(
        [
            [-1.0, 0.0],  # LCDM: on the boundary, kept
            [-0.9, 0.5],  # w0>=-1, sum=-0.4>=-1: kept
            [-1.2, 0.5],  # w0<-1 (phantom today): rejected
            [-0.9, -0.5],  # sum=-1.4<-1 (phantom early): rejected
        ]
    )
    assert list(inside(a)) == [True, True, False, False]


# --- box-normalised shaped densities ---------------------------------------


def test_box_normalisation_integrates_to_one():
    """A box-normalised density averages to ~1 over the base box (E_box[pi]=1)."""
    # Flat shape -> normalised density is exactly uniform, mean exp(logpi)=1.
    log_prior = box_normalised_logdensity(
        lambda a: np.zeros(len(a)),
        ("w", "wa"),
        BOUNDS,
        n_mc=200_000,
        rng=np.random.default_rng(0),
    )
    rng = np.random.default_rng(1)
    draws = rng.uniform([-3, -3], [1, 2], size=(100_000, 2))
    samples = {"w": draws[:, 0], "wa": draws[:, 1]}
    mean_density = np.mean(np.exp(log_prior(samples)))
    assert np.isclose(mean_density, 1.0, atol=0.02)


def test_gaussian_shrink_peaks_at_lcdm_and_vanishes_outside_box():
    """The shrink prior is densest at LCDM and -inf outside the base box."""
    log_prior = gaussian_shrink(BOUNDS, sigma=0.3, n_mc=100_000)
    samples = {
        "w": np.array([-1.0, -2.5, 2.0]),  # LCDM, far-but-in-box, outside box
        "wa": np.array([0.0, 1.5, 0.0]),
    }
    lp = log_prior(samples)
    assert lp[0] > lp[1]  # denser at LCDM than in the tail
    assert np.isneginf(lp[2])  # w=2 is outside [-3, 1]


def test_gaussian_shrink_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        gaussian_shrink(BOUNDS, sigma=0.0)


def test_thawing_band_peaks_on_the_track():
    """Density is higher on wa=-1.58(1+w0) than off it."""
    log_prior = thawing_band(BOUNDS, slope=-1.58, width=0.1, n_mc=100_000)
    w0 = np.array([-0.8, -0.8])
    on_track = -1.58 * (1.0 + w0[0])
    samples = {"w": w0, "wa": np.array([on_track, on_track + 1.0])}
    lp = log_prior(samples)
    assert lp[0] > lp[1]


def test_thawing_band_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        thawing_band(BOUNDS, width=0.0)


# --- integration: real chains, no Cobaya -----------------------------------
_EXT = os.environ.get("COSMOCOMPARE_TEST_EXT_CHAIN")
_BASE = os.environ.get("COSMOCOMPARE_TEST_BASE_CHAIN")


@pytest.mark.skipif(
    not (_EXT and _BASE),
    reason="set COSMOCOMPARE_TEST_EXT_CHAIN and _BASE_CHAIN to chain roots",
)
def test_sensitivity_panel_on_real_chains():
    """The panel runs, returns the baseline first, and flags low-ESS rows."""
    pytest.importorskip("anesthetic")
    from anesthetic import read_chains

    from cosmocompare.dark_energy import sensitivity_panel

    ext = read_chains(_EXT)
    base = read_chains(_BASE)
    rows = sensitivity_panel(
        ext, base, BOUNDS, sigmas=(1.0, 0.5), n_bootstrap=50, seed=0, n_mc=100_000
    )
    assert rows[0]["prior"].startswith("baseline")
    assert all("log_bayes_factor" in r and "effective_sample_size" in r for r in rows)
    assert all(isinstance(r["reliable"], bool) for r in rows)
