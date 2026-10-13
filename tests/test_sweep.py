"""Tests for the prior-narrowing sweep helpers (dependency-light part)."""

import os

import numpy as np
import pytest

from cosmocompare.sweep import flip_fraction, shrink_uniform


def test_full_fraction_recovers_base():
    """A fraction of 1 returns the original bounds."""
    out = shrink_uniform((-3.0, 1.0), 1.0)
    assert np.isclose(out["min"], -3.0) and np.isclose(out["max"], 1.0)


def test_homothetic_about_midpoint():
    """Halving about the midpoint halves the width, same centre."""
    out = shrink_uniform((-3.0, 1.0), 0.5)  # midpoint -1
    assert np.isclose(out["min"], -2.0) and np.isclose(out["max"], 0.0)


def test_homothetic_about_offset_centre():
    """Shrinking about an offset centre moves both bounds towards it."""
    out = shrink_uniform((-3.0, 2.0), 0.5, centre=0.0)  # wa-like, LCDM centre 0
    assert np.isclose(out["min"], -1.5) and np.isclose(out["max"], 1.0)


def test_result_contained_in_base():
    """Any valid shrink stays within the original bounds."""
    for f in (0.1, 0.5, 0.9, 1.0):
        out = shrink_uniform((-3.0, 1.0), f, centre=-1.0)
        assert out["min"] >= -3.0 - 1e-12 and out["max"] <= 1.0 + 1e-12


def test_centre_outside_base_raises():
    """A centre outside the base bounds is rejected."""
    with pytest.raises(ValueError):
        shrink_uniform((-3.0, 1.0), 0.5, centre=2.0)


def test_fraction_out_of_range_raises():
    """A fraction outside (0, 1] is rejected."""
    with pytest.raises(ValueError):
        shrink_uniform((-3.0, 1.0), 1.5)
    with pytest.raises(ValueError):
        shrink_uniform((-3.0, 1.0), 0.0)


def test_flip_fraction_interpolates_crossing():
    """The flip fraction is the linear-interpolated zero crossing of ln B."""
    records = [
        {"fraction": 1.0, "log_bayes_factor": -0.5},
        {"fraction": 0.5, "log_bayes_factor": +0.5},
        {"fraction": 0.25, "log_bayes_factor": +1.0},
    ]
    assert np.isclose(flip_fraction(records), 0.75)


def test_flip_fraction_none_without_crossing():
    """No sign change over the range returns None."""
    records = [
        {"fraction": 1.0, "log_bayes_factor": -0.5},
        {"fraction": 0.5, "log_bayes_factor": -0.2},
    ]
    assert flip_fraction(records) is None


def test_flip_fraction_exact_zero():
    """An exact zero at a sampled width is returned as the crossing."""
    records = [
        {"fraction": 1.0, "log_bayes_factor": -0.3},
        {"fraction": 0.5, "log_bayes_factor": 0.0},
        {"fraction": 0.25, "log_bayes_factor": 0.4},
    ]
    assert flip_fraction(records) == 0.5


# --- Integration: real chains, no Cobaya. Set both env vars to a chain root each.
_EXT = os.environ.get("COSMOCOMPARE_TEST_EXT_CHAIN")
_BASE = os.environ.get("COSMOCOMPARE_TEST_BASE_CHAIN")
_PARAM = os.environ.get("COSMOCOMPARE_TEST_SWEEP_PARAM", "w")
_BOUNDS = os.environ.get("COSMOCOMPARE_TEST_SWEEP_BOUNDS", "-3,-0.333")


@pytest.mark.skipif(
    not (_EXT and _BASE),
    reason="set COSMOCOMPARE_TEST_EXT_CHAIN and _BASE_CHAIN to chain roots",
)
def test_bayes_factor_sweep_on_real_chains():
    """A homothetic prior sweep runs, recovers f=1, and degrades ESS as it narrows."""
    pytest.importorskip("anesthetic")
    from anesthetic import read_chains

    from cosmocompare.sweep import bayes_factor_sweep

    ext = read_chains(_EXT)
    base = read_chains(_BASE)
    lo, hi = (float(x) for x in _BOUNDS.split(","))
    centre = 0.5 * (lo + hi)

    records = bayes_factor_sweep(
        ext,
        base,
        {_PARAM: (lo, hi)},
        fractions=[1.0, 0.5, 0.2],
        centres={_PARAM: centre},
        n_bootstrap=50,
        seed=0,
    )
    assert len(records) == 3
    # Full width is a finite Bayes factor; the window stays within the base prior.
    assert np.isfinite(records[0]["log_bayes_factor"])
    assert records[0]["bounds"][_PARAM][0] >= lo - 1e-9
    assert records[0]["bounds"][_PARAM][1] <= hi + 1e-9
    # Narrowing cannot increase the effective sample size.
    ess = [r["effective_sample_size"] for r in records]
    assert ess[0] >= ess[-1] > 0
