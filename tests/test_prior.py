"""Tests for the prior-change log ratio (dependency-light part)."""

import numpy as np
import pandas as pd
import pytest

from cosmocompare.prior import prior_log_ratio


def test_uniform_narrowing_ratio():
    """Narrowing a uniform prior gives log(width_old/width_new) inside, -inf out."""
    samples = pd.DataFrame({"x": [0.1, 0.5, 0.9]})
    changes = {
        "x": {
            "base": {"type": "uniform", "min": 0.0, "max": 1.0},
            "new": {"type": "uniform", "min": 0.25, "max": 0.75},
        }
    }
    log_r = prior_log_ratio(samples, changes)
    assert np.isneginf(log_r[0])  # 0.1 is outside the new range
    assert np.isclose(log_r[1], np.log(1.0 / 0.5))  # inside: width 1.0 -> 0.5
    assert np.isneginf(log_r[2])  # 0.9 is outside


def test_unchanged_prior_gives_zero():
    """An identical base and new prior contributes nothing."""
    samples = pd.DataFrame({"x": [0.2, 0.4, 0.6]})
    changes = {
        "x": {
            "base": {"type": "uniform", "min": 0.0, "max": 1.0},
            "new": {"type": "uniform", "min": 0.0, "max": 1.0},
        }
    }
    assert np.allclose(prior_log_ratio(samples, changes), 0.0)


def test_multiple_parameters_sum():
    """Ratios from several parameters add."""
    samples = pd.DataFrame({"x": [0.5], "y": [0.5]})
    changes = {
        "x": {
            "base": {"type": "uniform", "min": 0.0, "max": 1.0},
            "new": {"type": "uniform", "min": 0.0, "max": 0.5},
        },
        "y": {
            "base": {"type": "uniform", "min": 0.0, "max": 1.0},
            "new": {"type": "uniform", "min": 0.0, "max": 0.5},
        },
    }
    assert np.isclose(prior_log_ratio(samples, changes)[0], 2 * np.log(2.0))


def test_broadening_prior_is_rejected():
    """A new prior reaching beyond the base support is refused, not biased."""
    samples = pd.DataFrame({"x": [0.2, 0.5, 0.8]})
    changes = {
        "x": {
            "base": {"type": "uniform", "min": 0.0, "max": 1.0},
            "new": {"type": "uniform", "min": -0.5, "max": 1.0},  # wider than base
        }
    }
    with pytest.raises(ValueError):
        prior_log_ratio(samples, changes)
