"""Tests for the model augmentation guards."""

import numpy as np
import pandas as pd
import pytest

from cosmocompare.models import TargetModel


def test_new_parameter_is_drawn():
    """A genuinely new parameter is sampled and added as a column."""
    base = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    model = TargetModel(
        "m",
        {"new_params": {"w": {"prior": {"type": "uniform", "min": 0.0, "max": 1.0}}}},
    )
    augmented = model.augment_samples(base, rng=np.random.default_rng(0))
    assert "w" in augmented.columns
    assert len(augmented["w"]) == 3


def test_existing_parameter_without_proposal_is_rejected():
    """Listing an existing parameter with no base proposal is a silent-no-op trap."""
    base = pd.DataFrame({"x": [0.2, 0.5, 0.8]})
    model = TargetModel(
        "m",
        {"new_params": {"x": {"prior": {"type": "uniform", "min": 0.0, "max": 0.5}}}},
    )
    with pytest.raises(ValueError):
        model.augment_samples(base, rng=np.random.default_rng(0))


def test_existing_parameter_with_base_proposal_is_kept():
    """An existing parameter with its base prior as proposal is kept, not redrawn."""
    base = pd.DataFrame({"x": [0.2, 0.5, 0.8]})
    model = TargetModel(
        "m",
        {
            "new_params": {
                "x": {
                    "prior": {"type": "uniform", "min": 0.0, "max": 0.5},
                    "proposal": {"type": "uniform", "min": 0.0, "max": 1.0},
                }
            }
        },
    )
    augmented = model.augment_samples(base, rng=np.random.default_rng(0))
    # Original values are preserved (not resampled).
    assert np.allclose(augmented["x"], [0.2, 0.5, 0.8])
    # Correction is log(new/base) inside, -inf outside the narrowed range.
    corr = model.log_prior_correction(augmented)
    assert np.isclose(corr[0], np.log(1.0 / 0.5))
    assert np.isneginf(corr[2])
