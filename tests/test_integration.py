"""Full-pipeline check against a real nested run, without Cobaya.

This exercises the parts that the dependency-light unit tests cannot: anesthetic
column handling, augment-before-reweight alignment, and reading the evidence off
the reweighted nested run. It runs only when a chain root is provided through the
``COSMOCOMPARE_TEST_CHAIN`` environment variable, so the default suite stays
free of heavy dependencies and bundled data.
"""

import os

import numpy as np
import pytest

anesthetic = pytest.importorskip("anesthetic")

CHAIN = os.environ.get("COSMOCOMPARE_TEST_CHAIN")
pytestmark = pytest.mark.skipif(
    not CHAIN, reason="set COSMOCOMPARE_TEST_CHAIN to a nested-sampling chain root"
)


def test_augment_reweight_evidence_pipeline():
    """A synthetic likelihood flows through augment, reweight and evidence."""
    from anesthetic import read_chains

    from cosmocompare.evidence import importance_evidence, nested_evidence
    from cosmocompare.models import TargetModel
    from cosmocompare.reweight import importance_sample

    base = read_chains(CHAIN)
    model = TargetModel(
        "wtest",
        {
            "new_params": {
                "w": {"prior": {"type": "uniform", "min": -3.0, "max": -0.33}}
            }
        },
    )

    rng = np.random.default_rng(0)
    augmented = model.augment_samples(base, rng=rng)
    assert "w" in augmented.columns

    base_logL = np.asarray(augmented.logL, dtype=float)
    logL_new = base_logL + rng.normal(scale=0.5, size=len(augmented))
    correction = model.log_prior_correction(augmented)
    assert np.allclose(correction, 0.0)  # proposal defaults to the prior

    posterior = importance_sample(augmented, logL_new, correction)

    # The new column must survive the reweighting, aligned to the retained rows.
    w = np.asarray(posterior["w"], dtype=float)
    assert w.shape == (len(posterior),)
    assert w.min() >= -3.0 and w.max() <= -0.33

    # Primary evidence is finite and carries a positive error bar.
    logZ, logZ_err = nested_evidence(posterior)
    assert np.isfinite(logZ)
    assert logZ_err > 0.0

    # Diagnostic ratio estimator is finite with a sensible effective sample size.
    draws = np.asarray(base.logZ(100), dtype=float)
    diagnostic = importance_evidence(
        logL_new,
        base_logL,
        base_weights=np.asarray(augmented.get_weights(), dtype=float),
        log_prior_correction=correction,
        base_log_evidence=float(draws.mean()),
        base_log_evidence_err=float(draws.std()),
        n_bootstrap=100,
        seed=0,
    )
    assert np.isfinite(diagnostic.log_ratio)
    assert diagnostic.effective_sample_size > 0.0
    assert np.isfinite(diagnostic.log_evidence)
