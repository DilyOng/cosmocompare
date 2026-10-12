"""Full-pipeline checks against a real nested run, without Cobaya.

These exercise what the dependency-light unit tests cannot: anesthetic column
handling, weight-based re-weighting, the evidence-ratio estimator, and the
prior-narrowing route validated against its analytic value. They run only when a
chain root is given through the ``COSMOCOMPARE_TEST_CHAIN`` environment variable,
so the default suite stays free of heavy dependencies and bundled data.
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
    """A synthetic likelihood flows through augment, re-weight and evidence."""
    from anesthetic import read_chains

    from cosmocompare.evidence import importance_evidence
    from cosmocompare.models import TargetModel
    from cosmocompare.reweight import apply_log_weights, log_importance_weights

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

    # Re-weighted posterior keeps the new column aligned and in range.
    log_w = log_importance_weights(logL_new - base_logL, correction)
    posterior = apply_log_weights(augmented, log_w)
    w = np.asarray(posterior["w"], dtype=float)
    assert w.shape == (len(augmented),)
    assert w.min() >= -3.0 and w.max() <= -0.33

    # Evidence-ratio estimator is finite with a sensible effective sample size.
    draws = np.asarray(base.logZ(100), dtype=float)
    evidence = importance_evidence(
        logL_new,
        base_logL,
        base_weights=np.asarray(augmented.get_weights(), dtype=float),
        log_prior_correction=correction,
        base_log_evidence=float(draws.mean()),
        base_log_evidence_err=float(draws.std()),
        n_bootstrap=100,
        seed=0,
    )
    assert np.isfinite(evidence.log_ratio)
    assert evidence.effective_sample_size > 0.0
    assert np.isfinite(evidence.log_evidence)


def test_prior_narrowing_matches_analytic():
    """Narrowing a uniform prior reproduces (b-a)/(b'-a') * posterior-fraction."""
    from anesthetic import read_chains

    from cosmocompare.prior import reweight_prior

    base = read_chains(CHAIN)
    param = (
        base.columns[0][0] if isinstance(base.columns[0], tuple) else base.columns[0]
    )
    phi = np.asarray(base[param], dtype=float)
    a, b = float(phi.min()), float(phi.max())
    ap, bp = a + 0.3 * (b - a), b - 0.3 * (b - a)

    changes = {
        param: {
            "base": {"type": "uniform", "min": a, "max": b},
            "new": {"type": "uniform", "min": ap, "max": bp},
        }
    }
    evidence, posterior = reweight_prior(base, changes, n_bootstrap=200, seed=0)

    weights = np.asarray(base.get_weights(), dtype=float)
    inside = (phi >= ap) & (phi <= bp)
    fraction = weights[inside].sum() / weights.sum()
    analytic = np.log((b - a) / (bp - ap)) + np.log(fraction)

    assert np.isclose(evidence.log_ratio, analytic, atol=1e-6)
    # Re-weighted posterior has zero weight outside the new range.
    post_w = np.asarray(posterior.get_weights(), dtype=float)
    assert post_w[~inside].sum() == 0.0
