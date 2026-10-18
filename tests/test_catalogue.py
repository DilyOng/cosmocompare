"""Tests for the evidence-conditional-on-prior catalogue (persist + query)."""

import numpy as np

from cosmocompare.catalogue import (
    Catalogue,
    Cell,
    flag_ambiguous,
    flag_sign_flips,
    flag_unreliable,
)


class FakeSamples:
    """Minimal anesthetic-like sample set for posterior reconstruction."""

    def __init__(self, weights):
        self._w = np.asarray(weights, dtype=float)

    def get_weights(self):
        return self._w

    def set_weights(self, w, inplace=False):
        self._w = np.asarray(w, dtype=float)

    def copy(self):
        return FakeSamples(self._w.copy())


def _cell(**kw):
    base = dict(
        model="walcdm",
        base_model="lcdm",
        dataset="bao.desi_dr2+planck_2018_CamSpec",
        prior_method="homothetic",
        prior_params={"fraction": 0.5},
        estimator="is_ratio",
    )
    base.update(kw)
    return Cell(**base)


def test_cell_id_stable_and_key_sensitive():
    """The cell id is stable for the same key and changes with the key."""
    a = _cell()
    b = _cell()
    c = _cell(prior_params={"fraction": 0.25})
    assert a.cell_id() == b.cell_id()
    assert a.cell_id() != c.cell_id()


def test_save_appends_and_caches(tmp_path):
    """Saving writes a summary row and a dlogw cache that round-trips."""
    cat = Catalogue(tmp_path)
    dlogw = np.array([0.0, -1.0, -np.inf, 0.5])
    cid = cat.save(_cell(ln_bayes_factor=1.2, reliable=True, dlogw=dlogw))

    df = cat.load_summary()
    assert len(df) == 1
    assert df.iloc[0]["cell_id"] == cid
    back = cat.load_dlogw(cid)
    assert np.allclose(back[np.isfinite(back)], dlogw[np.isfinite(dlogw)])
    assert np.isneginf(back[2])


def test_save_is_idempotent(tmp_path):
    """Re-saving the same cell replaces its row rather than duplicating it."""
    cat = Catalogue(tmp_path)
    cat.save(_cell(ln_bayes_factor=1.0, dlogw=np.zeros(3)))
    cat.save(_cell(ln_bayes_factor=2.0, dlogw=np.zeros(3)))
    df = cat.load_summary()
    assert len(df) == 1
    assert df.iloc[0]["ln_bayes_factor"] == 2.0


def test_load_posterior_reconstructs_weights(tmp_path):
    """The stored posterior is the base chain reweighted by exp(dlogw)."""
    cat = Catalogue(tmp_path)
    dlogw = np.log(np.array([1.0, 2.0, 4.0]))  # ratios 1,2,4
    cid = cat.save(_cell(dlogw=dlogw))
    base = FakeSamples(np.array([1.0, 1.0, 1.0]))
    post = cat.load_posterior(base, cid)
    w = post.get_weights()
    # Weights proportional to base * ratio = [1, 2, 4].
    assert np.allclose(w / w.max(), np.array([0.25, 0.5, 1.0]))


def test_flag_sign_flips(tmp_path):
    """A (model, dataset) with both signs among reliable cells is flagged."""
    cat = Catalogue(tmp_path)
    cat.save(_cell(prior_method="baseline", ln_bayes_factor=-0.5, reliable=True,
                   dlogw=None))
    cat.save(_cell(prior_method="physicality_cut", ln_bayes_factor=+0.8,
                   reliable=True, dlogw=np.zeros(2)))
    flagged = flag_sign_flips(cat.load_summary())
    assert len(flagged) == 2


def test_shards_merge_into_summary(tmp_path):
    """Parallel shard writers merge into one summary, de-duplicated by cell_id."""
    a = Catalogue(tmp_path, shard="walcdm__d1")
    b = Catalogue(tmp_path, shard="wlcdm__d1")
    a.save(_cell(model="walcdm", ln_bayes_factor=1.0, dlogw=np.zeros(2)))
    b.save(_cell(model="wlcdm", ln_bayes_factor=2.0, dlogw=np.zeros(2)))
    # Each shard file holds only its own row.
    assert len(a.load_summary()) == 1
    assert len(b.load_summary()) == 1
    merged = Catalogue(tmp_path).merge_shards()
    assert len(merged) == 2
    assert set(merged["model"]) == {"walcdm", "wlcdm"}


def test_flag_ambiguous_and_unreliable(tmp_path):
    """Ambiguous (|lnB|<1) and unreliable (reliable False) filters work."""
    cat = Catalogue(tmp_path)
    cat.save(_cell(prior_method="a", ln_bayes_factor=0.3, reliable=True,
                   dlogw=np.zeros(2)))
    cat.save(_cell(prior_method="b", ln_bayes_factor=5.0, reliable=False,
                   dlogw=np.zeros(2)))
    df = cat.load_summary()
    assert len(flag_ambiguous(df)) == 1
    assert len(flag_unreliable(df)) == 1
