"""Tests for the two-root chain resolver (new_grid first, grid fallback)."""

import pytest

from cosmocompare.config import chain_stem, resolve_chain_stem


def _make_run(root, model, dataset, method="ns"):
    """Create a fake PolyChord output file for (model, dataset) under root."""
    stem = chain_stem(root, model, dataset, method=method)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # A representative PolyChord output file sharing the stem.
    (stem.parent / f"{dataset}.stats").write_text("fake")
    return stem


def test_chain_stem_structure(tmp_path):
    """Stem is <root>/<method>/<model>/<dataset>/<dataset>_polychord_raw/<dataset>."""
    stem = chain_stem(tmp_path, "walcdm", "bao.desi_dr2", method="ns")
    assert stem.as_posix().endswith(
        "ns/walcdm/bao.desi_dr2/bao.desi_dr2_polychord_raw/bao.desi_dr2"
    )


def test_primary_preferred_when_in_both(tmp_path):
    """When a run exists in both roots, the primary (new_grid) wins."""
    primary = tmp_path / "new_grid"
    fallback = tmp_path / "grid"
    _make_run(primary, "lcdm", "sn.pantheon")
    _make_run(fallback, "lcdm", "sn.pantheon")
    stem = resolve_chain_stem("lcdm", "sn.pantheon", primary=primary, fallback=fallback)
    assert "new_grid" in stem.as_posix()


def test_falls_back_to_grid_when_absent_from_primary(tmp_path):
    """A run only in the fallback root is still found."""
    primary = tmp_path / "new_grid"
    fallback = tmp_path / "grid"
    _make_run(fallback, "klcdm", "bao.desi_dr2")
    stem = resolve_chain_stem(
        "klcdm", "bao.desi_dr2", primary=primary, fallback=fallback
    )
    assert "grid" in stem.as_posix() and "new_grid" not in stem.as_posix()


def test_raises_when_absent_from_both(tmp_path):
    """A run in neither root raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resolve_chain_stem(
            "nope", "nodata", primary=tmp_path / "new_grid", fallback=tmp_path / "grid"
        )
