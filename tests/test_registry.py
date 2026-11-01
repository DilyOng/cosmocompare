"""Tests for the model registry and the neutrino-cell nmn filter."""

import textwrap

from cosmocompare import config
from cosmocompare.models_registry import REGISTRY, REQUIRED_NMN


def test_neutrino_models_are_registered():
    """The three neutrino models are sweepable, with the expected extra params."""
    assert {"mlcdm", "nlcdm", "Nlcdm"} <= set(REGISTRY)
    assert set(REGISTRY["mlcdm"].params) == {"mnu"}
    assert set(REGISTRY["nlcdm"].params) == {"nnu"}
    assert set(REGISTRY["Nlcdm"].params) == {"mnu", "nnu"}


def test_neutrino_centres_lie_inside_their_priors():
    """Each neutrino centre (the LCDM nesting point) is interior to its range."""
    for model in REQUIRED_NMN:
        for p in REGISTRY[model].params.values():
            assert p.low <= p.centre <= p.high
            assert p.scan == "interior"


def _write_cell(tmp_path, dataset, nmn):
    """Create a minimal PolyChord-style cell with a given num_massive_neutrinos."""
    cell = tmp_path / "ns" / "nlcdm" / dataset
    (cell / f"{dataset}_polychord_raw").mkdir(parents=True)
    (cell / f"{dataset}.input.yaml").write_text(
        textwrap.dedent(
            f"""
        theory:
          camb:
            extra_args:
              num_massive_neutrinos: {nmn}
        """
        )
    )
    return config.chain_stem(tmp_path, "nlcdm", dataset)


def test_cell_massive_neutrinos_reads_the_setting(tmp_path):
    """The helper returns the integer nmn from a cell's input.yaml."""
    stem = _write_cell(tmp_path, "bao.desi_dr2", 1)
    assert config.cell_massive_neutrinos(stem) == 1


def test_cell_massive_neutrinos_missing_returns_none(tmp_path):
    """A stem with no input.yaml yields None rather than raising."""
    stem = config.chain_stem(tmp_path, "nlcdm", "absent")
    assert config.cell_massive_neutrinos(stem) is None
