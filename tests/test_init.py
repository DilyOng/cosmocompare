"""Tests for the package's public surface."""

import re

import cosmocompare


def test_version_is_pep440_ish():
    """The package exposes a dotted numeric version string."""
    assert isinstance(cosmocompare.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", cosmocompare.__version__)


def test_public_names_are_importable():
    """Every name in __all__ resolves to an attribute on the package."""
    for name in cosmocompare.__all__:
        assert hasattr(cosmocompare, name), name
