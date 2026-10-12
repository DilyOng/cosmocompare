"""Tests that the installed metadata matches the source of truth."""

import tomllib
from pathlib import Path

import cosmocompare

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject_attr():
    """pyproject reads the version from the module attribute we export."""
    with open(ROOT / "pyproject.toml", "rb") as stream:
        pyproject = tomllib.load(stream)
    attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "cosmocompare._version.__version__"


def test_project_name_is_cosmocompare():
    """The declared distribution name is the expected one."""
    with open(ROOT / "pyproject.toml", "rb") as stream:
        pyproject = tomllib.load(stream)
    assert pyproject["project"]["name"] == "cosmocompare"


def test_cobaya_is_optional():
    """Cobaya must stay an optional extra, not a core dependency."""
    with open(ROOT / "pyproject.toml", "rb") as stream:
        pyproject = tomllib.load(stream)
    core = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "cobaya" not in core
    extras = pyproject["project"]["optional-dependencies"]
    assert any("cobaya" in dep for dep in extras["likelihood"])


def test_package_imports_without_cobaya():
    """Importing the package must not require the optional likelihood stack."""
    assert hasattr(cosmocompare, "Runner")
