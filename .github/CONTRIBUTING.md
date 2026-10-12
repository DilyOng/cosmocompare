# Contributing to cosmocompare

Contributions are welcome, whether that is a bug report, a documentation fix or
a new feature.

## Reporting problems

Open an issue at https://github.com/DilyOng/cosmocompare/issues. For a bug,
please include the version of `cosmocompare`, your Python version, and the
smallest example that reproduces the problem.

## Proposing changes

For anything larger than a small fix, please open an issue first so the approach
can be agreed before you spend time on it.

## Running the checks locally

The checks below are exactly what CI runs, so a clean run locally should mean a
green pull request.

```bash
pip install -e ".[test]"

pytest                                       # test suite
flake8 cosmocompare tests                    # settings in .flake8
pydocstyle --convention=numpy cosmocompare   # numpydoc docstrings
black --check cosmocompare tests             # formatting
isort --check-only cosmocompare tests        # import order
```

## Conventions

- Docstrings follow the numpydoc convention.
- Line length is 88 characters, matching `black`.
- The likelihood stack (Cobaya) is an **optional** dependency. It must never be
  imported at module scope, so that `pip install cosmocompare` stays usable for
  reweighting, evidence and diagnostics without it. `tests/test_packaging.py`
  enforces this.
- The importance weight always carries the full prior-over-proposal correction.
  Do not drop it on the assumption that the proposal equals the target prior.

## Support

This is research software maintained alongside a PhD, so responses may take a
little time. Issues are the best way to reach the maintainer.
