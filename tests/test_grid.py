"""Tests for the (model, prior) grid enumeration."""

import pytest

from cosmocompare.grid import all_jobs, index_of_job, job_by_index


def test_log_uniform_skipped_for_listed_models():
    """Models in the skip list get no log-uniform job."""
    jobs = all_jobs()
    assert ("wlcdm", "log_uniform") not in jobs
    assert ("mlcdm", "log_uniform") in jobs


def test_index_round_trip():
    """A job maps to an index and back to itself."""
    for i, job in enumerate(all_jobs()):
        assert job_by_index(i) == job
        assert index_of_job(*job) == i


def test_out_of_range_index_raises():
    """An index past the end of the grid raises IndexError."""
    with pytest.raises(IndexError):
        job_by_index(len(all_jobs()))


def test_unknown_job_raises():
    """A combination not in the grid raises ValueError."""
    with pytest.raises(ValueError):
        index_of_job("not_a_model", "uniform")


def test_custom_grid_dimensions():
    """A restricted grid enumerates only the requested combinations."""
    jobs = all_jobs(models=("a", "b"), prior_types=("uniform",), no_log_uniform=())
    assert jobs == [("a", "uniform"), ("b", "uniform")]
