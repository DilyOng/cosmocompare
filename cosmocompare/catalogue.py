r"""Persist and query evidence-conditional-on-prior results.

Computing a Bayes factor under a changed prior is cheap, but the whole point of a
database-wide prior-sensitivity map is to do it *once* and then look it up. This
module stores two things per result cell, keyed by
``(model, dataset, prior_method, prior_params, estimator)``:

- a **metrics row** (log evidence, log Bayes factor, errors, ESS, Pareto-k-hat,
  flip fraction, reliability) appended to a CSV summary table, and
- the per-sample **log prior ratio** ``dlogw = log(pi_new / pi_old)`` saved as a
  compressed ``.npz`` in a cache directory.

The normalised posterior under any stored prior is reconstructed on lookup by
attaching ``w_base * exp(dlogw)`` to the base nested-sampling chain, so no sample
matrices are duplicated and nothing is recomputed. A small query layer flags the
scientifically interesting cases (verdicts that flip across priors, ambiguous but
flippable, or unreliable).

Design choices (deliberately dependency-light): the summary is a CSV (pandas),
the cache is per-cell ``.npz`` (numpy). Both are trivially swappable for Parquet
or a single HDF5 later; the per-cell layout is parallel-write-safe for a
scheduler array.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns of the summary table, in order.
SUMMARY_COLUMNS = (
    "model",
    "base_model",
    "dataset",
    "prior_method",
    "prior_params",
    "estimator",
    "ln_evidence",
    "ln_evidence_err",
    "ln_bayes_factor",
    "ln_bayes_factor_err",
    "ess",
    "pareto_k",
    "degenerate_fraction",
    "flip_fraction",
    "reliable",
    "chain_source",
    "cell_id",
)


@dataclass
class Cell:
    """One evidence-conditional-on-prior result.

    Attributes
    ----------
    model, base_model, dataset : str
        The extended model, the base model it is compared against, and the data.
    prior_method : str
        The narrowing family, for example ``"homothetic"``, ``"physicality_cut"``,
        ``"no_phantom"`` or ``"baseline"``.
    prior_params : dict
        The parameters of the narrowing (for example ``{"fraction": 0.5}``).
    estimator : str
        How the evidence was computed: ``"direct_ns"``, ``"is_ratio"`` or
        ``"sddr"``.
    ln_evidence, ln_evidence_err : float
        The extended-model log evidence under this prior and its error.
    ln_bayes_factor, ln_bayes_factor_err : float
        ``log(Z_ext / Z_base)`` and its error.
    ess, pareto_k, degenerate_fraction : float
        Reliability diagnostics of the reweighting.
    flip_fraction : float or None
        For a sweep member, the width fraction at which the sign flips.
    reliable : bool
        Whether the cell passed the reliability gate.
    chain_source : str
        ``"new_grid"`` or ``"grid"`` (which root the chain came from).
    dlogw : numpy.ndarray or None
        Per-sample ``log(pi_new / pi_old)``. Cached so the posterior can be
        rebuilt; ``None`` for a ``direct_ns`` baseline.
    """

    model: str
    base_model: str
    dataset: str
    prior_method: str
    prior_params: dict
    estimator: str
    ln_evidence: float = float("nan")
    ln_evidence_err: float = float("nan")
    ln_bayes_factor: float = float("nan")
    ln_bayes_factor_err: float = float("nan")
    ess: float = float("nan")
    pareto_k: float = float("nan")
    degenerate_fraction: float = float("nan")
    flip_fraction: float = None
    reliable: bool = False
    chain_source: str = ""
    dlogw: np.ndarray = field(default=None, repr=False)

    def cell_id(self):
        """Return a stable hash of the identifying key.

        Returns
        -------
        str
            A 16-character hex digest of ``(model, base_model, dataset,
            prior_method, sorted prior_params, estimator)``.
        """
        key = json.dumps(
            [
                self.model,
                self.base_model,
                self.dataset,
                self.prior_method,
                self.prior_params,
                self.estimator,
            ],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(key.encode()).hexdigest()[:16]


class Catalogue:
    """A directory holding the summary table and the per-cell ``dlogw`` cache.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory for the catalogue. Created if absent. Holds ``summary.csv`` and
        a ``cache/`` subdirectory of ``.npz`` files.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root / "cache"
        self.summary_path = self.root / "summary.csv"
        self.cache.mkdir(parents=True, exist_ok=True)

    def save(self, cell):
        """Append a cell's metrics to the summary and cache its ``dlogw``.

        A cell with the same ``cell_id`` already present is replaced, so re-running
        is idempotent rather than duplicating rows.

        Parameters
        ----------
        cell : Cell
            The result to store.

        Returns
        -------
        str
            The cell id.
        """
        cid = cell.cell_id()

        if cell.dlogw is not None:
            np.savez_compressed(
                self.cache / f"{cid}.npz",
                dlogw=np.asarray(cell.dlogw, dtype=np.float32),
            )

        row = {
            "model": cell.model,
            "base_model": cell.base_model,
            "dataset": cell.dataset,
            "prior_method": cell.prior_method,
            "prior_params": json.dumps(cell.prior_params, sort_keys=True),
            "estimator": cell.estimator,
            "ln_evidence": cell.ln_evidence,
            "ln_evidence_err": cell.ln_evidence_err,
            "ln_bayes_factor": cell.ln_bayes_factor,
            "ln_bayes_factor_err": cell.ln_bayes_factor_err,
            "ess": cell.ess,
            "pareto_k": cell.pareto_k,
            "degenerate_fraction": cell.degenerate_fraction,
            "flip_fraction": cell.flip_fraction,
            "reliable": bool(cell.reliable),
            "chain_source": cell.chain_source,
            "cell_id": cid,
        }

        df = self.load_summary()
        df = df[df["cell_id"] != cid]  # replace an existing identical cell
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df[list(SUMMARY_COLUMNS)]
        df.to_csv(self.summary_path, index=False)
        return cid

    def load_summary(self):
        """Return the summary table as a DataFrame (empty with the right columns)."""
        if self.summary_path.exists():
            return pd.read_csv(self.summary_path)
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    def load_dlogw(self, cell_id):
        """Return the cached per-sample ``dlogw`` for a cell id, or None."""
        path = self.cache / f"{cell_id}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            return data["dlogw"].astype(float)

    def load_posterior(self, base_samples, cell_id):
        """Rebuild a stored posterior by reweighting the base chain.

        Parameters
        ----------
        base_samples : anesthetic samples
            The base nested-sampling chain the cell was computed from.
        cell_id : str
            The cell id (from :meth:`Cell.cell_id` or a summary row).

        Returns
        -------
        anesthetic samples
            The base chain reweighted by the cached ``dlogw``; the normalised
            posterior under the stored prior, at no recompute cost.

        Raises
        ------
        KeyError
            If no ``dlogw`` is cached for the cell (for example a baseline).
        """
        from cosmocompare.reweight import apply_log_weights

        dlogw = self.load_dlogw(cell_id)
        if dlogw is None:
            raise KeyError(f"No cached dlogw for cell {cell_id!r}.")
        return apply_log_weights(base_samples, dlogw)


# =============================================================================
# Flagging: turn the table into "scientifically interesting cases"
# =============================================================================

def flag_sign_flips(summary):
    """Return (model, dataset) groups whose verdict changes sign across priors.

    A sign change in ``ln_bayes_factor`` across the reliable cells of one
    (model, dataset) means the model-comparison verdict is prior-driven.

    Parameters
    ----------
    summary : pandas.DataFrame
        A catalogue summary table.

    Returns
    -------
    pandas.DataFrame
        The rows belonging to prior-sensitive (model, dataset) groups.
    """
    reliable = summary[summary["reliable"]]
    keep = []
    for (model, dataset), g in reliable.groupby(["model", "dataset"]):
        signs = np.sign(g["ln_bayes_factor"].to_numpy())
        if len(np.unique(signs[np.isfinite(signs)])) > 1:
            keep.append(g)
    return pd.concat(keep, ignore_index=True) if keep else summary.iloc[0:0]


def flag_ambiguous(summary, threshold=1.0):
    """Return cells whose |ln B| is below a decisiveness threshold.

    Parameters
    ----------
    summary : pandas.DataFrame
        A catalogue summary table.
    threshold : float, optional
        |ln B| below which a verdict is called ambiguous. Defaults to 1.

    Returns
    -------
    pandas.DataFrame
        The ambiguous cells.
    """
    return summary[summary["ln_bayes_factor"].abs() < threshold]


def flag_unreliable(summary):
    """Return cells that failed the reliability gate.

    Parameters
    ----------
    summary : pandas.DataFrame
        A catalogue summary table.

    Returns
    -------
    pandas.DataFrame
        Cells where ``reliable`` is False (small ESS or high Pareto-k-hat),
        which should be confirmed with a dedicated nested-sampling run.
    """
    return summary[~summary["reliable"].astype(bool)]
