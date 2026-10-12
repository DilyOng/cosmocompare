"""Re-evaluation of a target-model likelihood on existing samples.

The whole saving of importance sampling is that the samples already exist; only
the likelihood is new. This module wraps Cobaya so that an extended model's log
likelihood can be evaluated point by point on samples drawn under a different
model, in serial or across processes.

Cobaya is an optional dependency. It is imported lazily so that the rest of the
package - prior maths, reweighting, evidence, diagnostics - installs and imports
without it.
"""

import multiprocessing as mp

import numpy as np
import yaml


class LikelihoodCalculator:
    """Evaluate a Cobaya model's log likelihood on supplied samples.

    Parameters
    ----------
    yaml_path : str
        Path to the Cobaya input file describing theory, likelihoods and
        parameters. Any ``sampler`` and ``output`` blocks are dropped, since the
        model is only ever evaluated, never sampled.

    Attributes
    ----------
    sampled_params : list of str
        Names of the parameters the model expects, in Cobaya's order.
    """

    def __init__(self, yaml_path):
        from cobaya.model import get_model

        with open(yaml_path, "r") as stream:
            config = yaml.safe_load(stream)
        config.pop("sampler", None)
        config.pop("output", None)

        self.yaml_path = yaml_path
        self.model = get_model(config)
        self.sampled_params = list(self.model.parameterization.sampled_params())

    def log_likelihood_point(self, point):
        """Return the log likelihood at a single point.

        Parameters
        ----------
        point : dict
            Mapping from parameter name to value covering
            :attr:`sampled_params`.

        Returns
        -------
        float
            The log likelihood, or ``-inf`` if the point is rejected (for
            example as out of bounds or non-computable).
        """
        try:
            return self.model.logposterior(point).loglike
        except Exception:
            return -np.inf

    def _point_dicts(self, samples):
        """Extract per-sample parameter dictionaries from a frame.

        Parameters
        ----------
        samples : pandas.DataFrame
            Samples whose columns include every name in
            :attr:`sampled_params`, either directly or as the first element of a
            tuple column label.

        Returns
        -------
        list of dict
            One parameter dictionary per row.
        """
        points = []
        for _, row in samples.iterrows():
            point = {}
            for name in self.sampled_params:
                if name in row.index:
                    point[name] = row[name]
                    continue
                match = next(
                    (
                        col
                        for col in row.index
                        if isinstance(col, tuple) and col[0] == name
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"Parameter {name!r} not found in samples.")
                point[name] = row[match]
            points.append(point)
        return points

    def log_likelihood(self, samples, use_parallel=True, n_cores=None):
        """Return the log likelihood for every row of ``samples``.

        Parameters
        ----------
        samples : pandas.DataFrame
            Points to evaluate.
        use_parallel : bool, optional
            Evaluate across processes when there are more than 100 samples.
            Defaults to True.
        n_cores : int, optional
            Worker count for parallel evaluation. Defaults to one fewer than the
            machine's CPU count.

        Returns
        -------
        numpy.ndarray
            Log likelihood per sample, with ``-inf`` for rejected points.
        """
        points = self._point_dicts(samples)

        if use_parallel and len(points) > 100:
            from joblib import Parallel, delayed

            if n_cores is None:
                n_cores = max(1, mp.cpu_count() - 1)
            values = Parallel(n_jobs=n_cores)(
                delayed(self.log_likelihood_point)(point) for point in points
            )
        else:
            values = [self.log_likelihood_point(point) for point in points]

        return np.asarray(values, dtype=float)
