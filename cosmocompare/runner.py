"""End-to-end driver tying the reweighting pipeline together.

Given a configuration, the runner reads the base chains once and, for each target
model, augments the full nested-sampling run with the model's new parameters,
re-evaluates the likelihood, reweights, and records both the reweighted posterior
and the evidence.

The evidence is the importance-sampling ratio estimator
(:func:`cosmocompare.evidence.importance_evidence`): it integrates over the full
prior volume the base run traversed, marginalises the added parameters through
the one-draw-per-sample augmentation, and keeps the baseline ``log Z`` on the
same footing. It is *not* read from ``logZ`` of the re-weighted run: once a new
parameter is bolted on, the samples no longer form a nested sequence, so that
geometry is invalid. The re-weighted posterior is produced by multiplying the
weights, not by recomputing contours.
"""

from pathlib import Path

import numpy as np

from cosmocompare.config import chains_root, load_config, output_root
from cosmocompare.diagnostics import effective_sample_size, efficiency
from cosmocompare.evidence import importance_evidence
from cosmocompare.models import ModelFactory
from cosmocompare.reweight import apply_log_weights, log_importance_weights


class Runner:
    """Run importance sampling for every model in a configuration.

    Parameters
    ----------
    config : str, pathlib.Path or dict
        Either a path to a YAML configuration or an already-parsed configuration
        dictionary (see :func:`cosmocompare.config.load_config`).
    rng : numpy.random.Generator, optional
        Source of randomness for parameter augmentation and the evidence
        bootstrap, for reproducible runs.

    Notes
    -----
    The runner operates on the full nested-sampling run, not on posterior points,
    because the evidence integral needs the whole prior-to-posterior volume. The
    ``max_samples`` setting truncates the run and is a convenience for smoke-
    testing the plumbing only; the evidence it yields is not meaningful.
    """

    def __init__(self, config, rng=None):
        from anesthetic import read_chains

        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config
        self.settings = config["global_settings"]
        self.rng = np.random.default_rng() if rng is None else rng

        self.output_dir = Path(self.settings.get("output_dir") or output_root())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        chain_path = Path(self.settings["base_chain_path"])
        root = chains_root()
        if root is not None and not chain_path.is_absolute():
            chain_path = root / chain_path
        self.base_samples = read_chains(str(chain_path))

        max_samples = self.settings.get("max_samples")
        if max_samples:
            self.base_samples = self.base_samples.head(max_samples)

    def run(self):
        """Process every target model in the configuration.

        Returns
        -------
        dict
            Mapping from model name to its result dictionary (see
            :meth:`process_model`). A model that fails maps to a dict with an
            ``"error"`` key rather than aborting the batch.
        """
        results = {}
        for name, model_config in self.config.get("target_models", {}).items():
            try:
                results[name] = self.process_model(name, model_config)
            except Exception as exc:  # noqa: BLE001 - one model must not sink the batch
                results[name] = {"error": str(exc)}
        return results

    def process_model(self, name, model_config):
        """Reweight the base run onto one target model and get its evidence.

        Parameters
        ----------
        name : str
            Model identifier.
        model_config : dict
            Model configuration (see :class:`cosmocompare.models.TargetModel`).

        Returns
        -------
        dict
            Result with the re-weighted posterior, the evidence (ratio estimator
            with its bootstrap error and effective sample size), and the weight
            diagnostics.
        """
        from cosmocompare.likelihood import LikelihoodCalculator

        model = ModelFactory.create(name, model_config)

        # Augment the run with the new parameters. Existing parameters whose prior
        # is being changed keep their sampled values (see TargetModel).
        augmented = model.augment_samples(self.base_samples, rng=self.rng)

        calculator = LikelihoodCalculator(model.yaml_path)
        logL_new = calculator.log_likelihood(
            augmented,
            use_parallel=self.settings.get("use_parallel", True),
            n_cores=self.settings.get("n_cores"),
        )
        base_logL = np.asarray(augmented.logL, dtype=float)
        correction = model.log_prior_correction(augmented)

        # Evidence: the importance-sampling ratio estimator (the valid route).
        base_logZ_draws = np.asarray(self.base_samples.logZ(1000), dtype=float)
        evidence = importance_evidence(
            logL_new,
            base_logL,
            base_weights=np.asarray(augmented.get_weights(), dtype=float),
            log_prior_correction=correction,
            base_log_evidence=float(base_logZ_draws.mean()),
            base_log_evidence_err=float(base_logZ_draws.std()),
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )

        # Re-weighted posterior for parameter estimation: weights only, no
        # nested recompute. Its logZ is meaningless; the evidence is above.
        log_w = log_importance_weights(logL_new - base_logL, correction)
        posterior = apply_log_weights(augmented, log_w)

        weights = np.asarray(posterior.get_weights(), dtype=float)
        result = {
            "posterior": posterior,
            "logL_new": logL_new,
            "evidence": evidence,
            "log_evidence": evidence.log_evidence,
            "log_evidence_err": evidence.log_evidence_err,
            "effective_sample_size": effective_sample_size(weights),
            "efficiency": efficiency(weights),
        }

        if self.settings.get("save_results", True):
            self._save(name, result)
        return result

    def _save(self, name, result):
        """Write a model's posterior and likelihood array to disk.

        Parameters
        ----------
        name : str
            Model identifier, used as the output sub-directory name.
        result : dict
            The result dictionary from :meth:`process_model`.
        """
        out = self.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        result["posterior"].to_csv(out / f"{name}_posterior.csv")
        np.save(out / f"{name}_logL_new.npy", result["logL_new"])
