"""End-to-end driver tying the reweighting pipeline together.

Given a configuration, the runner reads the base chains once and, for each
target model, augments the samples with the model's new parameters, re-evaluates
the likelihood, reweights, and records both the reweighted posterior and the
evidence. It is the object a batch job or a notebook calls; the individual steps
live in the focused modules it imports.
"""

from pathlib import Path

import numpy as np

from cosmocompare.config import load_config
from cosmocompare.diagnostics import effective_sample_size, efficiency
from cosmocompare.evidence import importance_evidence, nested_evidence
from cosmocompare.models import ModelFactory
from cosmocompare.reweight import attach_new_parameters, importance_sample


class Runner:
    """Run importance sampling for every model in a configuration.

    Parameters
    ----------
    config : str, pathlib.Path or dict
        Either a path to a YAML configuration or an already-parsed
        configuration dictionary (see :func:`cosmocompare.config.load_config`).
    rng : numpy.random.Generator, optional
        Source of randomness for parameter augmentation and the evidence
        bootstrap, for reproducible runs.
    """

    def __init__(self, config, rng=None):
        from anesthetic import read_chains

        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config
        self.settings = config["global_settings"]
        self.rng = np.random.default_rng() if rng is None else rng

        self.output_dir = Path(self.settings.get("output_dir", "cosmocompare_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_samples = read_chains(self.settings["base_chain_path"])
        if self.settings.get("use_posterior", True):
            self.base_samples = self.base_samples.posterior_points()

        max_samples = self.settings.get("max_samples")
        if max_samples:
            self.base_samples = self.base_samples.head(max_samples)

        self.base_logL = np.asarray(self.base_samples.logL, dtype=float)
        self.base_weights = np.asarray(self.base_samples.get_weights(), dtype=float)

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
        """Reweight the base samples onto one target model.

        Parameters
        ----------
        name : str
            Model identifier.
        model_config : dict
            Model configuration (see :class:`cosmocompare.models.TargetModel`).

        Returns
        -------
        dict
            Result with the reweighted posterior, the evidence estimates and the
            weight diagnostics.
        """
        from cosmocompare.likelihood import LikelihoodCalculator

        model = ModelFactory.create(name, model_config)
        augmented = model.augment_samples(self.base_samples, rng=self.rng)

        calculator = LikelihoodCalculator(model.yaml_path)
        logL_new = calculator.log_likelihood(
            augmented,
            use_parallel=self.settings.get("use_parallel", True),
            n_cores=self.settings.get("n_cores"),
        )

        correction = model.log_prior_correction(augmented)
        finite_mask = np.isfinite(logL_new + correction)

        posterior = importance_sample(self.base_samples, logL_new, correction)
        attach_new_parameters(
            posterior, augmented, model.new_params.keys(), finite_mask
        )

        base_logZ_draws = np.asarray(self.base_samples.logZ(1000), dtype=float)
        evidence = importance_evidence(
            logL_new,
            self.base_logL,
            base_weights=self.base_weights,
            log_prior_correction=correction,
            base_log_evidence=float(base_logZ_draws.mean()),
            base_log_evidence_err=float(base_logZ_draws.std()),
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )
        nested_logZ, nested_logZ_err = nested_evidence(posterior)

        weights = np.asarray(posterior.get_weights(), dtype=float)
        result = {
            "posterior": posterior,
            "logL_new": logL_new,
            "evidence": evidence,
            "nested_log_evidence": nested_logZ,
            "nested_log_evidence_err": nested_logZ_err,
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
