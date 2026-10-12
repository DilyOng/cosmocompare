"""cosmocompare: Bayesian model comparison by importance sampling.

Nested sampling produces, at great expense, both posterior samples and the
Bayesian evidence for a cosmological model under a given prior. ``cosmocompare``
reuses those existing chains: it reweights them onto narrowed priors and onto
extended models, and recomputes the evidence, so that a model-comparison verdict
and its sensitivity to the prior can be obtained in seconds rather than months.

The import path stays light. Reading chains needs anesthetic; re-evaluating a
target likelihood needs Cobaya, which is an optional extra imported only when a
:class:`~cosmocompare.likelihood.LikelihoodCalculator` is constructed.
"""

from cosmocompare._version import __version__
from cosmocompare.config import chains_root, load_config, output_root
from cosmocompare.diagnostics import effective_sample_size, efficiency
from cosmocompare.evidence import (
    EvidenceResult,
    importance_evidence,
    log_bayes_factor,
    nested_evidence,
)
from cosmocompare.grid import all_jobs, index_of_job, job_by_index
from cosmocompare.models import ModelFactory, TargetModel
from cosmocompare.parameters import Parameter
from cosmocompare.reweight import importance_sample, log_importance_weights
from cosmocompare.runner import Runner

__all__ = [
    "__version__",
    "EvidenceResult",
    "ModelFactory",
    "Parameter",
    "Runner",
    "TargetModel",
    "all_jobs",
    "chains_root",
    "effective_sample_size",
    "efficiency",
    "importance_evidence",
    "importance_sample",
    "index_of_job",
    "job_by_index",
    "load_config",
    "log_bayes_factor",
    "log_importance_weights",
    "nested_evidence",
    "output_root",
]
