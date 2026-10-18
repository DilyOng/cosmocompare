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
from cosmocompare.catalogue import (
    Catalogue,
    Cell,
    flag_ambiguous,
    flag_sign_flips,
    flag_unreliable,
)
from cosmocompare.config import (
    chain_stem,
    chains_root,
    load_config,
    output_root,
    resolve_chain_stem,
)
from cosmocompare.dark_energy import (
    LCDM_CENTRE,
    box_normalised_logdensity,
    gaussian_shrink,
    no_phantom,
    physicality_cut,
    quintessence_prior,
    sensitivity_panel,
    thawing_band,
)
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
from cosmocompare.powerscale import power_scale_sweep, power_scaled_logdensity
from cosmocompare.prior import (
    prior_log_ratio,
    region_volume_fraction,
    reweight_density,
    reweight_prior,
    reweight_region,
)
from cosmocompare.psis import PSISResult, pareto_smooth, psis, psis_diagnostic
from cosmocompare.reweight import apply_log_weights, log_importance_weights
from cosmocompare.runner import Runner
from cosmocompare.sddr import SDDRResult, savage_dickey
from cosmocompare.sweep import (
    bayes_factor_sweep,
    flip_fraction,
    posterior_centres,
    shrink_uniform,
)

__all__ = [
    "__version__",
    "Catalogue",
    "Cell",
    "EvidenceResult",
    "LCDM_CENTRE",
    "flag_ambiguous",
    "flag_sign_flips",
    "flag_unreliable",
    "ModelFactory",
    "PSISResult",
    "Parameter",
    "Runner",
    "SDDRResult",
    "TargetModel",
    "all_jobs",
    "apply_log_weights",
    "bayes_factor_sweep",
    "box_normalised_logdensity",
    "chain_stem",
    "chains_root",
    "resolve_chain_stem",
    "effective_sample_size",
    "efficiency",
    "flip_fraction",
    "gaussian_shrink",
    "importance_evidence",
    "index_of_job",
    "job_by_index",
    "load_config",
    "log_bayes_factor",
    "log_importance_weights",
    "nested_evidence",
    "no_phantom",
    "output_root",
    "pareto_smooth",
    "physicality_cut",
    "posterior_centres",
    "power_scale_sweep",
    "power_scaled_logdensity",
    "prior_log_ratio",
    "psis",
    "psis_diagnostic",
    "quintessence_prior",
    "region_volume_fraction",
    "reweight_density",
    "reweight_prior",
    "reweight_region",
    "savage_dickey",
    "sensitivity_panel",
    "shrink_uniform",
    "thawing_band",
]
