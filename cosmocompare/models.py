"""Extended cosmological models expressed as additions to a base model.

An extended model is described here by the parameters it adds to the base model
and by the Cobaya input that defines its likelihood. Everything the sampler
needs from a model - how to draw its new parameters and how to score them under
the target prior - is reached through :class:`TargetModel`.
"""

import numpy as np

from cosmocompare.parameters import Parameter


class TargetModel:
    """An extended model defined relative to a base set of samples.

    Parameters
    ----------
    name : str
        Identifier for the model, used in logging and output paths (for example
        ``"wlcdm"``).
    config : dict
        Model configuration with keys:

        - ``yaml_path``: path to the Cobaya input defining the likelihood.
        - ``new_params``: mapping of parameter name to a dict carrying a
          ``prior`` block and an optional ``proposal`` block, each in the form
          understood by :class:`cosmocompare.parameters.Parameter`.
    """

    def __init__(self, name, config):
        self.name = name
        self.yaml_path = config.get("yaml_path")
        self.new_params = {
            p_name: Parameter(
                p_name,
                prior=p_config["prior"],
                proposal=p_config.get("proposal"),
            )
            for p_name, p_config in config.get("new_params", {}).items()
        }

    def augment_samples(self, base_samples, rng=None):
        """Return a copy of the base samples with new-parameter columns added.

        Each new parameter is drawn from its proposal and appended as a column.

        Parameters
        ----------
        base_samples : pandas.DataFrame or anesthetic samples
            Base-model samples to extend. Anything exposing ``copy`` is copied;
            otherwise ``to_pandas`` is tried.
        rng : numpy.random.Generator, optional
            Source of randomness, forwarded to each proposal.

        Returns
        -------
        pandas.DataFrame
            The augmented samples. The returned object is of the same type as
            the input when that type supports column assignment.
        """
        if hasattr(base_samples, "copy"):
            augmented = base_samples.copy()
        elif hasattr(base_samples, "to_pandas"):
            augmented = base_samples.to_pandas()
        else:
            raise TypeError("base_samples must support copy() or to_pandas().")

        n_samples = len(augmented)
        for name, param in self.new_params.items():
            # Only genuinely new parameters are drawn. A name that already exists
            # is an existing parameter whose prior is being changed, not added:
            # overwriting its sampled coordinate would corrupt the likelihood
            # ratio, so its real values are kept and only re-weighted.
            if name in augmented.columns:
                continue
            augmented[name] = param.sample(n_samples, rng=rng)
        return augmented

    def log_prior_correction(self, augmented_samples):
        """Return the summed ``log(prior / proposal)`` over all new parameters.

        Parameters
        ----------
        augmented_samples : pandas.DataFrame
            Samples carrying a column for each new parameter.

        Returns
        -------
        numpy.ndarray
            Per-sample correction term to add to the log likelihood ratio before
            reweighting. Identically zero when every proposal matches its prior.
        """
        total = np.zeros(len(augmented_samples))
        for name, param in self.new_params.items():
            total = total + param.log_correction(augmented_samples[name].to_numpy())
        return total

    def __repr__(self):
        """Return a concise, unambiguous representation."""
        return f"TargetModel(name={self.name!r}, new_params={list(self.new_params)})"


class ModelFactory:
    """Build :class:`TargetModel` instances from configuration."""

    @staticmethod
    def create(name, config):
        """Return a :class:`TargetModel` for ``name``.

        Parameters
        ----------
        name : str
            Model identifier.
        config : dict
            Model configuration, as documented on :class:`TargetModel`.

        Returns
        -------
        TargetModel
            The constructed model.
        """
        return TargetModel(name, config)
