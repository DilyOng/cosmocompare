===================================================================================
cosmocompare: Bayesian model comparison by importance sampling of existing chains
===================================================================================

:Author: Dily Duan Yi Ong
:License: MIT
:Homepage: https://github.com/DilyOng/cosmocompare

.. image:: https://img.shields.io/pypi/v/cosmocompare.svg
   :target: https://pypi.org/project/cosmocompare/
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/cosmocompare.svg
   :target: https://pypi.org/project/cosmocompare/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/DilyOng/cosmocompare/blob/main/LICENSE
   :alt: License: MIT

``cosmocompare`` performs Bayesian model comparison without rerunning nested
sampling. It takes the posterior samples a survey already produced under one
model and prior, reweights them onto a narrowed or changed prior and onto
extended models, and recomputes the Bayesian evidence with an uncertainty, so
that a model-comparison verdict and its sensitivity to the prior can be obtained
in seconds on a laptop.

The weight update carries the full prior-over-proposal correction, so narrowing a
prior or drawing new parameters from an informed proposal stays unbiased rather
than relying on the proposal and the target prior coinciding.

Developed under a UKRI-funded high performance computing (DiRAC) project.

Installation
------------

.. code:: bash

    pip install cosmocompare                  # reweighting, evidence, diagnostics
    pip install "cosmocompare[likelihood]"    # adds the Cobaya likelihood stack

Requires Python 3.11 or later.

Evidence
--------

The evidence is the importance-sampling ratio estimator (``importance_evidence``),
which integrates over the full prior volume of the base run. It is **not** read
from ``logZ`` of a re-weighted run: once a parameter is added or a prior changed,
the samples no longer form a nested sequence, so that geometry is invalid.

Narrowing a prior (no re-run, no likelihood calls)
--------------------------------------------------

To recompute the evidence for a narrower or changed prior on existing parameters,
re-weight an existing nested run directly. The likelihood is unchanged, so this is
instant:

.. code:: python

    from anesthetic import read_chains
    from cosmocompare import reweight_prior

    chains = read_chains("lcdm_planck")
    changes = {
        "omk": {
            "base": {"type": "uniform", "min": -0.3, "max": 0.3},   # original prior
            "new":  {"type": "uniform", "min": -0.05, "max": 0.05}, # narrowed prior
        }
    }
    evidence, posterior = reweight_prior(chains, changes)
    print("log Z_new =", evidence.log_evidence, "+/-", evidence.log_evidence_err)
    print("log(Z_new/Z_old) =", evidence.log_ratio, " ESS =", evidence.effective_sample_size)

Adding a parameter / full pipeline
----------------------------------

.. code:: python

    from cosmocompare import Runner

    results = Runner("config.yaml").run()
    for name, result in results.items():
        evidence = result["evidence"]
        print(name, evidence.log_ratio, "+/-", evidence.log_ratio_err)
