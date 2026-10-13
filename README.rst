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

Developed under a UKRI-funded high performance computing (DiRAC) project.

Installation
------------

.. code:: bash

    pip install cosmocompare                  # reweighting, evidence, diagnostics
    pip install "cosmocompare[likelihood]"    # adds the Cobaya likelihood stack

Requires Python 3.11 or later.
