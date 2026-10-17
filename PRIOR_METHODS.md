# Prior-changing and reweighting methods in `cosmocompare`

A catalogue of every method for changing a prior and recomputing the Bayesian
evidence by reusing existing nested-sampling chains, with its status in the
package (**implemented**, **cite-only**, or **rejected**) and its literature.

The package is organised by *operation*, not by this taxonomy: a user reweights,
scans, or cross-checks, and the methods below are the ways to specify the prior
and to judge reliability. Every implemented method shares one exact engine,
`Z_new/Z_old = <pi_new/pi_old>_posterior` (importance sampling of the weighted
nested samples), with the base prior's normalisation kept (it is the Ockham
prior-volume term, not a cancelling constant).

---

## A. What prior to change to (the prior menu)

All of these are **implemented**, in `dark_energy.py` (dark-energy specific) and
`prior.py`/`sweep.py` (generic). Every one is a *contraction* of the base box, so
reweighting stays valid (widening into unsampled regions is refused in `prior.py`).

| Method | Function | Note | Reference |
|---|---|---|---|
| Uniform width sweep (homothetic) | `sweep.bayes_factor_sweep`, `shrink_uniform` | `lnB(f)` curve; data-independent centre | Trotta 2008 |
| Generic shape change (uniform / Gaussian / truncated-Gaussian / log-uniform) | `parameters.Parameter`, `prior.reweight_prior` | per-parameter, factorised | PRIOR_TYPES_SUMMARY |
| Physicality cut `w0+wa<0` | `dark_energy.physicality_cut` | hard cut; ESS-safe | standard CPL |
| No-phantom `w0>=-1, w0+wa>=-1` | `dark_energy.no_phantom` | hard cut; ESS trap on DESI | standard |
| Gaussian shrink toward LCDM | `dark_energy.gaussian_shrink` | scan sigma -> `lnB(sigma)` | Trotta 2008 |
| Thawing band `wa ~ -1.58(1+w0)` | `dark_energy.thawing_band` | soft band, never a delta | thawing quintessence |
| Quintessence-induced (KDE) | `dark_energy.quintessence_prior` | theory prior, plugin table | Toomey+2025 |
| Arbitrary / correlated density | `prior.reweight_density`, `dark_energy.box_normalised_logdensity` | any normalised joint prior | - |
| Region of arbitrary shape | `prior.reweight_region` | MC-normalised, error propagated | - |

Literature-box priors are supported as a special case of `reweight_region`, but
are flagged arbitrary/basis-dependent and are not recommended as a primary result.

## B. How to deform the prior continuously

| Method | Function | Status | Reference |
|---|---|---|---|
| Power-scaling `pi_0^(1-a) pi_1^a` | `powerscale.power_scale_sweep`, `power_scaled_logdensity` | **implemented** | Kallioinen+2024 (`priorsense`), Vehtari+2024 |

## C. Reliability diagnostics for a reweight

| Method | Function | Status | Reference |
|---|---|---|---|
| Kish effective sample size | `diagnostics.effective_sample_size` | **implemented** | Kish 1965 |
| Bootstrap error (finite-replicate, degenerate-fraction flag) | `evidence.importance_evidence` | **implemented** | Efron 1979 |
| PSIS Pareto-k-hat | `psis.psis`, `psis.psis_diagnostic` | **implemented** | Vehtari+2024, Zhang & Stephens 2009 |

Use Pareto-k-hat for smooth reweightings; for hard cuts it is not meaningful (no
tail to fit, returns `nan`) and the Kish ESS plus support containment govern.

## D. Evidence estimators / cross-checks

| Method | Function | Status | Reference |
|---|---|---|---|
| Importance-ratio evidence | `evidence.importance_evidence` | **implemented** (core) | standard SNIS |
| Savage-Dickey density ratio | `sddr.savage_dickey` | **implemented** (nested cross-check, tail-gated) | Dickey 1971; Verdinelli & Wasserman 1995; Trotta 2008 |
| Nested-sampling `logZ` reader | `evidence.nested_evidence` | **implemented** (base evidence only) | Skilling 2006 |

## E. Cite as related work (deliberately NOT implemented)

These are sound methods, but they do not fit the "cheap post-processing of one
existing chain" regime, or they solve a different problem; implementing them would
be feature sprawl or statistically mismatched (confirmed with Gemini 3.1 Pro and
GPT-5.4).

| Method | Why not implemented | Reference |
|---|---|---|
| Bridge sampling | needs samples from *both* distributions | Meng & Wong 1996; Gronau+2017 |
| Path sampling / thermodynamic integration | needs a tempered chain, not post-hoc | Gelman & Meng 1998 |
| Learnt harmonic mean / floZ | evidence from posterior samples via a learned density; we already have exact `Z` | McEwen+2021; Srinivasan, Trotta+2024 |
| Local / infinitesimal prior sensitivity | derivative-based; too abstract for v0.x | Giordano+2018 |
| epsilon-contamination robustness | bounds over a prior class; future work | Berger 1990 |
| CosmoMC / GetDist importance sampling | cosmology prior art for chain reweighting | Lewis & Bridle 2002 |
| SMC / pocoMC / nautilus | samplers, not post-processing | Karamanis+2022; Lange 2023 |

## F. Rejected outright

| Method | Why | Reference |
|---|---|---|
| Classic harmonic mean estimator | infinite variance; unstable; discouraged | Newton & Raftery 1994 |

---

**Cross-model extension** (computing the evidence for a *new* model by reusing a
chain as a proposal and re-evaluating the likelihood with Cobaya) is a separate,
expensive operation handled by `models.py` + `likelihood.py`, not a prior change.
It is the subject of a second study; see the project plan.
