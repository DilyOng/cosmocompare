#!/usr/bin/env python3
"""Cross-model importance sampling: reweight a base run onto an extended model.

Reads a fully nested-sampled BASE model, augments each posterior sample with the
EXTENDED model's new parameter(s) drawn from their prior, re-evaluates the extended
likelihood with Cobaya, and estimates the extended evidence by importance sampling
- no new nested run. The base likelihood is re-evaluated in the base-model limit
with the same calculator (not read from the chain) so the ratio is internally
consistent. The result is compared with the DIRECT nested evidence of the extended
model as the gold-standard validation.

The driver is universal: the base and extended models are arbitrary as long as the
extension is nested in the base (it reduces to the base when its extra parameters
take their limit values). The new parameter(s) are inferred as those SAMPLED by the
extended model that the base model does not sample (read from the base yaml, not the
chain columns, which also carry derived parameters). Limit values come from
:data:`cosmocompare.extend.BASE_LIMITS` (override with --lcdm-value). Multi-parameter
extensions such as w0waCDM (adds w and wa) are handled.

By default the base chain is compressed to a fixed number of equal-weight posterior
draws (--posterior-samples), so the estimator is the plain mean of the likelihood
ratio and the cost is a few thousand likelihood calls rather than the whole nested
chain. The nested evidences used for the comparison are always read from the full
runs, before compression.

Needs a Cobaya environment (COBAYA_PACKAGES_PATH set, run under a Slurm allocation).
Examples::

    # LCDM -> wCDM (one new parameter, inferred)
    python extend_run.py --ext-model wlcdm --dataset bao.desi_dr2 --n-cores 1
    # LCDM -> w0waCDM (two new parameters, inferred)
    python extend_run.py --ext-model walcdm --dataset bao.desi_dr2 --n-cores 1
    # wCDM -> w0waCDM (arbitrary nested base)
    python extend_run.py --base-model wlcdm --ext-model walcdm --dataset bao.desi_dr2
"""

import argparse
import json
import os

import numpy as np
import yaml
from anesthetic import read_chains

from cosmocompare.config import resolve_chain_stem
from cosmocompare.evidence import nested_evidence
from cosmocompare.extend import (
    infer_new_parameters,
    resolve_base_limits,
    run_extension,
)
from cosmocompare.likelihood import LikelihoodCalculator
from cosmocompare.models import TargetModel


def _load_yaml(path):
    """Load a Cobaya input yaml, tolerating Cobaya's custom YAML tags."""
    with open(path) as stream:
        text = stream.read()
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        try:
            from cobaya.yaml import yaml_load
            return yaml_load(text)
        except Exception:
            return yaml.load(text, Loader=yaml.UnsafeLoader)


def resolve_input_yaml(stem):
    """Return the Cobaya input yaml for a chain stem.

    The ``.input.yaml`` may sit next to the polychord stem or one directory up
    (``<root>/<dataset>/<dataset>.input.yaml``); both layouts occur in the grid,
    so try each.
    """
    stem = str(stem)
    base = os.path.basename(stem)
    candidates = [
        f"{stem}.input.yaml",
        os.path.join(os.path.dirname(os.path.dirname(stem)), f"{base}.input.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise SystemExit(f"No .input.yaml found for {stem!r}; looked in: {candidates}")


def prior_from_yaml(yaml_path, name):
    """Return a uniform prior dict for ``name`` read from a Cobaya input file."""
    cfg = _load_yaml(yaml_path)
    p = cfg.get("params", {}).get(name, {})
    pr = p.get("prior", {}) if isinstance(p, dict) else {}
    lo, hi = pr.get("min"), pr.get("max")
    if lo is None or hi is None:
        raise SystemExit(
            f"Could not read a uniform (min/max) prior for {name!r} from "
            f"{yaml_path} (got {pr!r}); only uniform priors are supported, so a "
            f"non-uniform prior must be handled explicitly rather than coerced.")
    return {"type": "uniform", "min": float(lo), "max": float(hi)}


def sampled_params_from_yaml(yaml_path):
    """Return the SAMPLED parameter names (those with a prior block) in a yaml.

    New parameters are inferred as ``ext_sampled - base_sampled`` using this,
    rather than from the chain's columns: anesthetic columns also carry derived
    parameters, so a new parameter present as a derived column in the base chain
    (for example a fixed ``w``) would otherwise be misread as shared and never
    drawn, silently collapsing the Bayes factor.
    """
    cfg = _load_yaml(yaml_path)
    return [name for name, p in cfg.get("params", {}).items()
            if isinstance(p, dict) and isinstance(p.get("prior"), dict)]


def _column(samples, name):
    """Return a chain column by name, tolerating anesthetic tuple labels."""
    if name in samples.columns:
        return np.asarray(samples[name], dtype=float)
    for c in samples.columns:
        if isinstance(c, tuple) and c[0] == name:
            return np.asarray(samples[c], dtype=float)
    raise SystemExit(f"column {name!r} not found in the extended chain.")


def informed_proposal(ext_samples, name, prior, inflate=1.5):
    """Build a truncated-Gaussian proposal fitted to the extended posterior.

    The proposal is centred on the extended model's posterior mean of ``name``
    with its posterior standard deviation (inflated by ``inflate`` for safe tail
    coverage), truncated to the prior box. Drawing the new parameter from where
    the extended likelihood actually has support, rather than from the broad flat
    prior, raises the effective sample size; the ``log(prior/proposal)``
    correction keeps the estimator unbiased. Using the extended chain only sets
    the proposal (efficiency), not the estimand, so it is legitimate.
    """
    w = np.asarray(ext_samples.get_weights(), dtype=float)
    x = _column(ext_samples, name)
    mean = float(np.average(x, weights=w))
    var = float(np.average((x - mean) ** 2, weights=w))
    std = max(np.sqrt(var) * float(inflate), 1e-6)
    return {"type": "truncated_gaussian", "mean": mean, "std": std,
            "min": prior["min"], "max": prior["max"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="lcdm")
    parser.add_argument("--ext-model", required=True,
                        help="e.g. wlcdm, klcdm, walcdm")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--new-param", nargs="+", default=None,
                        help="new parameter name(s); inferred if omitted")
    parser.add_argument("--lcdm-value", nargs="+", type=float, default=None,
                        help="limit value(s) recovering the base, one per new "
                             "parameter; inferred from BASE_LIMITS if omitted")
    parser.add_argument("--posterior-samples", type=int, default=5000,
                        help="compress the base chain to this many equal-weight "
                             "posterior draws before reweighting (0 = full chain)")
    parser.add_argument("--proposal", choices=["prior", "informed"], default="prior",
                        help="new-parameter proposal: 'prior' (q=pi, weight "
                             "L_ext/L_base) or 'informed' (truncated Gaussian fitted "
                             "to the extended posterior, to raise the ESS)")
    parser.add_argument("--proposal-inflate", type=float, default=1.5,
                        help="inflate the informed-proposal std by this factor")
    parser.add_argument("--n-draws", type=int, default=1,
                        help="Rao-Blackwell: draws of the new parameter(s) per base "
                             "sample, averaged (>1 raises the ESS at n_draws x cost)")
    parser.add_argument("--n-cores", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1000,
                        help="logZ draws for nested evidences")
    parser.add_argument("--primary", default=None)
    parser.add_argument("--fallback", default=None)
    parser.add_argument("--outdir", default="cc_extend")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base_stem = resolve_chain_stem(args.base_model, args.dataset,
                                   primary=args.primary, fallback=args.fallback)
    ext_stem = resolve_chain_stem(args.ext_model, args.dataset,
                                  primary=args.primary, fallback=args.fallback)
    ext_yaml = resolve_input_yaml(ext_stem)
    base_yaml = resolve_input_yaml(base_stem)
    print(f"base: {base_stem}\next : {ext_stem}\nyaml: {ext_yaml}")

    base = read_chains(str(base_stem))
    ext = read_chains(str(ext_stem))
    # Nested evidences from the FULL runs (needed for the gold-standard comparison),
    # computed before any compression of the base chain.
    base_logZ, base_logZ_err = nested_evidence(base, args.nsamples)
    ext_logZ, ext_logZ_err = nested_evidence(ext, args.nsamples)
    nested_lnB = ext_logZ - base_logZ
    print(f"nested ln Z: base={base_logZ:.3f}+/-{base_logZ_err:.3f}  "
          f"ext={ext_logZ:.3f}+/-{ext_logZ_err:.3f}  nested ln B={nested_lnB:+.3f}")

    # Compress the base to equal-weight posterior draws: the estimator becomes the
    # plain mean of L_ext/L_base and the cost is a few thousand likelihood calls.
    if args.posterior_samples and len(base) > args.posterior_samples:
        base = base.compress(args.posterior_samples)
        print(f"compressed base to {len(base)} equal-weight posterior samples")

    calc = LikelihoodCalculator(ext_yaml)
    # Infer the new parameter(s) as ext_sampled - base_sampled, taking the base
    # sampled parameters from the base model's yaml (NOT the chain columns).
    base_sampled = set(sampled_params_from_yaml(base_yaml))
    ext_sampled = set(calc.sampled_params)
    try:
        new_params = infer_new_parameters(calc.sampled_params, base_sampled,
                                          requested=args.new_param)
    except ValueError as exc:
        raise SystemExit(str(exc))
    for p in new_params:
        if p not in ext_sampled:
            raise SystemExit(
                f"{p!r} is not a sampled parameter of {args.ext_model!r}.")
        if p in base_sampled:
            raise SystemExit(
                f"{p!r} is sampled in the base {args.base_model!r}; it is shared, "
                f"not new - do not pass it as --new-param.")
    try:
        limits = resolve_base_limits(new_params, values=args.lcdm_value)
    except ValueError as exc:
        raise SystemExit(str(exc))
    # The shared-prior cancellation is exact only when the shared-parameter priors
    # match between base and ext; compare the raw prior blocks (covers non-uniform)
    # and warn rather than silently bias ln B.
    ext_cfg, base_cfg = _load_yaml(ext_yaml), _load_yaml(base_yaml)

    def _prior(cfg, name):
        p = cfg.get("params", {}).get(name, {})
        return p.get("prior") if isinstance(p, dict) else None

    mismatched = [p for p in sorted(ext_sampled & base_sampled)
                  if _prior(ext_cfg, p) != _prior(base_cfg, p)]
    if mismatched:
        print(f"WARNING: shared-parameter priors differ between base and ext for "
              f"{mismatched}; shared-prior cancellation assumes they match, so "
              f"ln B may be biased.")
    print(f"new parameter(s): {new_params}")
    print(f"base limit(s): {limits}")

    # Build each new parameter's prior and (optionally) an informed proposal.
    # With q = prior the correction vanishes (weight = L_ext/L_base); with an
    # informed proposal the new parameter is drawn where the extended likelihood
    # has support, raising the ESS, and log(prior/proposal) keeps it unbiased.
    new_params_cfg = {}
    for p in new_params:
        prior = prior_from_yaml(ext_yaml, p)
        cfg_p = {"prior": prior}
        if args.proposal == "informed":
            cfg_p["proposal"] = informed_proposal(ext, p, prior,
                                                  inflate=args.proposal_inflate)
        new_params_cfg[p] = cfg_p
    print(f"proposal: {args.proposal}"
          + (f" (inflate {args.proposal_inflate})" if args.proposal == "informed"
             else ""))
    target = TargetModel(args.ext_model,
                         {"yaml_path": ext_yaml, "new_params": new_params_cfg})

    rng = np.random.default_rng(args.seed)
    result, _ = run_extension(
        base, target, calc, limits,
        base_log_evidence=base_logZ, base_log_evidence_err=base_logZ_err,
        rng=rng, n_cores=args.n_cores, n_bootstrap=args.n_bootstrap, seed=args.seed,
        n_draws=args.n_draws,
    )

    out = {
        "base_model": args.base_model, "ext_model": args.ext_model,
        "dataset": args.dataset, "new_params": new_params, "limits": limits,
        "proposal": args.proposal,
        "proposal_inflate": (args.proposal_inflate if args.proposal == "informed"
                             else None),
        "n_draws": args.n_draws,
        "posterior_samples": int(len(base)),
        "n_samples": result.n_samples,
        "is_log_bayes_factor": result.log_bayes_factor,
        "is_log_bayes_factor_err": result.log_bayes_factor_err,
        "nested_log_bayes_factor": float(nested_lnB),
        "nested_log_bayes_factor_err": float(np.hypot(ext_logZ_err, base_logZ_err)),
        "is_vs_nested_diff": float(result.log_bayes_factor - nested_lnB),
        "effective_sample_size": result.effective_sample_size,
        "ess_fraction": result.ess_fraction,
        "pareto_k": result.pareto_k,
        "max_weight": result.max_weight,
        "base_logL_consistency": result.base_logL_consistency,
        "reliable": result.reliable,
    }
    name = (f"{args.base_model}_to_{args.ext_model}_{args.dataset}"
            f"_{args.proposal}_N{len(base)}_m{args.n_draws}")
    with open(os.path.join(args.outdir, f"extend_{name}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== RESULT ===")
    print(f"  IS     ln B = {result.log_bayes_factor:+.3f} +/- "
          f"{result.log_bayes_factor_err:.3f}")
    print(f"  nested ln B = {nested_lnB:+.3f}  (diff {out['is_vs_nested_diff']:+.3f})")
    print(f"  ESS = {result.effective_sample_size:.1f}  "
          f"({100*result.ess_fraction:.3f}% of {result.n_samples})")
    print(f"  Pareto k-hat = {result.pareto_k:.2f}  "
          f"max weight = {result.max_weight:.3f}")
    print(f"  base logL consistency (max|stored-fresh|) = "
          f"{result.base_logL_consistency:.3g}")
    print(f"  reliable = {result.reliable}")
    if (result.base_logL_consistency is not None
            and np.isfinite(result.base_logL_consistency)
            and result.base_logL_consistency > 1.0):
        print("  WARNING: large base-logL inconsistency; the fresh base likelihood "
              "may not reproduce the chain's - interpret ln B with care.")


if __name__ == "__main__":
    main()
