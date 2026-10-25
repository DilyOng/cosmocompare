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
take their limit values). The new parameter(s) are inferred as those the extended
model samples that the base chain lacks, and their limit values come from
:data:`cosmocompare.extend.BASE_LIMITS` (override with --lcdm-value). Multi-parameter
extensions such as w0waCDM (which adds w and wa) are handled.

Reports Kish ESS, its fraction, the PSIS Pareto k-hat (the reliability diagnostic
that catches support mismatch), the largest single weight, and the stored-vs-fresh
base-likelihood consistency.

Needs a Cobaya environment. Examples::

    # LCDM -> wCDM (one new parameter, inferred)
    python extend_run.py --ext-model wlcdm --dataset bao.desi_dr2
    # LCDM -> w0waCDM (two new parameters, inferred)
    python extend_run.py --ext-model walcdm --dataset bao.desi_dr2
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
    with open(yaml_path) as stream:
        cfg = yaml.safe_load(stream)
    p = cfg.get("params", {}).get(name, {})
    pr = p.get("prior", {}) if isinstance(p, dict) else {}
    lo, hi = pr.get("min"), pr.get("max")
    if lo is None or hi is None:
        raise SystemExit(
            f"Could not read a uniform prior for {name!r} from {yaml_path}")
    return {"type": "uniform", "min": float(lo), "max": float(hi)}


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
    print(f"base: {base_stem}\next : {ext_stem}\nyaml: {ext_yaml}")

    base = read_chains(str(base_stem))
    ext = read_chains(str(ext_stem))
    base_logZ, base_logZ_err = nested_evidence(base, args.nsamples)
    ext_logZ, ext_logZ_err = nested_evidence(ext, args.nsamples)
    nested_lnB = ext_logZ - base_logZ
    print(f"nested ln Z: base={base_logZ:.3f}+/-{base_logZ_err:.3f}  "
          f"ext={ext_logZ:.3f}+/-{ext_logZ_err:.3f}  nested ln B={nested_lnB:+.3f}")

    calc = LikelihoodCalculator(ext_yaml)
    # Infer the new parameter(s): those the extended model samples that the base
    # chain lacks. Universal in the base/extended pair, and multi-parameter.
    base_cols = {c[0] if isinstance(c, tuple) else c for c in base.columns}
    try:
        new_params = infer_new_parameters(calc.sampled_params, base_cols,
                                          requested=args.new_param)
        limits = resolve_base_limits(new_params, values=args.lcdm_value)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"new parameter(s): {new_params}")
    print(f"base limit(s): {limits}")

    # Proposal = prior (q = pi), so the correction vanishes and we measure the
    # honest effective sample size of the bridge. The full nested chain is reused;
    # no truncation (which would bias the Bayes factor and blind k-hat).
    new_params_cfg = {p: {"prior": prior_from_yaml(ext_yaml, p)}
                      for p in new_params}
    target = TargetModel(args.ext_model,
                         {"yaml_path": ext_yaml, "new_params": new_params_cfg})

    rng = np.random.default_rng(args.seed)
    result, _ = run_extension(
        base, target, calc, limits,
        base_log_evidence=base_logZ, base_log_evidence_err=base_logZ_err,
        rng=rng, n_cores=args.n_cores, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    out = {
        "base_model": args.base_model, "ext_model": args.ext_model,
        "dataset": args.dataset, "new_params": new_params, "limits": limits,
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
    name = f"{args.base_model}_to_{args.ext_model}_{args.dataset}"
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


if __name__ == "__main__":
    main()
