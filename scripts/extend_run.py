#!/usr/bin/env python3
"""Cross-model importance sampling: reweight an LCDM run onto an extended model.

Reads a base (LCDM) nested run, augments it with the extended model's new
parameter drawn from its prior, re-evaluates the extended likelihood with Cobaya,
and estimates the extended evidence by importance sampling - no new nested run.
The base likelihood is re-evaluated in the LCDM limit with the same calculator
(not read from the chain) so the ratio is internally consistent. The result is
compared with the DIRECT nested evidence of the extended model (read from its own
run) as the gold-standard validation.

Reports Kish ESS, its fraction, the PSIS Pareto k-hat (the reliability diagnostic
that catches support mismatch), the largest single weight, and the stored-vs-fresh
base-likelihood consistency.

Needs a Cobaya environment (for example CSD3 venv_grid). Example::

    python extend_run.py --base-model lcdm --ext-model wlcdm \
        --dataset bao.desi_dr2 --new-param w --lcdm-value -1 \
        --outdir ~/cc_extend
"""

import argparse
import json
import os

import numpy as np
import yaml
from anesthetic import read_chains

from cosmocompare.config import resolve_chain_stem
from cosmocompare.evidence import nested_evidence
from cosmocompare.extend import run_extension
from cosmocompare.likelihood import LikelihoodCalculator
from cosmocompare.models import TargetModel

#: Value of each new parameter that recovers LCDM.
LCDM_LIMIT = {"w": -1.0, "omegak": 0.0, "omk": 0.0, "w0": -1.0, "wa": 0.0}


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
    parser.add_argument("--ext-model", required=True, help="e.g. wlcdm, klcdm")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--new-param", default=None,
                        help="new parameter name; inferred if omitted")
    parser.add_argument("--lcdm-value", type=float, default=None,
                        help="value recovering LCDM; inferred if omitted")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="cap to the top-weight N base samples (default: all)")
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
    ext_yaml = f"{ext_stem}.input.yaml"
    print(f"base: {base_stem}\next : {ext_stem}\nyaml: {ext_yaml}")

    base = read_chains(str(base_stem))
    ext = read_chains(str(ext_stem))
    base_logZ, base_logZ_err = nested_evidence(base, args.nsamples)
    ext_logZ, ext_logZ_err = nested_evidence(ext, args.nsamples)
    nested_lnB = ext_logZ - base_logZ
    print(f"nested ln Z: base={base_logZ:.3f}+/-{base_logZ_err:.3f}  "
          f"ext={ext_logZ:.3f}+/-{ext_logZ_err:.3f}  nested ln B={nested_lnB:+.3f}")

    calc = LikelihoodCalculator(ext_yaml)
    # Infer the new parameter: the sampled parameter the base chain lacks.
    base_cols = {c[0] if isinstance(c, tuple) else c for c in base.columns}
    new_param = args.new_param
    if new_param is None:
        extra = [p for p in calc.sampled_params if p not in base_cols]
        if len(extra) != 1:
            raise SystemExit(f"Cannot infer the new parameter; candidates: {extra}. "
                             f"Pass --new-param.")
        new_param = extra[0]
    lcdm_value = (args.lcdm_value if args.lcdm_value is not None
                  else LCDM_LIMIT.get(new_param))
    if lcdm_value is None:
        raise SystemExit(f"No LCDM limit known for {new_param!r}; pass --lcdm-value.")
    print(f"new parameter: {new_param}  (LCDM limit = {lcdm_value})")

    # Cap to the top-weight samples if asked (keeps the posterior bulk).
    if args.max_samples and len(base) > args.max_samples:
        w = np.asarray(base.get_weights(), dtype=float)
        keep = np.argsort(w)[-args.max_samples:]
        base = base.iloc[np.sort(keep)]
        print(f"capped base run to the top-weight {len(base)} samples")

    prior = prior_from_yaml(ext_yaml, new_param)
    # Proposal = prior (q = pi), so the correction vanishes and we measure the
    # honest effective sample size of the bridge.
    target = TargetModel(args.ext_model, {"yaml_path": ext_yaml,
                                          "new_params": {new_param: {"prior": prior}}})

    rng = np.random.default_rng(args.seed)
    result, _ = run_extension(
        base, target, calc, {new_param: lcdm_value},
        base_log_evidence=base_logZ, base_log_evidence_err=base_logZ_err,
        rng=rng, n_cores=args.n_cores, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    out = {
        "base_model": args.base_model, "ext_model": args.ext_model,
        "dataset": args.dataset, "new_param": new_param, "lcdm_value": lcdm_value,
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
    name = f"{args.ext_model}_{args.dataset}"
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
