#!/usr/bin/env python3
"""Full-grid prior-sensitivity map: every registered model, every dataset.

For each (extended model, dataset) this reweights the model's nested run onto a
homothetic contraction of its extra-parameter prior about the LCDM point, plus any
registered hard cuts, and recomputes the log Bayes factor against the LCDM run
(whose evidence is read directly from nested sampling). Results go to a
:class:`cosmocompare.catalogue.Catalogue` so posteriors and Bayes factors become a
later lookup.

Parallel-safe for a scheduler array: run with ``--task-index K`` to process the
single (model, dataset) pair K of the enumerated grid, writing to a per-pair shard.
After the array finishes, merge once with ``--merge``. Without ``--task-index`` it
runs all pairs sequentially (for a small validation subset on a login node).

Models and their priors come from :mod:`cosmocompare.models_registry` (neutrino
models are held out there). Nothing is submitted to a scheduler by this script.

Examples::

    # enumerate the grid
    python grid_prior_sweep.py --list
    # one array task
    python grid_prior_sweep.py --outdir ~/cc_grid --task-index $SLURM_ARRAY_TASK_ID
    # merge shards afterwards
    python grid_prior_sweep.py --outdir ~/cc_grid --merge
"""

import argparse
import os

import numpy as np
from anesthetic import read_chains

from cosmocompare.catalogue import Catalogue, Cell
from cosmocompare.config import available_datasets, resolve_chain_stem
from cosmocompare.evidence import importance_evidence, nested_evidence
from cosmocompare.models_registry import REGISTRY, model_names
from cosmocompare.prior import prior_log_ratio, region_volume_fraction
from cosmocompare.psis import psis
from cosmocompare.sweep import flip_fraction, shrink_uniform

FRACTIONS = [round(1.0 - 0.05 * i, 2) for i in range(19)]  # 1.00 -> 0.10
ESS_FLOOR = 1000.0


def read_logf(stem):
    """Return log(nprior/ndiscarded) for a chain stem, or 0.0 if unavailable."""
    info = f"{stem}.prior_info"
    if not os.path.exists(info):
        return 0.0
    nprior = ndiscarded = None
    with open(info) as stream:
        for line in stream:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "=":
                if parts[0] == "nprior":
                    nprior = float(parts[2])
                elif parts[0] == "ndiscarded":
                    ndiscarded = float(parts[2])
    if not nprior or not ndiscarded:
        return 0.0
    return float(np.log(nprior / ndiscarded))


def strip_bom_columns(samples):
    r"""Strip a leading UTF-8 BOM (``﻿``) from a chain's column labels.

    Some chains' ``.paramnames`` files were written with a byte-order mark, which
    anesthetic reads into the first parameter's name (for example
    ``'﻿Alens'`` instead of ``'Alens'``). Any lookup of that parameter by
    name then raises ``KeyError`` and the reweighting of that pair fails while the
    direct-nested baseline, which never indexes it, still succeeds. This renames
    the affected columns in place so the parameter is addressable again.
    """
    def fix(label):
        if isinstance(label, tuple):
            return tuple(x.lstrip("﻿") if isinstance(x, str) else x
                         for x in label)
        return label.lstrip("﻿") if isinstance(label, str) else label

    samples.columns = samples.columns.map(fix)
    return samples


def enumerate_pairs(models, base_model, primary=None, fallback=None):
    """Return the sorted list of (model, dataset) pairs with both chains present."""
    pairs = []
    base_ds = set(available_datasets(base_model, primary=primary, fallback=fallback))
    for model in models:
        for dataset in available_datasets(model, primary=primary, fallback=fallback):
            if dataset in base_ds:
                pairs.append((model, dataset))
    return sorted(pairs)


def _evaluate(ctx, log_r, logf, frac_sigma=0.0):
    """Reweight by a log prior ratio; return the metrics for a cell."""
    args = ctx["args"]
    ev = importance_evidence(
        ctx["ext_logL"], ctx["ext_logL"], base_weights=ctx["ext_weights"],
        log_prior_correction=log_r, base_log_evidence=ctx["ext_logZ"],
        base_log_evidence_err=ctx["ext_logZ_err"], n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    ln_evidence_err = float(np.hypot(ev.log_evidence_err or 0.0, frac_sigma))
    ln_b = ev.log_evidence - ctx["base_logZ"] + logf
    with np.errstate(divide="ignore"):
        khat = psis(np.log(ctx["ext_weights"]) + log_r).pareto_k
    return {
        "ln_evidence": ev.log_evidence, "ln_evidence_err": ln_evidence_err,
        "ln_bayes_factor": ln_b,
        "ln_bayes_factor_err": float(np.hypot(ln_evidence_err, ctx["base_logZ_err"])),
        "ess": ev.effective_sample_size, "pareto_k": khat,
        "degenerate_fraction": ev.bootstrap_degenerate_fraction,
    }


def _reliable(m):
    """Reliability gate: ESS above the floor and a finite Pareto k-hat < 0.7.

    A reweighted cell with enough surviving samples yields a finite k-hat (the base
    posterior weights vary within the region), so a non-finite k-hat means the fit
    failed for want of samples and the cell is unreliable. The direct-NS baseline,
    which has no k-hat, is marked reliable where it is saved, not here.
    """
    ess_ok = m["ess"] is not None and m["ess"] >= ESS_FLOOR
    k = m["pareto_k"]
    return bool(ess_ok and np.isfinite(k) and k < 0.7)


def _cell(ctx, method, params, m, log_r, flip=None):
    """Build a catalogue Cell from metrics and a per-sample log ratio."""
    return Cell(
        model=ctx["model"], base_model=ctx["base_model"], dataset=ctx["dataset"],
        prior_method=method, prior_params=params, estimator="is_ratio",
        ln_evidence=m["ln_evidence"], ln_evidence_err=m["ln_evidence_err"],
        ln_bayes_factor=m["ln_bayes_factor"],
        ln_bayes_factor_err=m["ln_bayes_factor_err"], ess=m["ess"],
        pareto_k=m["pareto_k"], degenerate_fraction=m["degenerate_fraction"],
        flip_fraction=flip, reliable=_reliable(m), chain_source=ctx["chain_source"],
        dlogw=log_r,
    )


def run_pair(model, dataset, args, catalogue):
    """Run baseline + homothetic sweep + cuts for one (model, dataset) pair."""
    spec = REGISTRY[model]
    ext_stem = resolve_chain_stem(model, dataset, primary=args.primary,
                                  fallback=args.fallback)
    base_stem = resolve_chain_stem(args.base_model, dataset, primary=args.primary,
                                   fallback=args.fallback)
    source = "new_grid" if "new_grid" in str(ext_stem) else "grid"
    ext = strip_bom_columns(read_chains(str(ext_stem)))
    base = strip_bom_columns(read_chains(str(base_stem)))
    logf = 0.0 if args.no_logf else read_logf(ext_stem) - read_logf(base_stem)

    base_logZ, base_logZ_err = nested_evidence(base, args.nsamples)
    ext_logZ, ext_logZ_err = nested_evidence(ext, args.nsamples)
    ctx = {
        "ext": ext, "model": model, "base_model": args.base_model,
        "dataset": dataset, "chain_source": source,
        "ext_logL": np.asarray(ext.logL, dtype=float),
        "ext_weights": np.asarray(ext.get_weights(), dtype=float),
        "ext_logZ": ext_logZ, "ext_logZ_err": ext_logZ_err,
        "base_logZ": base_logZ, "base_logZ_err": base_logZ_err, "args": args,
    }

    # Baseline (direct nested evidences).
    catalogue.save(Cell(
        model=model, base_model=args.base_model, dataset=dataset,
        prior_method="baseline", prior_params={}, estimator="direct_ns",
        ln_evidence=ext_logZ, ln_evidence_err=ext_logZ_err,
        ln_bayes_factor=ext_logZ - base_logZ + logf,
        ln_bayes_factor_err=float(np.hypot(ext_logZ_err, base_logZ_err)),
        reliable=True, chain_source=source, dlogw=None,
    ))

    # Homothetic contraction about each parameter's LCDM centre.
    sweep = []
    for f in args.fractions:
        changes = {
            p.column: {
                "base": {"type": "uniform", "min": p.low, "max": p.high},
                "new": shrink_uniform((p.low, p.high), f, p.centre),
            }
            for p in spec.params.values()
        }
        log_r = prior_log_ratio(ext, changes)
        sweep.append((float(f), log_r, _evaluate(ctx, log_r, logf)))
    records = [{"fraction": f, "log_bayes_factor": m["ln_bayes_factor"]}
               for f, _, m in sweep]
    ff = flip_fraction(records)
    for f, log_r, m in sweep:
        catalogue.save(_cell(ctx, "homothetic", {"fraction": f}, m, log_r, flip=ff))

    # Registered hard cuts (regions).
    for cut in spec.cuts:
        box = [(spec.params[c].low, spec.params[c].high) for c in cut.columns]
        frac = region_volume_fraction(cut.inside, box, n_mc=args.n_mc)
        values = np.column_stack([np.asarray(ext[c], float) for c in cut.columns])
        in_region = np.asarray(cut.inside(values), dtype=bool)
        log_r = np.where(in_region, -np.log(frac), -np.inf)
        frac_sigma = float(np.sqrt((1.0 - frac) / (frac * args.n_mc)))
        m = _evaluate(ctx, log_r, logf, frac_sigma=frac_sigma)
        catalogue.save(_cell(ctx, cut.method, {"kind": cut.kind}, m, log_r))

    return source, ext_logZ - base_logZ + logf, ff


def main():
    """Enumerate the grid and run one pair (array) or all pairs (sequential)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=model_names())
    parser.add_argument("--base-model", default="lcdm")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="restrict to these datasets (default: discover all)")
    parser.add_argument("--primary", default=None)
    parser.add_argument("--fallback", default=None)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-mc", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-logf", action="store_true")
    parser.add_argument("--outdir", default="cc_grid")
    parser.add_argument("--task-index", type=int, default=None,
                        help="run only pair K of the enumerated grid (array mode)")
    parser.add_argument("--list", action="store_true", help="print the grid and exit")
    parser.add_argument("--merge", action="store_true",
                        help="merge shards into summary.csv and exit")
    args = parser.parse_args()

    cat_root = os.path.join(args.outdir, "catalogue")

    if args.merge:
        merged = Catalogue(cat_root).merge_shards()
        print(f"Merged {len(merged)} cells into {cat_root}/summary.csv")
        return

    pairs = enumerate_pairs(args.models, args.base_model, args.primary, args.fallback)
    if args.datasets is not None:
        pairs = [(m, d) for (m, d) in pairs if d in set(args.datasets)]

    if args.list:
        for i, (m, d) in enumerate(pairs):
            print(f"{i:4d}  {m:10s}  {d}")
        print(f"\n{len(pairs)} pairs")
        return

    if args.task_index is not None:
        if not 0 <= args.task_index < len(pairs):
            raise SystemExit(f"task-index {args.task_index} out of range "
                             f"[0, {len(pairs) - 1}]")
        pairs = [pairs[args.task_index]]

    for model, dataset in pairs:
        shard = None if args.task_index is None else f"{model}__{dataset}"
        catalogue = Catalogue(cat_root, shard=shard)
        try:
            source, lnb, ff = run_pair(model, dataset, args, catalogue)
            print(f"[ok] {model:10s} {dataset:55s} [{source}] "
                  f"baseline lnB {lnb:+.3f}  flip {ff}")
        except Exception as exc:  # keep the array task alive, record the failure
            print(f"[FAIL] {model} {dataset}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
