#!/usr/bin/env python3
"""Prior-sensitivity sweep for the w0waCDM-vs-LCDM evidence, writing a catalogue.

For each dataset combination this re-weights the existing broad-prior w0waCDM
nested run onto a menu of narrower (w0, wa) priors and recomputes the log Bayes
factor against LCDM, answering: how does the evidence for dynamical dark energy
depend on the prior, and how narrow must it be to flip it?

The base-model (LCDM) evidence is read directly from its nested run (``direct_ns``);
only the extended model's evidence moves under narrowing, computed by exact
importance-sampling reweighting (``is_ratio``) of the nested dead points, with no
likelihood calls and no new sampling. Every result cell (its metrics and its
per-sample log prior ratio ``dlogw``) is written to a ``Catalogue`` so posteriors
and Bayes factors become a later lookup, not a recompute. The Kish ESS and the
Pareto-k-hat are reported so unreliable cells are visible.

Narrowing methods applied, all data-independent:
- ``homothetic``: contraction of the DESI box about the LCDM point (the lnB(f) curve);
- ``one_sided``: raising the lower (phantom) bounds of w0 and wa together;
- ``box``: discrete literature boxes;
- ``physicality_cut``: w0 + wa < 0 (early-time non-phantom);
- ``no_phantom``: w0 >= -1 and w0 + wa >= -1.

Nothing is submitted to a scheduler. Chains are read in place via the two-root
resolver (new_grid first, then grid). Example::

    python referee_prior_sweep.py --outdir ./prior_sweep_results
"""

import argparse
import json
import os

import numpy as np
from anesthetic import read_chains

from cosmocompare.catalogue import Catalogue, Cell
from cosmocompare.config import resolve_chain_stem
from cosmocompare.dark_energy import no_phantom, physicality_cut
from cosmocompare.evidence import importance_evidence, nested_evidence
from cosmocompare.prior import prior_log_ratio, region_volume_fraction
from cosmocompare.psis import psis
from cosmocompare.sweep import flip_fraction, shrink_uniform

#: Base prior bounds of the extended-model parameters (DESI / this work). In the
#: chains the present-day equation of state is named ``w`` (i.e. w0).
PARAM_BASES = {"w": (-3.0, 1.0), "wa": (-3.0, 2.0)}

#: Fixed, data-independent centre the prior shrinks towards (the LCDM point).
CENTRE = {"w": -1.0, "wa": 0.0}

#: Width fractions for the homothetic sweep (1.0 = original DESI prior).
FRACTIONS = [round(1.0 - 0.05 * i, 2) for i in range(19)]  # 1.00 -> 0.10 step 0.05

#: The four flagship dataset combinations (DES-SN5YR chain name is ``sn.desy5``).
DATASETS = [
    "bao.desi_dr2+planck_2018_CamSpec",
    "bao.desi_dr2+planck_2018_CamSpec+sn.desy5",
    "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie",
    "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
]

#: Discrete, externally-motivated narrower rectangular priors (within the base).
DISCRETE_PRIORS = {
    "popovic_w_-2_0": {"w": (-2.0, 0.0)},
    "quintessence_w_>=-1": {"w": (-1.0, 1.0)},
}

#: One-sided axis (Patel, Chakraborty & Amendola 2024): raise the lower (phantom)
#: bound of w0 and wa together, keeping the upper bounds.
LOWER_BOUNDS = [-3.0, -2.5, -2.0, -1.5, -1.0]

#: Effective-sample-size floor below which a reweighted cell is unreliable.
ESS_FLOOR = 1000.0


def read_logf(stem):
    """Return the PolyChord correction ``log(nprior/ndiscarded)`` for a chain stem.

    Parameters
    ----------
    stem : str or pathlib.Path
        The chain stem (``.prior_info`` is read beside it).

    Returns
    -------
    float
        ``log(nprior / ndiscarded)``, or 0.0 if the file is missing/incomplete.
    """
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


def _box_changes(new_bounds):
    """Build a prior-change dict narrowing the base box to ``new_bounds``."""
    changes = {}
    for name, base in PARAM_BASES.items():
        lo, hi = new_bounds.get(name, base)
        changes[name] = {
            "base": {"type": "uniform", "min": base[0], "max": base[1]},
            "new": {"type": "uniform", "min": lo, "max": hi},
        }
    return changes


def _evaluate(ctx, log_r, logf, frac_sigma=0.0):
    """Reweight by a per-sample log prior ratio and return the cell metrics.

    Parameters
    ----------
    ctx : dict
        Precomputed context: ext_logL, ext_weights, ext_logZ, ext_logZ_err,
        base_logZ, base_logZ_err, args.
    log_r : numpy.ndarray
        Per-sample ``log(pi_new / pi_old)``.
    logf : float
        The ``logF_ext - logF_base`` prior-initialisation correction.
    frac_sigma : float, optional
        Extra log-space error from a Monte-Carlo region-volume estimate.

    Returns
    -------
    dict
        ``{ln_evidence, ln_evidence_err, ln_bayes_factor, ln_bayes_factor_err,
        ess, pareto_k}``.
    """
    args = ctx["args"]
    ev = importance_evidence(
        ctx["ext_logL"],
        ctx["ext_logL"],
        base_weights=ctx["ext_weights"],
        log_prior_correction=log_r,
        base_log_evidence=ctx["ext_logZ"],
        base_log_evidence_err=ctx["ext_logZ_err"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    ln_evidence = ev.log_evidence
    ln_evidence_err = float(np.hypot(ev.log_evidence_err or 0.0, frac_sigma))
    ln_b = ln_evidence - ctx["base_logZ"] + logf
    ln_b_err = float(np.hypot(ln_evidence_err, ctx["base_logZ_err"]))

    with np.errstate(divide="ignore"):
        khat = psis(np.log(ctx["ext_weights"]) + log_r).pareto_k

    return {
        "ln_evidence": ln_evidence,
        "ln_evidence_err": ln_evidence_err,
        "ln_bayes_factor": ln_b,
        "ln_bayes_factor_err": ln_b_err,
        "ess": ev.effective_sample_size,
        "pareto_k": khat,
        "degenerate_fraction": ev.bootstrap_degenerate_fraction,
    }


def _reliable(metrics):
    """Reliability gate: ESS above the floor and a finite Pareto k-hat < 0.7."""
    ess_ok = metrics["ess"] is not None and metrics["ess"] >= ESS_FLOOR
    k = metrics["pareto_k"]
    return bool(ess_ok and np.isfinite(k) and k < 0.7)


def _cell(ctx, method, params, metrics, log_r, flip=None):
    """Assemble a catalogue Cell from metrics and the per-sample log ratio."""
    return Cell(
        model=ctx["ext_model"],
        base_model=ctx["base_model"],
        dataset=ctx["dataset"],
        prior_method=method,
        prior_params=params,
        estimator="is_ratio",
        ln_evidence=metrics["ln_evidence"],
        ln_evidence_err=metrics["ln_evidence_err"],
        ln_bayes_factor=metrics["ln_bayes_factor"],
        ln_bayes_factor_err=metrics["ln_bayes_factor_err"],
        ess=metrics["ess"],
        pareto_k=metrics["pareto_k"],
        degenerate_fraction=metrics["degenerate_fraction"],
        flip_fraction=flip,
        reliable=_reliable(metrics),
        chain_source=ctx["chain_source"],
        dlogw=log_r,
    )


def sweep_dataset(ctx, catalogue, logf):
    """Run every narrowing method for one dataset and write cells to the catalogue.

    Parameters
    ----------
    ctx : dict
        Dataset context (chains, evidences, model names, args).
    catalogue : cosmocompare.catalogue.Catalogue
        The store to write cells to.
    logf : float
        The prior-initialisation correction ``logF_ext - logF_base``.

    Returns
    -------
    dict
        A JSON-friendly summary (for the plotting script), including the sweep
        records, the flip fraction, and the discrete/one-sided/physical results.
    """
    ext = ctx["ext"]

    # Baseline: LCDM evidence is direct from nested sampling; the f = 1 extended
    # evidence is also direct (no reweighting). lnB is their difference + logF.
    baseline_lnb = ctx["ext_logZ"] - ctx["base_logZ"] + logf
    baseline_lnb_err = float(np.hypot(ctx["ext_logZ_err"], ctx["base_logZ_err"]))
    catalogue.save(
        Cell(
            model=ctx["ext_model"], base_model=ctx["base_model"],
            dataset=ctx["dataset"], prior_method="baseline", prior_params={},
            estimator="direct_ns", ln_evidence=ctx["ext_logZ"],
            ln_evidence_err=ctx["ext_logZ_err"], ln_bayes_factor=baseline_lnb,
            ln_bayes_factor_err=baseline_lnb_err,
            ess=float("nan"), pareto_k=float("nan"), reliable=True,
            chain_source=ctx["chain_source"], dlogw=None,
        )
    )

    # Homothetic contraction about the LCDM point: the lnB(f) curve. Compute each
    # fraction once (log ratio + metrics), derive the flip fraction, then save.
    sweep = []
    for f in ctx["args"].fractions:
        changes = {
            name: {
                "base": {"type": "uniform", "min": b[0], "max": b[1]},
                "new": shrink_uniform(b, f, CENTRE[name]),
            }
            for name, b in PARAM_BASES.items()
        }
        log_r = prior_log_ratio(ext, changes)
        sweep.append((float(f), log_r, _evaluate(ctx, log_r, logf)))

    sweep_records = [
        {
            "fraction": f,
            "log_bayes_factor": m["ln_bayes_factor"],
            "log_bayes_factor_err": m["ln_bayes_factor_err"],
            "effective_sample_size": m["ess"],
            "pareto_k": m["pareto_k"],
        }
        for f, _, m in sweep
    ]
    ff = flip_fraction(sweep_records)
    for f, log_r, m in sweep:
        catalogue.save(_cell(ctx, "homothetic", {"fraction": f}, m, log_r, flip=ff))

    # One-sided: raise the lower (phantom) bound of w0 and wa together.
    one_sided = []
    for low in LOWER_BOUNDS:
        changes = _box_changes({name: (low, b[1]) for name, b in PARAM_BASES.items()})
        log_r = prior_log_ratio(ext, changes)
        m = _evaluate(ctx, log_r, logf)
        catalogue.save(_cell(ctx, "one_sided", {"lower_bound": low}, m, log_r))
        one_sided.append({"lower_bound": low, **_public(m)})

    # Discrete literature boxes.
    discrete = {}
    for label, bounds in DISCRETE_PRIORS.items():
        changes = _box_changes(bounds)
        log_r = prior_log_ratio(ext, changes)
        m = _evaluate(ctx, log_r, logf)
        catalogue.save(
            _cell(ctx, "box", {"label": label, "bounds": bounds}, m, log_r)
        )
        discrete[label] = {"bounds": {k: list(v) for k, v in bounds.items()},
                           **_public(m)}

    # Hard physical cuts (regions): compute the normalised log ratio and the MC
    # volume-fraction error, then evaluate.
    base_box = [PARAM_BASES["w"], PARAM_BASES["wa"]]
    region = {}
    for method, (params, inside) in (
        ("physicality_cut", physicality_cut(("w", "wa"))),
        ("no_phantom", no_phantom(("w", "wa"))),
    ):
        frac = region_volume_fraction(inside, base_box, n_mc=ctx["args"].n_mc)
        values = np.column_stack([np.asarray(ext[p], float) for p in ("w", "wa")])
        in_region = np.asarray(inside(values), dtype=bool)
        log_r = np.where(in_region, -np.log(frac), -np.inf)
        frac_sigma = float(np.sqrt((1.0 - frac) / (frac * ctx["args"].n_mc)))
        m = _evaluate(ctx, log_r, logf, frac_sigma=frac_sigma)
        catalogue.save(_cell(ctx, method, {}, m, log_r))
        region[method] = _public(m)

    return {
        "sweep": sweep_records,
        "flip_fraction": ff,
        "one_sided": one_sided,
        "discrete": discrete,
        "region": region,
        "baseline": {"log_bayes_factor": baseline_lnb,
                     "log_bayes_factor_err": baseline_lnb_err},
        "logf_correction": logf,
    }


def _public(m):
    """Return the JSON-friendly subset of an _evaluate metrics dict."""
    return {
        "log_bayes_factor": m["ln_bayes_factor"],
        "log_bayes_factor_err": m["ln_bayes_factor_err"],
        "effective_sample_size": m["ess"],
        "pareto_k": m["pareto_k"],
    }


def main():
    """Resolve chains, run the sweep for each dataset, and write the catalogue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext-model", default="walcdm", help="w0waCDM model dir")
    parser.add_argument("--base-model", default="lcdm", help="LCDM model dir")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--primary", default=None, help="primary grid root override")
    parser.add_argument("--fallback", default=None, help="fallback grid root override")
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-mc", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-logf", action="store_true",
                        help="disable the log(nprior/ndiscarded) correction")
    parser.add_argument("--outdir", default="prior_sweep_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    catalogue = Catalogue(os.path.join(args.outdir, "catalogue"))
    all_results = {}

    for dataset in args.datasets:
        try:
            ext_stem = resolve_chain_stem(
                args.ext_model, dataset, primary=args.primary, fallback=args.fallback
            )
            base_stem = resolve_chain_stem(
                args.base_model, dataset, primary=args.primary, fallback=args.fallback
            )
        except FileNotFoundError as exc:
            print(f"[skip] {dataset}: {exc}")
            continue

        source = "new_grid" if "new_grid" in str(ext_stem) else "grid"
        print(f"\n=== {dataset}  [{source}] ===")
        ext = read_chains(str(ext_stem))
        base = read_chains(str(base_stem))
        logf = 0.0 if args.no_logf else read_logf(ext_stem) - read_logf(base_stem)

        base_logZ, base_logZ_err = nested_evidence(base, args.nsamples)
        ext_logZ, ext_logZ_err = nested_evidence(ext, args.nsamples)
        ctx = {
            "ext": ext, "ext_model": args.ext_model, "base_model": args.base_model,
            "dataset": dataset, "chain_source": source,
            "ext_logL": np.asarray(ext.logL, dtype=float),
            "ext_weights": np.asarray(ext.get_weights(), dtype=float),
            "ext_logZ": ext_logZ, "ext_logZ_err": ext_logZ_err,
            "base_logZ": base_logZ, "base_logZ_err": base_logZ_err,
            "args": args,
        }

        result = sweep_dataset(ctx, catalogue, logf)
        all_results[dataset] = result
        print(f"logF correction: {logf:+.3f}   baseline lnB "
              f"{result['baseline']['log_bayes_factor']:+.3f}   "
              f"flip f={result['flip_fraction']}")

    out_path = os.path.join(args.outdir, "prior_sweep.json")
    with open(out_path, "w") as stream:
        json.dump(all_results, stream, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Catalogue: {catalogue.root}  ({len(catalogue.load_summary())} cells)")


if __name__ == "__main__":
    main()
