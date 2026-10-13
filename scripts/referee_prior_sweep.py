#!/usr/bin/env python3
"""Prior-sensitivity sweep for the w0waCDM-vs-LCDM evidence (referee response).

For each requested dataset combination this re-weights the existing broad-prior
w0waCDM nested run onto progressively narrower (w0, wa) priors and recomputes the
log Bayes factor against LCDM, answering: how does the evidence for dynamical dark
energy depend on the prior bounds, and how narrow must they be to flip it?

The narrowing is homothetic about a fixed, data-independent centre (the LCDM point
w0 = -1, wa = 0), so the test is not post-hoc prior tuning. The reweighting is
exact (no likelihood calls, no new sampling); the effective sample size is
reported so that over-narrowed, unreliable points are visible, and any flip should
be confirmed with a dedicated nested run.

Nothing is submitted to a scheduler. The script reads chains and writes results
locally; run it on the node (or wrap it in a scheduler array) yourself.

Example
-------
    python referee_prior_sweep.py \
        --ns-root /home/dlo26/rds/rds-dirac-dp192-63QXlf5HuFo/dlo26/new_grid/ns \
        --outdir ./prior_sweep_results
"""

import argparse
import glob
import json
import os

from anesthetic import read_chains

from cosmocompare import nested_evidence, reweight_prior
from cosmocompare.sweep import bayes_factor_sweep, flip_fraction

#: Default base prior bounds of the extended-model parameters (DESI / this work).
PARAM_BASES = {"w0": (-3.0, 1.0), "wa": (-3.0, 2.0)}

#: Fixed, data-independent centre the prior shrinks towards (the LCDM point).
CENTRE = {"w0": -1.0, "wa": 0.0}

#: Width fractions for the homothetic sweep (1.0 = original DESI prior).
FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

#: The three dataset combinations the referee named (edit to match chain names).
DATASETS = [
    "bao.desi_dr2+planck_2018_CamSpec",
    "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie",
    "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
]

#: Discrete, externally-motivated narrower priors (rectangular, within the base).
#: Each maps a label to the narrowed bounds per parameter; unlisted parameters
#: keep their base bounds. Non-rectangular priors (e.g. the quintessence
#: constraint w0 + wa < 0) are not expressible here and would need a joint mask.
DISCRETE_PRIORS = {
    "popovic_w0_-2_0": {"w0": (-2.0, 0.0)},
    "quintessence_w0_>=-1": {"w0": (-1.0, 1.0)},
}


def find_chain_root(ns_root, model, dataset):
    """Return the anesthetic chain-root prefix for a (model, dataset), or None.

    Parameters
    ----------
    ns_root : str
        Root directory holding one sub-directory per model.
    model : str
        Model sub-directory name (for example ``"walcdm"`` or ``"lcdm"``).
    dataset : str
        Dataset-combination directory name.

    Returns
    -------
    str or None
        The path prefix to pass to ``anesthetic.read_chains`` (the ``.stats``
        file with its extension stripped), or None if no run is found.
    """
    pattern = os.path.join(ns_root, model, dataset, "**", "*.stats")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        return None
    return matches[0][: -len(".stats")]


def sweep_dataset(ext, base, args):
    """Run the homothetic sweep and the discrete priors for one dataset.

    Parameters
    ----------
    ext : anesthetic.NestedSamples
        The extended-model (w0waCDM) run.
    base : anesthetic.NestedSamples
        The base-model (LCDM) run.
    args : argparse.Namespace
        Parsed command-line options.

    Returns
    -------
    dict
        Sweep records, discrete-prior results, and the flip fraction.
    """
    records = bayes_factor_sweep(
        ext,
        base,
        PARAM_BASES,
        fractions=args.fractions,
        centres=CENTRE,
        nsamples=args.nsamples,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    base_logZ, _ = nested_evidence(base, args.nsamples)
    discrete = {}
    for label, bounds in DISCRETE_PRIORS.items():
        changes = {
            name: {
                "base": {
                    "type": "uniform",
                    "min": PARAM_BASES[name][0],
                    "max": PARAM_BASES[name][1],
                },
                "new": {"type": "uniform", "min": lo, "max": hi},
            }
            for name, (lo, hi) in bounds.items()
        }
        evidence, _ = reweight_prior(
            ext, changes, nsamples=args.nsamples,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        discrete[label] = {
            "bounds": {k: list(v) for k, v in bounds.items()},
            "log_bayes_factor": evidence.log_evidence - base_logZ,
            "effective_sample_size": evidence.effective_sample_size,
        }

    return {
        "sweep": records,
        "discrete": discrete,
        "flip_fraction": flip_fraction(records),
    }


def main():
    """Parse arguments, run the sweep for each dataset, and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns-root", required=True, help="root of the chain grid")
    parser.add_argument("--ext-model", default="walcdm", help="w0waCDM model dir")
    parser.add_argument("--base-model", default="lcdm", help="LCDM model dir")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="prior_sweep_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_results = {}

    for dataset in args.datasets:
        ext_root = find_chain_root(args.ns_root, args.ext_model, dataset)
        base_root = find_chain_root(args.ns_root, args.base_model, dataset)
        if ext_root is None or base_root is None:
            print(f"[skip] {dataset}: ext={ext_root is not None}, "
                  f"base={base_root is not None} (chain not found)")
            continue

        print(f"\n=== {dataset} ===")
        ext = read_chains(ext_root)
        base = read_chains(base_root)
        result = sweep_dataset(ext, base, args)
        all_results[dataset] = result

        print(f"{'f':>5} {'lnB':>9} {'+/-':>7} {'ESS':>8}")
        for r in result["sweep"]:
            print(f"{r['fraction']:5.2f} {r['log_bayes_factor']:9.3f} "
                  f"{r['log_bayes_factor_err']:7.3f} "
                  f"{r['effective_sample_size']:8.0f}")
        ff = result["flip_fraction"]
        print(f"flip fraction (lnB -> 0): {ff if ff is None else round(ff, 3)}")
        for label, d in result["discrete"].items():
            print(f"discrete {label}: lnB = {d['log_bayes_factor']:.3f} "
                  f"(ESS {d['effective_sample_size']:.0f})")

    out_path = os.path.join(args.outdir, "prior_sweep.json")
    with open(out_path, "w") as stream:
        json.dump(all_results, stream, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
