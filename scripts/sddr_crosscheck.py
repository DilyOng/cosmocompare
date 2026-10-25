"""Savage-Dickey cross-check for cross-model importance-sampling cases.

For each (base, ext) extension, estimate the log Bayes factor from the EXTENDED
chain alone with the Savage-Dickey density ratio, and compare to the nested lnB
(ext lnZ minus base lnZ). SDDR needs no likelihood re-evaluation, no MPI and no
Slurm: it is pure KDE post-processing and runs on a login node in seconds. It is
an independent cross-check on the importance-sampling evidence, and the estimator
of choice when the extension opens a direction the base posterior barely samples
(for example wCDM -> w0waCDM), where forward reweighting is biased.

It needs the EXTENDED chain, so it does not serve the "reuse the base chain only"
goal of cross-model IS; use it for validation and for poorly-overlapping cases.

Run on CSD3 (see memory extend-cross-model-is for the environment recipe):

    source .../venv_new_grid/bin/activate
    export PYTHONPATH=$HOME/cosmocompare_src
    export COBAYA_PACKAGES_PATH=.../new_grid/packages
    python scripts/sddr_crosscheck.py
"""
import argparse
import os
import sys

from anesthetic import read_chains

from cosmocompare.config import resolve_chain_stem
from cosmocompare.evidence import nested_evidence
from cosmocompare.extend import BASE_LIMITS
from cosmocompare.sddr import savage_dickey

# the yaml helpers live next to this script in extend_run.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extend_run import (  # noqa: E402
    prior_from_yaml,
    resolve_input_yaml,
    sampled_params_from_yaml,
)

DEFAULT_CASES = [
    ("lcdm", "klcdm"),
    ("lcdm", "wlcdm"),
    ("wlcdm", "walcdm"),
]


def crosscheck(base_model, ext_model, dataset):
    base_stem = resolve_chain_stem(base_model, dataset)
    ext_stem = resolve_chain_stem(ext_model, dataset)
    ext_y = resolve_input_yaml(ext_stem)
    base_s = set(sampled_params_from_yaml(resolve_input_yaml(base_stem)))
    added = [p for p in sampled_params_from_yaml(ext_y) if p not in base_s]
    if not added:
        raise SystemExit(f"{ext_model} adds no sampled parameters over {base_model}")

    base = read_chains(str(base_stem))
    ext = read_chains(str(ext_stem))
    nest_lnb = nested_evidence(ext)[0] - nested_evidence(base)[0]

    point = {p: BASE_LIMITS[p] for p in added}
    bounds = [(prior_from_yaml(ext_y, p)["min"],
               prior_from_yaml(ext_y, p)["max"]) for p in added]
    res = savage_dickey(ext, added, point, prior_bounds=bounds)
    return added, res, nest_lnb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bao.desi_dr2")
    ap.add_argument("--case", action="append", nargs=2, metavar=("BASE", "EXT"),
                    help="a (base ext) pair; repeatable. Defaults to the three "
                         "standard cases.")
    args = ap.parse_args()
    cases = args.case or DEFAULT_CASES

    hdr = "%-6s %-7s %-10s %8s %8s %9s %8s %8s %6s"
    print(hdr % ("base", "ext", "params", "sddrB", "nestB", "diff", "postDen",
                 "hdrMass", "ok"))
    for base_model, ext_model in cases:
        try:
            added, res, nest_lnb = crosscheck(base_model, ext_model, args.dataset)
            print(hdr % (
                base_model, ext_model, ",".join(added),
                "%+.3f" % res.log_bayes_factor,
                "%+.3f" % nest_lnb,
                "%+.3f" % (res.log_bayes_factor - nest_lnb),
                "%.3f" % res.log_posterior_density,
                "%.3f" % res.hdr_mass,
                "yes" if res.reliable else "NO",
            ))
        except Exception as exc:  # noqa: BLE001
            print("%-6s %-7s ERROR: %r" % (base_model, ext_model, exc))


if __name__ == "__main__":
    main()
