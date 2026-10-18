#!/usr/bin/env python3
"""Plot the Bayes factor as a function of prior-narrowing choices.

Reads the JSON written by ``referee_prior_sweep.py`` and produces two panels:

- left: the log Bayes factor along the homothetic prior-width sweep (continuous
  narrowing about the LCDM point), with bootstrap bands, the decision line
  ``lnB = 0``, the flip point, and open markers where a cell is unreliable
  (ESS below the floor or Pareto k-hat >= 0.7);
- right: the log Bayes factor under the discrete narrowing methods (the full
  prior, the physicality cut, the non-phantom region, and literature boxes),
  grouped by dataset.

The line is a point-to-point join of the computed cells (one exact reweighting per
fraction), not a fit; the shaded band is the per-point bootstrap standard
deviation. Usage::

    python plot_prior_sweep.py prior_sweep.json --out prior_sweep.pdf
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

#: Colour-blind-friendly palette, one colour per dataset.
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

#: Short labels for the dataset keys.
SHORT = {
    "bao.desi_dr2+planck_2018_CamSpec": "DESI DR2 + CMB",
    "bao.desi_dr2+planck_2018_CamSpec+sn.desy5": "DESI DR2 + CMB + DES-SN5YR",
    "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie": "DESI DR2 + CMB + DES-Dovekie",
    "bao.desi_dr2+planck_2018_CamSpec+sn.union3": "DESI DR2 + CMB + Union3",
}

#: Effective-sample-size floor below which a reweighted cell is unreliable.
ESS_FLOOR = 1000.0

#: Pareto k-hat above which a reweighted cell is unreliable.
KHAT_CEIL = 0.7


def _reliable(ess, khat):
    """Return a boolean mask: ESS above the floor and k-hat below the ceiling."""
    ess = np.asarray(ess, dtype=float)
    khat = np.asarray(khat, dtype=float)
    ess_ok = ess >= ESS_FLOOR
    khat_ok = ~np.isfinite(khat) | (khat < KHAT_CEIL)
    return ess_ok & khat_ok


def plot_sweep(ax, results):
    """Draw the homothetic prior-width sweep panel."""
    for colour, (name, r) in zip(COLOURS, results.items()):
        s = sorted(r["sweep"], key=lambda x: x["fraction"])
        f = np.array([p["fraction"] for p in s])
        lnB = np.array([p["log_bayes_factor"] for p in s])
        err = np.array([p["log_bayes_factor_err"] for p in s])
        ess = np.array([p["effective_sample_size"] for p in s])
        khat = np.array([p.get("pareto_k", np.nan) for p in s])
        reliable = _reliable(ess, khat)

        ax.fill_between(f, lnB - err, lnB + err, color=colour, alpha=0.15, lw=0)
        ax.plot(f, lnB, color=colour, lw=1.8, label=SHORT.get(name, name))
        ax.plot(f[reliable], lnB[reliable], "o", color=colour, ms=4)
        ax.plot(f[~reliable], lnB[~reliable], "o", mfc="white", mec=colour, ms=4)

        ff = r.get("flip_fraction")
        if ff is not None:
            ax.axvline(ff, color=colour, ls=":", lw=1, alpha=0.7)

    ax.axhline(0.0, color="0.3", ls="--", lw=1)
    ax.set_xlim(1.02, 0.08)
    ax.set_xlabel(r"prior width fraction $f$  (narrower $\rightarrow$)")
    ax.set_ylabel(r"$\ln B\ \ (w_0w_a\mathrm{CDM}$ vs $\Lambda\mathrm{CDM})$")
    ax.text(0.985, 0.975, r"$\uparrow$ favours $w_0w_a$CDM", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="0.4")
    ax.text(0.985, 0.025, r"$\downarrow$ favours $\Lambda$CDM", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.4")
    ax.text(0.015, 0.025, "open markers: ESS < %d or $\\hat{k}\\geq$ %.1f (unreliable)"
            % (ESS_FLOOR, KHAT_CEIL), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color="0.5")
    ax.set_title("Homothetic narrowing about the $\\Lambda$CDM point", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)


def _method_values(r):
    """Return (labels, values, reliable) for the discrete narrowing methods."""
    labels, values, reliable = [], [], []
    labels.append("full")
    values.append(r["baseline"]["log_bayes_factor"])
    reliable.append(True)
    region = r.get("region", {})
    for key, lab in (("physicality_cut", r"$w_0{+}w_a{<}0$"),
                     ("no_phantom", "non-\nphantom")):
        if key in region:
            labels.append(lab)
            values.append(region[key]["log_bayes_factor"])
            reliable.append(_reliable(region[key]["effective_sample_size"],
                                      region[key].get("pareto_k", np.nan)).item())
    for key, lab in (("popovic_w_-2_0", r"$w\in$" "\n" r"$[-2,0]$"),
                     ("quintessence_w_>=-1", r"$w\geq-1$")):
        d = r.get("discrete", {}).get(key)
        if d is not None:
            labels.append(lab)
            values.append(d["log_bayes_factor"])
            reliable.append(_reliable(d["effective_sample_size"],
                                      d.get("pareto_k", np.nan)).item())
    return labels, values, reliable


def plot_choices(ax, results):
    """Draw the discrete narrowing-method panel as grouped bars."""
    names = list(results.items())
    nb = len(names)
    labels = _method_values(names[0][1])[0]
    width = 0.8 / nb

    for i, (colour, (name, r)) in enumerate(zip(COLOURS, names)):
        _, vals, reliable = _method_values(r)
        x = np.arange(len(vals)) + (i - (nb - 1) / 2) * width
        for xi, v, ok in zip(x, vals, reliable):
            ax.bar(xi, v, width=width, color=colour, alpha=0.9 if ok else 0.35,
                   hatch=None if ok else "//")

    ax.axhline(0.0, color="0.3", ls="--", lw=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\ln B$")
    ax.set_title("Discrete narrowing methods", fontsize=10)
    ax.text(0.98, 0.02, "faded/hatched: unreliable", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="0.5")


def main():
    """Read the sweep JSON and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", help="prior_sweep.json from referee_prior_sweep.py")
    parser.add_argument("--out", default="prior_sweep.pdf")
    args = parser.parse_args()

    with open(args.json) as stream:
        results = json.load(stream)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    plot_sweep(ax1, results)
    plot_choices(ax2, results)
    fig.suptitle(
        "Prior sensitivity of the dynamical dark energy evidence "
        "(prior-volume corrected)", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", args.out, "and", args.out.replace(".pdf", ".png"))


if __name__ == "__main__":
    main()
