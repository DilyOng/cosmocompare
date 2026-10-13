#!/usr/bin/env python3
"""Plot the Bayes factor as a function of prior-narrowing choices.

Reads the JSON written by ``referee_prior_sweep.py`` and produces two panels:

- left: the log Bayes factor along the homothetic prior-width sweep (continuous
  narrowing about the LCDM point), with bootstrap bands, the decision line
  ``lnB = 0``, the flip point, and a marker where the effective sample size
  drops (results below it are unreliable);
- right: the log Bayes factor under discrete, externally-motivated prior choices
  (the full DESI prior, narrower literature boxes, and the physical non-phantom
  region), grouped by dataset.

Usage
-----
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

#: Effective-sample-size floor below which a reweighted point is unreliable.
ESS_FLOOR = 1000.0


def plot_sweep(ax, results):
    """Draw the homothetic prior-width sweep panel."""
    for colour, (name, r) in zip(COLOURS, results.items()):
        s = sorted(r["sweep"], key=lambda x: x["fraction"])
        f = np.array([p["fraction"] for p in s])
        lnB = np.array([p["log_bayes_factor"] for p in s])
        err = np.array([p["log_bayes_factor_err"] for p in s])
        ess = np.array([p["effective_sample_size"] for p in s])
        reliable = ess >= ESS_FLOOR

        ax.fill_between(f, lnB - err, lnB + err, color=colour, alpha=0.15, lw=0)
        ax.plot(f, lnB, color=colour, lw=1.8, label=SHORT.get(name, name))
        # Open markers where the ESS has collapsed (unreliable).
        ax.plot(f[reliable], lnB[reliable], "o", color=colour, ms=4)
        ax.plot(f[~reliable], lnB[~reliable], "o", mfc="white", mec=colour, ms=4)

        ff = r.get("flip_fraction")
        if ff is not None:
            ax.axvline(ff, color=colour, ls=":", lw=1, alpha=0.7)

    ax.axhline(0.0, color="0.3", ls="--", lw=1)
    ax.set_xlim(1.02, 0.08)  # narrower to the right
    ax.set_xlabel(r"prior width fraction $f$  (narrower $\rightarrow$)")
    ax.set_ylabel(r"$\ln B\ \ (w_0w_a\mathrm{CDM}$ vs $\Lambda\mathrm{CDM})$")
    ax.text(
        0.985,
        0.975,
        r"$\uparrow$ favours $w_0w_a$CDM",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.4",
    )
    ax.text(
        0.985,
        0.025,
        r"$\downarrow$ favours $\Lambda$CDM",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.4",
    )
    ax.text(
        0.015,
        0.025,
        "open markers: ESS < %d (unreliable)" % ESS_FLOOR,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.5",
    )
    ax.set_title("Homothetic narrowing about the $\\Lambda$CDM point", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)


def plot_choices(ax, results):
    """Draw the discrete prior-choice panel as grouped bars."""
    labels = [
        "DESI\n(full)",
        "Popovic\n$w\\in[-2,0]$",
        "quint.\n$w\\geq-1$",
        "non-phantom",
    ]
    names = list(results.items())
    nb = len(names)
    width = 0.8 / nb
    ymin, ymax = -3.0, 3.5

    for i, (colour, (name, r)) in enumerate(zip(COLOURS, names)):
        vals = [
            r["sweep"][0]["log_bayes_factor"],  # full prior (f = 1)
            r["discrete"]["popovic_w_-2_0"]["log_bayes_factor"],
            r["discrete"]["quintessence_w_>=-1"]["log_bayes_factor"],
            r["physical"]["log_bayes_factor"],
        ]
        ess = [None, None, None, r["physical"]["effective_sample_size"]]
        x = np.arange(len(labels)) + (i - (nb - 1) / 2) * width
        for xi, v, e in zip(x, vals, ess):
            clipped = max(v, ymin + 0.05)
            ax.bar(xi, clipped - 0, width=width, color=colour, bottom=0, alpha=0.9)
            if v < ymin:  # off-scale (physical region)
                ax.annotate(
                    "%.1f\n(ESS %d)" % (v, e),
                    (xi, ymin + 0.1),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="white",
                )

    ax.axhline(0.0, color="0.3", ls="--", lw=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r"$\ln B$")
    ax.set_title("Discrete prior choices", fontsize=10)


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
        "(prior-volume corrected)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", args.out, "and", args.out.replace(".pdf", ".png"))


if __name__ == "__main__":
    main()
