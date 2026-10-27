r"""Per-model prior registry for the full-grid prior-sensitivity map.

Each extended model in the ``unimpeded`` grid adds one or two parameters to
:math:`\Lambda\mathrm{CDM}`. To sweep every model's prior we need, per model, the
chain column name of each extra parameter, its prior range, the value that
recovers :math:`\Lambda\mathrm{CDM}` (the homothetic centre), whether that centre
is interior or on a boundary, and any physically- or theory-motivated hard cuts.

The entries below were checked with Gemini 3.1 Pro and GPT-5.4. Their corrections
are encoded:

- ``w >= -1`` (``wlcdm``) and the ``w0, wa`` non-phantom region are *theory*
  restrictions (quintessence-like), not mandatory physicality, so they are tagged
  ``kind="theory"``; the early-time ``w0 + wa < 0`` condition is a dark-energy
  *viability* cut, tagged ``kind="viability"``.
- ``r`` has its :math:`\Lambda\mathrm{CDM}` limit (:math:`r = 0`) on the lower
  boundary, so its sweep is tagged ``scan="boundary"`` and must be read as a
  boundary-nested contraction, not compared like-for-like with interior scans. It
  stays a linear-width scan: reweighting a uniform chain onto a log-uniform prior
  is a different prior family, not a narrower one, and diverges at zero.
- The neutrino models are **excluded**: ``mlcdm`` (``mnu`` in ``[0.06, 2.0]``)
  does not even contain the true :math:`\Lambda\mathrm{CDM}` point
  :math:`\sum m_\nu = 0`, and ``nlcdm``/``Nlcdm`` have an ambiguous/duplicated
  ``mnu``/``nnu`` naming. They await the user confirming the correct runs.
"""

from dataclasses import dataclass, field

import numpy as np

#: Neutrino models held out until the user confirms the correct runs and that the
#: prior contains the LCDM nesting point.
EXCLUDED_MODELS = ("mlcdm", "nlcdm", "Nlcdm")


@dataclass
class ParamSpec:
    """One extra parameter of an extended model.

    Attributes
    ----------
    column : str
        The parameter's column name in the chains.
    low, high : float
        The prior range it was sampled under.
    centre : float
        The value recovering LCDM (the homothetic contraction centre).
    scan : str
        ``"interior"`` when ``centre`` is inside the range, ``"boundary"`` when it
        sits on an edge (contraction is then one-sided toward that edge).
    """

    column: str
    low: float
    high: float
    centre: float
    scan: str = "interior"


@dataclass
class Cut:
    """A hard sub-region prior for an extended model.

    Attributes
    ----------
    method : str
        Label for the cut (the catalogue ``prior_method``).
    kind : str
        ``"viability"`` (a dark-energy viability condition) or ``"theory"`` (a
        theory-class restriction such as non-phantom). Reported, not enforced as
        "physical necessity".
    columns : tuple of str
        The parameter columns the cut acts on, in the order ``inside`` expects.
    inside : callable
        ``inside(array)`` for an ``(n, len(columns))`` array -> boolean mask.
    """

    method: str
    kind: str
    columns: tuple
    inside: callable


@dataclass
class ModelSpec:
    """An extended model: its extra parameters and its hard cuts."""

    name: str
    params: dict
    cuts: list = field(default_factory=list)


#: The validated registry of models to sweep (neutrinos excluded, see module doc).
REGISTRY = {
    "wlcdm": ModelSpec(
        "wlcdm",
        {"w": ParamSpec("w", -3.0, -0.333, -1.0, "interior")},
        cuts=[
            Cut("non_phantom", "theory", ("w",), lambda a: np.asarray(a)[:, 0] >= -1.0),
        ],
    ),
    "klcdm": ModelSpec(
        "klcdm", {"omk": ParamSpec("omk", -0.3, 0.3, 0.0, "interior")}, cuts=[]
    ),
    "walcdm": ModelSpec(
        "walcdm",
        {
            "w": ParamSpec("w", -3.0, 1.0, -1.0, "interior"),
            "wa": ParamSpec("wa", -3.0, 2.0, 0.0, "interior"),
        },
        cuts=[
            Cut(
                "physicality_cut",
                "viability",
                ("w", "wa"),
                lambda a: np.asarray(a)[:, 0] + np.asarray(a)[:, 1] < 0.0,
            ),
            Cut(
                "no_phantom",
                "theory",
                ("w", "wa"),
                lambda a: (np.asarray(a)[:, 0] >= -1.0)
                & (np.asarray(a)[:, 0] + np.asarray(a)[:, 1] >= -1.0),
            ),
        ],
    ),
    "Alcdm": ModelSpec(
        "Alcdm", {"Alens": ParamSpec("Alens", 0.0, 10.0, 1.0, "interior")}, cuts=[]
    ),
    "nrunlcdm": ModelSpec(
        "nrunlcdm", {"nrun": ParamSpec("nrun", -1.0, 1.0, 0.0, "interior")}, cuts=[]
    ),
    "rlcdm": ModelSpec(
        "rlcdm", {"r": ParamSpec("r", 0.0, 3.0, 0.0, "boundary")}, cuts=[]
    ),
}


def model_names():
    """Return the list of registered (sweepable) model names."""
    return list(REGISTRY)
