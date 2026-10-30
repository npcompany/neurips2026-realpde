"""Analytic NACA4418 geometry: a signed distance field at any angle of attack.

**numpy only** — same rule as :mod:`rpde_t1.features`, because a submission needs
this at inference and must not import anything the evaluation container lacks.

Why analytic rather than measured. The simulated invalid mask *is* the solid body,
but only a small interior core of it: the CFD writes exact zeros in roughly 20 cells
where the analytic section covers ~200, because the foil is 18% thick (12.5 mm, about
7 cells) and only its thickest middle region fully contains cells. So the mask cannot
be used as an outline — fitting one to it drove the chord to 26 mm against the
physically known 69.7 mm. It can only be used to *check* an outline by containment.

How the placement was determined, and how well:

* **Chord 69.7 mm** — the one independently established number. It comes from
  inverting ``Re = U c / nu`` on the 82 real trajectories, which gave c/nu = 69654
  in water. Not fitted here.
* **Orientation** — the section is mirrored in y before rotation (``MIRROR_Y``) and
  rotated with ``AOA_SIGN = +1``. Established against the simulated *low-speed body*
  (time-mean speed below 2% of freestream at native resolution, ~200-245 cells per
  angle): a free fit of the un-mirrored section lands at the negative of the labelled
  angle at every rung and the mirrored camber wins at all five angles (FINDINGS 113).
  The earlier ``AOA_SIGN = -1``, no-mirror placement was the mirror image of the body
  about the pivot row; the ~20-cell exact-zero core it was checked against sits along
  the chord near the pivot and is covered either way, which is how it passed.
* **Pivot and position** — fitted with that orientation by maximising IoU against the
  low-speed body across all five angles: ``pivot_frac 0.45``, mean IoU **0.84** (0.74 at
  AoA 0, where the boundary layer is counted as body by the threshold, 0.81-0.92 at
  5-20). Reproduce with ``scripts/eval/geometry_orientation_check.py``.
* **Row order.** :func:`grid_coordinates` assigns row 0 the *lowest* y while the released
  files store row 0 at the *highest* y. The placement below is fitted in this module's own
  frame, so it is self-consistent; do not mix its ``Y`` with a file's ``y`` array.

The cross-stream length scale used for Strouhal reasoning is
:func:`projected_height`, which is exact. An earlier bounding-rectangle estimate
(``c*sin(a) + t_max*cos(a)``) overestimated it by ~38% at 10 degrees and is retained
nowhere.

Honest limits, and they matter:

* The optimum is **shallow and under-determined**. The analytic section covers ~200
  cells where the mask has ~20, so many placements score similarly; an earlier fit
  against a coarser polygon (n=70) preferred a quarter-chord pivot and reported 1.00
  everywhere, which did **not** survive refitting at this resolution. Treat the pivot
  as a fitted parameter, not as a recovered physical convention.
* Real cells cannot pin the placement: the PIV hole is a thin strip plus the laser
  shadow, and the fast-flow edge above the body is a separated shear layer at incidence.
  The temporal-variance map puts the real body a few native cells higher toward the
  trailing edge at 15-20 degrees than this sim-fitted section (FINDINGS 113).
* Net: the field is good to roughly ±1 cell (3.4 mm at the scored resolution). That is
  fine for a smooth conditioning feature and **not** fine for anything depending on
  the exact zero level set.

Why it is worth having: it makes geometry available at **any** angle of attack,
including ones never seen in training. The private test uses unseen AoA, so a learned
angle embedding must extrapolate whereas this is exact by construction.
"""

from __future__ import annotations

import numpy as np

#: NACA 4418: 4% maximum camber, at 40% chord, 18% thick.
NACA_CAMBER = 0.04
NACA_CAMBER_POS = 0.40
NACA_THICKNESS = 0.18

#: Chord in metres, from inverting Re = U*c/nu over the real trajectories.
CHORD_M = 0.0697

#: Pivot as a fraction of chord, and its position in metres. Fitted against the
#: n=120 polygon this module actually builds; mean containment 0.917. A shallow
#: optimum -- see the module docstring before treating these as physical constants.
PIVOT_FRAC = 0.45
PIVOT_XY_M = (0.03574, 0.13752)

#: The NACA section is built with its upper surface at +y; on this module's grid (row 0 at
#: the lowest y) the body's camber points toward *smaller* y, so the section is mirrored
#: before rotation. Rotation sense then follows: positive angle of attack puts the trailing
#: edge at smaller y. Both measured against the simulated body, FINDINGS 113.
MIRROR_Y = np.array([1.0, -1.0])
AOA_SIGN = 1.0

#: The measurement grid, identical in every trajectory (verified across the release).
#: Native is 128 columns x 64 rows; the scored resolution takes every second point.
GRID_X_M = (0.0017, 0.2190)
GRID_Y_M = (0.0821, 0.1899)
GRID_NATIVE_HW = (64, 128)


def naca4418_polygon(n: int = 120) -> np.ndarray:
    """Closed unit-chord section as ``(2n-1, 2)`` points, upper surface then lower.

    Cosine spacing puts points where curvature is highest, at the leading edge. The
    ``-0.1036`` coefficient closes the trailing edge (the classic ``-0.1015`` leaves
    it open, which would make the polygon non-closed and break the inside test).
    """
    m, p, t = NACA_CAMBER, NACA_CAMBER_POS, NACA_THICKNESS
    x = (1 - np.cos(np.linspace(0.0, np.pi, n))) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.where(
        x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
    )
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    theta = np.arctan(dyc)
    upper = np.stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)], axis=1)
    lower = np.stack([x + yt * np.sin(theta), yc - yt * np.cos(theta)], axis=1)[::-1][1:]
    return np.concatenate([upper, lower])


def place_polygon(
    aoa_deg: float,
    polygon: np.ndarray | None = None,
    chord: float = CHORD_M,
    pivot_frac: float = PIVOT_FRAC,
    pivot_xy: tuple[float, float] = PIVOT_XY_M,
    sign: float = AOA_SIGN,
) -> np.ndarray:
    """Scale, rotate about the pivot and translate the section into world metres."""
    poly = naca4418_polygon() if polygon is None else polygon
    a = sign * np.deg2rad(aoa_deg)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    centred = (poly - np.array([pivot_frac, 0.0])) * chord * MIRROR_Y
    return centred @ rot.T + np.asarray(pivot_xy)


def grid_coordinates(h: int = 32, w: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` in metres for an ``h x w`` view of the fixed measurement grid."""
    y = np.linspace(GRID_Y_M[0], GRID_Y_M[1], GRID_NATIVE_HW[0])
    x = np.linspace(GRID_X_M[0], GRID_X_M[1], GRID_NATIVE_HW[1])
    ys = y[:: max(1, GRID_NATIVE_HW[0] // h)][:h]
    xs = x[:: max(1, GRID_NATIVE_HW[1] // w)][:w]
    return np.meshgrid(xs, ys)


def _point_in_polygon(poly: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Ray-casting test, vectorised over query points."""
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    dy = np.where(y2 == y, 1e-12, y2 - y)
    crossings = ((y > py[:, None]) != (y2 > py[:, None])) & (
        px[:, None] < (x2 - x) * (py[:, None] - y) / dy + x
    )
    return crossings.sum(axis=1) % 2 == 1


def _distance_to_polygon(poly: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Unsigned distance from each query point to the polygon boundary."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    seg = b - a  # (E, 2)
    len2 = (seg**2).sum(axis=1)
    len2 = np.where(len2 == 0, 1e-24, len2)
    q = np.stack([px, py], axis=1)[:, None, :]  # (P, 1, 2)
    t = (((q - a) * seg).sum(axis=2) / len2).clip(0.0, 1.0)  # (P, E)
    closest = a + t[..., None] * seg
    return np.sqrt(((q - closest) ** 2).sum(axis=2)).min(axis=1)


def airfoil_sdf(aoa_deg: float, h: int = 32, w: int = 64, normalize: bool = True) -> np.ndarray:
    """Signed distance to the foil surface on the scored grid: ``(h, w)``, float32.

    Negative inside the solid, positive in the fluid. With ``normalize`` the distance
    is expressed in chords, which keeps the feature scale-free and consistent with the
    model working in non-dimensional velocity.
    """
    X, Y = grid_coordinates(h, w)
    poly = place_polygon(aoa_deg)
    px, py = X.ravel(), Y.ravel()
    dist = _distance_to_polygon(poly, px, py)
    signed = np.where(_point_in_polygon(poly, px, py), -dist, dist).reshape(X.shape)
    if normalize:
        signed = signed / CHORD_M
    return signed.astype(np.float32)


def projected_height(
    aoa_deg: float, polygon: np.ndarray | None = None, chord: float = CHORD_M
) -> float:
    """Exact extent of the section perpendicular to the flow, in metres.

    This is the length scale the Strouhal number needs: the width the body presents
    *across* the stream, which is what sets the wake width and hence the shedding
    frequency.

    Each surface point is projected onto ``(-sin a, cos a)`` -- the unit vector normal
    to the flow direction ``(cos a, sin a)`` -- and the spread of those projections is
    taken. Equivalently: rotate the body by ``-a`` and measure its vertical extent.
    Exact for the polygon, no geometric approximation.

    Prefer this over the bounding-rectangle estimate ``c*sin(a) + t_max*cos(a)``, which
    treats the section as a filled rectangle and therefore counts corners the airfoil
    does not have. Measured error of that approximation: **+41% at 5 deg, +38% at
    10 deg, +30% at 20 deg**, and -2.3% at 0 deg (where camber makes the true y-extent
    0.1842c exceed t_max = 0.18c). It is exact only at 0 and 90 degrees, and wrong
    everywhere in between -- including the whole range this dataset covers.
    """
    poly = naca4418_polygon() if polygon is None else polygon
    a = np.deg2rad(aoa_deg)
    v = -poly[:, 0] * np.sin(a) + poly[:, 1] * np.cos(a)
    return float((v.max() - v.min()) * chord)


def strouhal_period(aoa_deg: float, u_fs: float, strouhal: float = 0.123) -> float:
    """Estimated vortex-shedding period in seconds, from ``St = f*h/U``.

    ``h`` is the exact projected height. The default ``strouhal`` is the mean measured
    on this dataset **using that exact height** (0.089-0.151 across cells). Note this
    sits near the square-cylinder value (~0.13) rather than the circular-cylinder 0.2;
    an earlier estimate of ~0.19 was an artefact of the bounding-rectangle height.
    """
    return projected_height(aoa_deg) / max(1e-9, strouhal * u_fs)


#: Angle, in degrees, mapped to half a period of the lowest Fourier frequency. Set to
#: the top of the released range so the basis is monotone across 0-20 and still
#: bounded and smooth beyond it, which is where the private test is expected to sit.
AOA_FOURIER_SCALE = 20.0


def inlet_direction(aoa_deg: float) -> np.ndarray:
    """Freestream direction as a unit 2-vector ``(cos a, sin a)``, float32.

    A continuous encoding of incidence: unlike a five-way categorical it interpolates
    and extrapolates to unseen angles, which the private test requires. Paired with
    :func:`airfoil_sdf` this separates "where the body is" from "which way the flow
    comes in".
    """
    a = np.deg2rad(aoa_deg)
    return np.array([np.cos(a), AOA_SIGN * np.sin(a)], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Geometry as extra model input channels
# --------------------------------------------------------------------------- #
#
# Today the angle of attack reaches a model as the bare scalar ``AoA/20`` inside the
# conditioning vector -- and the vendored baselines ignore the conditioning vector
# entirely, so for FNO it reaches the model as *nothing at all*. Measured cost of an
# unseen angle: about -4 final points, worst on ``mvpe`` (-5.52), which is a
# wake-position metric. A scalar cannot express where the body is.
#
# These planes can, and every one of them is computable from the input window alone,
# which is the binding constraint: ``metadata`` is an empty dict on scored calls.

#: Channel groups :func:`geometry_channels` understands, with their widths.
GEOMETRY_CHANNELS: dict[str, int] = {
    #: Signed distance to the foil surface. Exact at *any* angle, including unseen
    #: ones, which is the whole point -- a five-way categorical cannot extrapolate.
    "sdf": 1,
    #: Freestream direction as ``(cos a, sin a)``, broadcast to constant planes.
    #: Separates "where the body is" from "which way the flow comes in".
    "inlet": 2,
    #: Fourier features of the angle, ``sin/cos`` at 2 frequencies over the 0-20 deg
    #: range. A linear scalar forces the network to build any nonlinearity in the angle
    #: out of its own layers; these hand it a basis that interpolates smoothly and stays
    #: bounded outside the fitted range.
    "aoa_fourier": 4,
}


def n_geometry_channels(names: tuple[str, ...]) -> int:
    """Total width of the requested channel groups."""
    unknown = set(names) - set(GEOMETRY_CHANNELS)
    if unknown:
        raise ValueError(f"unknown geometry channel(s) {sorted(unknown)}")
    return sum(GEOMETRY_CHANNELS[n] for n in names)


def parse_geom_channels(spec: str | None) -> tuple[str, ...]:
    """``"sdf,inlet"`` -> ``("sdf", "inlet")``; empty, ``None`` or trailing commas -> ``()``.

    One parser for the ``geom_channels`` config field, shared by the dataset, the model
    builder and the submission. The field is a string rather than a list because it
    arrives from a CLI flag and is round-tripped through ``config.json``; splitting it in
    three places is how the training input and the served input end up different widths.
    """
    return tuple(n for n in str(spec or "").split(",") if n)


def geometry_channels(
    aoa_deg: float,
    names: tuple[str, ...],
    h: int = 32,
    w: int = 64,
) -> np.ndarray:
    """``(h, w, K)`` float32 planes encoding the geometry at ``aoa_deg``.

    Constant in time, so a caller broadcasts them over the window's ``T`` axis. Built
    from the angle alone: at training time that is the label, at inference the estimate
    from :func:`rpde_t1.features.estimate_aoa` (1.77 deg MAE on held-out cells). Both
    paths must use the *same* function, which is why this returns planes rather than
    being inlined in the dataset.

    The ``sdf`` plane comes from :func:`airfoil_sdf_fast`, not the exact
    :func:`airfoil_sdf`. Same reason the second-moment head uses it: the exact field
    costs 52 ms per call and inference needs one per sample, which measured as **-7.7
    time_score** there -- a cost that would apply here too, since these planes are
    rebuilt for every window. The 0.5 deg table step is far inside the 1.77 deg MAE of
    the angle estimator that indexes it, so the quantisation is not the binding error.
    Training reads the table as well, so the planes a model is trained on and the planes
    it is served are identical by construction rather than by argument.
    """
    n_geometry_channels(names)
    planes: list[np.ndarray] = []
    for name in names:
        if name == "sdf":
            planes.append(airfoil_sdf_fast(aoa_deg, h, w)[..., None])
        elif name == "inlet":
            planes.append(np.broadcast_to(inlet_direction(aoa_deg), (h, w, 2)))
        elif name == "aoa_fourier":
            t = np.pi * aoa_deg / AOA_FOURIER_SCALE
            feats = np.array([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)], dtype=np.float32)
            planes.append(np.broadcast_to(feats, (h, w, 4)))
    return np.ascontiguousarray(np.concatenate(planes, axis=-1), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Precomputed SDF table
# --------------------------------------------------------------------------- #
#
# :func:`airfoil_sdf` costs **52 ms per call** -- it measures 2,048 grid points against a
# 120-vertex polygon. Called once per sample at inference that is 7.6 s over a 147-window
# eval set, which measured as a **-7.7 time_score** hit: more than the second-moment head
# it feeds is worth. The head's own forward pass is 0.06 ms/sample by comparison, so the
# geometry was 800x the cost of the network using it.
#
# The competition FAQ is explicit that "model construction and checkpoint loading run
# outside the timed region; only inference is timed", and the separate wall-clock limit is
# 5 minutes against a real submission's measured 1.9 s. So the fix is to build the table
# once at load and look it up.

#: Angle grid for the table, in degrees. The step is far below the 1.77 deg MAE of the
#: angle estimator that indexes it, so quantisation costs nothing measurable; the range
#: extends past the released 0-20 because the private test is expected at unseen angles.
SDF_TABLE_STEP = 0.5
SDF_TABLE_RANGE = (-5.0, 30.0)


def _sdf_table_angles() -> np.ndarray:
    lo, hi = SDF_TABLE_RANGE
    return np.arange(lo, hi + 0.5 * SDF_TABLE_STEP, SDF_TABLE_STEP, dtype=np.float32)


_SDF_TABLE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def sdf_table(h: int = 32, w: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """``(angles, stack)`` of precomputed signed-distance fields, built once per shape.

    Call :func:`warm_geometry_tables` from a submission's module scope so the ~2.7 s of
    construction lands outside the timed region rather than inside the first batch.
    """
    key = (h, w)
    if key not in _SDF_TABLE:
        angles = _sdf_table_angles()
        _SDF_TABLE[key] = (
            angles,
            np.stack([airfoil_sdf(float(a), h, w) for a in angles]).astype(np.float32),
        )
    return _SDF_TABLE[key]


def airfoil_sdf_fast(aoa_deg: float, h: int = 32, w: int = 64) -> np.ndarray:
    """Nearest tabulated :func:`airfoil_sdf`. Same contract, ~4 orders of magnitude cheaper.

    Nearest rather than interpolated on purpose: interpolating two signed-distance fields
    does not give the signed distance of the interpolated geometry, and at a 0.5 deg step
    the difference is well inside the field's own ~1-cell accuracy.
    """
    angles, stack = sdf_table(h, w)
    return stack[int(np.argmin(np.abs(angles - aoa_deg)))]


def warm_geometry_tables(h: int = 32, w: int = 64) -> None:
    """Build the SDF table eagerly. For a submission's load phase, which is untimed."""
    sdf_table(h, w)
