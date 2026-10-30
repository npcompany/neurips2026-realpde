"""A calibrated width rule: turn a bound head's *shape* into SPS-optimal widths.

The problem this solves. A trained interval head predicts a half-width field that ranks
the local error well and sizes it badly: FINDINGS 10 fits a single multiplier of 0.60
independently on U-Net, FNO and CNO, i.e. every head trained by that recipe comes out
~40% too wide. A global multiplier removes the average error and nothing else, and the
residual is large -- binned by the head's own predicted ``rho = half / sigma``, the best
multiplier on ``val_interp`` runs **0.30 at the narrow end to 0.90 at the wide end**. One
scalar cannot express that, so it buys about a third of what is there.

What the rule is. Per bin of *dimensionless, inference-available* features, a multiplier
pair ``(a_lo, a_up)`` applied to the head's own offsets:

    lower = pred - a_lo(bin) * (pred - lower_head)
    upper = pred + a_up(bin) * (upper_head - pred)

and each bin's pair is chosen by maximising the scorer's own per-element credit on a
fitting sample,

    mean_i  1[-a_lo h_i <= err_i <= a_up h_i] * exp(-(a_lo + a_up) h_i / sigma)

with no distributional assumption anywhere. Two properties matter more than the form:

* **The global multiplier is the one-bin special case**, so the rule cannot do worse than
  the calibration it replaces except through fitting noise.
* **Asymmetry is free.** The scorer requires only ``lower <= upper``; it does *not*
  require the interval to contain the prediction. So ``a_lo != a_up`` shifts the interval
  toward the conditional median of the target at no cost, and the reported prediction --
  which the accuracy axes score -- is left untouched.

Binning features are restricted to things a submission has: the head's own predicted
width (as ``log rho``, which is dimensionless), the channel, and the lead index. The
freestream speed is deliberately *not* a binning axis -- it already enters through the
head's conditioning, and adding it would spend bins on a quantity the private test set
can place outside our fitted range.

Numpy only, and evaluated with two ``searchsorted`` calls plus a gather, so it is safe and
free inside a submission payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import N_MEASURED, SIGMA_GLOBAL

#: Multipliers searched per bin. Geometric, because the credit is a function of
#: ``a * h / sigma`` and the interesting range spans an order of magnitude.
#:
#: The range is wide on purpose. A candidate scale field need not be in the units of an
#: error at all -- the input window's temporal spread, or an exponentiated regression of
#: ``log|err|``, can sit an order of magnitude off -- and a multiplier that saturates at
#: the edge of the grid silently reports the *grid bound* as the optimum. Measured: four
#: of seven arms in the first scale ladder pinned at 2.0, and their held-out SPS then
#: collapsed for that reason rather than for any property of the ranking. Callers should
#: also pre-normalise (see ``normalise``), which makes the optimum land mid-grid.
DEFAULT_GRID = np.geomspace(0.02, 8.0, 72)

#: Elements per chunk when evaluating a multiplier grid.
#:
#: Not a tuning knob -- a correctness guard. The obvious implementation broadcasts the
#: grid against every element at once, which on the pooled fitting set (21M scored
#: elements) allocates a ``(40, 21M)`` float array: 6.7 GB, times several boolean and
#: ``where`` temporaries. On a 15.8 GB box with no swap that reaches the *global* OOM
#: killer, which then kills an unrelated process -- measured, four times, taking the
#: terminal session with it. Peak here is ``len(grid) * CHUNK * 4`` bytes ~ 40 MB.
CHUNK = 250_000

#: Elements per chunk on the asymmetric path, which holds *two* float64
#: ``(chunk, len(grid))`` factors at once rather than one float32 one. Chosen so its peak
#: matches ``CHUNK``'s ~40 MB rather than inheriting a 4x larger one.
CHUNK_ASYM = 32_768

#: Cap on the elements a single fit consumes, and on the elements per bin. The credit is
#: an average over elements, so its standard error at 10^5 samples is already far below
#: the spacing of the multiplier grid; the millions beyond that buy nothing and cost
#: memory. Subsampling is seeded, so a refit reproduces.
MAX_FIT_ELEMENTS = 3_000_000
MAX_BIN_ELEMENTS = 200_000
MAX_BIN_ELEMENTS_ASYM = 80_000
FIT_SEED = 0


@dataclass
class WidthRule:
    """Per-bin multipliers on a head's half-widths, with the bin edges they assume.

    ``rho_edges`` are interior edges on ``log(half / sigma)``; ``lead_edges`` interior
    edges on the lead index. ``a_lo`` / ``a_up`` have shape
    ``(n_rho, N_MEASURED, n_lead)``.
    """

    rho_edges: np.ndarray
    lead_edges: np.ndarray
    a_lo: np.ndarray
    a_up: np.ndarray
    n_fit: int = 0
    note: str = ""

    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.a_lo.shape)  # type: ignore[return-value]

    @property
    def is_uniform(self) -> bool:
        """Whether ``rho_edges`` is an evenly spaced grid, so a bin index needs no search."""
        if len(self.rho_edges) < 2:
            return True
        d = np.diff(self.rho_edges)
        return bool(np.allclose(d, d[0], rtol=1e-6, atol=1e-12))

    def to_uniform(self, n_bins: int = 512, pad: float = 0.5) -> WidthRule:
        """Resample onto an evenly spaced ``log rho`` grid, for a search-free deployment.

        Quantile edges are the right thing to *fit* on -- they put equal data in every bin --
        and the wrong thing to *evaluate*, because a bin index then costs a binary search per
        element. Measured on a batch of 32 windows (2.6M elements): ``searchsorted`` against 47
        quantile edges is 1.72 ms/window and the broadcast gather another 1.30, i.e. ~9.4
        ms/window for the whole transform against a board budget of ~21 ms/window. That is the
        L11 failure exactly -- a cheap-looking numpy step costing more than the model it serves.

        On an evenly spaced grid the index is ``(log rho - lo) / step`` truncated, which is
        arithmetic. The step function is piecewise constant, so resampling it is *exact*
        wherever a uniform bin lies inside one quantile bin, and the only error is at the
        boundaries -- bounded by making ``n_bins`` comfortably larger than ``n_rho``. ``pad``
        extends the grid half a decade past the fitted range so an unseen operating point
        clamps to the end bins rather than to an extrapolated multiplier.
        """
        if len(self.rho_edges) == 0:
            return self
        lo = float(self.rho_edges[0]) - pad
        hi = float(self.rho_edges[-1]) + pad
        edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
        # Sample the existing step function at each new bin's centre.
        centres = 0.5 * (np.concatenate([[lo], edges]) + np.concatenate([edges, [hi]]))
        src = np.searchsorted(self.rho_edges, centres)
        return WidthRule(
            rho_edges=edges,
            lead_edges=self.lead_edges,
            a_lo=self.a_lo[src].astype(np.float32),
            a_up=self.a_up[src].astype(np.float32),
            n_fit=self.n_fit,
            note=f"{self.note} uniform({n_bins})",
        )

    def rho_index(self, half: np.ndarray) -> np.ndarray:
        """The ``log rho`` bin subscript of a half-width field of **any** shape.

        Separate from :meth:`bin_index` only because the channel and lead subscripts need the
        ``(N, T, H, W, C)`` layout and this does not: a caller holding the scored elements flat
        (see ``score.sps_lab.ruled_sps``) needs this half and must not reimplement it, since a
        bin index that is wrong by one silently applies a neighbouring multiplier.

        ``int32`` from both branches. Bin subscripts are bounded by ``len(rho_edges)``, so the
        width is never load-bearing, and ``searchsorted``'s int64 doubles a per-element index
        array that on a 40M-element sweep is pure memory traffic.
        """
        log_rho = np.log(np.maximum(half, 1e-30) / np.float32(SIGMA_GLOBAL))
        if self.is_uniform and len(self.rho_edges) >= 2:
            step = float(self.rho_edges[1] - self.rho_edges[0])
            return np.clip(
                ((log_rho - np.float32(self.rho_edges[0])) / np.float32(step) + 1.0).astype(
                    np.int32
                ),
                0,
                len(self.rho_edges),
            )
        return np.searchsorted(self.rho_edges, log_rho).astype(np.int32)

    def bin_index(self, half: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin subscripts for a ``(N, T, H, W, N_MEASURED)`` half-width field."""
        t = half.shape[1]
        i_rho = self.rho_index(half)
        lead = np.arange(t).reshape(1, t, 1, 1, 1)
        i_lead = np.broadcast_to(np.searchsorted(self.lead_edges, lead), half.shape)
        i_ch = np.broadcast_to(np.arange(half.shape[-1]).reshape(1, 1, 1, 1, -1), half.shape)
        return i_rho, i_ch, i_lead

    def _gather(
        self, table: np.ndarray, i_rho: np.ndarray, n_ch: int, ch: np.ndarray | None = None
    ) -> np.ndarray | None:
        """``table[i_rho, ch, i_lead]`` without materialising the broadcast index arrays.

        Fancy-indexing with three broadcast index arrays costs 1.30 ms/window; flattening the
        table and taking a single index built by arithmetic costs 0.48, and when the lead axis
        is degenerate (the selected configuration) the index is just ``i_rho * n_ch + ch``.

        ``ch`` may be passed as an explicit per-element channel subscript, for a caller whose
        elements are flat rather than laid out with the channel on the last axis. Left to the
        default it is that axis, which is the ``predict()`` path.

        Returns ``None`` when the lead axis is *not* degenerate, so the caller falls back to the
        general path rather than this one silently indexing the wrong bin.
        """
        if table.shape[2] != 1:
            return None
        flat = np.ascontiguousarray(table[:, :, 0]).reshape(-1)
        if ch is None:
            ch = np.arange(n_ch, dtype=np.int32).reshape(*([1] * (i_rho.ndim - 1)), n_ch)
        return flat.take(i_rho * np.int32(n_ch) + ch)

    def apply(
        self, pred: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recalibrated ``(lower, upper)``; unmeasured channels are left pinned to ``pred``.

        Written against the *contiguity* of the arrays rather than for brevity, because this
        runs inside ``predict()`` on every element of every window and ``time_score`` is a
        budget. The measured fields are ``[..., :2]`` of a 3-channel array, so every operation
        on that slice is strided: ``np.minimum`` on two channels costs 1.23 ms/window against
        0.21 for all three. Taking one contiguous copy up front and writing back once turns
        eleven strided passes into three, which with the uniform-grid index (see
        :meth:`to_uniform`) brings the transform from 9.4 to 3.5 ms/window.
        """
        c = N_MEASURED
        p = np.ascontiguousarray(pred[..., :c])
        below = p - lower[..., :c]
        above = upper[..., :c] - p
        half = 0.5 * (below + above)

        i_rho, i_ch, i_lead = self.bin_index(half)
        m_lo = self._gather(self.a_lo, i_rho, c)
        m_up = self._gather(self.a_up, i_rho, c)
        if m_lo is None or m_up is None:
            m_lo, m_up = self.a_lo[(i_rho, i_ch, i_lead)], self.a_up[(i_rho, i_ch, i_lead)]

        new_lo = p - below * m_lo
        new_hi = p + above * m_up
        # The scorer zeroes every subscore on a reversed interval, so order them -- reusing
        # ``half`` as the output buffer, which is dead by this point.
        np.minimum(new_lo, new_hi, out=half)
        np.maximum(new_lo, new_hi, out=new_hi)

        lo, hi = pred.copy(), pred.copy()
        lo[..., :c] = half
        hi[..., :c] = new_hi
        return lo, hi

    def apply_torch(self, pred, lower, upper):
        """:meth:`apply` on whatever device the tensors are already on.

        Same transform, moved to where the data is. The bounds leave the backbone on the
        GPU, and running the rule on the host meant a copy down, a serial numpy pass over
        2.6M elements per batch, and nothing else to overlap it with: **measured at +5.62
        ms/window against a ~46 ms budget, which was 100% of this artefact's latency
        regression** (the wider head costs −0.08, i.e. nothing). Every step here is an
        elementwise op or a gather, so on-device it is a rounding error.

        Returns ``None`` -- rather than a subtly different answer -- when the rule is not on
        a uniform grid, because the bin index is then a binary search this path does not
        implement. Declining is the same contract :meth:`_gather` follows for a
        non-degenerate lead axis, and for the same reason: a wrong multiplier is a wrong
        interval that no subscore would flag. ``make_predict`` uniformises at construction,
        so the shipped path always qualifies.

        A second implementation of a transform is exactly the duplication PROTOCOL 5
        records going wrong, so ``tests/test_sps_lab.py`` asserts the two agree elementwise
        on random fields rather than trusting that they were written to match.
        """
        import torch  # noqa: PLC0415 - keeps the numpy fitting path torch-free

        if not (self.is_uniform and len(self.rho_edges) >= 2):
            return None

        c = N_MEASURED
        dev, dt = pred.device, pred.dtype
        p = pred[..., :c]
        below = p - lower[..., :c]
        above = upper[..., :c] - p
        half = 0.5 * (below + above)

        # The same arithmetic index as ``bin_index``: on an evenly spaced grid the bin is a
        # division and a truncation, and ``.to(int32)`` truncates toward zero as ``astype``
        # does. Kept in float32 throughout so the two paths round identically.
        step = float(self.rho_edges[1] - self.rho_edges[0])
        log_rho = torch.log(half.clamp_min(1e-30) / float(SIGMA_GLOBAL))
        i_rho = ((log_rho - float(self.rho_edges[0])) / step + 1.0).to(torch.int32)
        i_rho = i_rho.clamp_(0, len(self.rho_edges)).long()

        n_ch, n_lead = self.a_lo.shape[1], self.a_lo.shape[2]
        t = pred.shape[1]
        lead = torch.arange(t, device=dev)
        edges = torch.as_tensor(np.asarray(self.lead_edges), device=dev, dtype=lead.dtype)
        i_lead = torch.bucketize(lead, edges).view(1, t, 1, 1, 1)
        i_ch = (torch.arange(n_ch, device=dev) * n_lead).view(1, 1, 1, 1, n_ch)
        flat = i_rho * (n_ch * n_lead) + i_ch + i_lead

        def gather(table: np.ndarray) -> torch.Tensor:
            f = torch.as_tensor(table.reshape(-1), device=dev, dtype=dt)
            return f[flat]

        new_lo = p - below * gather(self.a_lo)
        new_hi = p + above * gather(self.a_up)

        lo, hi = pred.clone(), pred.clone()
        lo[..., :c] = torch.minimum(new_lo, new_hi)
        hi[..., :c] = torch.maximum(new_lo, new_hi)
        return lo, hi

    # ------------------------------------------------------------------ #

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "rho_edges": self.rho_edges.tolist(),
                    "lead_edges": self.lead_edges.tolist(),
                    "a_lo": self.a_lo.tolist(),
                    "a_up": self.a_up.tolist(),
                    "n_fit": self.n_fit,
                    "note": self.note,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_json(cls, path: Path) -> WidthRule:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            rho_edges=np.asarray(d["rho_edges"], dtype=np.float64),
            lead_edges=np.asarray(d["lead_edges"], dtype=np.int64),
            a_lo=np.asarray(d["a_lo"], dtype=np.float32),
            a_up=np.asarray(d["a_up"], dtype=np.float32),
            n_fit=int(d.get("n_fit", 0)),
            note=str(d.get("note", "")),
        )

    def absorb_prescale(self, ref: float) -> WidthRule:
        """Fold a fitting-time prescale into the rule, so it applies to the raw scale.

        Fitting pre-divides the candidate scale by ``ref`` to land the optimal multiplier
        mid-grid. Rather than making every caller remember that constant -- the kind of
        split definition PROTOCOL 5 records going wrong four times -- the rule absorbs it:
        the multipliers take the factor, and the ``log rho`` edges shift by ``log ref`` so
        they still select the same elements.
        """
        if ref == 1.0:
            return self
        return WidthRule(
            rho_edges=self.rho_edges + np.log(ref),
            lead_edges=self.lead_edges,
            a_lo=(self.a_lo / ref).astype(np.float32),
            a_up=(self.a_up / ref).astype(np.float32),
            n_fit=self.n_fit,
            note=f"{self.note} prescale={ref:.6g}",
        )

    @classmethod
    def constant(cls, k: float) -> WidthRule:
        """The single-multiplier calibration, as a one-bin rule. The control arm."""
        return cls(
            rho_edges=np.zeros(0),
            lead_edges=np.zeros(0, dtype=np.int64),
            a_lo=np.full((1, N_MEASURED, 1), k, dtype=np.float32),
            a_up=np.full((1, N_MEASURED, 1), k, dtype=np.float32),
            note=f"constant k={k}",
        )


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _subsample(n: int, cap: int) -> np.ndarray | slice:
    """Indices of at most ``cap`` of ``n`` elements, seeded; a slice when no cut is needed."""
    if n <= cap:
        return slice(None)
    return np.random.default_rng(FIT_SEED).choice(n, size=cap, replace=False)


def _credit_symmetric(
    err: np.ndarray,
    half: np.ndarray,
    grid: np.ndarray,
    h_up: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """``mean_i 1[|err_i| <= a h_i] exp(-2 a h_i / sigma)`` for every ``a`` in ``grid``.

    Chunked over elements (see :data:`CHUNK`); the result is identical to the broadcast
    form up to float32 rounding.

    ``h_up`` makes the input band two-sided: ``half`` is then the offset *below* the
    prediction and ``h_up`` the offset above, and one common multiplier ``a`` scales both, so
    the credit is ``1[-a h_lo <= err <= a h_up] exp(-a (h_lo + h_up) / sigma)``. With
    ``h_up=None`` this is exactly the one-sided form above. The lower offset may be negative
    (a band that sits entirely above the prediction); the algebra is unchanged.

    ``weight`` (optional, one per element) makes it a weighted mean -- the scorer weights every
    element's credit by its window's accuracy factor ``q`` (``sps_lab.quality_weights``).
    """
    g = np.asarray(grid, dtype=np.float32).reshape(-1, 1)
    total = np.zeros(g.size, dtype=np.float64)
    n = err.size
    sigma = np.float32(SIGMA_GLOBAL)
    for s in range(0, n, CHUNK):
        e = err[s : s + CHUNK].astype(np.float32).reshape(1, -1)
        w_lo = g * half[s : s + CHUNK].astype(np.float32).reshape(1, -1)
        w_up = w_lo if h_up is None else g * h_up[s : s + CHUNK].astype(np.float32).reshape(1, -1)
        inside = (e >= -w_lo) & (e <= w_up)
        credit = np.where(inside, np.exp(-(w_lo + w_up) / sigma), 0.0)
        if weight is not None:
            credit = credit * weight[s : s + CHUNK].astype(np.float32).reshape(1, -1)
        total += credit.sum(axis=1, dtype=np.float64)
    if weight is not None:
        return total / max(1e-12, float(np.sum(weight, dtype=np.float64)))
    return total / max(1, n)


def _credit_matrix(
    err: np.ndarray,
    half: np.ndarray,
    grid: np.ndarray,
    h_up: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """``credit[i, j]`` for ``(a_lo, a_up) = (grid[i], grid[j])``, the full outer product.

    ``h_up`` (optional) is the head's own offset *above* the prediction, with ``half`` then
    the offset below; each side's multiplier scales its own offset. Default: one shared
    offset, the one-sided form every existing caller uses.

    The objective **factorises over the two multipliers**, because both the indicator and
    the width penalty do::

        1[-a_lo h <= e <= a_up h] exp(-(a_lo + a_up) h / sigma)
          = (1[e >= -a_lo h] exp(-a_lo h / sigma)) * (1[e <= a_up h] exp(-a_up h / sigma))

    so the whole matrix is ``A.T @ B / n`` for two ``(n, len(grid))`` factors. The
    exponential is then evaluated once per ``(element, multiplier)`` instead of once per
    ``(element, multiplier, multiplier)``, and the quadratic part is a GEMM rather than a
    Python loop over ``a_lo`` -- measured **54x** faster at ``MAX_BIN_ELEMENTS_ASYM``
    elements, agreeing with the loop to 6e-9 (float32 rounding) with an identical argmax.

    That factor is why the asymmetric arm is affordable at all: it was 32 of the 36 minutes
    a full ``sps_select.py`` sweep spent, and the sweep fits one of these per rho bin per
    configuration.

    Accumulated in float64, matching the ``sum(dtype=np.float64)`` the loop form used;
    chunked over elements (see :data:`CHUNK_ASYM`) so peak memory does not grow with ``n``.
    """
    g = np.asarray(grid, dtype=np.float32).reshape(1, -1)
    sigma = np.float32(SIGMA_GLOBAL)
    n = err.size
    out = np.zeros((g.size, g.size), dtype=np.float64)
    for s in range(0, n, CHUNK_ASYM):
        e = err[s : s + CHUNK_ASYM].astype(np.float32).reshape(-1, 1)
        h = half[s : s + CHUNK_ASYM].astype(np.float32).reshape(-1, 1)
        w = h * g
        lo = np.where(e >= -w, np.exp(-w / sigma), np.float32(0.0)).astype(np.float64)
        if h_up is not None:
            w = h_up[s : s + CHUNK_ASYM].astype(np.float32).reshape(-1, 1) * g
        up = np.where(e <= w, np.exp(-w / sigma), np.float32(0.0)).astype(np.float64)
        if weight is not None:
            up = up * weight[s : s + CHUNK_ASYM].astype(np.float64).reshape(-1, 1)
        out += lo.T @ up
    if weight is not None:
        return out / max(1e-12, float(np.sum(weight, dtype=np.float64)))
    return out / max(1, n)


def _credit_asymmetric(
    err: np.ndarray,
    half: np.ndarray,
    grid: np.ndarray,
    h_up: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Best ``(a_lo, a_up, credit)`` over the outer product of ``grid`` with itself.

    ``argmax`` on the C-ordered matrix scans ``a_lo`` outer and ``a_up`` inner, so ties
    break exactly as the ``a_lo``-outer loop this replaced did.
    """
    g = np.asarray(grid, dtype=np.float32)
    m = _credit_matrix(err, half, g, h_up, weight)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(g[i]), float(g[j]), float(m[i, j])


def fit_width_rule(
    err: np.ndarray,
    half: np.ndarray,
    scored: np.ndarray,
    n_rho: int = 24,
    n_lead: int = 4,
    grid: np.ndarray = DEFAULT_GRID,
    asymmetric: bool = False,
    min_count: int = 2000,
    note: str = "",
    above: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> WidthRule:
    """Fit per-bin multipliers by direct maximisation of the scorer's credit.

    ``err`` and ``half`` are ``(N, T, H, W, N_MEASURED)`` in **physical** units -- the
    same units as ``SIGMA_GLOBAL``, which is the one conversion that has been got wrong
    repeatedly elsewhere in this codebase. ``scored`` is the scorer's own non-zero-target
    mask, since fitting on masked elements would calibrate against targets the metric
    ignores.

    Bins with fewer than ``min_count`` scored elements fall back to the pooled optimum,
    so a sparsely populated tail cannot contribute a wild multiplier.

    ``above`` makes the head's band two-sided: ``half`` is then the offset below the
    prediction (``pred - lower``) and ``above`` the offset above (``upper - pred``), which is
    exactly how :meth:`WidthRule.apply` reads a band, so a rule fitted this way applies to the
    same band without any symmetrisation. Bins are on the mean half-width, as ``apply`` bins.
    Omitted, the band is symmetric and everything reduces to the one-sided fit.

    ``weight`` (optional, same shape as ``err``) weights each element's credit; passing the
    window's accuracy factor ``q`` fits the rule to the scorer's own objective,
    ``mean(credit * q)``, instead of the unweighted credit. Omitted, every element counts once.
    """
    return fit_prepared(
        prepare_fit_sample(err, half, scored, n_lead=n_lead, above=above, weight=weight),
        n_rho=n_rho,
        grid=grid,
        asymmetric=asymmetric,
        min_count=min_count,
        note=note,
    )


def fit_prescaled(
    err: np.ndarray,
    half: np.ndarray,
    scored: np.ndarray,
    n_rho: int = 24,
    n_lead: int = 1,
    asymmetric: bool = False,
    above: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> WidthRule:
    """:func:`fit_width_rule` with the median prescale applied and then absorbed.

    The prescale is not optional and not a tuning knob: a candidate scale field need not be in
    the units of an error at all, and one whose level sits far from the error scale pins its
    multiplier at the edge of :data:`DEFAULT_GRID` and reports the *grid bound* as the optimum --
    measured on a width-64 head, which read 43.62 instead of a real number for exactly that
    reason. Mapping the scale onto the fitting set's median ``|err|`` first lands the optimum
    mid-grid, and :meth:`WidthRule.absorb_prescale` folds the constant back in so callers never
    have to remember it.

    Exists because this five-line procedure had been written out **four times** -- in
    ``train.scale_fit.size_and_score``, ``scripts/sps/sps_anchor.py``, ``scripts/sps/sps_apply.py``
    and
    ``scripts/sps/sps_headroom.py`` -- and PROTOCOL 5's first entry is two definitions of one thing
    disagreeing. Three of the four omitted the zero guard.
    """
    mean_half = half if above is None else 0.5 * (half + above)
    med_scale = float(np.median(mean_half[scored]))
    med_err = float(np.median(np.abs(err[scored])))
    ref = med_scale / med_err if med_scale > 0 and med_err > 0 else 1.0
    return fit_width_rule(
        err,
        half / ref,
        scored,
        n_rho=n_rho,
        n_lead=n_lead,
        asymmetric=asymmetric,
        above=None if above is None else above / ref,
        weight=weight,
    ).absorb_prescale(ref)


@dataclass
class FitSample:
    """The elements a width-rule fit consumes: scored, flattened, log-binned, capped.

    Split out of :func:`fit_width_rule` because it depends only on the *candidate scale* and
    not on ``n_rho`` or ``asymmetric`` -- while a configuration sweep varies only those two.
    The masking gathers, the ``log`` and the seeded cap cost ~9 s on the pooled ladder, and
    ``sps_select.py`` was paying them once per configuration to produce an identical sample;
    it now pays them once per arm. Immutable by convention: :func:`fit_prepared` only reads.
    """

    e: np.ndarray
    h: np.ndarray
    lr: np.ndarray
    ch: np.ndarray
    lead: np.ndarray | None
    t: int
    n_lead: int
    #: The offset above the prediction when the band is two-sided; ``h`` is then the offset
    #: below. ``None`` for the symmetric case, where ``h`` serves both sides.
    h_up: np.ndarray | None = None
    #: Per-element credit weight (the window's accuracy factor), or ``None`` for unweighted.
    w: np.ndarray | None = None


def prepare_fit_sample(
    err: np.ndarray,
    half: np.ndarray,
    scored: np.ndarray,
    n_lead: int = 4,
    above: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> FitSample:
    """Mask, flatten and cap a fitting set once, for reuse across rule configurations.

    Same arrays and units as :func:`fit_width_rule`, whose docstring documents them,
    including ``above``.
    """
    t = half.shape[1]
    half = np.asarray(half, dtype=np.float32)
    e = np.asarray(err, dtype=np.float32)[scored]
    h = half[scored]
    h_up = None if above is None else np.asarray(above, dtype=np.float32)[scored]
    w = None if weight is None else np.broadcast_to(weight, half.shape).astype(np.float32)[scored]
    # The bin variable is the mean half-width -- what ``WidthRule.apply`` bins on.
    mean_h = h if h_up is None else 0.5 * (h + h_up)
    lr = np.log(np.maximum(mean_h, np.float32(1e-30)) / np.float32(SIGMA_GLOBAL))
    # Only materialise the lead subscript if it can select anything. A single lead bin --
    # what every caller in the selection sweep asks for -- otherwise costs a broadcast and a
    # boolean gather over the whole field (55M elements on the pooled fitting set) to
    # produce a column of zeros.
    lead = (
        np.broadcast_to(np.arange(t, dtype=np.int16).reshape(1, t, 1, 1, 1), half.shape)[scored]
        if n_lead > 1
        else None
    )
    ch = np.broadcast_to(
        np.arange(half.shape[-1], dtype=np.int16).reshape(1, 1, 1, 1, -1), half.shape
    )[scored]

    # Cap the fit before any grid evaluation: the credit is a mean over elements, so a few
    # million samples already resolve it far finer than the multiplier grid.
    keep = _subsample(e.size, MAX_FIT_ELEMENTS)
    e, h, lr, ch = e[keep], h[keep], lr[keep], ch[keep]
    lead = None if lead is None else lead[keep]
    h_up = None if h_up is None else h_up[keep]
    w = None if w is None else w[keep]
    return FitSample(e=e, h=h, lr=lr, ch=ch, lead=lead, t=t, n_lead=n_lead, h_up=h_up, w=w)


def fit_prepared(
    sample: FitSample,
    n_rho: int = 24,
    grid: np.ndarray = DEFAULT_GRID,
    asymmetric: bool = False,
    min_count: int = 2000,
    note: str = "",
) -> WidthRule:
    """:func:`fit_width_rule` on an already-prepared sample. See it for the contract."""
    e, h, lr, ch, lead = sample.e, sample.h, sample.lr, sample.ch, sample.lead
    t, n_lead = sample.t, sample.n_lead
    h_up = sample.h_up
    w = sample.w

    def side(x: np.ndarray | None, idx) -> np.ndarray | None:
        return None if x is None else x[idx]

    rho_edges = np.quantile(lr, np.linspace(0, 1, n_rho + 1)[1:-1]) if n_rho > 1 else np.zeros(0)
    lead_edges = (
        np.linspace(0, t, n_lead + 1)[1:-1].astype(np.int64)
        if n_lead > 1
        else np.zeros(0, np.int64)
    )
    i_rho = np.searchsorted(rho_edges, lr)
    i_lead = 0 if lead is None else np.searchsorted(lead_edges, lead)

    # Pooled fallback, also the control this rule has to beat. Subsampled to the same cap a
    # single *bin*'s fit gets: it is the same one-multiplier estimate, so there is no reason
    # for it to consume the whole fitting set -- which cost 3.9 s per fit against 0.09 s for
    # a capped one, and by MAX_BIN_ELEMENTS' own rationale buys no resolution.
    pool_cap = MAX_BIN_ELEMENTS_ASYM if asymmetric else MAX_BIN_ELEMENTS
    pool_sub = _subsample(e.size, pool_cap)
    if asymmetric:
        p_lo, p_up, _ = _credit_asymmetric(
            e[pool_sub], h[pool_sub], grid, side(h_up, pool_sub), side(w, pool_sub)
        )
    else:
        pooled = _credit_symmetric(
            e[pool_sub], h[pool_sub], grid, side(h_up, pool_sub), side(w, pool_sub)
        )
        p_lo = p_up = float(grid[int(np.argmax(pooled))])

    a_lo = np.full((n_rho, N_MEASURED, n_lead), p_lo, dtype=np.float32)
    a_up = np.full((n_rho, N_MEASURED, n_lead), p_up, dtype=np.float32)

    flat = (i_rho * N_MEASURED + ch) * n_lead + i_lead
    order = np.argsort(flat, kind="stable")
    fs, es, hs = flat[order], e[order], h[order]
    hs_up = side(h_up, order)
    ws = side(w, order)
    starts = np.searchsorted(fs, np.arange(n_rho * N_MEASURED * n_lead), side="left")
    ends = np.searchsorted(fs, np.arange(n_rho * N_MEASURED * n_lead), side="right")

    cap = MAX_BIN_ELEMENTS_ASYM if asymmetric else MAX_BIN_ELEMENTS
    for b, (s, x) in enumerate(zip(starts, ends, strict=True)):
        if x - s < min_count:
            continue
        sub = _subsample(x - s, cap)
        eb, hb = es[s:x][sub], hs[s:x][sub]
        hb_up = None if hs_up is None else hs_up[s:x][sub]
        wb = None if ws is None else ws[s:x][sub]
        i, j, k = np.unravel_index(b, (n_rho, N_MEASURED, n_lead))
        if asymmetric:
            lo, up, _ = _credit_asymmetric(eb, hb, grid, hb_up, wb)
        else:
            lo = up = float(grid[int(np.argmax(_credit_symmetric(eb, hb, grid, hb_up, wb)))])
        a_lo[i, j, k], a_up[i, j, k] = lo, up

    return WidthRule(
        rho_edges=rho_edges,
        lead_edges=lead_edges,
        a_lo=a_lo,
        a_up=a_up,
        n_fit=int(e.size),
        note=note or f"n_rho={n_rho} n_lead={n_lead} asym={asymmetric}",
    )


__all__ = [
    "DEFAULT_GRID",
    "FitSample",
    "WidthRule",
    "fit_prepared",
    "fit_prescaled",
    "fit_width_rule",
    "prepare_fit_sample",
]
