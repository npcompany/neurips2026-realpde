"""Fit the second-moment head on cached TKE maps.

The whole point of the design in ``models/tke_head.py`` is that the objective touches
the backbone only through ``ke(pred)``, one ``(H, W)`` map per window. So this module
runs the backbone **once** per split, caches three maps per window, and then trains a
small CNN on images. A backbone-in-the-loop version of the same head would cost a
GPU-hour per arm; this costs minutes, which is what makes it affordable to A/B properly.

Discipline carried over from the interval head: the head is fitted on ``train`` windows
and reported on splits it never saw, and its honest baseline is not "no head" but the
**closed-form geometric blend**, which the head strictly generalises. A head that does
not beat the blend has learned nothing worth its inference cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import N_MEASURED, T_IN
from ..data.dataset import make_dataset
from ..features import invalid_mask
from ..geometry import airfoil_sdf
from ..models.predictor import kinetic_energy_map, kinetic_energy_map_components
from ..models.tke_head import (
    FEATURE_NAMES,
    N_BASE_FEATURES,
    TkeHead,
    apply_gain,
    blend_log_ratio,
    tke_features,
    tke_rel_l2,
)
from .seeds import seed_everything

#: Blend exponents the closed-form baseline is scanned over when reporting.
BASELINE_BLENDS = (0.0, 0.3, 0.5, 0.6, 0.75)

#: Exponent used for the blend the head's output is shrunk toward. 0.5 rather than the
#: 0.6 that won on ``val_time``: the two are within noise of each other on every split
#: and 0.5 was the better of the two on ``val_extrap``, which is the regime the private
#: test is expected to resemble.
ANCHOR_BLEND = 0.5


def collect_maps(
    predict,
    split: str,
    kind: str = "real",
    stride: int = 40,
    max_windows: int | None = None,
    cond_mode: str = "operating_point",
    stream_state: str = "",
    dt_aug: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    """Run the backbone once over a split and cache what the head needs.

    Returns ``ke_pred``, ``ke_input``, ``ke_target`` ``(N, H, W)``, plus ``valid``,
    ``sdf`` and ``cond``. ``stride`` defaults to the leak-free 40 -- overlapping windows
    would let one window's target sit inside another's input, and the maps inherit that
    defect from the windows they are computed on.

    ``dt_aug``: whole-step time-step augmentation of the collected windows (FINDINGS 141's
    reopen "the slot on the second-moment head"). ``(2,)`` collects every window at step 2
    where it fits (falling back to 1, with the slot self-consistently 0); pair with
    ``cond_mode="operating_point_dt"`` so the step lands in the cond column the head's
    horizon plane reads. Scored serving is unchanged: ``infer_conditioning`` writes the
    reference value 0 there.
    """
    # ``normalizer="none"`` because ``predict`` takes physical fields and does its own
    # non-dimensionalisation -- the same path a submission runs.
    ds = make_dataset(
        split,
        kind=kind,
        stride=stride,
        normalizer="none",
        cond_mode=cond_mode,
        dt_aug=dt_aug,
        # A stream-state backbone (TrainConfig.stream_state) reads two causal pseudo-frames;
        # the dataset synthesises them on the physical path exactly as the Track-2 port does.
        stream_state=stream_state,
    )
    n = len(ds) if max_windows is None else min(len(ds), max_windows)
    take = np.linspace(0, len(ds) - 1, n).astype(int) if n < len(ds) else np.arange(len(ds))

    inputs, targets, conds = [], [], []
    for i in take:
        item = ds[int(i)]
        inputs.append(item["input"].numpy())
        targets.append(item["target"].numpy())
        conds.append(item["cond"].numpy())
    x = np.stack(inputs).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    cond = np.stack(conds).astype(np.float32)

    pred = predict(x)
    if isinstance(pred, dict):
        pred = pred["prediction"]
    pred = np.asarray(pred, dtype=np.float32)

    # The model reads the full input (state pseudo-frames included); every map derived from
    # the OBSERVATION below reads the measured window only, which for a plain checkpoint is
    # ``x`` itself.
    xr = x[:, -T_IN:]

    sdf_by_aoa: dict[float, np.ndarray] = {}
    sdf = np.zeros(xr.shape[:1] + xr.shape[2:4], dtype=np.float32)
    valid = np.zeros_like(sdf)
    for i in range(xr.shape[0]):
        aoa = float(cond[i, 1]) * 20.0
        if aoa not in sdf_by_aoa:
            sdf_by_aoa[aoa] = airfoil_sdf(aoa, *xr.shape[2:4]).astype(np.float32)
        sdf[i] = sdf_by_aoa[aoa]
        valid[i] = (~invalid_mask(xr[i])).astype(np.float32)

    ke_pred_c, ke_input_c = kinetic_energy_map_components(pred), kinetic_energy_map_components(xr)
    return {
        "ke_pred": ke_pred_c.sum(-1),
        "ke_input": ke_input_c.sum(-1),
        "ke_target": kinetic_energy_map(y),
        # Per-component maps, read only by a per-component head (``--per-component``). The
        # scorer sees their sum, which is exactly the freedom that head exploits.
        "ke_pred_c": ke_pred_c,
        "ke_input_c": ke_input_c,
        "valid": valid,
        "sdf": sdf,
        "cond": cond,
        "pred": pred,
        "target": y,
    }


def _features(
    maps: dict[str, np.ndarray],
    drop: tuple[str, ...] = (),
    per_component: bool = False,
    n_extra: int = 0,
) -> np.ndarray:
    """The head's input planes. A single-gain head gets the shipped seven; a per-component
    head gets all of them, the four trailing ones filled from the per-component maps.

    ``n_extra`` appends that many planes from ``maps["extra"]``, an ``(N, K, H, W)`` array a
    campaign supplies (2026-09-17: the interval head's predicted error size, for the damping
    arm). Experimental input: the serving path does not build these planes, so a head fitted
    with them is an analysis object until it is wired through ``inference_log_ratio``.
    """
    feats = tke_features(
        maps["ke_pred"],
        maps["ke_input"],
        maps["valid"],
        maps["sdf"],
        maps["cond"],
        drop=drop,
        ke_pred_c=maps.get("ke_pred_c") if per_component else None,
        ke_input_c=maps.get("ke_input_c") if per_component else None,
    )
    feats = feats if per_component else feats[:, :N_BASE_FEATURES]
    if n_extra:
        feats = np.concatenate([feats, maps["extra"][:, :n_extra].astype(np.float32)], axis=1)
    return feats


def base_in_channels(per_component: bool) -> int:
    """Input width of a head without extra planes."""
    return len(FEATURE_NAMES) if per_component else N_BASE_FEATURES


def _tensors(
    maps: dict[str, np.ndarray],
    drop: tuple[str, ...],
    device: str,
    per_component: bool = False,
    n_extra: int = 0,
):
    """Features, the energy map the gain acts on (per component for a per-component head, so
    that ``exp(r) * ke_pred`` is the rescaled energy in both cases), the pooled target map
    and the mask."""
    ke_pred = maps["ke_pred_c"] if per_component else maps["ke_pred"]
    return (
        torch.from_numpy(_features(maps, drop, per_component, n_extra)).to(device),
        torch.from_numpy(ke_pred).to(device),
        torch.from_numpy(maps["ke_target"]).to(device),
        torch.from_numpy(maps["valid"]).to(device),
    )


def ceiling_from(sub: dict[str, float]) -> float:
    """The SPS accuracy ceiling implied by a row of subscores.

    ``100 * sum_b w_b * 0.5/(0.5+err_b)`` over rel_l2/tke/mvpe at weights .5/.3/.2. This
    is the quantity a fluctuation rescaling actually reaches the final score through --
    ``final ~ 0.272*SPS`` and ``SPS = ceiling * efficiency`` -- so it, not ``tke`` alone,
    is what an arm should be selected on.

    Selecting on tke alone is what let the head trade 0.27 rel_l2 for 2.67 tke and call it
    a win: rel_l2 carries weight **0.5** here against tke's **0.3**, so that trade is worth
    about a fifth of what the tke number suggests.
    """
    from ..score.report import invert_score_error

    w = {"rel_l2_score": 0.5, "tke_score": 0.3, "mvpe_score": 0.2}
    return 100.0 * sum(wt * 0.5 / (0.5 + invert_score_error(sub[k])) for k, wt in w.items())


def score_maps(maps: dict[str, np.ndarray], log_ratio: np.ndarray) -> dict[str, float]:
    """Score a candidate log-ratio through the vendored scorer, not a reimplementation."""
    from ..score.diagnostics import _subscores, load_official_scoring

    sc = load_official_scoring()
    target = maps["target"]
    c = sc.measured_channels(target)
    q = apply_gain(maps["pred"], log_ratio)
    sub = _subscores(sc, q, target, c)
    sub["ceiling"] = ceiling_from(sub)
    return sub


def rescaled_rel_l2_terms(maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps that make the *rescaled* rel-L2 a closed form in the gain.

    The head's cheapness comes from touching only ``(H, W)`` maps, so a rel_l2 term looks
    like it would force the fields back into the loop. It does not. With
    ``q = m + (pred - m) * g`` and ``f = pred - m``:

        |q - t|^2 = |m - t|^2 + 2 g <m - t, f> + g^2 |f|^2

    and each term is a map. Summing over time and the two measured channels:

        SSE(g) / T = dmean2 + 2*ke_target - 2*g*cov + 2*g^2*ke_pred

    where ``cov`` is the time-covariance of the prediction's and target's fluctuations and
    ``dmean2`` the squared difference of their time means. So the exact rel-L2 under any
    per-pixel gain costs two extra cached maps and no extra backbone passes.

    ``cov`` is returned PER COMPONENT, ``(N, H, W, 2)``: with a gain ``g_c`` per component
    the cross term is ``sum_c g_c cov_c``, and a shared gain is the special case
    ``g * sum_c cov_c``. :func:`ceiling_objective` handles both from this one layout.
    """
    p_, t_ = maps["pred"][..., :N_MEASURED], maps["target"][..., :N_MEASURED]
    pm, tm = p_.mean(1, keepdims=True), t_.mean(1, keepdims=True)
    cov = ((p_ - pm) * (t_ - tm)).mean(1)
    dmean2 = ((pm - tm)[:, 0] ** 2).sum(-1)
    tnorm2 = (t_**2).sum(axis=(1, 2, 3, 4))
    return cov.astype(np.float32), dmean2.astype(np.float32), tnorm2.astype(np.float32)


#: Objective modes ``fit_head`` accepts, and the ``(w_rel_l2, w_tke)`` each weights the two
#: axes a rescaling can move with. ``mvpe`` is invariant by construction and ``time`` does
#: not change, so only these two matter -- and only their *ratio* affects the argmin.
#:
#: * ``tke``     -- the scorer's tke error alone. The shipped default, and what every
#:   recorded head was fitted on. Lets the head trade rel_l2 away for free.
#: * ``ceiling`` -- the SPS accuracy ceiling, i.e. the scorer's own published branch
#:   weights (``SPS_BRANCH_WEIGHTS``). Ratio 1.67.
#: * ``board``   -- the **fitted** final-score marginals from ``score.report``, which add
#:   each axis's direct board weight to the part that arrives through the SPS ceiling.
#:   Ratio ~3.42 at the current operating point, i.e. it values rel_l2 about twice as
#:   highly relative to tke as ``ceiling`` does. This is the objective that matches how
#:   the leaderboard has actually paid; it is also the only one containing a fitted
#:   estimate rather than a published constant, which is the tradeoff in choosing it.
#: * ``rel_l2``  -- the rescaled rel_l2 error alone (w_tke = 0). Not a shipping objective: it
#:   measures how much of the rel_l2-optimal damping (FINDINGS 151) a head can predict, and
#:   what tke that costs, so the trade can be priced afterwards.
OBJECTIVE_MODES = ("tke", "ceiling", "board", "rel_l2")


def objective_weights(objective_mode: str, normalize: bool = True) -> tuple[float, float]:
    """``(w_rel_l2, w_tke)`` for an objective mode. See :data:`OBJECTIVE_MODES`.

    ``normalize`` rescales the pair to sum to 1. Only the *ratio* changes the argmin, but
    the *scale* multiplies the gradient, so without this a mode with smaller weights is
    simply less converged at the shared ``lr``/``steps`` -- which would confound the very
    comparison these modes exist to make. Measured on the wiring smoke test: unnormalised,
    ``board`` (weights summing to 0.63) barely left its zero initialisation in 200 steps
    while ``tke`` (an unweighted relative L2, effective scale 1) moved freely. Normalising
    is therefore not cosmetic; it is what makes an A/B between modes honest.
    """
    from ..score.report import SPS_BRANCH_WEIGHTS, board_accuracy_weights

    if objective_mode == "ceiling":
        pair = (SPS_BRANCH_WEIGHTS["rel_l2_score"], SPS_BRANCH_WEIGHTS["tke_score"])
    elif objective_mode == "rel_l2":
        pair = (1.0, 0.0)
    elif objective_mode == "board":
        w = board_accuracy_weights()
        pair = (w["rel_l2_score"], w["tke_score"])
    else:
        raise ValueError(f"objective_mode {objective_mode!r} has no axis weights")
    total = sum(pair)
    return (pair[0] / total, pair[1] / total) if normalize else pair


def ceiling_objective(
    log_ratio: torch.Tensor,
    ke_pred: torch.Tensor,
    ke_target: torch.Tensor,
    valid: torch.Tensor,
    cov: torch.Tensor,
    dmean2: torch.Tensor,
    tnorm2: torch.Tensor,
    t_out: int,
    w_rel: float = 0.5,
    w_tke: float = 0.3,
) -> torch.Tensor:
    """Negative weighted sum of the two scores a rescaling can move.

    Optimising ``tke`` alone lets the head trade rel_l2 away for free, and it did -- 0.27
    rel_l2 for 2.67 tke, on axes weighted 0.5 and 0.3, so about a fifth of the apparent
    gain survived. ``mvpe`` is invariant by construction and drops out of the gradient.

    ``w_rel``/``w_tke`` default to the scorer's published SPS branch weights, which makes
    this the SPS accuracy ceiling. Passing the fitted board marginals instead
    (:func:`objective_weights` with ``"board"``) makes it the first-order final-score
    change, which weights rel_l2 about twice as heavily relative to tke. Only the ratio
    matters to the argmin; the scale changes the printed objective value, so histories
    from two modes are not comparable as levels.
    """
    g = torch.exp(0.5 * log_ratio)
    if g.dim() == 4:
        # Per-component: ``ke_pred`` and ``cov`` are ``(N, H, W, 2)`` like ``g``.
        cross = (g * cov).sum(-1)
        quad = (g * g * ke_pred).sum(-1)
    else:
        cross = g * cov.sum(-1)
        quad = g * g * ke_pred
    sse = t_out * (dmean2 + 2 * ke_target - 2 * cross + 2 * quad).clamp_min(0)
    err_rel = (sse.sum(dim=(-2, -1)) / tnorm2.clamp_min(1e-12)).clamp_min(1e-12).sqrt().mean()
    err_tke = tke_rel_l2(log_ratio, ke_pred, ke_target, valid)
    return -(w_rel * 0.5 / (0.5 + err_rel) + w_tke * 0.5 / (0.5 + err_tke))


def fit_head(
    train_maps: dict[str, np.ndarray],
    objective_mode: str = "tke",
    steps: int = 3000,
    lr: float = 3e-3,
    width: int = 32,
    depth: int = 3,
    batch_size: int = 32,
    seed: int = 0,
    device: str | None = None,
    drop: tuple[str, ...] = (),
    select_maps: dict[str, np.ndarray] | None = None,
    eval_every: int = 100,
    weight_decay: float = 0.0,
    per_component: bool = False,
    n_extra: int = 0,
) -> tuple[TkeHead, list[dict]]:
    """Train the head against the scorer's own tke relative L2.

    ``per_component`` fits a head with two output log-ratios, one per velocity component,
    reading the four per-component input planes as well. The scorer's tke sees only the sum
    of the two components' energies, so this head can put the fluctuation energy into the
    component that agrees with the truth; the in-sample bound for that freedom is worth
    about +0.15 rel_l2 at the shipped head's tke level (2026-09-17). Whether a fitted head
    keeps any of it is what the arm measures. Only meaningful with an objective that prices
    rel_l2 (``ceiling``/``board``): on ``tke`` alone the two gains simply match each
    component's own variance, which the bound says is worth nothing over a shared gain.

    ``select_maps``, when given, is a split the head is **not** fitted on and whose
    objective is used to pick the step to stop at. 1322 training windows against 3000
    steps at batch 32 is ~73 epochs, and the first version of this overfitted hard enough
    to lose to the closed-form blend on Reynolds extrapolation -- so the stopping point
    has to be chosen rather than assumed. Returns the head and the eval history.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    feats, ke_pred, ke_target, valid = _tensors(train_maps, drop, device, per_component, n_extra)
    val = (
        _tensors(select_maps, drop, device, per_component, n_extra)
        if select_maps is not None
        else None
    )
    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"objective_mode must be one of {'|'.join(OBJECTIVE_MODES)}, got {objective_mode!r}"
        )
    # ``None`` rather than a ``(0.0, 0.0)`` placeholder: the tke branch below never reads
    # it, and a zero weight pair would silently make the objective identically zero if that
    # ever stopped being true.
    axis_w = None if objective_mode == "tke" else objective_weights(objective_mode)
    t_out = int(train_maps["target"].shape[1])
    extra = [torch.as_tensor(a, device=device) for a in rescaled_rel_l2_terms(train_maps)]
    val_extra = (
        [torch.as_tensor(a, device=device) for a in rescaled_rel_l2_terms(select_maps)]
        if select_maps is not None
        else None
    )

    def objective(log_ratio, idx, tensors=None, ex=None):
        kp, kt, vd = tensors or (ke_pred, ke_target, valid)
        cv, dm, tn = ex or extra
        if axis_w is None:
            return tke_rel_l2(log_ratio, kp[idx], kt[idx], vd[idx])
        return ceiling_objective(
            log_ratio, kp[idx], kt[idx], vd[idx], cv[idx], dm[idx], tn[idx], t_out, *axis_w
        )

    head = TkeHead(
        in_channels=base_in_channels(per_component) + n_extra,
        width=width,
        depth=depth,
        out_channels=2 if per_component else 1,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = feats.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    history: list[dict] = []
    best = (float("inf"), 0, {k: v.detach().clone() for k, v in head.state_dict().items()})

    for step in range(steps + 1):
        if val is not None and step % eval_every == 0:
            head.eval()
            with torch.no_grad():
                vi = torch.arange(val[1].shape[0], device=device)
                v = float(objective(head(val[0]), vi, (val[1], val[2], val[3]), val_extra))
            history.append({"step": step, "select": v})
            if v < best[0]:
                best = (v, step, {k: t.detach().clone() for k, t in head.state_dict().items()})
            head.train()
        if step == steps:
            break

        idx = torch.randint(0, n, (min(batch_size, n),), generator=g).to(device)
        loss = objective(head(feats[idx]), idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(
                f"  step {step:5d}  train={float(loss):.4f}"
                + (f"  select={history[-1]['select']:.4f}" if history else "")
                + f"  lr={sched.get_last_lr()[0]:.2e}"
            )

    if val is not None:
        print(f"  selected step {best[1]} ({objective_mode} objective {best[0]:.4f})")
        head.load_state_dict(best[2])
    head.eval()
    return head, history


@torch.no_grad()
def head_log_ratio(
    head: TkeHead,
    maps: dict[str, np.ndarray],
    device: str | None = None,
    drop: tuple[str, ...] = (),
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Extra planes are inferred from the weights: whatever the head takes beyond its base width.
    n_extra = int(head.body[0].weight.shape[1]) - base_in_channels(head.per_component)
    feats = torch.from_numpy(_features(maps, drop, head.per_component, n_extra)).to(device)
    return head.to(device)(feats).cpu().numpy()


def report(
    heads: dict[str, tuple[TkeHead, tuple[str, ...]]],
    maps_by_split: dict[str, dict[str, np.ndarray]],
    fit_split: str,
) -> dict:
    """Compare: no head, the closed-form blend at each exponent, every head arm, oracle.

    The oracle row is not an achievable target -- it is the denominator that says what
    fraction of the available headroom an arm actually captured. The blend rows are the
    honest baseline, because the head generalises the blend: an arm that does not beat it
    has learned nothing worth its inference cost.
    """
    out: dict = {"fit_split": fit_split, "splits": {}}
    for split, maps in maps_by_split.items():
        feats = _features(maps)
        rows = {"none": score_maps(maps, np.zeros_like(maps["ke_pred"]))}
        for b in BASELINE_BLENDS:
            if b:
                rows[f"blend {b:g}"] = score_maps(maps, blend_log_ratio(feats, b))
        for name, (head, drop) in heads.items():
            r = head_log_ratio(head, maps, drop=drop)
            rows[f"head {name}"] = score_maps(maps, r)
            # The shipped serving spelling is blend 0 with a shrink, i.e. shrinkage toward
            # the IDENTITY (FINDINGS 80d); 0.5 is what ships. Reported so an arm can be read
            # at the operating point it would actually be served at.
            for shrink in (0.5, 0.7):
                rows[f"head {name} x{shrink:g}"] = score_maps(maps, shrink * r)
            # Shrinking the learned correction toward the closed-form blend. The head is
            # strong where it was fitted and fragile at the ladder ends, while the blend
            # is weak and robust, so a convex mix is the cheapest way to ask whether the
            # two failure modes are independent. Reported at fixed weights rather than
            # fitted, so no split is spent on choosing one.
            anchor = blend_log_ratio(feats, ANCHOR_BLEND)
            if r.ndim == 4:  # per-component head: the pooled blend broadcasts over components
                anchor = anchor[..., None]
            for shrink in (0.25, 0.5, 0.75):
                mixed = shrink * r + (1.0 - shrink) * anchor
                rows[f"head {name} x{shrink:g}+blend"] = score_maps(maps, mixed)
        oracle = np.log(np.clip(maps["ke_target"], 1e-12, None)) - np.log(
            np.clip(maps["ke_pred"], 1e-12, None)
        )
        rows["ORACLE"] = score_maps(maps, oracle)
        out["splits"][split] = rows
    return out


def format_report(rep: dict) -> str:
    lines = []
    for split, rows in rep["splits"].items():
        role = "FITTED HERE" if split == rep["fit_split"] else "held out"
        base = rows["none"]
        oracle = rows["ORACLE"]
        head_line = f"{split}  ({role})"
        lines += [head_line, "-" * len(head_line)]
        lines.append(f"{'estimator':<14}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'d_tke':>8}{'%ceil':>8}")
        span = oracle["tke_score"] - base["tke_score"]
        for name, s in rows.items():
            d = s["tke_score"] - base["tke_score"]
            frac = 100.0 * d / span if span > 0 else float("nan")
            lines.append(
                f"{name:<14}{s['rel_l2_score']:>9.2f}{s['tke_score']:>8.2f}"
                f"{s['mvpe_score']:>8.2f}{d:>+8.2f}{frac:>8.1f}"
            )
        lines.append("")
    lines.append("%ceil is the fraction of the oracle's tke headroom captured.")
    lines.append("mvpe must be identical down the column: the gain is constant in time.")
    return "\n".join(lines)


def save_report(rep: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return path
