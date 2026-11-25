"""Rep segmentation and per-rep / per-set metrics.

Coordinates: everything is converted to cm with y pointing UP (image y points down).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Config, estimate_rpe
from .filters import fill_gaps, savgol


@dataclass
class Rep:
    n: int
    start: int            # frame: descent begins
    bottom: int           # frame: lowest bar position
    end: int              # frame: lockout
    # depth
    depth_margin_cm: float
    depth: str
    thigh_angle_deg: float          # + = below horizontal
    time_below_parallel_s: float
    # speed
    travel_cm: float
    descent_s: float
    concentric_s: float
    mcv_ms: float
    peak_v_ms: float
    min_v_ms: float                 # slowest point of the ascent (sticking)
    sticking_height_cm: float       # above the bottom
    # technique
    torso_bottom_deg: float
    fold_deg: float
    hips_first_ratio: float
    drift_bottom_cm: float          # + = bar in front of midfoot (towards the toes)
    flags: list = field(default_factory=list)


@dataclass
class SetResult:
    reps: list
    fps: float
    px_per_cm: float
    scale_source: str
    side: str
    plate_coverage: float
    velocity_loss_pct: float | None = None
    est_rpe: float | None = None
    logged_rpe: float | None = None
    load_kg: float | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def _to_cm(p_px: np.ndarray, px_per_cm: float) -> np.ndarray:
    out = np.asarray(p_px, float) / px_per_cm
    out[:, 1] *= -1.0
    return out


def _smooth_xy(p: np.ndarray, win: int) -> np.ndarray:
    return np.column_stack([savgol(fill_gaps(p[:, 0]), win), savgol(fill_gaps(p[:, 1]), win)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def find_reps(hip_y: np.ndarray, hip_v: np.ndarray, fps: float,
              cfg: Config) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Segment reps on the hip height (cleaner than the bar on real footage).

    A rep is an excursion of the hip more than `min_rep_travel_cm` below standing height.
    Start = the lifter is settled at the top just before the descent (walkout and pauses
    between reps excluded); end = lockout, the hip back at its standing height and stopped.
    Returns ((start, bottom, end) frames per rep, warnings).
    """
    n = len(hip_y)
    top = np.percentile(hip_y, 90)
    down_thr = top - cfg.min_rep_travel_cm
    look = int(round(cfg.top_search_s * fps))
    tol, still = cfg.settle_tolerance_cm, cfg.lockout_velocity_ms
    runs = _runs(hip_y < down_thr)
    reps, warnings, prev_end = [], [], 0
    for k, (a, z) in enumerate(runs):
        bottom = a + int(np.argmin(hip_y[a:z + 1]))
        nxt = runs[k + 1][0] if k + 1 < len(runs) else n
        if z + 1 >= n:
            warnings.append(f"Rep {len(reps) + 1} never comes back up before the video ends; not counted.")
            break
        # start: walk back from the fastest point of the descent until the hip has (almost)
        # stopped moving down. Speed-based, so standing sway, a walkout or a sag while bracing
        # before the rep is not counted, whatever the scale.
        lo = max(prev_end, bottom - look)
        pre_top = float(np.max(hip_y[lo:bottom + 1]))
        fastest = lo + int(np.argmin(hip_v[lo:bottom + 1]))
        stop_v = max(still, cfg.descent_start_fraction * -float(hip_v[fastest]))
        s = fastest
        while s > lo and hip_v[s - 1] < -stop_v:
            s -= 1
        # end: first frame within `tol` of the standing height reached after the bottom,
        # then on while the hip is still rising
        hi = min(nxt, bottom + look)
        post_top = float(np.max(hip_y[bottom:hi]))
        e = bottom
        while e + 1 < hi and hip_y[e] < post_top - tol:
            e += 1
        while e + 1 < hi and hip_v[e] > still:
            e += 1
        drop = min(pre_top, post_top) - float(hip_y[bottom])
        reps.append((int(s), int(bottom), int(e), drop))
        prev_end = e
    # a dip much shallower than the real reps is a walkout step, a re-rack or a pose glitch
    if reps:
        deepest = max(r[3] for r in reps)
        kept = [r for r in reps if r[3] >= cfg.min_rep_depth_fraction * deepest]
        for r in reps:
            if r not in kept:
                warnings.append(f"Ignored a {r[3]:.0f} cm hip dip at {r[1] / fps:.1f} s "
                                "(much shallower than the reps; walkout, re-rack or tracking glitch).")
        reps = kept
    return [r[:3] for r in reps], warnings


def angle_from_vertical(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    d = upper - lower
    return np.degrees(np.arctan2(np.abs(d[:, 0]), d[:, 1]))


def analyse(j: dict, bar_px: np.ndarray, fps: float, px_per_cm: float, cfg: Config,
            side: str, scale_source: str, plate_coverage: float) -> SetResult:
    win = max(5, int(round(cfg.smooth_window_s * fps)))
    P = {k: _smooth_xy(_to_cm(j[k], px_per_cm), win)
         for k in ("shoulder", "hip", "knee", "ankle", "heel", "toe")}
    bar = _smooth_xy(_to_cm(bar_px, px_per_cm), win)
    dt = 1.0 / fps
    vel = savgol(bar[:, 1], win, deriv=1, dt=dt) / 100.0          # m/s, + = up
    margin = P["knee"][:, 1] - P["hip"][:, 1]                     # + = hip below knee
    torso = angle_from_vertical(P["shoulder"], P["hip"])
    midfoot_x = (P["heel"][:, 0] + P["toe"][:, 0]) / 2
    facing = np.sign(np.nanmedian(P["toe"][:, 0] - P["heel"][:, 0])) or 1.0

    res = SetResult(reps=[], fps=fps, px_per_cm=px_per_cm, scale_source=scale_source,
                    side=side, plate_coverage=plate_coverage)
    if plate_coverage < 0.3:
        res.warnings.append("Plate not tracked; bar speed uses shoulder/wrist (less accurate).")

    hip_v = savgol(P["hip"][:, 1], win, deriv=1, dt=dt) / 100.0
    found, seg_warnings = find_reps(P["hip"][:, 1], hip_v, fps, cfg)
    res.warnings += seg_warnings
    for k, (s, hb, e) in enumerate(found, start=1):
        # hip bottom (anchor for depth and technique); bar bottom (anchor for speed) can differ,
        # e.g. when the torso folds the bar keeps sinking after the hips have turned around
        w = int(0.4 * fps)
        lo, hi = max(s, hb - w), min(e, hb + w) + 1
        b = lo + int(np.argmin(bar[lo:hi, 1]))
        db = lo + int(np.argmax(margin[lo:hi]))
        m = float(margin[db])
        depth = "PASS" if m >= cfg.depth_pass_cm else ("HIGH" if m < cfg.depth_high_cm else "BORDERLINE")
        dthigh = P["hip"][db] - P["knee"][db]
        thigh = float(np.degrees(np.arctan2(-dthigh[1], abs(dthigh[0]))))
        below = float(np.sum(margin[s:e + 1] > 0) / fps)

        # speed
        travel = float(bar[e, 1] - bar[b, 1])
        conc = (e - b) / fps
        mcv = travel / 100.0 / conc if conc > 0 else float("nan")
        v_up = vel[b:e + 1]
        # sticking: slowest point between 15% and 85% of the ascent
        h_rel = bar[b:e + 1, 1] - bar[b, 1]
        mid = (h_rel > 0.15 * travel) & (h_rel < 0.85 * travel)
        if mid.any():
            si = int(np.argmin(np.where(mid, v_up, np.inf)))
            vmin, stick_h = float(v_up[si]), float(h_rel[si])
        else:
            vmin, stick_h = float("nan"), float("nan")

        # technique
        t_bot = float(torso[hb])
        half = hb + max(1, (e - hb) // 2)
        fold = float(np.max(torso[hb:half + 1]) - t_bot)
        rise10 = np.nonzero(h_rel >= 10.0)[0]
        if len(rise10):
            f10 = b + int(rise10[0])
            hips_first = float((P["hip"][f10, 1] - P["hip"][b, 1]) / (bar[f10, 1] - bar[b, 1]))
        else:
            hips_first = float("nan")
        drift = float((bar[hb, 0] - midfoot_x[hb]) * facing)

        res.reps.append(Rep(
            n=k, start=int(s), bottom=int(b), end=int(e),
            depth_margin_cm=round(m, 1) + 0.0, depth=depth, thigh_angle_deg=round(thigh, 1),
            time_below_parallel_s=round(below, 2),
            travel_cm=round(travel, 1), descent_s=round((hb - s) / fps, 2), concentric_s=round(conc, 2),
            mcv_ms=round(mcv, 3), peak_v_ms=round(float(np.max(v_up)), 3),
            min_v_ms=round(vmin, 3), sticking_height_cm=round(stick_h, 1),
            torso_bottom_deg=round(t_bot, 1), fold_deg=round(fold, 1),
            hips_first_ratio=round(hips_first, 2), drift_bottom_cm=round(drift, 1)))

    _set_summary(res, cfg)
    return res


def _set_summary(res: SetResult, cfg: Config) -> None:
    reps = res.reps
    if not reps:
        res.warnings.append("No reps found. Check the bar is tracked and the whole squat is in frame.")
        return
    mcvs = np.array([r.mcv_ms for r in reps])
    res.velocity_loss_pct = round(float((mcvs.max() - mcvs[-1]) / mcvs.max() * 100), 1) if len(reps) > 1 else 0.0
    res.est_rpe = estimate_rpe(float(mcvs[-1]), cfg)

    # plausibility: catch segmentation or tracking errors before they turn into a wrong RPE
    med_travel = float(np.median([r.travel_cm for r in reps]))
    for r in reps:
        if r.descent_s > cfg.max_descent_s:
            res.warnings.append(f"Rep {r.n}: {r.descent_s:.1f} s descent looks too long; check the start frame.")
        if r.concentric_s > cfg.max_concentric_s:
            res.warnings.append(f"Rep {r.n}: {r.concentric_s:.1f} s ascent looks too long; check the end frame.")
        if r.travel_cm < cfg.min_rep_travel_cm:
            res.warnings.append(f"Rep {r.n}: bar travel {r.travel_cm:.0f} cm is too short for a squat; "
                                "bar track is probably wrong, so its speed is not reliable.")
        elif len(reps) > 1 and abs(r.travel_cm - med_travel) > cfg.travel_spread * med_travel:
            res.warnings.append(f"Rep {r.n}: bar travel {r.travel_cm:.0f} cm differs from the set median "
                                f"{med_travel:.0f} cm; bar track may be off.")

    med_fold = float(np.median([r.fold_deg for r in reps]))
    med_torso = float(np.median([r.torso_bottom_deg for r in reps]))
    for r in reps:
        if r.depth != "PASS":
            r.flags.append(f"depth {r.depth.lower()}")
        if r.fold_deg > cfg.fold_abs_flag_deg:
            r.flags.append("fold")
        elif len(reps) >= 3 and r.fold_deg - med_fold > cfg.fold_rel_flag_deg:
            r.flags.append("fold vs set")
        if len(reps) >= 3 and abs(r.torso_bottom_deg - med_torso) > cfg.torso_rel_flag_deg:
            r.flags.append("torso angle off")
        if abs(r.drift_bottom_cm) > cfg.drift_flag_cm:
            r.flags.append("bar drift")
        if np.isfinite(r.hips_first_ratio) and r.hips_first_ratio > cfg.hips_first_flag:
            r.flags.append("hips first")


def check_rpe(res: SetResult, cfg: Config) -> None:
    if res.logged_rpe is not None and res.est_rpe is not None:
        if abs(res.logged_rpe - res.est_rpe) >= cfg.rpe_disagreement_flag:
            res.warnings.append(
                f"Logged RPE {res.logged_rpe:g} vs bar-speed estimate {res.est_rpe:g}: "
                "worth a second look (estimate uses group averages until your profile exists).")
