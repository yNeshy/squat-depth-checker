"""Bar position and pixel scale.

Primary: find the plate as a circle near the shoulder (low-bar sits just below it) with
HoughCircles. Its centre is the bar end; its radius gives cm-per-pixel.
Fallback: shoulder/wrist midpoint from the pose, scaled from athlete height.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from .config import Config
from .filters import fill_gaps
from .pose import iter_frames


def standing_body_px(j: dict) -> float:
    """Nose-to-ankle pixel distance in the most upright frames."""
    d = j["ankle"][:, 1] - j["nose"][:, 1]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise SystemExit("No person detected in the video.")
    return float(np.percentile(d, 90))


def detect_plate(video, j: dict, cfg: Config, assumed_height_cm: float = 178.0,
                 progress: bool = True) -> dict:
    """Per-frame plate circle (cx, cy, r) in px, NaN where not found."""
    body = standing_body_px(j)
    px_per_cm_guess = body / (cfg.nose_to_ankle_ratio * assumed_height_cm)
    r_guess = cfg.plate_diameter_cm / 2 * px_per_cm_guess
    r_min, r_max = int(r_guess * 0.6), int(r_guess * 1.6)

    n = len(j["shoulder"])
    out = np.full((n, 3), np.nan)
    prev = None
    for i, frame in enumerate(iter_frames(video)):
        if i >= n:
            break
        sx, sy = j["shoulder"][i]
        if not np.isfinite(sx):
            continue
        # search window around the shoulder
        half = int(r_max * 2.2)
        x0, y0 = max(int(sx - half), 0), max(int(sy - half), 0)
        x1, y1 = min(int(sx + half), frame.shape[1]), min(int(sy + half), frame.shape[0])
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        gray = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=r_min,
                                   param1=120, param2=40, minRadius=r_min, maxRadius=r_max)
        if circles is None:
            continue
        c = circles[0].copy()
        c[:, 0] += x0
        c[:, 1] += y0
        # nearest to previous detection, else nearest to the shoulder
        ref = prev if prev is not None else np.array([sx, sy])
        best = c[np.argmin(np.hypot(c[:, 0] - ref[0], c[:, 1] - ref[1]))]
        # plate centre should sit within ~1 radius of the shoulder (bar on the back)
        if np.hypot(best[0] - sx, best[1] - sy) > 1.3 * best[2]:
            continue
        out[i] = best
        prev = best[:2]
        if progress and i % 30 == 0:
            print(f"\r  plate: frame {i}/{n}", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    return dict(circles=out, r_guess=r_guess)


def calibrate(j: dict, cfg: Config, plate: dict | None, height_cm: float | None) -> tuple[float, str]:
    """Return (px_per_cm, source)."""
    if plate is not None:
        r = plate["circles"][:, 2]
        r = r[np.isfinite(r)]
        if len(r) >= 10:
            med = np.median(r)
            r = r[np.abs(r - med) < 0.15 * med]
            return float(2 * np.median(r) / cfg.plate_diameter_cm), f"plate ({len(r)} frames)"
    if height_cm:
        return standing_body_px(j) / (cfg.nose_to_ankle_ratio * height_cm), f"height {height_cm:.0f} cm"
    h = cfg.assumed_height_cm
    return (standing_body_px(j) / (cfg.nose_to_ankle_ratio * h),
            f"ASSUMED height {h:.0f} cm (pass --height-cm for real cm values)")


def plate_problem(j: dict, plate: dict, cfg: Config,
                  windows: list[tuple[int, int]] | None = None) -> str | None:
    """Why the plate track should not be trusted, or None if it looks like the real plate.

    Catches the tracker locking onto another round thing (rack hardware, a plate on the
    floor, the athlete's torso) when the real plate is out of frame: the radius jumps
    around, and during the reps the 'plate' does not rise and fall with the shoulders.
    `windows` are (start, end) frames of the reps; without them the whole clip is used."""
    c = plate["circles"]
    ok = np.isfinite(c[:, 0])
    if ok.sum() < 10:
        return None
    r = c[ok, 2]
    spread = float(np.std(r) / np.median(r))
    if spread > cfg.plate_radius_spread:
        return f"plate size jumps between frames ({spread * 100:.0f}% spread)"
    sh_y = j["shoulder"][:, 1]
    use = ok & np.isfinite(sh_y)
    if windows:
        in_rep = np.zeros(len(use), bool)
        for s, e in windows:
            in_rep[s:e + 1] = True
        use &= in_rep
    if use.sum() >= 10 and np.std(sh_y[use]) > 0:
        corr = float(np.corrcoef(c[use, 1], sh_y[use])[0, 1])
        if not corr >= cfg.plate_min_corr:
            return f"plate does not move with the shoulders (correlation {corr:.2f})"
    for s, e in windows or []:
        m = use[s:e + 1]
        if m.sum() < 5:
            continue
        plate_dy = float(np.ptp(c[s:e + 1, 1][m]))
        sh_dy = float(np.nanmax(sh_y[s:e + 1]) - np.nanmin(sh_y[s:e + 1]))
        if sh_dy > 0 and abs(plate_dy / sh_dy - 1) > cfg.plate_travel_tolerance:
            return (f"plate moves {plate_dy:.0f} px in a rep where the shoulders move {sh_dy:.0f} px")
    return None


def bar_track(j: dict, plate: dict | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Bar (x, y) px per frame. Plate centre where found, pose fallback elsewhere.

    Returns (xy[N,2], used_plate_mask[N], plate_coverage)."""
    sh, wr = j["shoulder"], j["wrist"]
    fallback = np.where(np.isfinite(wr), (sh + wr) / 2, sh)
    if plate is None:
        return fallback, np.zeros(len(sh), bool), 0.0
    c = plate["circles"][:, :2]
    ok = np.isfinite(c[:, 0])
    cov = float(ok.mean())
    if cov < 0.3:
        return fallback, np.zeros(len(sh), bool), cov
    # use plate track, fill small gaps by interpolation, offset-corrected fallback for long gaps
    offset = np.nanmedian(c[ok] - fallback[ok], axis=0)
    xy = np.where(ok[:, None], c, fallback + offset)
    xy = np.column_stack([fill_gaps(xy[:, 0]), fill_gaps(xy[:, 1])])
    return xy, ok, cov
