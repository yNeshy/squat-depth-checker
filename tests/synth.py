"""Synthetic squatter with known ground truth, for testing without real footage.

Side view, athlete faces +x, image y points down. Two-link legs, rigid torso.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SHANK, THIGH, TORSO = 45.0, 45.0, 55.0
PX_PER_CM = 4.0
ANKLE = np.array([120.0, 20.0])         # cm, (x, y-up) in a 300 x 250 cm scene
SCENE_H_CM = 250.0


@dataclass
class RepSpec:
    bottom_margin_cm: float   # + = hip below knee at the bottom
    down_s: float
    up_s: float
    fold_deg: float = 0.0     # extra torso lean added in the first half of the ascent


def _ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


def _leg(hip):
    """Knee position (forward of the ankle-hip line) by two-link IK."""
    d = hip - ANKLE
    L = np.linalg.norm(d)
    L = min(L, SHANK + THIGH - 1e-6)
    a = (SHANK ** 2 - THIGH ** 2 + L ** 2) / (2 * L)
    h = np.sqrt(max(SHANK ** 2 - a ** 2, 0))
    base = ANKLE + a * d / L
    perp = np.array([d[1], -d[0]]) / L       # points forward (+x) for a hip above the ankle
    return base + h * perp


def _pose(hip_h, torso_deg):
    """Joint positions (cm, y-up) for a hip height above the ankle and torso lean."""
    mid_x = ANKLE[0] + 6.0
    th = np.radians(torso_deg)
    hip = np.array([mid_x - TORSO * np.sin(th) - 2.0, ANKLE[1] + hip_h])
    knee = _leg(hip)
    sh = hip + TORSO * np.array([np.sin(th), np.cos(th)])
    return dict(hip=hip, knee=knee, shoulder=sh)


def _hip_height_for_margin(margin, torso_deg):
    """Find hip height giving the wanted hip-below-knee margin."""
    lo, hi = 20.0, 89.0
    for _ in range(60):
        m = (lo + hi) / 2
        p = _pose(m, torso_deg)
        if p["knee"][1] - p["hip"][1] > margin:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def make_set(reps: list[RepSpec], fps: float = 60.0, gap_s: float = 1.0,
             walkout_s: float = 0.0, pause_dip_cm: float = 0.0):
    """Return (landmarks[N,33,3] in px, truth list, bar_px[N,2])."""
    stand_h, stand_t, bot_t = 88.0, 12.0, 45.0
    frames, truth = [], []

    def push(h, t):
        frames.append(_pose(h, t))

    # walkout: steps back from the rack, hip bobbing a few cm with each step
    for u in np.arange(0, walkout_s, 1 / fps):
        push(stand_h - 4.0 * abs(np.sin(np.pi * 3 * u / walkout_s)), stand_t)
    for _ in range(int(gap_s * fps)):
        push(stand_h, stand_t)
    for spec in reps:
        bh = _hip_height_for_margin(spec.bottom_margin_cm, bot_t)
        start = len(frames)
        for u in np.arange(0, spec.down_s, 1 / fps):
            e = _ease(u / spec.down_s)
            push(stand_h + (bh - stand_h) * e, stand_t + (bot_t - stand_t) * e)
        bottom = len(frames)
        up_n = int(round(spec.up_s * fps))
        for k in range(up_n + 1):
            u = k / up_n
            e = _ease(u)
            fold = spec.fold_deg * np.sin(np.pi * min(u / 0.6, 1.0))   # peaks ~30% up, gone by 60%
            push(bh + (stand_h - bh) * e, bot_t + (stand_t - bot_t) * e + fold)
        # concentric for speed = from the bar's lowest point (can be after the hip bottom when folding)
        ys = [f["shoulder"][1] for f in frames[bottom:]]
        bar_lo = bottom + int(np.argmin(ys))
        p0, p1 = frames[bar_lo]["shoulder"], frames[-1]["shoulder"]
        up_s = (len(frames) - 1 - bar_lo) / fps
        truth.append(dict(start=start, bottom=bottom, travel_cm=p1[1] - p0[1],
                          mcv=(p1[1] - p0[1]) / 100 / up_s, margin=spec.bottom_margin_cm,
                          fold=spec.fold_deg, end=len(frames) - 1,
                          descent_s=spec.down_s, up_s=spec.up_s))
        # pause between reps: breathe and brace, hip sags slightly then resets
        for u in np.arange(0, gap_s, 1 / fps):
            push(stand_h - pause_dip_cm * np.sin(np.pi * u / gap_s), stand_t)

    n = len(frames)
    lm = np.full((n, 33, 3), np.nan)
    lm[:, :, 2] = 0.1

    def px(p):
        return np.array([p[0] * PX_PER_CM, (SCENE_H_CM - p[1]) * PX_PER_CM])

    bar_px = np.zeros((n, 2))
    for i, f in enumerate(frames):
        sh, hip, knee = f["shoulder"], f["hip"], f["knee"]
        wrist = sh + np.array([-4.0, -18.0])
        elbow = sh + np.array([-14.0, -10.0])
        nose = sh + np.array([10.0, 22.0])
        pts = {11: sh, 13: elbow, 15: wrist, 23: hip, 25: knee, 27: ANKLE,
               29: ANKLE + [-7, -6], 31: ANKLE + [19, -6], 0: nose}
        for idx, p in pts.items():
            lm[i, idx, :2] = px(p)
            lm[i, idx, 2] = 0.95
            # mirror side with low visibility (mimics the far side)
            if idx:
                lm[i, idx + 1, :2] = px(p)
        bar_px[i] = px(sh + np.array([-3.0, -3.0]))   # bar sits just behind/below the shoulder
    return lm, truth, bar_px


def render_video(path, lm, bar_px, fps=60.0, plate_cm=45.0):
    """Draw stick figure + plate so the CLI's plate detection can be exercised."""
    w, h = int(300 * PX_PER_CM), int(SCENE_H_CM * PX_PER_CM)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    r = int(plate_cm / 2 * PX_PER_CM)
    for i in range(len(lm)):
        img = np.full((h, w, 3), 200, np.uint8)
        for a, b in [(11, 23), (23, 25), (25, 27), (11, 13), (13, 15)]:
            pa, pb = lm[i, a, :2].astype(int), lm[i, b, :2].astype(int)
            cv2.line(img, tuple(pa), tuple(pb), (60, 60, 60), 10)
        c = tuple(bar_px[i].astype(int))
        cv2.circle(img, c, r, (30, 30, 30), -1)
        cv2.circle(img, c, int(r * 0.2), (180, 180, 180), -1)
        vw.write(img)
    vw.release()
