"""Annotated output video: skeleton, bar path, depth line, live torso angle, rep MCV."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .analysis import SetResult
from .pose import iter_frames

GREEN, AMBER, RED, WHITE, CYAN = (80, 200, 80), (0, 190, 255), (60, 60, 230), (255, 255, 255), (255, 220, 0)
BONES = [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "toe"),
         ("ankle", "toe"), ("shoulder", "elbow"), ("elbow", "wrist")]


def _pt(p):
    return None if not np.all(np.isfinite(p)) else (int(p[0]), int(p[1]))


def _text(img, s, org, color=WHITE, scale=0.8):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def write_annotated(video: Path, out: Path, j: dict, bar_px: np.ndarray, res: SetResult) -> Path:
    fps = res.fps
    writer = None
    trail: list = []
    rep_at = {}
    for r in res.reps:
        for f in range(r.start, r.end + 1):
            rep_at[f] = r
    for i, frame in enumerate(iter_frames(video)):
        if i >= len(bar_px):
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        scale = frame.shape[0] / 1080.0
        # skeleton
        for a, b in BONES:
            pa, pb = _pt(j[a][i]), _pt(j[b][i])
            if pa and pb:
                cv2.line(frame, pa, pb, WHITE, max(2, int(3 * scale)), cv2.LINE_AA)
        # depth line hip -> knee height
        hip, knee = _pt(j["hip"][i]), _pt(j["knee"][i])
        if hip and knee:
            margin_px = hip[1] - knee[1]           # + = hip below knee (image y down)
            margin_cm = margin_px / res.px_per_cm
            col = GREEN if margin_cm >= 2 else (AMBER if margin_cm >= -2 else RED)
            cv2.line(frame, (knee[0] - int(60 * scale), knee[1]), (knee[0] + int(60 * scale), knee[1]), col, 2)
            cv2.circle(frame, hip, int(8 * scale) + 2, col, -1)
        # bar trail (current rep only)
        r = rep_at.get(i)
        bp = _pt(bar_px[i])
        if r is None:
            trail = []
        elif bp:
            trail.append(bp)
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, CYAN, max(2, int(3 * scale)), cv2.LINE_AA)
        if bp:
            cv2.circle(frame, bp, int(7 * scale) + 2, CYAN, -1)
        # torso angle
        sh = j["shoulder"][i]
        if hip and np.all(np.isfinite(sh)):
            ang = np.degrees(np.arctan2(abs(sh[0] - hip[0]), hip[1] - sh[1]))
            _text(frame, f"torso {ang:.0f} deg", (int(20 * scale), int(80 * scale)), WHITE, scale)
        # rep label
        if r is not None:
            col = GREEN if r.depth == "PASS" else (AMBER if r.depth == "BORDERLINE" else RED)
            _text(frame, f"rep {r.n}", (int(20 * scale), int(40 * scale)), WHITE, scale)
            if i >= r.bottom:
                _text(frame, f"depth {r.depth} {r.depth_margin_cm:+.1f} cm", (int(20 * scale), int(120 * scale)), col, scale)
            if i >= r.end - 1 or i >= r.bottom + (r.end - r.bottom) // 2:
                _text(frame, f"MCV {r.mcv_ms:.2f} m/s  fold {r.fold_deg:.0f} deg",
                      (int(20 * scale), int(160 * scale)), WHITE, scale)
        writer.write(frame)
    if writer is not None:
        writer.release()
    return out
