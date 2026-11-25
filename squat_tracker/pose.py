"""Video reading + MediaPipe pose landmarks.

Output is a plain dict of numpy arrays so it can be cached to .npz and re-analysed
without re-running the (slow) pose model.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
HOME = Path.home() / ".squat-track"

# MediaPipe landmark indices
NOSE = 0
L = dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27, heel=29, toe=31)
R = dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28, heel=30, toe=32)


def ensure_model(variant: str = "heavy") -> Path:
    path = HOME / "models" / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model ({variant}) to {path} ...", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    return path


def video_info(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read frames from: {video}")
    h, w = frame.shape[:2]
    return dict(fps=float(fps), n_frames=n, width=w, height=h)


def iter_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def run_pose(video: Path, variant: str = "heavy", progress: bool = True) -> dict:
    """Return dict(fps, width, height, lm[N,33,3]) with lm = (x_px, y_px, visibility); NaN when no person."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    info = video_info(video)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model(variant))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rows = []
    with vision.PoseLandmarker.create_from_options(opts) as det:
        for i, frame in enumerate(iter_frames(video)):
            h, w = frame.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = det.detect_for_video(img, int(round(i * 1000.0 / info["fps"])))
            if res.pose_landmarks:
                rows.append([(p.x * w, p.y * h, p.visibility) for p in res.pose_landmarks[0]])
            else:
                rows.append([(np.nan, np.nan, 0.0)] * 33)
            if progress and i % 30 == 0:
                print(f"\r  pose: frame {i}/{info['n_frames']}", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    return dict(fps=info["fps"], width=info["width"], height=info["height"],
                lm=np.asarray(rows, dtype=float))


def pick_side(lm: np.ndarray) -> str:
    """The side facing the camera has higher landmark visibility."""
    keys = ["shoulder", "hip", "knee", "ankle"]
    vl = np.nanmean([lm[:, L[k], 2] for k in keys])
    vr = np.nanmean([lm[:, R[k], 2] for k in keys])
    return "left" if vl >= vr else "right"


def joints(lm: np.ndarray, side: str) -> dict:
    idx = L if side == "left" else R
    out = {k: lm[:, i, :2].copy() for k, i in idx.items()}
    out["nose"] = lm[:, NOSE, :2].copy()
    out["vis"] = {k: lm[:, i, 2].copy() for k, i in idx.items()}
    return out
