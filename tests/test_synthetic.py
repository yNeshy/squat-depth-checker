"""Checks the metric layer against a synthetic squatter with known answers.

Run: python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from squat_tracker import pose  # noqa: E402
from squat_tracker.analysis import analyse  # noqa: E402
from squat_tracker.config import Config  # noqa: E402
from synth import PX_PER_CM, RepSpec, make_set, render_video  # noqa: E402

SPECS = [RepSpec(5.0, 1.4, 0.9), RepSpec(0.0, 1.4, 1.2), RepSpec(-6.0, 1.4, 1.6, fold_deg=18.0)]


def _run(fps=60.0):
    lm, truth, bar_px = make_set(SPECS, fps=fps)
    j = pose.joints(lm, pose.pick_side(lm))
    res = analyse(j, bar_px, fps, PX_PER_CM, Config(), "left", "synthetic", 1.0)
    return res, truth


def test_rep_count_and_depth():
    res, truth = _run()
    assert len(res.reps) == 3
    assert [r.depth for r in res.reps] == ["PASS", "BORDERLINE", "HIGH"]
    for r, t in zip(res.reps, truth):
        assert abs(r.depth_margin_cm - t["margin"]) < 1.0, (r.depth_margin_cm, t["margin"])


def test_velocity():
    for fps in (30.0, 60.0):
        res, truth = _run(fps)
        for r, t in zip(res.reps, truth):
            assert abs(r.mcv_ms - t["mcv"]) / t["mcv"] < 0.08, (fps, r.mcv_ms, t["mcv"])
        assert res.velocity_loss_pct > 30


def test_fold_flag():
    res, _ = _run()
    assert res.reps[2].fold_deg > 8
    assert "fold" in res.reps[2].flags
    assert res.reps[0].fold_deg < 3 and "fold" not in res.reps[0].flags


def test_cli_end_to_end(tmp_path):
    """Video with a drawn plate + cached landmarks: exercises plate scale, report and annotation."""
    from squat_tracker.cli import main

    fps = 30.0
    lm, truth, bar_px = make_set(SPECS, fps=fps)
    video = tmp_path / "synthetic.mp4"
    render_video(video, lm, bar_px, fps)
    np.savez_compressed(tmp_path / "synthetic.squat.pose.npz", fps=fps, width=1200, height=1000, lm=lm)
    rc = main(["analyse", str(video), "--load", "190", "--rpe", "9", "--annotate", "--no-history"])
    assert rc == 0
    import json
    out = json.loads((tmp_path / "synthetic.squat.json").read_text())
    assert abs(out["px_per_cm"] - PX_PER_CM) / PX_PER_CM < 0.05, out["px_per_cm"]
    assert len(out["reps"]) == 3
    assert (tmp_path / "synthetic.squat.annotated.mp4").stat().st_size > 10000
