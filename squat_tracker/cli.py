"""squat-track: per-rep depth, bar speed and torso metrics from a side-view squat video."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import bar, pose, report
from .analysis import analyse, check_rpe
from .config import Config


def _pose_cached(video: Path, variant: str, cache: Path, refresh: bool) -> dict:
    if cache.exists() and not refresh:
        z = np.load(cache)
        print(f"Using cached pose: {cache}", file=sys.stderr)
        return dict(fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]), lm=z["lm"])
    p = pose.run_pose(video, variant)
    np.savez_compressed(cache, **p)
    return p


def cmd_analyse(a) -> int:
    cfg = Config(plate_diameter_cm=a.plate_cm)
    video = Path(a.video)
    out_dir = Path(a.out) if a.out else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{video.stem}.squat"

    p = _pose_cached(video, a.model, Path(f"{base}.pose.npz"), a.refresh)
    lm = p["lm"]
    if np.isfinite(lm[:, :, 0]).mean() < 0.5:
        print("! Person found in fewer than half the frames; results may be unreliable.", file=sys.stderr)
    side = a.side if a.side != "auto" else pose.pick_side(lm)
    j = pose.joints(lm, side)

    def run(plate):
        px_per_cm, src = bar.calibrate(j, cfg, plate, a.height_cm)
        bar_px, _, cov = bar.bar_track(j, plate)
        return analyse(j, bar_px, p["fps"], px_per_cm, cfg, side, src, cov), bar_px

    plate = None if a.no_plate else bar.detect_plate(video, j, cfg, a.height_cm or cfg.assumed_height_cm)
    pre_warnings = []
    if plate is not None:
        # reps are segmented on the hip, so they do not depend on the plate; use them to check it
        why = bar.plate_problem(j, plate, cfg)
        if not why:
            first, _ = run(plate)
            why = bar.plate_problem(j, plate, cfg, [(r.start, r.end) for r in first.reps])
        if why:
            pre_warnings.append(f"Plate track rejected: {why}. Bar speed uses the shoulders instead "
                                "(less accurate). Film side-on with the whole plate in frame.")
            plate = None
    res, bar_px = run(plate)
    if res.scale_source.startswith("ASSUMED"):
        pre_warnings.append("No plate and no --height-cm: scale assumes a "
                            f"{cfg.assumed_height_cm:.0f} cm athlete, so cm and m/s values are approximate.")
    res.warnings[:0] = pre_warnings
    res.load_kg, res.logged_rpe = a.load, a.rpe
    check_rpe(res, cfg)

    print()
    print(report.table(res))
    print()
    print("Log line:")
    print("  " + report.log_line(res))
    files = report.write_files(res, base)
    if a.annotate:
        from .annotate import write_annotated
        files.append(write_annotated(video, Path(f"{base}.annotated.mp4"), j, bar_px, res))
    if a.load is not None and res.reps and not a.no_history:
        files.append(report.append_history(res, video))
    print("\nWrote: " + ", ".join(str(f) for f in files))
    return 0


def cmd_history(_a) -> int:
    print(report.show_history())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="squat-track", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyse", aliases=["analyze"], help="analyse one set from a video")
    an.add_argument("video")
    an.add_argument("--load", type=float, help="bar load in kg (saved to history)")
    an.add_argument("--rpe", type=float, help="RPE you logged for the set (compared to the bar-speed estimate)")
    an.add_argument("--plate-cm", type=float, default=45.0, help="plate diameter for scale (default 45)")
    an.add_argument("--height-cm", type=float, help="your height; used as scale when the plate is not found")
    an.add_argument("--no-plate", action="store_true", help="skip plate detection, use pose + height")
    an.add_argument("--side", choices=["auto", "left", "right"], default="auto", help="side facing the camera")
    an.add_argument("--model", choices=["lite", "full", "heavy"], default="heavy", help="pose model size")
    an.add_argument("--annotate", action="store_true", help="write an annotated video")
    an.add_argument("--out", help="output folder (default: next to the video)")
    an.add_argument("--refresh", action="store_true", help="ignore cached pose landmarks")
    an.add_argument("--no-history", action="store_true", help="do not append to ~/.squat-track/history.jsonl")
    an.set_defaults(fn=cmd_analyse)

    hi = sub.add_parser("history", help="list analysed sets")
    hi.set_defaults(fn=cmd_history)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
