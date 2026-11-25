# squat-tracker (prototype v0.1)

Per-rep **depth**, **bar speed** and **torso folding** from a side-view squat video. Runs offline.
Plan and reasoning: see the "Squat motion tracker plan" doc in the Powerlifting journey project.

## Install (Windows, PowerShell)

```powershell
cd squat-tracker
py -3.12 -m venv .venv          # Python 3.10-3.12 recommended for MediaPipe wheels
.venv\Scripts\Activate.ps1
pip install -e .
```

The first run downloads the MediaPipe pose model (~30 MB) to `%USERPROFILE%\.squat-track\models`.

## Use

```powershell
squat-track analyse top_set.mp4 --load 190 --rpe 9 --annotate
squat-track analyse top_set.mp4 --height-cm 180 --no-plate     # plate hidden or not round on video
squat-track history
```

Outputs next to the video: `<name>.squat.json`, `.csv`, `.annotated.mp4` (with `--annotate`) and
`.pose.npz` (cached landmarks, so re-runs are fast; `--refresh` ignores it).
With `--load`, each set is appended to `~/.squat-track/history.jsonl` for the personal velocity profile.

## Filming (matters more than anything in the code)

1. Side-on, lens in line with the bar sleeve (plate looks like a circle, not an ellipse).
2. Tripod at hip height, 3-4 m away, outside the rack. Nothing in front of the hip or knee at the bottom.
3. Landscape, 1080p, 60 fps if possible. Whole body and the full plate in frame the whole set.
4. Fitted shorts or tights at the hip.

## What the numbers mean

| Column | Meaning |
| --- | --- |
| Depth / Margin cm | Hip joint centre below knee joint centre at the deepest point. PASS >= +2 cm, BORDERLINE -2..+2, HIGH < -2 |
| MCV m/s | Mean concentric velocity: bar rise / time from lowest point to lockout |
| Up s | Time from the bottom to lockout |
| Slowest m/s | Slowest bar speed in the middle of the ascent (sticking point) |
| Torso deg | Shoulder-hip line from vertical at the hip's lowest point |
| Fold deg | How much the torso leans further in the first half of the ascent (the "good morning") |
| Hips 1st | Hip rise / bar rise over the first 10 cm. ~1 = together, > 1.5 = hips shoot |
| Drift cm | Bar ahead (+) or behind (-) midfoot at the bottom |

Velocity loss = drop from the fastest rep to the last. The RPE estimate uses group-average squat
velocities (in `squat_tracker/config.py`) until enough of your own sets are logged.

## Status

- Tested on a synthetic squatter with known depth, speed and fold (`python -m pytest tests`), including
  plate detection, scale, report and annotated video.
- **Not yet tested on real footage.** Expect to tune plate detection and thresholds on your first clips.

## Layout

```
squat_tracker/
  cli.py        argument parsing, orchestration
  pose.py       video reading, MediaPipe pose, side selection
  bar.py        plate detection (HoughCircles), scale, bar track with pose fallback
  analysis.py   rep segmentation and all metrics
  report.py     table, log line, JSON/CSV, history
  annotate.py   annotated video
  config.py     all thresholds
tests/          synthetic squatter + tests
```
