# Plan: video review agent for squat-depth-checker

Goal: an agent watches a squat video (local file before upload, or a YouTube link) and writes a short
comment on depth, bar speed, RPE (bar-speed estimate vs logged) and technique flags.
Squat only for now. Deadlift is a possible later step.

Status: plan only. No code has been changed yet; each step waits for a go-ahead.

## Step 0. Fix rep detection (do this first)

Why: on the first real clip (190 kg x 2, 23.11.2026) the output does not look right:

| Rep | Descent s | Bar travel cm | MCV m/s | Slowest m/s |
| --- | --- | --- | --- | --- |
| 1 | 9.42 | 66.2 | 0.235 | -1.64 |
| 2 | 5.42 | 34.6 | 0.121 | -0.28 |

- A 9.4 s descent means the walkout and setup were counted inside rep 1.
- Rep 2 travels about half as far as rep 1, so its start or end is cut in the wrong place
  (rep 2 starts at the exact frame rep 1 ends, so the between-reps pause is not separated).
- "Slowest" speeds are negative, so the concentric window includes the bar still going down.
- Result: MCV is too low, so the bar-speed RPE estimate (10) is not trustworthy yet.

Changes in `analysis.py`:
1. Descent start = last frame before the bottom where downward bar speed crosses a small threshold
   (e.g. -0.05 m/s) after a still period, not "last frame near the pre-rep height".
2. Concentric start = first frame after the bottom where bar speed turns positive and stays positive;
   concentric end = first frame back near standing height where speed drops below the lockout threshold.
3. Standing height per rep (local max between reps), not one 90th percentile for the whole clip,
   so a walkout or re-rack does not shift the reference.
4. Clamp the sticking-point search to frames with positive velocity.
5. Sanity checks that add a warning: descent > 4 s, concentric > 5 s, travel differs between reps by > 20%.
6. Add a synthetic test with a walkout and a pause between reps; re-check the real clip by hand
   against frame numbers from the annotated video.

## Step 1. File or YouTube input

- New command: `squat-track review FILE_OR_URL [--load KG] [--rpe RPE] [--json] [--comment]`.
- URL: download with `yt-dlp` (added to dependencies) into a new empty folder per video,
  `~/.squat-track/downloads/VIDEO_ID/`. Limits: best mp4 up to 1080p, max ~3 min / 500 MB.
  The downloaded file is only passed as a path argument; nothing from it is executed.
- Local file: analysed in place, as today.

## Step 2. Load and logged RPE

The video does not carry them. Sources, highest priority first:
1. CLI flags `--load` and `--rpe`.
2. Sidecar text file next to the video (`NAME.lift.txt`) with a line like `SQ 190x2 @9`.
3. For a YouTube link: the same pattern in the video title or description.
If none is found, the comment gives the bar-speed estimate only and says the logged RPE is missing.

## Step 3. Machine-readable summary and comment

- `--json`: one JSON object on stdout (no table), for agents. Fields: source, load_kg, reps,
  logged_rpe, est_rpe, rpe_delta, velocity_loss_pct, per-rep {depth, margin_cm, mcv_ms, slowest_ms,
  flags}, warnings, confidence (plate coverage, scale source).
- `--comment`: 3-6 plain-language lines, for example:

  > 190 kg x 2 @9 logged. Depth: rep 1 borderline (+1 cm), rep 2 deep (+11 cm).
  > Bar speed 0.42 then 0.33 m/s, 21% velocity loss. Bar-speed estimate RPE 9.5, matches your 9.
  > Rep 1: torso folded 11 deg out of the hole and bar 14 cm ahead of midfoot at the bottom.
  > Cue: sit 2-3 cm deeper on rep 1 and keep the chest up off the floor of the squat.

  (Numbers above are illustrative, not from a real run.)
- The comment states low confidence when the plate was tracked in less than 30% of frames.

## Step 4. Agent README

`docs/AGENT.md`: install, exact commands, JSON schema, exit codes, and the rule that the tool
only prints text.

Posting to YouTube is out of scope. If wanted later it needs: a Google Cloud project with the
YouTube Data API v3 enabled, an OAuth consent screen, an OAuth client, and a user token with the
`youtube.force-ssl` scope; then `commentThreads.insert` on the video. That would be a separate,
opt-in command.

## Later

- Personal velocity profile from `~/.squat-track/history.jsonl` replacing the group-average RPE map.
- Deadlift support.
