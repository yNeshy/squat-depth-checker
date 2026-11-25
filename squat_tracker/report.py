"""Terminal table, JSON/CSV files, paste-ready log line, history file."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from .analysis import SetResult
from .pose import HOME

COLS = [
    ("Rep", "n", "{}"),
    ("Depth", "depth", "{}"),
    ("Margin cm", "depth_margin_cm", "{:+.1f}"),
    ("MCV m/s", "mcv_ms", "{:.2f}"),
    ("Up s", "concentric_s", "{:.2f}"),
    ("Slowest m/s", "min_v_ms", "{:.2f}"),
    ("Torso deg", "torso_bottom_deg", "{:.0f}"),
    ("Fold deg", "fold_deg", "{:.0f}"),
    ("Hips 1st", "hips_first_ratio", "{:.2f}"),
    ("Drift cm", "drift_bottom_cm", "{:+.1f}"),
    ("Flags", "flags", "{}"),
]


def _fmt(fmt, v):
    if isinstance(v, list):
        return ", ".join(v) or "-"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def table(res: SetResult) -> str:
    rows = [[h for h, _, _ in COLS]]
    for r in res.reps:
        d = asdict(r)
        rows.append([_fmt(f, d[k]) for _, k, f in COLS])
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLS))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    out = "\n".join(lines)
    summary = [
        f"Scale: {res.px_per_cm:.2f} px/cm from {res.scale_source}; side filmed: {res.side}; "
        f"plate tracked in {res.plate_coverage * 100:.0f}% of frames",
    ]
    if res.reps:
        summary.append(f"Velocity loss: {res.velocity_loss_pct:.0f}%   Bar-speed RPE estimate: {res.est_rpe:g}"
                       + (f"   Logged RPE: {res.logged_rpe:g}" if res.logged_rpe is not None else ""))
    summary += [f"! {w}" for w in res.warnings]
    return out + "\n\n" + "\n".join(summary)


def log_line(res: SetResult) -> str:
    reps = res.reps
    if not reps:
        return "SQ | no reps detected"
    head = "SQ"
    if res.load_kg:
        head += f" {res.load_kg:g}x{len(reps)}"
    if res.logged_rpe is not None:
        head += f" @{res.logged_rpe:g}"
    ok = sum(r.depth == "PASS" for r in reps)
    parts = [
        head,
        f"depth {ok}/{len(reps)} ({', '.join(f'{round(r.depth_margin_cm) + 0:+d}' for r in reps)} cm)",
        "MCV " + ", ".join(f"{r.mcv_ms:.2f}" for r in reps),
        f"VL {res.velocity_loss_pct:.0f}%",
        f"est RPE {res.est_rpe:g}",
        "fold " + ", ".join(f"{r.fold_deg:.0f}" for r in reps) + " deg",
    ]
    flagged = [f"rep {r.n}: {', '.join(r.flags)}" for r in reps if r.flags]
    if flagged:
        parts.append("; ".join(flagged))
    return " | ".join(parts)


def write_files(res: SetResult, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    jp = Path(f"{out_base}.json")
    jp.write_text(json.dumps(res.to_dict(), indent=2))
    cp = Path(f"{out_base}.csv")
    with cp.open("w", newline="") as f:
        if res.reps:
            w = csv.DictWriter(f, fieldnames=list(asdict(res.reps[0]).keys()))
            w.writeheader()
            for r in res.reps:
                d = asdict(r)
                d["flags"] = "; ".join(d["flags"])
                w.writerow(d)
    return [jp, cp]


def append_history(res: SetResult, video: Path) -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    hp = HOME / "history.jsonl"
    rec = dict(
        analysed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), video=str(video), load_kg=res.load_kg,
        logged_rpe=res.logged_rpe, est_rpe=res.est_rpe, velocity_loss_pct=res.velocity_loss_pct,
        reps=[dict(mcv=r.mcv_ms, depth=r.depth, margin=r.depth_margin_cm, fold=r.fold_deg,
                   torso=r.torso_bottom_deg) for r in res.reps])
    with hp.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return hp


def show_history() -> str:
    hp = HOME / "history.jsonl"
    if not hp.exists():
        return "No history yet. Run `squat-track analyse` with --load and --rpe to start one."
    lines = ["Date                 Load  Reps  Logged  Est  First MCV  Last MCV  Depth"]
    for raw in hp.read_text().splitlines():
        r = json.loads(raw)
        reps = r["reps"]
        if not reps:
            continue
        lines.append(
            f"{r['analysed_at']:<20} {r['load_kg'] or '-':>5} {len(reps):>5} "
            f"{r['logged_rpe'] if r['logged_rpe'] is not None else '-':>7} {r['est_rpe']:>4} "
            f"{reps[0]['mcv']:>10.2f} {reps[-1]['mcv']:>9.2f}  "
            f"{sum(x['depth'] == 'PASS' for x in reps)}/{len(reps)}")
    return "\n".join(lines)
