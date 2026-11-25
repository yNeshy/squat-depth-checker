"""Tunable thresholds. Starting defaults; tune them against your own judged reps."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- calibration ---
    plate_diameter_cm: float = 45.0          # 20 kg / 25 kg competition-size plate
    nose_to_ankle_ratio: float = 0.88        # nose-to-ankle distance / standing height

    # --- depth (hip joint centre vs knee joint centre, cm, + = hip below knee) ---
    depth_pass_cm: float = 2.0               # margin >= this -> PASS
    depth_high_cm: float = -2.0              # margin < this  -> HIGH (between = BORDERLINE)

    # --- rep detection ---
    min_rep_travel_cm: float = 20.0          # bar must drop at least this far to count as a rep
    top_tolerance_cm: float = 6.0            # back within this of standing height = rep finished
    lockout_velocity_ms: float = 0.03        # concentric ends when bar slows below this near the top

    # --- velocity -> RPE (last-rep MCV, m/s). Group averages; replace with your own profile. ---
    # list of (min_mcv, rpe); first row whose min_mcv <= mcv wins
    rpe_map: list = field(default_factory=lambda: [
        (0.50, 7.0), (0.42, 8.0), (0.35, 9.0), (0.0, 10.0)])
    rpe_disagreement_flag: float = 1.0

    # --- technique flags ---
    fold_abs_flag_deg: float = 8.0           # torso opens/folds more than this out of the hole
    fold_rel_flag_deg: float = 5.0           # ...or this much more than the set median
    torso_rel_flag_deg: float = 5.0          # bottom torso angle this far from set median
    drift_flag_cm: float = 5.0               # bar this far from midfoot at the bottom
    hips_first_flag: float = 1.5             # hip rise / bar rise over first 10 cm

    # --- smoothing ---
    smooth_window_s: float = 0.15            # Savitzky-Golay window


def estimate_rpe(mcv: float, cfg: Config) -> float:
    for min_mcv, rpe in cfg.rpe_map:
        if mcv >= min_mcv:
            return rpe
    return 10.0
