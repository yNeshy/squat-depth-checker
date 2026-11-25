"""Small numpy-only signal helpers (no scipy dependency)."""
import numpy as np


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs; edges take the nearest valid value."""
    x = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def savgol_coeffs(window: int, order: int = 2, deriv: int = 0) -> np.ndarray:
    half = window // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vander(t, order + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    fact = float(np.prod(np.arange(1, deriv + 1))) if deriv else 1.0
    return pinv[deriv] * fact


def savgol(x: np.ndarray, window: int, order: int = 2, deriv: int = 0, dt: float = 1.0) -> np.ndarray:
    """Savitzky-Golay smoothing / derivative. window is forced odd and > order."""
    window = int(window)
    window = window if window % 2 else window + 1
    window = max(window, order + 2 if order % 2 else order + 3)
    if len(x) < window:
        window = len(x) if len(x) % 2 else len(x) - 1
        if window <= order:
            return np.gradient(x, dt) if deriv else np.asarray(x, float).copy()
    c = savgol_coeffs(window, order, deriv)[::-1]
    half = window // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    y = np.convolve(padded, c, mode="valid")
    return y / (dt ** deriv)
