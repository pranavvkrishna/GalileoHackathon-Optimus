"""Parameter-free forecast: future frame at time tau = S^m(history at tau - m*T/2), T = 2*pi/k, S = y-reflection
with sign flip (half-period spatio-temporal symmetry), m = smallest number of half periods that lands inside the
history window; linear interpolation in time between history frames. Also: full-period copy (no reflection)."""
import sys, math, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/team/person3/repo/shared"); import local_eval as le
def S(f): return -f[..., ::-1, :]
def forecast(vort, tf, s, k, mode):
    hist_t = tf[s-10:s]; hist = vort[s-10:s].astype(np.float64)
    T = 2 * math.pi / k; step = T / 2 if mode == "half" else T
    out = np.empty((20,) + hist.shape[1:])
    for j in range(20):
        tau = tf[s + j]; m = math.ceil((tau - hist_t[-1]) / step - 1e-9)
        src = tau - m * step
        if src < hist_t[0] - 1e-9: return None          # period too long for this history
        i = min(max(np.searchsorted(hist_t, src) - 1, 0), 8); w = (src - hist_t[i]) / (hist_t[i+1] - hist_t[i])
        f = (1 - w) * hist[i] + w * hist[i+1]
        out[j] = S(f) if (mode == "half" and m % 2 == 1) else f
    return out
def evaluate(path, sel_cases=None, name=""):
    t = le.read_validation(Path(path)) if "validation" in path else None
    return t
