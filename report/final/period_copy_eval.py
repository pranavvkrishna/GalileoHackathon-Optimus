"""Period-copy evidence on training windows (CPU, no fitting). Mirrors P3's WakeHybrid copy and residual.

    python period_copy_eval.py --train <train.npz> --variant <variants/Bplus/pitching_airfoil_wake> \
        --cnn <Bplus_s0 final_model.pt> --out period_copy_data.npz

Part 1 (lengthened history): every training case with k in [3.5, 7], windows starting at frames 60/160/260/360
(the validation start frames), history = the last L frames with (L-1)*dt >= T = 2*pi/k. Copy: future frame at
history-clock time tau = span + j*dt equals the history at tau - m*T (m = smallest whole number of periods that lands
inside the history), linear interpolation in time. Residual r: mean over i (i*dt - T >= 0) of
mean((h_i - h(i*dt - T))^2), divided by the history's variance about its time mean. NMSE: official per-lead NMSE
(variance of the true frames about the case mean field), clipped at 4, averaged over 20 leads.
Part 2 (deployed 10-frame history): training windows with k >= 6.98 at the same starts; copy vs the mirror-averaged
B+ network (C+); example = the gated window (r < 0.3) with the median r, lead 20.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

H, HZ, DT, CLIP = 10, 20, 0.1, 4.0
STARTS = (60, 160, 260, 360)


def sample(hist, t):
    u = min(max(t / DT, 0.0), len(hist) - 1.0)
    i = min(int(math.floor(u)), len(hist) - 2)
    w = u - i
    return (1 - w) * hist[i] + w * hist[i + 1]


def copy_and_residual(hist, T):
    span = (len(hist) - 1) * DT
    errs = [np.mean((hist[i] - sample(hist, i * DT - T)) ** 2) for i in range(len(hist)) if i * DT - T >= -1e-9]
    r = float(np.mean(errs) / max(np.mean((hist - hist.mean(0)) ** 2), 1e-12))
    out = []
    for j in range(HZ):
        tau = span + (j + 1) * DT
        m = math.ceil((tau - span) / T - 1e-9)
        out.append(sample(hist, tau - m * T))
    return np.stack(out), r


def nmse(pred, truth, mean_field):
    var = max(float(np.mean((truth - mean_field) ** 2)), 1e-12)
    lead = np.mean((pred - truth) ** 2, axis=(1, 2)) / var
    return np.minimum(lead, CLIP)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.train, allow_pickle=False) as d:
        k, th, reg = d["k"].astype(np.float64), d["theta0_deg"].astype(np.float64), d["regime"].astype(int)
        tf, mv, vort = d["t_frames"].astype(np.float64), d["mean_vort"].astype(np.float64), d["vort"]
    assert np.allclose(np.diff(tf), DT, atol=1e-4)
    rows = []
    for c in np.flatnonzero((k >= 3.5) & (k <= 7.0)):
        T = 2 * math.pi / k[c]
        L = int(math.ceil(T / DT - 1e-9)) + 1
        for s in STARTS:
            if s - L < 0 or s + HZ > vort.shape[1]:
                continue
            hist = vort[c, s - L:s].astype(np.float64)
            truth = vort[c, s:s + HZ].astype(np.float64)
            pred, r = copy_and_residual(hist, T)
            rows.append((c, s, k[c], th[c], reg[c], r, float(nmse(pred, truth, mv[c]).mean())))
    R = np.array(rows, dtype=np.float64)
    gate = R[:, 5] < 0.3
    summary = {"windows": int(len(R)), "cases": int(len(np.unique(R[:, 0]))), "gate_r_lt_0.3": int(gate.sum()),
               "copy_nmse_gated_mean": float(R[gate, 6].mean()), "copy_nmse_rejected_mean": float(R[~gate, 6].mean()),
               "gated_above_0.5": int((R[gate, 6] > 0.5).sum()), "gated_max": float(R[gate, 6].max())}
    per_reg = {}
    for g, name in enumerate(("periodic", "quasi-periodic", "aperiodic")):
        m = R[:, 4] == g
        per_reg[name] = {"n": int(m.sum()), "pass_rate": float(gate[m].mean()) if m.any() else None,
                         "copy_nmse_mean": float(R[m, 6].mean()) if m.any() else None}
    bins = np.arange(3.5, 7.01, 0.5)
    per_k = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (R[:, 2] >= lo) & (R[:, 2] < hi + (1e-9 if hi >= 7 else 0)) & (R[:, 4] == 0)
        per_k.append({"k_bin": [float(lo), float(hi)], "periodic_n": int(m.sum()),
                      "periodic_copy_nmse": float(R[m, 6].mean()) if m.any() else None})
    summary["by_regime"] = per_reg
    summary["periodic_by_k_bin"] = per_k

    # Part 2: deployed 10-frame history, k >= 6.98, copy vs C+ (mirror-averaged B+), CPU
    sys.path.insert(0, str(a.variant / "baseline"))
    spec = importlib.util.spec_from_file_location("rb", a.variant / "baseline/run_baseline.py")
    rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
    import torch
    torch.set_num_threads(1)
    net = rb.WakeMirrorAvg(); net.load_state_dict(rb.load_state(a.cnn)); net.eval()
    dep = []
    for c in np.flatnonzero(k >= 6.98):
        T = 2 * math.pi / k[c]
        for s in STARTS:
            hist = vort[c, s - H:s].astype(np.float64)
            truth = vort[c, s:s + HZ].astype(np.float64)
            pred, r = copy_and_residual(hist, T)
            x = np.empty((14, 48, 96), np.float32)
            x[:H] = vort[c, s - H:s].astype(np.float32)
            x[H], x[H + 1] = k[c] / (2 * math.pi), math.radians(th[c])
            x[H + 2], x[H + 3] = math.sin(k[c] * tf[s]), math.cos(k[c] * tf[s])
            with torch.no_grad():
                f = net(torch.from_numpy(x)[None])[0, :HZ * 4608].reshape(HZ, 48, 96).double().numpy()
            dep.append((c, s, k[c], reg[c], r, float(nmse(pred, truth, mv[c]).mean()), float(nmse(f, truth, mv[c]).mean()),
                        truth[-1], f[-1], pred[-1]))
    D = np.array([d[:7] for d in dep], dtype=np.float64)
    g2 = D[:, 4] < 0.3
    hyb = np.where(g2, D[:, 5], D[:, 6])
    summary["deployed_k_ge_6.98"] = {"windows": int(len(D)), "gated": int(g2.sum()),
                                     "copy_nmse_gated_mean": float(D[g2, 5].mean()) if g2.any() else None,
                                     "cnn_nmse_all_mean": float(D[:, 6].mean()), "hybrid_nmse_all_mean": float(hyb.mean()),
                                     "cnn_on_gated_mean": float(D[g2, 6].mean()) if g2.any() else None}
    gi = np.flatnonzero(g2)
    ex = int(gi[np.argsort(D[gi, 4])[len(gi) // 2]])
    summary["example"] = {"case_index": int(D[ex, 0]), "start": int(D[ex, 1]), "k": float(D[ex, 2]),
                          "regime": int(D[ex, 3]), "r": float(D[ex, 4]), "copy_nmse": float(D[ex, 5]),
                          "cnn_nmse": float(D[ex, 6]), "rule": "gated (r<0.3) deployed window with the median r"}
    np.savez_compressed(a.out, part1=R, part1_cols=np.array(["case", "start", "k", "theta0_deg", "regime", "r", "copy_nmse"]),
                        part2=D, part2_cols=np.array(["case", "start", "k", "regime", "r", "copy_nmse", "cnn_nmse"]),
                        example_fields=np.stack([dep[ex][7], dep[ex][8], dep[ex][9]]))
    Path(str(a.out).replace(".npz", ".json")).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
