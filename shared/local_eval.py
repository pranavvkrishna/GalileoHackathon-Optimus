"""Local reimplementation of the official pitching-wake validation scoring (no SIDERIUS imports).

    python shared/local_eval.py --baseline_dir variants/B/pitching_airfoil_wake/baseline \
        --data_dir data --model original/official/final_model.pt --out runs/X [--mirror_avg]

Everything here is copied from the release (20261121-v3), with framework plumbing removed:
  * window list      = WakeTable.windows(training=False)    task/plugins/_wake_task.py:134-137
  * inputs           = WakeTable.features                   task/plugins/_wake_task.py:141-150
  * table dtypes     = read_table casts                     task/plugins/_wake_task.py:202-208
  * output split     = unpack                               task/plugins/_wake_task.py:83-102
  * inference        = write_predictions (batches of 16, float32)  baseline/run_baseline.py:196-210
  * metrics          = _softmax, window_nmse, case_pool, balanced_accuracy, score
                                                            task/plugins/_wake_metrics.py:32-107
Results are labelled "local reimplementation", not official scores.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HISTORY, HORIZON, NY, NX, N_BINS, N_REGIMES = 10, 20, 48, 96, 50, 3
FIELD = NY * NX
NMSE_CLIP = 4.0  # _wake_metrics.py:32


# ---------------------------------------------------------------- data (_wake_task.py)
def read_validation(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        t = {name: d[name] for name in ("case_id", "k", "theta0_deg", "t_frames", "vort", "mean_vort", "log_psd",
                                        "st_dom", "regime", "win_case", "win_start")}
    # dtype casts as in read_table, _wake_task.py:202-208
    t["case_ids"] = tuple(str(v) for v in t.pop("case_id"))
    for name in ("k", "theta0_deg", "t_frames", "mean_vort", "log_psd", "st_dom"):
        t[name] = t[name].astype(np.float32)
    t["regime"] = t["regime"].astype(np.int64)
    t["win_case"] = t["win_case"].astype(np.int64)
    t["win_start"] = t["win_start"].astype(np.int64)
    return t


def features(t: dict, case: int, start: int) -> np.ndarray:  # _wake_task.py:141-150
    x = np.empty((HISTORY + 4, NY, NX), dtype=np.float32)
    x[:HISTORY] = t["vort"][case, start - HISTORY: start].astype(np.float32)
    k = float(t["k"][case])
    ts = float(t["t_frames"][start])
    x[HISTORY] = k / (2 * math.pi)
    x[HISTORY + 1] = math.radians(float(t["theta0_deg"][case]))
    x[HISTORY + 2] = math.sin(k * ts)
    x[HISTORY + 3] = math.cos(k * ts)
    return x


def unpack(out: np.ndarray) -> dict:  # _wake_task.py:83-102
    lead = out.shape[:-1]
    a = HORIZON * FIELD
    b = a + FIELD
    c = b + N_BINS
    return {"forecast": out[..., :a].reshape(*lead, HORIZON, NY, NX), "mean_vort": out[..., a:b].reshape(*lead, NY, NX),
            "log_psd": out[..., b:c], "st_dom": out[..., c], "regime_logits": out[..., c + 1: c + 1 + N_REGIMES]}


# ---------------------------------------------------------------- metrics (_wake_metrics.py)
def _softmax(z):  # :35-38
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def window_nmse(pred, t):  # :58-67, [W, 20] unclipped
    rows = []
    for w, (c, s) in enumerate(pred["pairs"]):
        truth = t["vort"][c, s: s + pred["forecast"].shape[1]].astype(np.float64)
        ref = t["mean_vort"][c].astype(np.float64)
        var = np.mean((truth - ref) ** 2)
        err = np.mean((pred["forecast"][w] - truth) ** 2, axis=(1, 2))
        rows.append(err / max(var, 1e-12))
    return np.asarray(rows)


def case_pool(pred):  # :70-81
    pooled = {}
    cases = [c for c, _ in pred["pairs"]]
    for c in sorted(set(cases)):
        idx = [i for i, cc in enumerate(cases) if cc == c]
        pooled[c] = {"mean_vort": pred["mean_vort"][idx].mean(axis=0), "log_psd": pred["log_psd"][idx].mean(axis=0),
                     "st_dom": float(np.median(pred["st_dom"][idx])),
                     "regime_prob": _softmax(pred["regime_logits"][idx]).mean(axis=0)}
    return pooled


def balanced_accuracy(true, pred):  # :84-86
    return float(np.mean([float(np.mean(pred[true == c] == c)) for c in np.unique(true)]))


def score(name, pred, t):  # :89-107
    if name in ("forecast_nmse", "forecast_nmse_final"):
        e = np.minimum(window_nmse(pred, t), NMSE_CLIP)  # clip each lead, then average
        return float(e.mean()) if name == "forecast_nmse" else float(e[:, -1].mean())
    pooled = case_pool(pred)
    cases = sorted(pooled)
    if name == "mean_field_rel_l2":
        return float(np.mean([np.linalg.norm(pooled[c]["mean_vort"] - t["mean_vort"][c])
                              / max(np.linalg.norm(t["mean_vort"][c]), 1e-12) for c in cases]))
    if name == "psd_log_rmse":
        return float(np.mean([np.sqrt(np.mean((pooled[c]["log_psd"] - t["log_psd"][c]) ** 2)) for c in cases]))
    if name == "st_abs_error":
        return float(np.mean([abs(pooled[c]["st_dom"] - float(t["st_dom"][c])) for c in cases]))
    if name == "regime_balanced_accuracy":
        tr = np.asarray([int(t["regime"][c]) for c in cases])
        p = np.asarray([int(np.argmax(pooled[c]["regime_prob"])) for c in cases])
        return balanced_accuracy(tr, p)
    raise KeyError(name)


METRICS = ("forecast_nmse", "forecast_nmse_final", "mean_field_rel_l2", "psd_log_rmse", "st_abs_error",
           "regime_balanced_accuracy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mirror_avg", action="store_true")
    a = ap.parse_args()
    started = time.perf_counter()

    sys.path.insert(0, str(a.baseline_dir.resolve()))
    import torch
    import run_baseline as rb  # framework imports in run_baseline live inside functions only
    assert not any(m.split(".")[0] in ("workflows", "execute_tools", "core") for m in sys.modules), "framework imported"

    device = rb.setup_torch()
    cls = rb.WakeMirrorAvg if a.mirror_avg else rb.WakeBaseline
    model = cls()
    model.load_state_dict(rb.load_state(a.model))
    model.to(device).eval()

    t = read_validation(a.data_dir / "validation.npz")
    pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]  # _wake_task.py:136-137
    X = torch.from_numpy(np.stack([features(t, c, s) for c, s in pairs]))

    outs, outs_g = [], []
    with torch.no_grad():  # run_baseline.py:202-205: batches of 16, float32 to numpy
        for s in range(0, len(pairs), 16):
            xb = X[s: s + 16].to(device)
            outs.append(model(xb).float().cpu().numpy())
            outs_g.append(model(rb.mirror_x(xb)).float().cpu().numpy())
    flat = np.concatenate(outs)
    flat_g = np.concatenate(outs_g)
    # deliverable stores float32 parts, metrics read float64 (_wake_task.py:369-374, _wake_metrics.py:48-49)
    parts = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in unpack(flat).items()}
    for name, value in parts.items():
        if not np.isfinite(value).all():
            raise ValueError(f"predicted {name} contains NaN or infinity")
    pred = dict(parts, pairs=pairs)

    metrics = {name: score(name, pred, t) for name in METRICS}

    # diagnostics
    lead = window_nmse(pred, t)  # [48, 20] unclipped
    F = parts["forecast"]
    Fg = unpack(flat_g)["forecast"].astype(np.float64)
    SF = -F[:, :, ::-1, :]
    defect = ((Fg - SF) ** 2).sum((1, 2, 3)) / np.maximum((F ** 2).sum((1, 2, 3)) + (Fg ** 2).sum((1, 2, 3)), 1e-12)
    case = np.array([c for c, _ in pairs])
    start = np.array([s for _, s in pairs])
    win = np.minimum(lead, NMSE_CLIP).mean(1)
    case_nmse = {t["case_ids"][c]: float(win[case == c].mean()) for c in np.unique(case)}
    case_defect = {t["case_ids"][c]: float(defect[case == c].mean()) for c in np.unique(case)}

    # latency: batch 1, eval/no_grad, after warmup, full forward only
    x1 = X[:1].to(device)
    ts = []
    with torch.no_grad():
        for _ in range(5):
            model(x1)
        for _ in range(20):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "diag.npz", case=case, case_id=np.array([t["case_ids"][c] for c in case]), start=start,
             lead_nmse=lead, defect=defect, k=t["k"][case], theta0_deg=t["theta0_deg"][case], regime=t["regime"][case])
    report = {
        "provenance": "local reimplementation (shared/local_eval.py), not the official SIDERIUS scorer",
        "model": str(a.model), "class": cls.__name__, "device": str(device), "windows": len(pairs),
        "metrics": metrics,
        "reflection_defect": float(np.mean(list(case_defect.values()))),
        "latency_ms_median": float(np.median(ts)),
        "run_seconds": round(time.perf_counter() - started, 1),
        "case_nmse": case_nmse, "case_defect": case_defect,
    }
    (a.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("class", "metrics", "reflection_defect", "latency_ms_median",
                                             "run_seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
