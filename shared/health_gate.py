"""P3 gates for one model (spec §5, §8): validity, official Health gate, latency, forecast equivariance.

    python shared/health_gate.py --baseline_dir variants/B/pitching_airfoil_wake/baseline \
        --plugins_dir variants/B/pitching_airfoil_wake/task/plugins --data_dir $DATA_DIR \
        --model <final_model.pt> --out runs/X/gates [--mirror_avg] [--device cuda|cpu]

Checks (all must pass for a valid candidate):
  * output shape [48, 96822] and every part finite
  * forecast_nmse <= 1 (shared/local_eval.score, matches the official scorer)
  * Health gate from declared/task_health.yaml, run with the REAL code: the task's view provider
    (_wake_health.WakePredictionViews: per-window temporal RMS of the 20 predicted frames) feeding
    the framework's SampleDispersionFloorCheck with min_dispersion read from the yaml; an
    independent numpy reimplementation must agree
  * median batch-1 forward latency <= 100 ms (eval, no_grad, 5 warm-up + 50 timed, CUDA synchronised,
    no host copies), with the device and any other GPU processes recorded
Diagnostic (reported, not a gate): forecast reflection equivariance error max|F(Gx) - S(F(x))|,
absolute and relative to max|F(x)|. For WakeMirrorAvg it should be at float32 round-off.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LATENCY_LIMIT_MS = 100.0


def read_health_yaml(plugins_dir: Path) -> dict:
    import yaml
    cfg = yaml.safe_load((plugins_dir.parent / "declared" / "task_health.yaml").read_text())
    (gate,) = [g for g in cfg["roster"] if g["check"] == "sample_dispersion_floor"]
    assert len(cfg["roster"]) == 1, "task_health.yaml roster changed; update this script"
    return gate


def official_health(forecast: np.ndarray, gate: dict) -> dict:
    """Run the task's view provider + framework check exactly as the gate does."""
    import _wake_health
    from execute_tools.health_checks.sample_dispersion_floor import SampleDispersionFloorCheck

    class Ctx:  # the provider only calls ctx.load_evaluation_payload()
        def load_evaluation_payload(self):
            return {"forecast": forecast}

    view = _wake_health.WakePredictionViews().materialize(_wake_health.CONTINUOUS_SAMPLES, Ctx())
    res = SampleDispersionFloorCheck().run(Ctx(), gate["parameters"], view=view)
    return {"gate_id": gate["gate_id"], "disposition": gate["disposition"], "passed": bool(res.passed),
            "verdict": str(res.verdict), "reason": res.reason,
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.metrics.items()},
            "samples": np.asarray(view.payload.samples, dtype=np.float64)}


def other_gpu_processes() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    import os
    return [line.strip() for line in out.splitlines() if line.strip() and line.split(",")[0].strip() != str(os.getpid())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=Path, required=True)
    ap.add_argument("--plugins_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mirror_avg", action="store_true")
    ap.add_argument("--device", choices=("cuda", "cpu"), default=None)
    a = ap.parse_args()
    sys.path[:0] = [str(a.plugins_dir.resolve()), str(a.baseline_dir.resolve()), str(HERE)]
    import torch
    import local_eval as le
    import run_baseline as rb

    rb.setup_torch()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cls = rb.WakeMirrorAvg if a.mirror_avg else rb.WakeBaseline
    model = cls()
    model.load_state_dict(rb.load_state(a.model))
    model.to(device).eval()

    t = le.read_validation(a.data_dir / "validation.npz")
    pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
    X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
    with torch.no_grad():
        flat = np.concatenate([model(X[s:s + 16].to(device)).float().cpu().numpy() for s in range(0, len(pairs), 16)])
        flat_g = np.concatenate([model(rb.mirror_x(X[s:s + 16].to(device))).float().cpu().numpy()
                                 for s in range(0, len(pairs), 16)])
    parts = {k: v.astype(np.float32).astype(np.float64) for k, v in le.unpack(flat).items()}
    finite = {k: bool(np.isfinite(v).all()) for k, v in parts.items()}
    nmse = le.score("forecast_nmse", dict(parts, pairs=pairs), t) if all(finite.values()) else float("nan")

    gate = read_health_yaml(a.plugins_dir)
    health = official_health(parts["forecast"], gate)
    rms = np.sqrt(np.mean(parts["forecast"].var(axis=1), axis=(1, 2)))           # independent reimplementation
    disp = float(np.std(rms, dtype=np.float64, ddof=0))
    assert np.allclose(rms, health["samples"], rtol=1e-5, atol=1e-6), "health view reimplementation disagrees"
    assert abs(disp - health["metrics"]["dispersion"]) < 1e-5 * max(disp, 1), "dispersion reimplementation disagrees"

    F = parts["forecast"]
    Fg = le.unpack(flat_g)["forecast"].astype(np.float64)
    eq_abs = float(np.abs(Fg - (-F[:, :, ::-1, :])).max())
    eq_rel = eq_abs / max(float(np.abs(F).max()), 1e-12)

    others = other_gpu_processes() if device.type == "cuda" else []
    x1 = X[:1].to(device)
    ts = []
    with torch.no_grad():
        for _ in range(5):
            model(x1)
        for _ in range(50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    lat = float(np.median(ts))

    checks = {
        "shape [48,96822]": flat.shape == (len(pairs), 96822),
        "all outputs finite": all(finite.values()),
        "forecast_nmse <= 1": bool(nmse <= 1),
        f"health {gate['gate_id']} ({gate['disposition']})": health["passed"],
        f"latency median <= {LATENCY_LIMIT_MS:g} ms": lat <= LATENCY_LIMIT_MS,
    }
    report = {
        "model": str(a.model), "class": cls.__name__, "windows": len(pairs), "forecast_nmse": nmse,
        "health": {k: v for k, v in health.items() if k != "samples"},
        "health_window_rms": {"min": float(rms.min()), "max": float(rms.max()), "mean": float(rms.mean())},
        "latency_ms": {"median": lat, "p90": float(np.percentile(ts, 90)), "n": len(ts),
                       "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "torch_threads": torch.get_num_threads(), "other_gpu_processes": others,
                       "note": "measured while other GPU jobs ran; re-measure on an idle GPU" if others else ""},
        "equivariance": {"max_abs": eq_abs, "max_rel": eq_rel},
        "checks": checks, "all_passed": all(checks.values()),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
