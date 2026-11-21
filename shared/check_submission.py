"""Check a submission package the way its manifest declares it (no SIDERIUS imports).

    python shared/check_submission.py --pkg submission/C_on_A --data_dir data [--expect_nmse 0.2562]

Loads submission.json -> plugin file + symbol (no-argument class) + weights, runs the 48 validation windows,
scores them with shared/local_eval.py (matches the official scorer), and checks shape, finiteness and
batch-1 latency (eval/no_grad, warm-up, plugin's own setup_torch thread setting).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import local_eval as le  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--expect_nmse", type=float)
    a = ap.parse_args()
    man = json.loads((a.pkg / "submission.json").read_text())
    for key in ("plugin", "symbol", "weights", "train_command", "train_dir", "seed", "kind", "initial_weights"):
        assert key in man, f"submission.json lacks {key}"
    for key in ("plugin", "weights", "initial_weights"):
        assert (a.pkg / man[key]).is_file(), f"{man[key]} missing"
    assert (a.pkg / man["train_dir"]).is_dir() and any((a.pkg / man["train_dir"]).rglob("*.py")), "train/ empty"

    spec = importlib.util.spec_from_file_location("submitted_plugin", a.pkg / man["plugin"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch
    assert not any(m.split(".")[0] in ("workflows", "execute_tools", "core") for m in sys.modules)
    device = mod.setup_torch()
    model = getattr(mod, man["symbol"])()  # no-argument constructor
    model.load_state_dict(mod.load_state(a.pkg / man["weights"]))
    model.to(device).eval()

    t = le.read_validation(a.data_dir / "validation.npz")
    pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
    X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
    with torch.no_grad():
        flat = np.concatenate([model(X[s:s + 16].to(device)).float().cpu().numpy() for s in range(0, len(pairs), 16)])
    shape_ok = flat.shape == (len(pairs), 96822)
    finite_ok = bool(np.isfinite(flat).all())
    parts = {k: v.astype(np.float32).astype(np.float64) for k, v in le.unpack(flat).items()}
    metrics = {n: le.score(n, dict(parts, pairs=pairs), t) for n in le.METRICS}

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
    checks = {"shape [48,96822]": shape_ok, "finite": finite_ok, "forecast_nmse<=1": metrics["forecast_nmse"] <= 1,
              "latency_median<100ms": lat < 100}
    if a.expect_nmse is not None:
        checks[f"forecast_nmse≈{a.expect_nmse}"] = abs(metrics["forecast_nmse"] - a.expect_nmse) < 5e-5
    print(json.dumps({"symbol": man["symbol"], "class": type(model).__name__, "device": str(device),
                      "torch_threads": torch.get_num_threads(), "metrics": metrics,
                      "latency_ms_median": round(lat, 2), "checks": checks}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
