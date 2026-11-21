"""P3 acceptance check (spec §5): the forecast metric clips each lead at 4, THEN averages.

    python shared/test_clip_order.py --plugins_dir variants/B/pitching_airfoil_wake/task/plugins \
        [--diag runs/A_official/local_eval/diag.npz --metrics runs/A_official/metrics.json]

1. Synthetic table where every per-lead NMSE is known exactly:
     window 0: leads 1-10 NMSE 5, leads 11-20 NMSE 0 -> clip-then-mean 2.0, mean-then-clip 2.5
     window 1: all leads NMSE 1                     -> 1.0 either way
   official forecast_nmse must be (2.0 + 1.0)/2 = 1.5 (wrong order would give 1.75);
   forecast_nmse_final must be (0 + 1)/2 = 0.5.
   Checked against the OFFICIAL _wake_metrics.score and against shared/local_eval.score.
2. Optional: real diag.npz lead errors, clipped per lead, reproduce the official metrics.json value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
H, NY, NX = 20, 48, 96


def synthetic():
    rng = np.random.default_rng(0)
    sign = rng.choice([-1.0, 1.0], size=(2, 30, NY, NX))       # truth^2 == 1 everywhere -> variance about 0 is 1
    table = SimpleNamespace(case_ids=("c0", "c1"), vort=sign.astype(np.float16), mean_vort=np.zeros((2, NY, NX), np.float32))
    pairs = [(0, 10), (1, 10)]
    target = np.array([[5.0] * 10 + [0.0] * 10, [1.0] * 20])   # [W, 20] per-lead NMSE
    truth = np.stack([table.vort[c, s:s + H].astype(np.float64) for c, s in pairs])
    forecast = truth + np.sqrt(target)[:, :, None, None]       # constant offset per lead -> MSE == target
    return table, pairs, target, forecast


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins_dir", type=Path, required=True)
    ap.add_argument("--diag", type=Path)
    ap.add_argument("--metrics", type=Path)
    a = ap.parse_args()
    sys.path[:0] = [str(a.plugins_dir.resolve()), str(HERE)]
    import _wake_metrics as official
    import local_eval as le

    # sanity of the spec's two-element example
    assert np.minimum([[5.0, 0.0]], 4.0).mean() == 2.0
    assert min(np.mean([5.0, 0.0]), 4.0) == 2.5

    table, pairs, target, forecast = synthetic()
    pred = {"forecast": forecast, "pairs": pairs}
    lead = official.window_nmse(pred, table)
    assert np.allclose(lead, target, atol=1e-9), np.abs(lead - target).max()

    t_local = {"vort": table.vort, "mean_vort": table.mean_vort}
    results = {}
    for name in ("forecast_nmse", "forecast_nmse_final"):
        off = official.score(name, pred, table)
        loc = le.score(name, pred, t_local)
        results[name] = {"official": off, "local_eval": loc}
        assert abs(off - loc) < 1e-12, (name, off, loc)
    assert abs(results["forecast_nmse"]["official"] - 1.5) < 1e-9, results       # clip-then-mean
    assert abs(results["forecast_nmse_final"]["official"] - 0.5) < 1e-9, results
    wrong = float(np.minimum(target.mean(1), 4.0).mean())
    assert abs(wrong - 1.75) < 1e-9                                              # the order the spec forbids
    print("synthetic:", json.dumps(results), "| wrong order would give", wrong)

    if a.diag:
        d = np.load(a.diag)
        mine = float(np.minimum(d["lead_nmse"], 4.0).mean())
        mine_final = float(np.minimum(d["lead_nmse"], 4.0)[:, -1].mean())
        ref = json.loads(a.metrics.read_text())
        ref = ref.get("metrics", ref)
        dn, df = abs(mine - ref["forecast_nmse"]), abs(mine_final - ref["forecast_nmse_final"])
        print(f"real diag: forecast_nmse {mine:.8f} vs {ref['forecast_nmse']:.8f} (|d|={dn:.1e}); "
              f"final {mine_final:.8f} vs {ref['forecast_nmse_final']:.8f} (|d|={df:.1e})")
        assert dn < 1e-6 and df < 1e-6
    print("CLIP ORDER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
