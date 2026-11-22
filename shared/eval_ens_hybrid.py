"""Seed-1 model and a 2-seed WakeHybrid ensemble vs the submitted C_from_Bplus_hybrid (#95). Mac CPU only.

    python shared/eval_ens_hybrid.py --hybrid_plugin ~/src/ref_hybrid/pkg/run_baseline.py \
        --ref_weights ~/src/ref_hybrid/pkg/final_model.pt --seed1 runs/Bplus_s1b/final_model.pt \
        --variant_dir variants/Bplus/pitching_airfoil_wake --data_dir data --out runs/ens_hybrid_s01

Ensemble = P3's WakeHybrid code, unchanged (period copy, distance-scaled OOD shrink, its K_RANGE), with
one class inserted below WakeMirrorAvgOOD in the MRO: the mirror-averaged forecast becomes the mean over
seeds {0, 1} of 0.5 * (F_m(x) + S F_m(G x)); statistics head = seed 0 (the module's own statistics()).
Seed 0 = the #95 weights. Official scores: run_baseline.evaluate_model (SIDERIUS metrics). Paired
bootstrap: per-window lead NMSE from shared/local_eval.py (checked equal to the official forecast_nmse).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_ensemble(hy, torch):
    class _EnsMirrorAvg(hy.WakeMirrorAvg):
        """Replaces WakeMirrorAvg.forward: mean over members of the mirror-averaged forecast."""

        def forward(self, x):
            fut = torch.stack([0.5 * (hy.WakeBaseline.forecast(m, x)
                                      + hy.mirror_field(hy.WakeBaseline.forecast(m, hy.mirror_x(x))))
                               for m in (self, self.m1)]).mean(0)
            return torch.cat([fut.flatten(1), self.statistics(x)], dim=1)

    class WakeHybridEnsemble2(hy.WakeHybrid, _EnsMirrorAvg):
        def __init__(self):
            super().__init__()
            self.m1 = hy.WakeBaseline()   # seed-1 member; seed 0 = self

    mro = [c.__name__ for c in WakeHybridEnsemble2.__mro__]
    assert mro[:6] == ["WakeHybridEnsemble2", "WakeHybrid", "WakeMirrorAvgOOD", "_EnsMirrorAvg", "WakeMirrorAvg",
                       "WakeBaseline"], mro
    return WakeHybridEnsemble2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid_plugin", type=Path, required=True)
    ap.add_argument("--ref_weights", type=Path, required=True)
    ap.add_argument("--seed1", type=Path, required=True)
    ap.add_argument("--variant_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path[:0] = [str((a.variant_dir / "task" / "plugins").resolve()), str(HERE)]
    import torch
    import local_eval as le
    import health_gate as hg
    from paired_stats import paired

    vb = load(a.variant_dir / "baseline" / "run_baseline.py", "rb_variant")      # official scoring path
    hy = load(a.hybrid_plugin, "rb_hybrid")                                       # P3 code, unchanged
    composition = vb.compose_task()
    dev = torch.device("cpu")
    sd0, sd1 = hy.load_state(a.ref_weights), hy.load_state(a.seed1)

    def build(kind):
        if kind == "ens":
            m = make_ensemble(hy, torch)()
            m.load_state_dict(sd0, strict=False)
            m.m1.load_state_dict(sd1)
            missing = [k for k in m.state_dict() if not k.startswith("m1.") and k not in sd0]
            assert not missing, missing
            return m.eval()
        cls, sd = {"ref": (hy.WakeHybrid, sd0), "s1": (hy.WakeBaseline, sd1), "s1_mirror": (hy.WakeMirrorAvg, sd1)}[kind]
        m = cls()
        m.load_state_dict(sd)
        return m.eval()

    names = {"ref": "C_from_Bplus_hybrid (#95, seed 0)", "s1": "Bplus_s1b", "s1_mirror": "C_from_Bplus_s1b",
             "ens": "ens_hybrid_seeds01"}
    t = le.read_validation(a.data_dir / "validation.npz")
    pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
    case = np.array([c for c, _ in pairs])
    X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
    gate = hg.read_health_yaml(a.variant_dir / "task" / "plugins")

    res, lead = {}, {}
    for kind, name in names.items():
        model = build(kind)
        torch.set_num_threads(os.cpu_count())
        outdir = a.out / kind
        official = vb.evaluate_model(model, composition, a.data_dir, outdir, dev)  # writes deliverable + metrics.json
        with torch.no_grad():
            flat = np.concatenate([model(X[s:s + 16]).float().numpy() for s in range(0, len(pairs), 16)])
            flat_g = np.concatenate([model(hy.mirror_x(X[s:s + 16])).float().numpy() for s in range(0, len(pairs), 16)])
        parts = {k: v.astype(np.float32).astype(np.float64) for k, v in le.unpack(flat).items()}
        pred = dict(parts, pairs=pairs)
        lead[kind] = np.minimum(le.window_nmse(pred, t), le.NMSE_CLIP)
        local_nmse = float(lead[kind].mean())
        health = hg.official_health(parts["forecast"], gate)
        F = parts["forecast"]
        Fg = le.unpack(flat_g)["forecast"].astype(np.float64)
        eq = float(np.abs(Fg + F[:, :, ::-1, :]).max())
        lat = {}
        for threads in (1, os.cpu_count()):
            torch.set_num_threads(threads)
            ts = []
            with torch.no_grad():
                for _ in range(5):
                    model(X[:1])
                for _ in range(30):
                    t0 = time.perf_counter()
                    model(X[:1])
                    ts.append((time.perf_counter() - t0) * 1e3)
            lat[f"cpu_{threads}thr_median_ms"] = round(float(np.median(ts)), 2)
        checks = {"shape": flat.shape == (len(pairs), 96822), "finite": bool(np.isfinite(flat).all()),
                  "forecast_nmse<=1": official["forecast_nmse"] <= 1, "health_" + gate["gate_id"]: bool(health["passed"]),
                  "latency<=100ms (1 CPU thread)": lat["cpu_1thr_median_ms"] <= 100}
        res[name] = {"official_metrics": official, "local_forecast_nmse": local_nmse,
                     "local_matches_official": abs(local_nmse - official["forecast_nmse"]) < 1e-6,
                     "health": {k: v for k, v in health.items() if k != "samples"}, "equivariance_max_abs": eq,
                     "latency": lat, "checks": checks, "all_passed": all(checks.values())}
        print(f"{name:36s} official forecast_nmse {official['forecast_nmse']:.5f} final {official['forecast_nmse_final']:.5f}"
              f" st {official['st_abs_error']:.6f} | local match {res[name]['local_matches_official']} | gates "
              f"{'PASS' if res[name]['all_passed'] else 'FAIL ' + str(checks)} | latency {lat}")

    reg = np.array([int(t["regime"][c]) for c in np.unique(case)])
    REG = ("periodic", "quasi-periodic", "aperiodic")

    def pc(L):
        cs = np.unique(case)
        return np.array([L[case == c].mean() for c in cs]), np.array([L[case == c][:, -1].mean() for c in cs])

    fr, gr = pc(lead["ref"])
    comps = {}
    for kind in ("ens", "s1_mirror"):
        f, g = pc(lead[kind])
        d = f - fr
        comps[f"{names[kind]} - {names['ref']}"] = {
            "forecast_nmse": paired(d), "forecast_nmse_final": paired(g - gr),
            "by_regime": {REG[r]: {"n": int((reg == r).sum()), "mean": float(d[reg == r].mean()),
                                   "better": int((d[reg == r] < -1e-9).sum()), "worse": int((d[reg == r] > 1e-9).sum())}
                          for r in np.unique(reg)}}
    s0 = res[names["ref"]]["official_metrics"]["forecast_nmse"]
    s1 = res[names["s1_mirror"]]["official_metrics"]["forecast_nmse"]
    out = {"provenance": "Mac CPU; official SIDERIUS metrics via run_baseline.evaluate_model; bootstrap 2000 case resamples, seed 0",
           "hybrid_plugin": str(a.hybrid_plugin), "ref_weights": str(a.ref_weights), "seed1_weights": str(a.seed1),
           "results": res, "paired_vs_ref": comps,
           "seed_spread_mirror_avg_forecast_nmse": {"seed0": s0, "seed1": s1, "abs_diff": abs(s1 - s0)}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ens_hybrid_results.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    for k, v in comps.items():
        s = v["forecast_nmse"]
        print(f"{k}: {s['mean']:+.5f} CI95 [{s['ci95'][0]:+.5f}, {s['ci95'][1]:+.5f}] better {s['improved']}/{s['n_cases']}"
              f" | by regime " + "; ".join(f"{r} {x['mean']:+.4f} ({x['better']}b/{x['worse']}w)" for r, x in v["by_regime"].items()))
    print("seed spread (mirror-avg forecast_nmse):", out["seed_spread_mirror_avg_forecast_nmse"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
