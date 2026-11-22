"""Seed spread + 3-seed mirror-averaged OOD ensemble vs the submitted C_from_Bplus_ood (#52). CPU, validation only.

    python shared/ensemble_eval.py --baseline_dir variants/Bplus/pitching_airfoil_wake/baseline --data_dir data \
        --models runs/Bplus_s0/final_model.pt runs/Bplus_s1/final_model.pt runs/Bplus_s2/final_model.pt \
        --ref_pkg ~/src/ref_C_from_Bplus_ood/pkg --out runs/ensemble_Bplus_s012

Ensemble forecast = mean over members of 0.5 * (F_m(x) + S F_m(G x)), then the same out-of-range shrinkage as
P3's WakeMirrorAvgOOD (k outside the training range: linear ramp 0 -> 0.7 over leads toward the history mean;
theta0 rule with strength 0.0). Statistics head = member 0 (seed 0). The OOD range uses its own constant names,
so the statistics head keeps the training-time K_RANGE normalisation (WakeMirrorAvgOOD rebinds K_RANGE).
Scored with shared/local_eval.py (matches the official scorer). Bootstrap: 2000 case resamples, seed 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import local_eval as le  # noqa: E402

# P3 WakeMirrorAvgOOD constants (submission #52, run_baseline.py:151-156), renamed
OOD_K_TRAIN = (2.0016000270843506, 6.997600078582764)
OOD_TH_TRAIN_DEG = (5.008999824523926, 34.994998931884766)
OOD_STRENGTH_K, OOD_STRENGTH_TH = 0.7, 0.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_ensemble_class(rb):
    class WakeEnsembleMirrorOOD(nn.Module):
        def __init__(self, n: int = 3) -> None:
            super().__init__()
            self.members = nn.ModuleList([rb.WakeBaseline() for _ in range(n)])

        def forward(self, x):
            fut = torch.stack([0.5 * (m.forecast(x) + rb.mirror_field(m.forecast(rb.mirror_x(x))))
                               for m in self.members]).mean(0)
            k = x[:, rb.HISTORY, 0, 0] * (2 * np.pi)
            th = torch.rad2deg(x[:, rb.HISTORY + 1, 0, 0])
            a = torch.zeros_like(k)
            a = torch.where((k > OOD_K_TRAIN[1]) | (k < OOD_K_TRAIN[0]), torch.full_like(a, OOD_STRENGTH_K), a)
            th_out = (th > OOD_TH_TRAIN_DEG[1]) | (th < OOD_TH_TRAIN_DEG[0])
            a = torch.where(th_out, torch.maximum(a, torch.full_like(a, OOD_STRENGTH_TH)), a)
            if bool((a > 0).any()):
                hist_mean = x[:, :rb.HISTORY].mean(dim=1, keepdim=True)
                ramp = torch.linspace(0, 1, rb.HORIZON, dtype=x.dtype).view(1, rb.HORIZON, 1, 1)
                w = a.view(-1, 1, 1, 1) * ramp
                fut = (1 - w) * fut + w * hist_mean
            return torch.cat([fut.flatten(1), self.members[0].statistics(x)], dim=1)

    return WakeEnsembleMirrorOOD


def run(model, X, t, pairs):
    with torch.no_grad():
        flat = np.concatenate([model(X[s:s + 16]).float().numpy() for s in range(0, len(pairs), 16)])
    assert flat.shape == (len(pairs), 96822) and np.isfinite(flat).all()
    parts = {k: v.astype(np.float32).astype(np.float64) for k, v in le.unpack(flat).items()}
    pred = dict(parts, pairs=pairs)
    metrics = {n: le.score(n, pred, t) for n in le.METRICS}
    lead = np.minimum(le.window_nmse(pred, t), le.NMSE_CLIP)  # [48, 20] clipped per lead
    return metrics, lead


def per_case(lead, case):
    cs = np.unique(case)
    return np.array([lead[case == c].mean() for c in cs]), np.array([lead[case == c][:, -1].mean() for c in cs])


def paired(d, n=2000, seed=0):
    b = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "better": int((d < -1e-9).sum()), "worse": int((d > 1e-9).sum()), "n_cases": len(d)}  # |d|<=1e-9: tie


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--ref_pkg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, __import__("os").cpu_count()))  # local diagnostic only

    rb = load_module(a.baseline_dir / "run_baseline.py", "rb_bplus")
    ref = load_module(a.ref_pkg / "run_baseline.py", "rb_ref")
    ref_man = json.loads((a.ref_pkg / "submission.json").read_text())
    t = le.read_validation(a.data_dir / "validation.npz")
    pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
    case = np.array([c for c, _ in pairs])
    X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
    reg = np.array([int(t["regime"][c]) for c in np.unique(case)])
    REG = ("periodic", "quasi-periodic", "aperiodic")

    results, leads = {}, {}
    # reference: the submitted package's own class and weights
    m = getattr(ref, ref_man["symbol"])()
    m.load_state_dict(ref.load_state(a.ref_pkg / ref_man["weights"]))
    results["C_from_Bplus_ood (#52)"], leads["ref"] = run(m.eval(), X, t, pairs)

    states = [rb.load_state(p) for p in a.models]
    for i, (p, sd) in enumerate(zip(a.models, states)):
        for cls in (rb.WakeBaseline, rb.WakeMirrorAvg):
            m = cls()
            m.load_state_dict(sd)
            name = f"{'Bplus' if cls is rb.WakeBaseline else 'C_from_Bplus'}_s{i}"
            results[name], leads[name] = run(m.eval(), X, t, pairs)
    Ens = make_ensemble_class(rb)
    ens = Ens(len(states))
    for mem, sd in zip(ens.members, states):
        mem.load_state_dict(sd)
    ens_name = f"ensemble_{len(states)}seed_mirror_ood"
    results[ens_name], leads["ens"] = run(ens.eval(), X, t, pairs)

    def spread(prefix):
        v = np.array([results[f"{prefix}_s{i}"]["forecast_nmse"] for i in range(len(states))])
        return {"values": v.tolist(), "mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1)) if len(v) > 1 else None,
                "min": float(v.min()), "max": float(v.max())}

    fr, gr = per_case(leads["ref"], case)
    comps = {}
    for name, key in [(ens_name, "ens")] + [(f"C_from_Bplus_s{i}", f"C_from_Bplus_s{i}") for i in range(len(states))]:
        f, g = per_case(leads[key], case)
        d = f - fr
        comps[f"{name} - C_from_Bplus_ood"] = {
            "forecast_nmse": paired(d), "forecast_nmse_final": paired(g - gr),
            "by_regime": {REG[r]: {"n": int((reg == r).sum()), "mean": float(d[reg == r].mean()),
                                   "better": int((d[reg == r] < 0).sum()), "worse": int((d[reg == r] > 0).sum())}
                          for r in np.unique(reg)}}
    stats_same = all(results[ens_name][k] == results["Bplus_s0"][k]
                     for k in ("mean_field_rel_l2", "psd_log_rmse", "st_abs_error", "regime_balanced_accuracy"))
    out = {"provenance": "local reimplementation of the official scorer (shared/local_eval.py), CPU, validation 12 cases / 48 windows",
           "models": [str(p) for p in a.models], "reference_pkg": str(a.ref_pkg), "metrics": results,
           "seed_spread_forecast_nmse": {"Bplus": spread("Bplus"), "C_from_Bplus": spread("C_from_Bplus")},
           "paired_vs_C_from_Bplus_ood": comps, "ensemble_stats_head_equals_seed0": stats_same,
           "ood_cases_in_validation": int(sum((float(k) < OOD_K_TRAIN[0]) or (float(k) > OOD_K_TRAIN[1]) for k in t["k"]))}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ensemble_results.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    np.savez(a.out / "lead_nmse_clipped.npz", case=case, **{k.replace(" ", "_"): v for k, v in leads.items()})

    for name, r in results.items():
        print(f"{name:28s} forecast_nmse {r['forecast_nmse']:.5f} final {r['forecast_nmse_final']:.5f} st {r['st_abs_error']:.6f}")
    for k, v in out["seed_spread_forecast_nmse"].items():
        print(f"seed spread {k}: {v}")
    for k, v in comps.items():
        s = v["forecast_nmse"]
        print(f"{k}: {s['mean']:+.5f} CI95 [{s['ci95'][0]:+.5f}, {s['ci95'][1]:+.5f}] better {s['better']}/{s['n_cases']}")
    print("ensemble stats head == seed 0:", stats_same, "| OOD cases in validation:", out["ood_cases_in_validation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
