"""Extend the frozen plot data with the post-draft models and record every report number with its source.

    python report/final/prepare_update.py --snapshot <wake_remote_backup_20261121_1845> --data_dir data \
        --ens_zip <snapshot>/person3/C_ens_hybrid.zip

No fitting. Inputs: plot_data.npz (frozen, from the earlier report bundle), saved per-window diagnostics
(shared/local_eval.py, which matches the official scorer), saved official predictions, and the submitted
C_ens_hybrid package (run on CPU). Outputs: plot_data_final.npz and report_numbers.json in this folder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "shared"), str(REPO / "variants/Bplus/pitching_airfoil_wake/task/plugins")]
import local_eval as le  # noqa: E402

CLIP = 4.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def boot(delta: np.ndarray) -> dict:
    """Same rule as build_figures.py: 2000 case resamples, default_rng(0), 2.5/97.5% quantiles."""
    r = np.random.default_rng(0)
    b = delta[r.integers(len(delta), size=(2000, len(delta)))].mean(1)
    ci = np.quantile(b, [.025, .975])
    return {"mean": float(delta.mean()), "ci95": [float(ci[0]), float(ci[1])],
            "better": int((delta < 0).sum()), "worse": int((delta > 0).sum()), "n_cases": int(len(delta))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--ens_zip", type=Path, required=True)
    a = ap.parse_args()
    import torch

    base = {k: v for k, v in np.load(HERE / "plot_data.npz", allow_pickle=False).items()}
    t = le.read_validation(a.data_dir / "validation.npz")
    pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
    keys = [(t["case_ids"][c], s) for c, s in pairs]
    assert list(base["case_id"]) == [k[0] for k in keys], "frozen window order differs from validation order"
    sources, numbers = {}, {}

    def from_diag(name, path):
        d = np.load(path, allow_pickle=False)
        idx = {(str(c), int(s)): i for i, (c, s) in enumerate(zip(d["case_id"], d["start"]))}
        take = [idx[k] for k in keys]
        sources[name] = {"file": str(path), "sha256": sha(path)}
        return d["lead_nmse"][take].astype(np.float64), d["defect"][take].astype(np.float64)

    P3 = a.snapshot / "person3/runs"
    leads = {n: base[f"lead_{n}"].astype(np.float64) for n in ("A_ref", "B", "C", "C+", "E")}
    defects = {}
    for name, path in [("A_avg", REPO / "runs/C_on_A/local_eval/diag.npz"),
                       ("B+", REPO / "runs/Bplus_s0/local_eval/diag.npz"),
                       ("B_nmse", P3 / "nmse_s0/local_eval/diag.npz"),
                       ("C_nmse", P3 / "C_from_nmse_s0/local_eval/diag.npz"),
                       ("C_w96", P3 / "C_from_w96_s0/local_eval/diag.npz"),
                       ("C_w96b", P3 / "C_from_w96b_s0/local_eval/diag.npz")]:
        leads[name], defects[name] = from_diag(name, path)

    # ---- C_ens: the submitted package, run on CPU (no fitting)
    zsha = sha(a.ens_zip)
    assert zsha.startswith("85377301"), zsha
    tmp = Path(tempfile.mkdtemp(prefix="cens_"))
    with zipfile.ZipFile(a.ens_zip) as z:
        z.extractall(tmp)
    man = json.loads((tmp / "submission.json").read_text())
    pk = load(tmp / man["plugin"], "cens_pkg")
    model = getattr(pk, man["symbol"])()
    model.load_state_dict(pk.load_state(tmp / man["weights"]))
    model.eval()
    X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
    torch.set_num_threads(8)
    with torch.no_grad():
        flat = np.concatenate([model(X[s:s + 16]).float().numpy() for s in range(0, len(pairs), 16)])
        flat_g = np.concatenate([model(pk.mirror_x(X[s:s + 16])).float().numpy() for s in range(0, len(pairs), 16)])
    parts = {k: v.astype(np.float32).astype(np.float64) for k, v in le.unpack(flat).items()}
    pred = dict(parts, pairs=pairs)
    leads["C_ens"] = le.window_nmse(pred, t)
    F, Fg = parts["forecast"], le.unpack(flat_g)["forecast"].astype(np.float64)
    defects["C_ens"] = ((Fg + F[:, :, ::-1, :]) ** 2).sum((1, 2, 3)) / np.maximum((F ** 2).sum((1, 2, 3)) + (Fg ** 2).sum((1, 2, 3)), 1e-12)
    # official SIDERIUS scorer on the same outputs
    vb = load(REPO / "variants/Bplus/pitching_airfoil_wake/baseline/run_baseline.py", "rb_variant")
    official = vb.evaluate_model(model, vb.compose_task(), a.data_dir, tmp / "eval", torch.device("cpu"))
    import health_gate as hg
    gate = hg.read_health_yaml(REPO / "variants/Bplus/pitching_airfoil_wake/task/plugins")
    health = hg.official_health(parts["forecast"], gate)
    lat = {}
    for th in (1, 8):
        torch.set_num_threads(th)
        ts = []
        with torch.no_grad():
            for _ in range(5):
                model(X[:1])
            for _ in range(30):
                t0 = time.perf_counter(); model(X[:1]); ts.append((time.perf_counter() - t0) * 1e3)
        lat[f"mac_cpu_{th}thr_median_ms"] = round(float(np.median(ts)), 1)
    sources["C_ens"] = {"file": str(a.ens_zip), "sha256": zsha, "symbol": man["symbol"],
                        "final_model_sha256": sha(tmp / man["weights"]), "run": "Mac CPU inference, no fitting"}
    numbers["C_ens_package"] = {"official_metrics": official, "health_passed": bool(health["passed"]),
                                "health_gate": gate["gate_id"], "latency": lat,
                                "max_abs_equivariance": float(np.abs(Fg + F[:, :, ::-1, :]).max()),
                                "finite": bool(np.isfinite(flat).all()), "shape": list(flat.shape)}

    # ---- figure-3 fields: declared case (nearest median A_ref error), first window, leads 1 and 20
    meta = json.loads((HERE / "figure_provenance.json").read_text())
    wi, ci, s0 = meta["window_index"], meta["case_index"], meta["window_start"]
    assert keys[wi] == (meta["selected_case"], s0)
    truth = t["vort"][ci, s0:s0 + 20].astype(np.float64)
    fields_ens = np.stack([truth[[0, 19]], F[wi][[0, 19]]], axis=0)  # [method(truth, C_ens), lead(1,20), y, x]

    # ---- validation summary per configuration (official clip order)
    names = ["A_ref", "A_avg", "B", "C", "B+", "C+", "B_nmse", "C_nmse", "C_ens", "E", "C_w96", "C_w96b"]
    case_ids = np.array([k[0] for k in keys])
    order = list(base["case_order"])
    val = {}
    for n in names:
        L = np.minimum(leads[n], CLIP)
        val[n] = {"forecast_nmse": float(L.mean()), "forecast_nmse_final": float(L[:, -1].mean())}
        if n in defects:
            dd = defects[n]
            val[n]["reflection_defect"] = float(np.mean([dd[case_ids == c].mean() for c in order]))
    numbers["validation"] = val
    # check the frozen leads against the saved official values quoted in the draft
    for n, ref in {"A_ref": 0.2571, "B": 0.2094, "C": 0.1856, "C+": 0.1440, "E": 0.1897, "A_avg": 0.2562,
                   "B_nmse": 0.1526, "C_nmse": 0.1261, "C_ens": 0.1181}.items():
        assert abs(val[n]["forecast_nmse"] - ref) < 6e-5, (n, val[n]["forecast_nmse"], ref)
    assert abs(official["forecast_nmse"] - val["C_ens"]["forecast_nmse"]) < 1e-9

    # ---- paired case changes (same rule as the figures)
    S = {n: np.array([np.minimum(leads[n], CLIP)[case_ids == c].mean() for c in order]) for n in names}
    SF = {n: np.array([np.minimum(leads[n], CLIP)[case_ids == c][:, -1].mean() for c in order]) for n in names}
    numbers["paired"] = {f"{x} - {y}": {"forecast_nmse": boot(S[x] - S[y]), "forecast_nmse_final": boot(SF[x] - SF[y])}
                         for x, y in [("A_avg", "A_ref"), ("C", "B"), ("C+", "C"), ("C_nmse", "C+"), ("C_ens", "C_nmse"),
                                      ("C", "E"), ("C_ens", "A_ref"), ("C_w96", "C_nmse"), ("C_w96b", "C_nmse")]}

    # ---- forecast mixtures recomputed from saved official predictions (validation-scored)
    def preds(run):
        p = P3 / run / "official/pitching_wake_predictions.npz"
        d = np.load(p, allow_pickle=False)
        idx = {(str(c), int(s)): i for i, (c, s) in enumerate(zip(d["case_id"], d["win_start"]))}
        sources[f"pred_{run}"] = {"file": str(p), "sha256": sha(p)}
        return d["forecast"][[idx[k] for k in keys]].astype(np.float64)

    sc = lambda Fm: le.score("forecast_nmse", {"forecast": Fm, "pairs": pairs}, t)
    N, Bp, W, Wb = preds("C_from_nmse_s0"), preds("C_from_Bplus_s0"), preds("C_from_w96_s0"), preds("C_from_w96b_s0")
    mixes = {"0.6 C_nmse + 0.4 C+ (C_ens forecast)": sc(0.6 * N + 0.4 * Bp)}
    for tag, Wx in (("w96", W), ("w96b", Wb)):
        for wa, wn, wb in ((0.5, 0.5, 0), (0.6, 0.4, 0), (0.4, 0.4, 0.2), (0.5, 0.3, 0.2), (0.34, 0.33, 0.33)):
            mixes[f"{wa} {tag} + {wn} C_nmse + {wb} C+"] = sc(wa * Wx + wn * N + wb * Bp)
    numbers["mixtures_validation"] = mixes

    out = dict(base)
    out["fields_ens"] = fields_ens
    for n, L in leads.items():
        out[f"lead_{n}"] = L
    out["defect_names"] = np.array(sorted(defects))
    out["defect_values"] = np.array([float(np.mean([defects[n][case_ids == c].mean() for c in order])) for n in sorted(defects)])
    np.savez_compressed(HERE / "plot_data_final.npz", **out)
    report = {"inputs": {"validation": {"file": str(a.data_dir / "validation.npz"), "sha256": sha(a.data_dir / "validation.npz")},
                         "frozen_base": {"file": str(HERE / "plot_data.npz"), "sha256": sha(HERE / "plot_data.npz")}, **sources},
              "rules": "official clip order (per-lead NMSE clipped at 4, then averaged); per-case means over 4 windows; "
                       "case bootstrap 2000 resamples, numpy default_rng(0), 2.5/97.5% quantiles",
              **numbers}
    (HERE / "report_numbers.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    for n in names:
        print(f"{n:8s} {val[n]}")
    for k, v in numbers["paired"].items():
        f = v["forecast_nmse"]
        print(f"{k:16s} {f['mean']:+.4f} [{f['ci95'][0]:+.4f}, {f['ci95'][1]:+.4f}] better {f['better']}/12 | final {v['forecast_nmse_final']['mean']:+.4f}")
    print(json.dumps(numbers["C_ens_package"], indent=1, default=float))
    print(json.dumps(mixes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
