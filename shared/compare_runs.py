"""P3: paired case-level comparison of any runs + report figures 1 and 2 (spec §5, A10).

    python shared/compare_runs.py --data_dir $DATA_DIR --out_dir report/abc \
        --run A_s0=runs/A_s0 --run B_s0=runs/B_s0 --run C_from_B_s0=runs/C_from_B_s0 \
        --pair B_s0-A_s0 --pair C_from_B_s0-B_s0 --pair C_from_B_s0-A_s0

Each --run NAME=DIR needs DIR/local_eval/diag.npz (shared/local_eval.py) and, for figure 2,
DIR/pitching_wake_predictions.npz (official `run_baseline.py evaluate` deliverable); override the
prediction path with --pred NAME=FILE. The FIRST run is the reference: figure 2 shows the case whose
reference case-NMSE is nearest the median (first in case order on ties, no cherry-picking).

Statistics match shared/paired_stats.py exactly (per-lead clip at 4, then window mean, then case mean;
2000-resample paired case bootstrap, seed 0). The interval describes case-sampling variability over these
12 validation cases only: not training-seed noise, not a population claim.
Writes OUT_DIR/paired.json, OUT_DIR/fig1.pdf, OUT_DIR/fig2.pdf (fig2 skipped with --no_fig2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLIP = 4.0
REGIMES = ("periodic", "quasi-periodic", "aperiodic")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # categorical slots 1-4, fixed order (validated, light)
MARKERS = ("o", "s", "^", "D")
X_EXT, Y_EXT = (-0.5, 5.5), (-1.5, 1.5)                  # chord units, TASK_CARD.md
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"


def per_case(d):
    L = np.minimum(d["lead_nmse"], CLIP)
    cases = np.unique(d["case"])
    return (cases, np.array([L[d["case"] == c].mean() for c in cases]),
            np.array([L[d["case"] == c][:, -1].mean() for c in cases]),
            np.array([L[d["case"] == c].mean(0) for c in cases]))


def paired(diff, n=2000, seed=0):  # identical to shared/paired_stats.py
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "improved": int((diff < 0).sum()), "worsened": int((diff > 0).sum()), "unchanged": int((diff == 0).sum()),
            "n_cases": int(len(diff)), "bootstrap": {"resamples": n, "seed": seed, "unit": "case"}}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def kv(items, what):
    out = {}
    for it in items:
        name, sep, val = it.partition("=")
        if not sep:
            raise SystemExit(f"{what} must be NAME=PATH, got {it!r}")
        out[name] = Path(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--pair", action="append", default=[], help="X-Y means X minus Y (negative = X better)")
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--data_dir", type=Path)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--no_fig2", action="store_true")
    a = ap.parse_args()
    runs = kv(a.run, "--run")
    names = list(runs)
    if len(names) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per figure")
    pred_paths = {n: runs[n] / "pitching_wake_predictions.npz" for n in names} | kv(a.pred, "--pred")
    pairs = [tuple(p.split("-", 1)) for p in a.pair]
    for x, y in pairs:
        assert x in runs and y in runs, f"pair {x}-{y} uses an unknown run"
    plt.rcParams.update({"font.size": 8, "axes.labelcolor": INK, "axes.titlesize": 8, "pdf.fonttype": 42})

    D = {n: np.load(runs[n] / "local_eval" / "diag.npz") for n in names}
    ref = D[names[0]]
    for n in names[1:]:  # identical window alignment is required for pairing
        for key in ("case", "start", "case_id"):
            assert np.array_equal(D[n][key], ref[key]), f"{n} {key} misaligned with {names[0]}"
    S = {n: per_case(D[n]) for n in names}
    cases = S[names[0]][0]
    first = {c: int(np.where(ref["case"] == c)[0][0]) for c in cases}
    ids = [str(ref["case_id"][first[c]]) for c in cases]
    k = np.array([float(ref["k"][first[c]]) for c in cases])
    th = np.radians([float(ref["theta0_deg"][first[c]]) for c in cases])
    kth = k * th
    reg = np.array([int(ref["regime"][first[c]]) for c in cases])

    out = {"provenance": "per-window lead NMSE from shared/local_eval.py (matches official scorer); official clip order",
           "runs": {n: str(runs[n]) for n in names},
           "per_run": {n: {"forecast_nmse": float(S[n][1].mean()), "forecast_nmse_final": float(S[n][2].mean())}
                       for n in names},
           "pairs": {}}
    for x, y in pairs:
        dfull, dfinal = S[x][1] - S[y][1], S[x][2] - S[y][2]
        out["pairs"][f"{x}-{y}"] = {
            "comparison": f"{x} - {y} (negative = {x} better)",
            "forecast_nmse": {y: float(S[y][1].mean()), x: float(S[x][1].mean()), **paired(dfull)},
            "forecast_nmse_final": {y: float(S[y][2].mean()), x: float(S[x][2].mean()), **paired(dfinal)},
            "by_regime": {REGIMES[r]: {"n": int((reg == r).sum()),
                                       "mean_delta_forecast_nmse": float(dfull[reg == r].mean()),
                                       "mean_delta_forecast_nmse_final": float(dfinal[reg == r].mean())}
                          for r in np.unique(reg)},
            "lead_delta_case_avg": [float(v) for v in (S[x][3] - S[y][3]).mean(0)],
            "cases": [{"case_id": ids[i], "k": float(k[i]), "theta0_deg": float(np.degrees(th[i])),
                       "k_theta0_rad": float(kth[i]), "regime": REGIMES[reg[i]],
                       "delta_nmse": float(dfull[i]), "delta_nmse_final": float(dfinal[i])} for i in range(len(cases))],
        }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "paired.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    # ---------------- Figure 1: (a) lead curves, (b) per-case paired change vs k*theta0
    color = {n: SERIES[i] for i, n in enumerate(names)}   # colour follows the run, fixed order
    fig, ax = plt.subplots(1, 2 if pairs else 1, figsize=(7.2, 2.7), squeeze=False)
    ax = ax[0]
    t = np.arange(1, 21) * 0.1
    ends = []
    for i, n in enumerate(names):
        m = S[n][3].mean(0)
        ax[0].plot(t, m, color=color[n], lw=2, label=n, marker=MARKERS[i], markevery=[19], ms=4, mec="white", mew=1)
        ends.append((m[-1], n))
    # direct end labels, nudged apart by ~one text line of the final axis range so they never collide
    ax[0].set_ylim(bottom=0)
    lo, hi = ax[0].get_ylim()
    gap = 0.075 * (hi - lo)
    ends.sort()
    placed = []
    for v, n in ends:
        y = v if not placed or v - placed[-1] >= gap else placed[-1] + gap
        placed.append(y)
        ax[0].annotate(f"{n} {v:.3f}", (t[-1], v), xytext=(t[-1] + 0.06, y), textcoords="data", va="center",
                       fontsize=6.5, color=INK)
    ax[0].set_ylim(0, max(hi, placed[-1] + gap))   # keep the top label inside the axes, clear of the title
    ax[0].set_xlabel("lead time (c/U)")
    ax[0].set_ylabel("NMSE (clipped per lead, case-avg)")
    ax[0].set_xlim(0, 2.75)
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].set_title(f"(a) forecast error vs lead, {len(cases)} validation cases", loc="left", color=INK)
    style(ax[0])
    if pairs:
        for j, (x, y) in enumerate(pairs):
            dfull = S[x][1] - S[y][1]
            # panel (b) has its own legend: each pair takes the next fixed slot plus its own marker
            ax[1].scatter(kth, dfull, marker=MARKERS[j], s=30, facecolor=SERIES[j], edgecolor="white", lw=1,
                          label=f"{x} − {y}", zorder=3)
        ax[1].axhline(0, color=INK, lw=0.8, zorder=2)
        ax[1].set_xlabel(r"$k\,\theta_0$ (rad)")
        ax[1].set_ylabel(r"$\Delta$ case NMSE")
        ax[1].legend(fontsize=7, frameon=False)
        ax[1].set_title("(b) per-case paired change; below 0 = first run better", loc="left", color=INK)
        style(ax[1])
    fig.tight_layout()
    fig.savefig(a.out_dir / "fig1.pdf")
    plt.close(fig)

    # ---------------- Figure 2: truth vs every run at lead 1 and lead 20
    fig2_case = None
    if not a.no_fig2:
        fA = S[names[0]][1]
        ci = int(np.argmin(np.abs(fA - np.median(fA))))
        c, w = cases[ci], first[cases[ci]]
        s0 = int(ref["start"][w])
        fig2_case = {"case_id": ids[ci], "window_start": s0, "rule": f"case nearest median {names[0]} case NMSE"}
        with np.load(a.data_dir / "validation.npz", allow_pickle=False) as v:
            assert str(v["case_id"][c]) == ids[ci]
            truth = v["vort"][c, s0:s0 + 20].astype(np.float64)
        rows = [("truth", truth, None)]
        for n in names:
            with np.load(pred_paths[n], allow_pickle=False) as p:
                idx = [i for i, (cid, st) in enumerate(zip(p["case_id"], p["win_start"]))
                       if str(cid) == ids[ci] and int(st) == s0]
                assert len(idx) == 1, f"{n}: window {ids[ci]}@{s0} not found once in {pred_paths[n]}"
                rows.append((n, p["forecast"][idx[0]].astype(np.float64), n))
        leads = (0, 19)
        vmax = float(np.percentile(np.abs(truth[list(leads)]), 99.5))   # one shared symmetric scale
        fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.55 * len(rows) + 0.3), sharex=True, sharey=True,
                                 squeeze=False)
        for i, (label, f, run) in enumerate(rows):
            for j, lead in enumerate(leads):
                axx = axes[i, j]
                im = axx.imshow(f[lead], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="equal",
                                extent=(*X_EXT, *Y_EXT), interpolation="nearest")
                if i == 0:
                    axx.set_title(f"lead {lead + 1} ({(lead + 1) * 0.1:.1f} c/U)", fontsize=8, color=INK)
                if run is not None:
                    axx.text(0.99, 0.96, f"NMSE {D[run]['lead_nmse'][w, lead]:.3f}", transform=axx.transAxes,
                             ha="right", va="top", fontsize=7, color=INK,
                             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
                axx.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
                if i == len(rows) - 1:
                    axx.set_xlabel("x / c")
                if j == 0:
                    axx.set_ylabel(f"{label}\ny / c", color=INK)
        fig.suptitle(f"case {ids[ci]} (k={k[ci]:.2f}, θ0={np.degrees(th[ci]):.1f}°, {REGIMES[reg[ci]]}), "
                     f"window start {s0}: case nearest median {names[0]} error", fontsize=8, color=INK)
        fig.colorbar(im, ax=axes, shrink=0.8, extend="both", label=r"$\omega_z c/U$ (shared, symmetric)")
        fig.savefig(a.out_dir / "fig2.pdf")
        plt.close(fig)
    out["fig2"] = fig2_case
    (a.out_dir / "paired.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    for n in names:
        print(f"{n}: forecast_nmse {out['per_run'][n]['forecast_nmse']:.4f} final {out['per_run'][n]['forecast_nmse_final']:.4f}")
    for key, p in out["pairs"].items():
        for m in ("forecast_nmse", "forecast_nmse_final"):
            s = p[m]
            print(f"{key} {m}: delta {s['mean']:+.4f} CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                  f"improved {s['improved']}/{s['n_cases']}")
    print("fig2:", fig2_case, "| wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
