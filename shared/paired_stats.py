"""Paired C_on_A vs A statistics and report figures 1 and 2.

    python shared/paired_stats.py --runs runs --data_dir data --report report

Inputs: runs/{A_official,C_on_A}/local_eval/diag.npz (per-window per-lead NMSE from shared/local_eval.py,
identical to the official scorer) and runs/{A_official,C_on_A}/pitching_wake_predictions.npz (official
deliverables, for the field plots). Clipping follows _wake_metrics.score: clip each lead at 4, then average.
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
REGIMES = ("periodic", "quasi-periodic", "aperiodic")  # index order of regime_logits (_wake_task.unpack)
CONFIGS = ("A_official", "C_on_A", "B_s0", "C_from_B_s0")
COL = {"A_official": "#2a78d6", "C_on_A": "#eb6834", "B_s0": "#1baf7a", "C_from_B_s0": "#e87ba4"}  # slots 1,2,3,5
LABEL = {"A_official": "A (official CNN)", "C_on_A": "C_on_A (mirror-averaged)",
         "B_s0": "B (mirror-augmented training)", "C_from_B_s0": "C_from_B (B + mirror-averaged)"}
SHORT = {"A_official": "A", "C_on_A": "C_on_A", "B_s0": "B", "C_from_B_s0": "C_from_B"}
PAIRS = [("C_on_A", "A_official"), ("B_s0", "A_official"), ("C_from_B_s0", "B_s0"), ("C_from_B_s0", "A_official")]
FIG_PAIRS = [("B_s0", "A_official"), ("C_from_B_s0", "B_s0")]
MARK = {0: "o", 1: "s", 2: "^"}
X_EXT, Y_EXT = (-0.5, 5.5), (-1.5, 1.5)  # chord units, TASK_CARD.md:36-37
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"


def per_case(d):
    L = np.minimum(d["lead_nmse"], CLIP)               # [48, 20] clipped per lead
    cases = np.unique(d["case"])
    full = np.array([L[d["case"] == c].mean() for c in cases])
    final = np.array([L[d["case"] == c][:, -1].mean() for c in cases])
    curve = np.array([L[d["case"] == c].mean(0) for c in cases])  # [12, 20]
    return cases, full, final, curve


def paired(diff, n=2000, seed=0):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--report", type=Path, default=Path("report"))
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.labelcolor": INK, "axes.titlesize": 8, "pdf.fonttype": 42})

    avail = [r for r in CONFIGS if (a.runs / r / "local_eval" / "diag.npz").is_file()]
    D = {r: np.load(a.runs / r / "local_eval" / "diag.npz") for r in avail}
    for r in avail:
        assert np.array_equal(D["A_official"]["case"], D[r]["case"]) and np.array_equal(D["A_official"]["start"], D[r]["start"])
    dA = D["A_official"]
    PC = {r: per_case(D[r]) for r in avail}          # r -> (cases, full, final, curve)
    cases, fA, gA, cA = PC["A_official"]
    first = {c: int(np.where(dA["case"] == c)[0][0]) for c in cases}
    ids = [str(dA["case_id"][first[c]]) for c in cases]
    k = np.array([float(dA["k"][first[c]]) for c in cases])
    th = np.radians([float(dA["theta0_deg"][first[c]]) for c in cases])
    kth = k * th
    reg = np.array([int(dA["regime"][first[c]]) for c in cases])

    def compare(r1, r0):
        dfull, dfinal = PC[r1][1] - PC[r0][1], PC[r1][2] - PC[r0][2]
        return {
            "comparison": f"{r1} - {r0} (negative = {r1} better)",
            "forecast_nmse": {r0: float(PC[r0][1].mean()), r1: float(PC[r1][1].mean()), **paired(dfull)},
            "forecast_nmse_final": {r0: float(PC[r0][2].mean()), r1: float(PC[r1][2].mean()), **paired(dfinal)},
            "by_regime": {REGIMES[g]: {"n": int((reg == g).sum()),
                                       "mean_delta_forecast_nmse": float(dfull[reg == g].mean()),
                                       "mean_delta_forecast_nmse_final": float(dfinal[reg == g].mean()),
                                       "better": int((dfull[reg == g] < 0).sum()), "worse": int((dfull[reg == g] > 0).sum())}
                          for g in np.unique(reg)},
            "lead_delta_case_avg": [float(v) for v in (PC[r1][3] - PC[r0][3]).mean(0)],
            "cases": [{"case_id": ids[i], "k_theta0_rad": float(kth[i]), "regime": REGIMES[reg[i]],
                       "delta_nmse": float(dfull[i]), "delta_nmse_final": float(dfinal[i])} for i in range(len(cases))],
        }

    pairs = [p for p in PAIRS if p[0] in avail and p[1] in avail]
    allres = {"provenance": "per-window lead NMSE from shared/local_eval.py (matches official scorer); official clip order",
              "per_case": [{"case_id": ids[i], "k": float(k[i]), "theta0_deg": float(np.degrees(th[i])),
                            "k_theta0_rad": float(kth[i]), "regime": REGIMES[reg[i]],
                            **{f"{r}_nmse": float(PC[r][1][i]) for r in avail},
                            **{f"{r}_nmse_final": float(PC[r][2][i]) for r in avail}} for i in range(len(cases))],
              "pairs": {f"{r1}-{r0}": compare(r1, r0) for r1, r0 in pairs}}
    (a.runs / "paired_stats.json").write_text(json.dumps(allres, indent=1) + "\n", encoding="utf-8")

    # ---------------- Figure 1
    a.report.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    t = np.arange(1, 21) * 0.1
    for r in avail:
        m = PC[r][3].mean(0)
        ax[0].plot(t, m, color=COL[r], lw=2, ls="--" if r.startswith("C_") else "-",  # dashed = mirror-averaged
                   label=f"{LABEL[r]}  {PC[r][1].mean():.3f}")
        ax[0].plot(t[-1], m[-1], "o", color=COL[r], ms=4, mec="white", mew=1)
    ax[0].set_xlabel("lead time (c/U)")
    ax[0].set_ylabel("NMSE (clipped per lead, case-avg)")
    ax[0].set_xlim(0, 2.1)
    ax[0].set_ylim(bottom=0)
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower right", title="config  forecast_nmse", title_fontsize=6.5)
    ax[0].set_title("(a) forecast error vs lead, 12 validation cases", loc="left", color=INK)
    style(ax[0])
    from matplotlib.lines import Line2D
    handles = []
    for r1, r0 in FIG_PAIRS:
        if r1 not in avail or r0 not in avail:
            continue
        d = PC[r1][1] - PC[r0][1]
        hollow = r1.startswith("C_")  # secondary encoding: hollow = averaging step, filled = training step
        fc, ec = ("white", COL[r1]) if hollow else (COL[r1], "white")
        for g in np.unique(reg):
            s = reg == g
            ax[1].scatter(kth[s], d[s], marker=MARK[g], s=34, facecolor=fc, edgecolor=ec, lw=1.5 if hollow else 1, zorder=3)
        handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc=fc, mec=ec, mew=1.5 if hollow else 1,
                              label=f"{SHORT[r1]} − {SHORT[r0]}"))
    handles += [Line2D([], [], ls="", marker=MARK[g], ms=6, mfc=MUTED, mec="white", label=REGIMES[g]) for g in np.unique(reg)]
    ax[1].axhline(0, color=INK, lw=0.8, zorder=2)
    ax[1].set_xlabel(r"$k\,\theta_0$ (rad)")
    ax[1].set_ylabel(r"$\Delta$ case NMSE")
    ax[1].legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left", ncol=1)
    ax[1].set_title("(b) per-case change; below 0 = first config better", loc="left", color=INK)
    style(ax[1])
    fig.tight_layout()
    fig.savefig(a.report / "fig1.pdf")
    plt.close(fig)

    # ---------------- Figure 2: case whose A case-NMSE is nearest the median (first in order on ties)
    ci = int(np.argmin(np.abs(fA - np.median(fA))))
    c = cases[ci]
    w = first[c]
    s0 = int(dA["start"][w])
    with np.load(a.data_dir / "validation.npz", allow_pickle=False) as v:
        truth = v["vort"][c, s0:s0 + 20].astype(np.float64)
        assert str(v["case_id"][c]) == ids[ci]
    preds = {}
    for r in ("A_official", "C_on_A"):
        with np.load(a.runs / r / "pitching_wake_predictions.npz", allow_pickle=False) as p:
            idx = [i for i, (cid, st) in enumerate(zip(p["case_id"], p["win_start"])) if str(cid) == ids[ci] and int(st) == s0]
            assert len(idx) == 1
            preds[r] = p["forecast"][idx[0]].astype(np.float64)
    rows = [("truth", truth, None), ("A", preds["A_official"], "A_official"), ("C_on_A", preds["C_on_A"], "C_on_A")]
    leads = (0, 19)
    # one shared symmetric scale for all six panels; 99.5th percentile of |truth| so the wake is visible
    # (near-airfoil peaks saturate; colorbar marked with extend="both")
    vmax = float(np.percentile(np.abs(truth[list(leads)]), 99.5))
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for i, (name, f, run) in enumerate(rows):
        for j, l in enumerate(leads):
            ax = axes[i, j]
            im = ax.imshow(f[l], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="equal",
                           extent=(*X_EXT, *Y_EXT), interpolation="nearest")
            if i == 0:
                ax.set_title(f"lead {l + 1} ({(l + 1) * 0.1:.1f} c/U)", fontsize=8, color=INK)
            if run is not None:
                ax.text(0.99, 0.96, f"NMSE {D[run]['lead_nmse'][w, l]:.3f}", transform=ax.transAxes, ha="right",
                        va="top", fontsize=7, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
            if i == 2:
                ax.set_xlabel("x / c")
            if j == 0:
                ax.set_ylabel(f"{name}\ny / c", color=INK)
    fig.suptitle(f"case {ids[ci]} (k={k[ci]:.2f}, θ0={np.degrees(th[ci]):.1f}°, {REGIMES[reg[ci]]}), window start {s0}"
                 f" — case nearest median A error", fontsize=8, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, extend="both", label=r"$\omega_z c/U$ (shared, symmetric)")
    fig.savefig(a.report / "fig2.pdf")
    plt.close(fig)

    # ---------------- console summary
    for name, res in allres["pairs"].items():
        print(name)
        for key in ("forecast_nmse", "forecast_nmse_final"):
            s = res[key]
            print(f"  {key:20s} delta {s['mean']:+.4f} CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
                  f" better {s['improved']}/{s['n_cases']} worse {s['worsened']}")
        print("  by regime: " + "; ".join(f"{g} (n={v['n']}) {v['mean_delta_forecast_nmse']:+.4f} [{v['better']}b/{v['worse']}w]"
                                         f" final {v['mean_delta_forecast_nmse_final']:+.4f}" for g, v in res["by_regime"].items()))
    print(f"fig2 case {ids[ci]} start {s0}; wrote {a.report/'fig1.pdf'}, {a.report/'fig2.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
