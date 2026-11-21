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
COL = {"A_official": "#2a78d6", "C_on_A": "#eb6834"}   # categorical slots 1, 2
LABEL = {"A_official": "A (official CNN)", "C_on_A": "C_on_A (mirror-averaged)"}
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

    D = {r: np.load(a.runs / r / "local_eval" / "diag.npz") for r in ("A_official", "C_on_A")}
    assert np.array_equal(D["A_official"]["case"], D["C_on_A"]["case"])
    assert np.array_equal(D["A_official"]["start"], D["C_on_A"]["start"])
    dA = D["A_official"]
    cases, fA, gA, cA = per_case(dA)
    _, fC, gC, cC = per_case(D["C_on_A"])
    first = {c: int(np.where(dA["case"] == c)[0][0]) for c in cases}
    ids = [str(dA["case_id"][first[c]]) for c in cases]
    k = np.array([float(dA["k"][first[c]]) for c in cases])
    th = np.radians([float(dA["theta0_deg"][first[c]]) for c in cases])
    kth = k * th
    reg = np.array([int(dA["regime"][first[c]]) for c in cases])

    dfull, dfinal = fC - fA, gC - gA
    by_regime = {REGIMES[r]: {"n": int((reg == r).sum()),
                              "mean_delta_forecast_nmse": float(dfull[reg == r].mean()),
                              "mean_delta_forecast_nmse_final": float(dfinal[reg == r].mean())}
                 for r in np.unique(reg)}
    lead_delta = (cC - cA).mean(0)
    out = {
        "provenance": "per-window lead NMSE from shared/local_eval.py (matches official scorer); official clip order",
        "comparison": "C_on_A - A_official (negative = C better)",
        "forecast_nmse": {"A": float(fA.mean()), "C": float(fC.mean()), **paired(dfull)},
        "forecast_nmse_final": {"A": float(gA.mean()), "C": float(gC.mean()), **paired(dfinal)},
        "by_regime": by_regime,
        "lead_delta_case_avg": [float(v) for v in lead_delta],
        "cases": [{"case_id": ids[i], "k": float(k[i]), "theta0_deg": float(np.degrees(th[i])),
                   "k_theta0_rad": float(kth[i]), "regime": REGIMES[reg[i]],
                   "A_nmse": float(fA[i]), "C_nmse": float(fC[i]), "delta_nmse": float(dfull[i]),
                   "A_nmse_final": float(gA[i]), "C_nmse_final": float(gC[i]), "delta_nmse_final": float(dfinal[i])}
                  for i in range(len(cases))],
    }
    (a.runs / "paired_C_on_A_vs_A.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    # ---------------- Figure 1
    a.report.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    t = np.arange(1, 21) * 0.1
    for r, curve in (("A_official", cA), ("C_on_A", cC)):
        m = curve.mean(0)
        ax[0].plot(t, m, color=COL[r], lw=2, label=LABEL[r])
        ax[0].plot(t[-1], m[-1], "o", color=COL[r], ms=4, mec="white", mew=1)
        ax[0].annotate(f"{m[-1]:.3f}", (t[-1], m[-1]), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=7, color=INK)
    ax[0].set_xlabel("lead time (c/U)")
    ax[0].set_ylabel("NMSE (clipped per lead, case-avg)")
    ax[0].set_xlim(0, 2.25)
    ax[0].set_ylim(bottom=0)
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].set_title("(a) forecast error vs lead, 12 validation cases", loc="left", color=INK)
    style(ax[0])
    for r in np.unique(reg):
        s = reg == r
        ax[1].scatter(kth[s], dfull[s], marker=MARK[r], s=36, facecolor=COL["C_on_A"], edgecolor="white", lw=1,
                      label=f"{REGIMES[r]} (n={s.sum()})", zorder=3)
    ax[1].axhline(0, color=INK, lw=0.8, zorder=2)
    ax[1].set_xlabel(r"$k\,\theta_0$ (rad)")
    ax[1].set_ylabel(r"$\Delta$ case NMSE (C$-$A)")
    ax[1].legend(fontsize=7, frameon=False, title="true regime", title_fontsize=7)
    ax[1].set_title("(b) per-case change; below 0 = C better", loc="left", color=INK)
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
    for key in ("forecast_nmse", "forecast_nmse_final"):
        s = out[key]
        print(f"{key}: A {s['A']:.4f} C {s['C']:.4f} delta {s['mean']:+.4f} CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
              f" improved {s['improved']}/{s['n_cases']} worsened {s['worsened']}")
    print("by regime:", json.dumps(by_regime))
    print("lead delta (case avg):", " ".join(f"{v:+.3f}" for v in lead_delta))
    for i in np.argsort(kth):
        print(f"  {ids[i]:7s} kθ0={kth[i]:.2f} {REGIMES[reg[i]]:14s} Δ={dfull[i]:+.4f} Δfinal={dfinal[i]:+.4f}")
    print(f"fig2 case {ids[ci]} start {s0}; wrote {a.report/'fig1.pdf'}, {a.report/'fig2.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
