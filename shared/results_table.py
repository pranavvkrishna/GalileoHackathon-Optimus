"""P3: the spec §5 results table, one row per configuration, as Markdown + LaTeX + JSON.

    python shared/results_table.py --out report/results_table \
        --row "A_official|0|organizer weights (not retrained locally)|runs/A_official" \
        --row "C_on_A|0|organizer weights + mirror averaging|runs/C_on_A" \
        --row "A_s0|0|local train, seed 0|runs/A_s0|runs/A_s0/train_log.txt"

Row = NAME|SEED|PROVENANCE|RUN_DIR[|TRAIN_LOG]. RUN_DIR must contain
  metrics.json             official `run_baseline.py evaluate` (or the train-time evaluation) metrics
  local_eval/metrics.json  shared/local_eval.py (reflection defect)
  gates/gates.json         shared/health_gate.py (health + latency); missing -> "not run"
--metrics NAME=FILE overrides the official metrics path for a row (e.g. runs/C_on_A/official/metrics.json).
train_seconds comes from the last JSON line of TRAIN_LOG ("-" if no log: no local training).

utility = [2*clip(1-forecast_nmse,0,1) + 0.5*clip(1-st_abs_error/0.3,0,1)] / 2.5   (spec §3)
The script also asserts that the official forecast_nmse and the local_eval value agree to 1e-6.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ("configuration", "seed", "provenance", "split", "forecast_nmse", "forecast_nmse_final", "st_abs_error",
        "utility", "reflection_defect", "health_status", "latency_ms", "train_seconds")


def clip01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def utility(nmse: float, st: float) -> float:
    return (2 * clip01(1 - nmse) + 0.5 * clip01(1 - st / 0.3)) / 2.5


def train_seconds(log: Path | None):
    if log is None:
        return None
    for line in reversed(log.read_text(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{") and "train_seconds" in line:
            return float(json.loads(line)["train_seconds"])
    raise ValueError(f"no train_seconds JSON line in {log}")


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--metrics", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True, help="path stem; writes .md, .tex, .json")
    a = ap.parse_args()
    override = dict(m.split("=", 1) for m in a.metrics)

    rows = []
    for spec in a.row:
        f = spec.split("|")
        if len(f) not in (4, 5):
            raise SystemExit(f"bad --row {spec!r}")
        name, seed, prov, rdir = f[:4]
        rdir = Path(rdir)
        log = Path(f[4]) if len(f) == 5 and f[4] else None
        m = json.loads(Path(override.get(name, rdir / "metrics.json")).read_text())
        m = m.get("metrics", m)
        le = json.loads((rdir / "local_eval" / "metrics.json").read_text())
        assert abs(le["metrics"]["forecast_nmse"] - m["forecast_nmse"]) < 1e-6, \
            f"{name}: local_eval {le['metrics']['forecast_nmse']} != official {m['forecast_nmse']}"
        gp = rdir / "gates" / "gates.json"
        g = json.loads(gp.read_text()) if gp.is_file() else None
        rows.append({
            "configuration": name, "seed": seed, "provenance": prov, "split": "validation (12 cases, 48 windows)",
            "forecast_nmse": m["forecast_nmse"], "forecast_nmse_final": m["forecast_nmse_final"],
            "st_abs_error": m["st_abs_error"], "utility": utility(m["forecast_nmse"], m["st_abs_error"]),
            "reflection_defect": le["reflection_defect"],
            "health_status": ("pass" if g["all_passed"] else "FAIL") if g else "not run",
            "latency_ms": g["latency_ms"]["median"] if g else None,
            "latency_device": (g["latency_ms"]["gpu"] or g["latency_ms"]["device"]) if g else None,
            "train_seconds": train_seconds(log),
        })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    md = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        md.append("| " + " | ".join(fmt(r[c], 1 if c in ("latency_ms", "train_seconds") else 4) for c in COLS) + " |")
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def tex(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    head = ["Config", "Seed", "Provenance", "NMSE", r"NMSE$_{20}$", "St err", "Utility", "Refl.\\ defect",
            "Health", "Lat.\\ (ms)", "Train (s)"]
    tl = [r"\begin{tabular}{lllrrrrrlrr}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in rows:
        tl.append(" & ".join([tex(r["configuration"]), r["seed"], tex(r["provenance"]), fmt(r["forecast_nmse"]),
                              fmt(r["forecast_nmse_final"]), fmt(r["st_abs_error"]), fmt(r["utility"]),
                              fmt(r["reflection_defect"], 3), r["health_status"], fmt(r["latency_ms"], 1),
                              fmt(r["train_seconds"], 0)]) + r" \\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    a.out.with_suffix(".tex").write_text("\n".join(tl) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
