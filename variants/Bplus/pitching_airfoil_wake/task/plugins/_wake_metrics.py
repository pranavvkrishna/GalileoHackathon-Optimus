"""Scores for the pitching-airfoil wake task (ORG-MAE-2).

Every metric compares the deliverable with the simulated truth of the same
evaluation windows, read from the data folder through the bound task data path
(``WakeTaskDataPath.windows``). Three layers, one per scientific output:

* short-term forecast (per window): normalised mean-squared error of the next
  20 vorticity frames (2 convective times). The normaliser is the variance of
  the true frames about the case's long-time mean field, so 1.0 equals
  predicting the mean wake; per-window values are clipped at 4 so one diverged
  window cannot dominate.
* long-time statistics (per case): the window predictions of a case are pooled
  (mean field and log-spectrum averaged, St median) and compared with the
  statistics of the full simulated record.
* regime (per case): argmax of the window-averaged softmax probabilities;
  balanced accuracy over the three classes present in the evaluation split.

A case, not a window, is the independent unit for statistics and regime.
The hidden-test scorer and the standalone baseline call these same classes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np

from execute_tools.evaluation_metric import EvaluationMetric
from execute_tools.task_data_path import require_bound_task_data_path

NMSE_CLIP = 4.0


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def aligned(payload: Mapping[str, np.ndarray], table: Any, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Predictions aligned to the evaluation windows; raises on a missing or non-finite prediction."""
    index = {(str(c), int(s)): i for i, (c, s) in enumerate(zip(payload["case_id"], payload["win_start"]))}
    keys = [(table.case_ids[c], s) for c, s in pairs]
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError(f"{len(missing)} evaluation window(s) have no prediction, e.g. {missing[:3]}")
    take = [index[k] for k in keys]
    out = {name: np.asarray(payload[name], dtype=np.float64)[take]
           for name in ("forecast", "mean_vort", "log_psd", "st_dom", "regime_logits")}
    for name, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f"predicted {name} contains NaN or infinity")
    out["pairs"] = pairs
    return out


def window_nmse(pred: Mapping[str, Any], table: Any) -> np.ndarray:
    """[W, 20] per-window, per-lead NMSE (unclipped)."""
    rows = []
    for w, (c, s) in enumerate(pred["pairs"]):
        truth = table.vort[c, s : s + pred["forecast"].shape[1]].astype(np.float64)
        ref = table.mean_vort[c].astype(np.float64)
        var = np.mean((truth - ref) ** 2)
        err = np.mean((pred["forecast"][w] - truth) ** 2, axis=(1, 2))
        rows.append(err / max(var, 1e-12))
    return np.asarray(rows)


def case_pool(pred: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    pooled: dict[int, dict[str, Any]] = {}
    cases = [c for c, _ in pred["pairs"]]
    for c in sorted(set(cases)):
        idx = [i for i, cc in enumerate(cases) if cc == c]
        pooled[c] = {
            "mean_vort": pred["mean_vort"][idx].mean(axis=0),
            "log_psd": pred["log_psd"][idx].mean(axis=0),
            "st_dom": float(np.median(pred["st_dom"][idx])),
            "regime_prob": _softmax(pred["regime_logits"][idx]).mean(axis=0),
        }
    return pooled


def balanced_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    recalls = [float(np.mean(pred[true == c] == c)) for c in np.unique(true)]
    return float(np.mean(recalls))


def score(name: str, pred: Mapping[str, Any], table: Any) -> float:
    if name in ("forecast_nmse", "forecast_nmse_final"):
        e = np.minimum(window_nmse(pred, table), NMSE_CLIP)
        return float(e.mean()) if name == "forecast_nmse" else float(e[:, -1].mean())
    pooled = case_pool(pred)
    cases = sorted(pooled)
    if name == "mean_field_rel_l2":
        v = [np.linalg.norm(pooled[c]["mean_vort"] - table.mean_vort[c]) / max(np.linalg.norm(table.mean_vort[c]), 1e-12)
             for c in cases]
        return float(np.mean(v))
    if name == "psd_log_rmse":
        return float(np.mean([np.sqrt(np.mean((pooled[c]["log_psd"] - table.log_psd[c]) ** 2)) for c in cases]))
    if name == "st_abs_error":
        return float(np.mean([abs(pooled[c]["st_dom"] - float(table.st_dom[c])) for c in cases]))
    if name == "regime_balanced_accuracy":
        t = np.asarray([int(table.regime[c]) for c in cases])
        p = np.asarray([int(np.argmax(pooled[c]["regime_prob"])) for c in cases])
        return balanced_accuracy(t, p)
    raise KeyError(name)


def _evaluate(name: str, payload: Mapping[str, np.ndarray], task_scope: Any, data_dir: str | None) -> float:
    if data_dir is None:
        raise ValueError("these metrics read the simulated fields from the data folder; data_dir is None")
    windows: Callable[..., Any] | None = getattr(require_bound_task_data_path(), "windows", None)
    if not callable(windows):
        raise TypeError("the bound task data path does not provide windows(scope, data_dir)")
    table, pairs = windows(task_scope, data_dir)
    return score(name, aligned(payload, table, pairs), table)


class _WakeMetric(EvaluationMetric):
    NAME: ClassVar[str] = ""

    def _compute(self, deliverables: Mapping[int, str], /, *, evaluation_payload: Mapping[str, np.ndarray],
                 task_scope: Any, data_dir: str | None = None) -> tuple[float, list[float | None] | None, tuple[str, ...]]:
        return _evaluate(self.NAME, evaluation_payload, task_scope, data_dir), None, ()


class ForecastNmse(_WakeMetric):
    """Primary: NMSE of the 20-frame vorticity forecast, mean over leads and windows."""

    NAME = "forecast_nmse"
    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("forecast_nmse",)


class ForecastNmseFinal(_WakeMetric):
    NAME = "forecast_nmse_final"
    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("forecast_nmse_final",)


class MeanFieldRelL2(_WakeMetric):
    NAME = "mean_field_rel_l2"
    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("mean_field_rel_l2",)


class PsdLogRmse(_WakeMetric):
    NAME = "psd_log_rmse"
    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("psd_log_rmse",)


class StAbsError(_WakeMetric):
    NAME = "st_abs_error"
    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("st_abs_error",)


class RegimeBalancedAccuracy(_WakeMetric):
    NAME = "regime_balanced_accuracy"
    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("regime_balanced_accuracy",)
