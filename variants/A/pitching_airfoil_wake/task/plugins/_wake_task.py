"""Task data path for the pitching-airfoil wake task (ORG-MAE-2).

Adapted from the SIDERIUS CSV classification template
(examples/task_templates/csv_classification/plugins/_csv_task.py, v1.1.0) via
the organizer field-surrogate pattern: one sample is one forecast window of one
simulated case instead of one CSV row.

Data layout. The run's data folder holds ``train.npz`` and ``validation.npz``
written by ``fetch_data.py``. Each holds, for ``n`` cases (one pitching
kinematics each), ``F`` stored frames and ``W`` evaluation windows:

  case_id     [n]            str      case name
  k           [n]            float32  reduced frequency k = omega c / U
  theta0_deg  [n]            float32  pitch amplitude (degrees), theta(t) = theta0 sin(k t)
  t_frames    [F]            float32  frame times (convective units c/U), 0.1 apart
  vort        [n,F,48,96]    float16  spanwise vorticity omega_z c / U on a 16-cells-per-chord
                                      grid, x in [-0.5, 5.5] c from the leading edge (columns),
                                      y in [-1.5, 1.5] c (rows, row 0 at y = -1.5 c)
  probe_t     [S]            float32  probe sample times, 0.05 apart, from t = 0
  probe_v     [n,S,5]        float32  cross-stream velocity v/U at five wake probes
  mean_vort   [n,48,96]      float32  time-mean vorticity over t in [15, 60]   (target)
  log_psd     [n,50]         float32  log10 normalised wake spectrum in 50 bins of f c/U in
                                      [0, 2.5]                                 (target)
  st_dom      [n]            float32  dominant wake frequency f c / U          (target)
  regime      [n]            int64    0 periodic, 1 quasi-periodic, 2 aperiodic  (target)
  win_case    [W]            int64    case index of each evaluation window
  win_start   [W]            int64    index of the first forecast frame of the window

Model input per window is ``[14, 48, 96]`` float32: channels 0-9 are the ten
frames before the forecast start (oldest first), channel 10 is k / (2 pi)
(forcing frequency, constant map), 11 is theta0 in radians, 12 and 13 are
sin(k t_s) and cos(k t_s) with t_s the time of the first forecast frame (the
forcing phase). Output (and supervision) per window is a flat ``[96822]``
float32 vector, see ``unpack``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.seeding import MissingRunSeed
from execute_tools.task_data_path import (
    DeliverableWriteRequest,
    EpochSamplingParams,
    EvalMaterializationParams,
    EvaluationReadRequest,
    HealthCoverageRequest,
    HealthCoverageResult,
    ScopeBuildRequest,
    StorageReadScope,
    TaskOutputArtifactInventory,
    ValidationScopeError,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset

TASK_DATA_PATH_ID = "pitching_airfoil_wake"
TRAINING_PARTITION = 0
VALIDATION_PARTITION = 1
SCOPE_KIND = "pitching_wake_windows"
NY, NX = 48, 96
HISTORY = 10
HORIZON = 20
N_BINS = 50
N_REGIMES = 3
N_IN = HISTORY + 4
FIELD = NY * NX
OUT_DIM = HORIZON * FIELD + FIELD + N_BINS + 1 + N_REGIMES  # 96822
TRAIN_STRIDE = 5
DELIVERABLE_FILE = "pitching_wake_predictions.npz"


def unpack(out: np.ndarray) -> dict[str, np.ndarray]:
    """Split flat outputs ``[..., OUT_DIM]`` into named parts.

    forecast  [..., 20, 48, 96]  vorticity of the next 20 frames (0.1 c/U apart)
    mean_vort [..., 48, 96]      long-time mean vorticity of the case
    log_psd   [..., 50]          log10 normalised wake spectrum
    st_dom    [...]              dominant wake frequency f c / U
    regime_logits [..., 3]       unnormalised log-probabilities (periodic, quasi-periodic, aperiodic)
    """
    lead = out.shape[:-1]
    a = HORIZON * FIELD
    b = a + FIELD
    c = b + N_BINS
    return {
        "forecast": out[..., :a].reshape(*lead, HORIZON, NY, NX),
        "mean_vort": out[..., a:b].reshape(*lead, NY, NX),
        "log_psd": out[..., b:c],
        "st_dom": out[..., c],
        "regime_logits": out[..., c + 1 : c + 1 + N_REGIMES],
    }


class WindowSelection(BaseModel):
    """A scope: which split file to read and how to sample its windows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["pitching_wake_windows"] = SCOPE_KIND
    file: str = Field(min_length=1)
    training: bool
    portion: float = Field(gt=0.0, le=1.0)
    seed: int
    max_samples: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class WakeTable:
    """One split file."""

    case_ids: tuple[str, ...]
    k: np.ndarray
    theta0_deg: np.ndarray
    t_frames: np.ndarray
    vort: np.ndarray  # float16 [n, F, 48, 96]
    mean_vort: np.ndarray
    log_psd: np.ndarray
    st_dom: np.ndarray
    regime: np.ndarray
    win_case: np.ndarray
    win_start: np.ndarray

    def windows(self, training: bool) -> list[tuple[int, int]]:
        """(case, start) pairs: all strided windows for training, declared windows for scoring."""
        if not training:
            return [(int(c), int(s)) for c, s in zip(self.win_case, self.win_start)]
        n_frames = self.vort.shape[1]
        return [(c, s) for c in range(len(self.case_ids)) for s in range(HISTORY, n_frames - HORIZON + 1, TRAIN_STRIDE)]

    def features(self, case: int, start: int) -> np.ndarray:
        x = np.empty((N_IN, NY, NX), dtype=np.float32)
        x[:HISTORY] = self.vort[case, start - HISTORY : start].astype(np.float32)
        k = float(self.k[case])
        ts = float(self.t_frames[start])
        x[HISTORY] = k / (2 * math.pi)
        x[HISTORY + 1] = math.radians(float(self.theta0_deg[case]))
        x[HISTORY + 2] = math.sin(k * ts)
        x[HISTORY + 3] = math.cos(k * ts)
        return x

    def target(self, case: int, start: int) -> np.ndarray:
        """Supervision: true future frames, case mean field, spectrum, St and one-hot regime."""
        y = np.empty(OUT_DIM, dtype=np.float32)
        parts = unpack(y)
        n_frames = self.vort.shape[1]
        stop = min(start + HORIZON, n_frames)
        parts["forecast"][: stop - start] = self.vort[case, start:stop].astype(np.float32)
        parts["forecast"][stop - start :] = np.nan
        parts["mean_vort"][...] = self.mean_vort[case]
        parts["log_psd"][...] = self.log_psd[case]
        y[HORIZON * FIELD + FIELD + N_BINS] = self.st_dom[case]
        onehot = np.zeros(N_REGIMES, dtype=np.float32)
        onehot[int(self.regime[case])] = 1.0
        parts["regime_logits"][...] = onehot
        return y


_CACHE: dict[tuple[str, float, int], WakeTable] = {}

REQUIRED = ("case_id", "k", "theta0_deg", "t_frames", "vort", "mean_vort", "log_psd", "st_dom", "regime",
            "win_case", "win_start")


def read_table(path: str) -> WakeTable:
    """Parse and validate one split file; cached per (path, mtime, size)."""
    if not os.path.isfile(path):
        raise ValidationScopeError(f"split file not found in the data folder: {path}")
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime, stat.st_size)
    if key in _CACHE:
        return _CACHE[key]
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED if name not in data.files]
        if missing:
            raise ValueError(f"{path} lacks arrays {missing}")
        arrays = {name: data[name] for name in REQUIRED}
    n = len(arrays["case_id"])
    if n == 0:
        raise ValueError(f"{path} holds no cases")
    n_frames = len(arrays["t_frames"])
    if arrays["vort"].shape != (n, n_frames, NY, NX):
        raise ValueError(f"{path}: vort has shape {arrays['vort'].shape}")
    if arrays["log_psd"].shape != (n, N_BINS):
        raise ValueError(f"{path}: log_psd has shape {arrays['log_psd'].shape}")
    ids = tuple(str(v) for v in arrays["case_id"])
    if len(set(ids)) != n:
        raise ValueError(f"{path} repeats a case_id")
    wc, ws = arrays["win_case"].astype(np.int64), arrays["win_start"].astype(np.int64)
    if len(wc) and (wc.min() < 0 or wc.max() >= n or ws.min() < HISTORY or ws.max() + HORIZON > n_frames):
        raise ValueError(f"{path}: evaluation windows out of range")
    table = WakeTable(
        case_ids=ids, k=arrays["k"].astype(np.float32), theta0_deg=arrays["theta0_deg"].astype(np.float32),
        t_frames=arrays["t_frames"].astype(np.float32), vort=arrays["vort"],
        mean_vort=arrays["mean_vort"].astype(np.float32), log_psd=arrays["log_psd"].astype(np.float32),
        st_dom=arrays["st_dom"].astype(np.float32), regime=arrays["regime"].astype(np.int64),
        win_case=wc, win_start=ws,
    )
    _CACHE.clear()
    _CACHE[key] = table
    return table


class _WindowDataset:
    """Lazy torch-style dataset: windows are cut from the frame stack on access."""

    def __init__(self, table: WakeTable, pairs: list[tuple[int, int]]) -> None:
        self.table, self.pairs = table, pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import torch

        case, start = self.pairs[index]
        return torch.from_numpy(self.table.features(case, start)), torch.from_numpy(self.table.target(case, start))


class WakeTaskDataPath:
    """Reads split files; writes predicted windows as one npz deliverable."""

    task_data_path_id: ClassVar[str] = TASK_DATA_PATH_ID

    def __init__(self, *, train_file: str, validation_file: str) -> None:
        for name, value in (("train_file", train_file), ("validation_file", validation_file)):
            if not value or os.path.isabs(value) or os.path.normpath(value).startswith(".."):
                raise ValueError(f"{name} must be a file name inside the data folder; got {value!r}")
        if train_file == validation_file:
            raise ValueError("train_file and validation_file must be different files")
        self._train_file = train_file
        self._validation_file = validation_file

    @staticmethod
    def _subsample(count: int, portion: float | None, max_samples: int | None, seed: int | None) -> list[int]:
        keep = count
        if portion is not None and portion < 1.0:
            keep = max(1, math.floor(count * portion))
        if max_samples is not None:
            keep = min(keep, max_samples)
        if keep >= count:
            return list(range(count))
        if seed is None:
            raise MissingRunSeed("subsampling windows needs a seed; none was supplied")
        drawn = np.random.default_rng(seed).permutation(count)[:keep]
        return sorted(int(index) for index in drawn)

    def windows(self, scope: object, data_dir: str) -> tuple[WakeTable, list[tuple[int, int]]]:
        """The split table and the selected (case, start) windows; the single selection rule."""
        selection = self._require_scope(scope)
        table = read_table(os.path.join(data_dir, selection.file))
        pairs = table.windows(selection.training)
        keep = self._subsample(len(pairs), selection.portion, selection.max_samples, selection.seed)
        return table, [pairs[i] for i in keep]

    def training_dataset(self, scope: object, params: EpochSamplingParams) -> Dataset[Any]:
        table, pairs = self.windows(scope, params.data_dir)
        kept = self._subsample(len(pairs), params.train_portion, params.max_samples, params.epoch_seed)
        return _WindowDataset(table, [pairs[i] for i in kept])  # type: ignore[return-value]

    def validation_dataset(self, scope: object, params: EvalMaterializationParams) -> Dataset[Any]:
        table, pairs = self.windows(scope, params.data_dir)
        return _WindowDataset(table, pairs)  # type: ignore[return-value]

    @staticmethod
    def deliverable_name(*, model_type: str, run_name: str, exp_id: str) -> str:
        return DELIVERABLE_FILE

    def write_deliverable(self, outputs: Any, request: DeliverableWriteRequest) -> None:
        """Pair each window's flat output with its (case id, start) and save the parts."""
        if request.task_scope is None or request.source_context is None:
            raise ValueError("positional outputs need request.task_scope and request.source_context")
        table, pairs = self.windows(request.task_scope, request.source_context.data_dir)
        flat = _stack_outputs(outputs)
        if flat.shape[0] != len(pairs) or len(pairs) != request.source_context.sample_count:
            raise ValueError(f"{flat.shape[0]} outputs for {len(pairs)} windows; refusing to pair")
        if flat.shape[1:] != (OUT_DIM,):
            raise ValueError(f"each output must be [{OUT_DIM}]; got {tuple(flat.shape[1:])}")
        os.makedirs(request.output_dir, exist_ok=True)
        save_deliverable(os.path.join(request.output_dir, DELIVERABLE_FILE),
                         case_ids=[table.case_ids[c] for c, _ in pairs], starts=[s for _, s in pairs], flat=flat)

    def read_evaluation_payload(self, request: EvaluationReadRequest) -> object:
        return load_deliverable(os.path.join(request.deliverable_dir, DELIVERABLE_FILE))

    def enumerate_output_artifacts(self, request: EvaluationReadRequest) -> TaskOutputArtifactInventory:
        path = Path(request.deliverable_dir) / DELIVERABLE_FILE
        return TaskOutputArtifactInventory(
            run_name=request.run_name, exp_id=request.exp_id, model_type=request.model_type,
            relative_paths=(DELIVERABLE_FILE,) if path.exists() or path.is_symlink() else (),
        )

    def _require_scope(self, scope: object) -> WindowSelection:
        if not isinstance(scope, WindowSelection):
            raise ValueError(f"expected a WindowSelection scope, got {type(scope).__name__}")
        return scope

    def _selection(self, file: str, request: ScopeBuildRequest, training: bool) -> WindowSelection:
        if request.subset_ref is not None:
            raise ValueError(f"{TASK_DATA_PATH_ID} has no data subsets; got {request.subset_ref!r}")
        seed = request.seed
        if seed is None:
            if request.portion < 1.0 or request.max_samples is not None:
                raise MissingRunSeed("a sampled window scope (portion < 1 or max_samples) requires request.seed")
            seed = 0
        return WindowSelection(file=file, training=training, portion=request.portion, seed=seed,
                               max_samples=request.max_samples)

    def build_training_scope(self, request: ScopeBuildRequest) -> object:
        if request.selection_strategy == "target" and tuple(request.target_partitions) != (TRAINING_PARTITION,):
            raise ValueError(f"the only training partition is {TRAINING_PARTITION}; "
                             f"got target partitions {list(request.target_partitions)}")
        return self._selection(self._train_file, request, training=True)

    def build_eval_scope(self, request: ScopeBuildRequest) -> object:
        return self._selection(self._validation_file, request, training=False)

    def serialize_scope(self, scope: object) -> str:
        return json.dumps(self._require_scope(scope).model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def deserialize_scope(self, payload: str) -> object:
        return WindowSelection.model_validate_json(payload)

    def storage_read_scope(self, data_dir: str, scope: object) -> StorageReadScope:
        path = os.path.abspath(os.path.join(data_dir, self._require_scope(scope).file))
        return StorageReadScope(file_paths=(path,), expected_on_disk_bytes=os.path.getsize(path))

    def validate_health_coverage(self, request: HealthCoverageRequest) -> HealthCoverageResult:
        scope = self._require_scope(request.evaluation_scope)
        override = request.health_gate_files
        if override is not None and tuple(override) != (VALIDATION_PARTITION,):
            return HealthCoverageResult(applicable=True, covered=False,
                                        reason=f"Health reads only partition {VALIDATION_PARTITION}; "
                                               f"override {list(override)} unsupported")
        if scope.file != self._validation_file:
            return HealthCoverageResult(applicable=True, covered=False,
                                        reason=f"evaluation scope reads {scope.file}, not {self._validation_file}")
        return HealthCoverageResult(applicable=True, covered=True,
                                    reason=f"Health checks read the predicted windows for {scope.file}")


def _stack_outputs(outputs: Any) -> np.ndarray:
    import torch

    if isinstance(outputs, Mapping):
        raise ValueError("outputs must be positional (one flat vector per window in dataset order)")
    items = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            output = output.detach().to("cpu", torch.float32).numpy()
        items.append(np.asarray(output, dtype=np.float32).reshape(-1))
    if not items:
        raise ValueError("no outputs to write")
    return np.stack(items)


def save_deliverable(path: str, *, case_ids: list[str], starts: list[int], flat: np.ndarray) -> None:
    """The one writer of the deliverable format (also used by the standalone baseline)."""
    parts = unpack(np.asarray(flat, dtype=np.float32))
    temporary = path + ".partial.npz"
    np.savez(temporary, case_id=np.asarray(case_ids, dtype=str), win_start=np.asarray(starts, dtype=np.int64),
             forecast=parts["forecast"].astype(np.float32), mean_vort=parts["mean_vort"].astype(np.float32),
             log_psd=parts["log_psd"].astype(np.float32), st_dom=parts["st_dom"].astype(np.float32),
             regime_logits=parts["regime_logits"].astype(np.float32))
    os.replace(temporary, path)


def load_deliverable(path: str) -> dict[str, np.ndarray]:
    """Decode the deliverable into named arrays; refuses a malformed file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"prediction file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        decoded = {name: data[name] for name in data.files}
    shapes = {"forecast": (HORIZON, NY, NX), "mean_vort": (NY, NX), "log_psd": (N_BINS,), "st_dom": (),
              "regime_logits": (N_REGIMES,)}
    for name in ("case_id", "win_start", *shapes):
        if name not in decoded:
            raise ValueError(f"{path} lacks array {name!r}")
    n = len(decoded["case_id"])
    for name, shape in shapes.items():
        if decoded[name].shape != (n, *shape):
            raise ValueError(f"{path}: {name} has shape {decoded[name].shape}, expected {(n, *shape)}")
    decoded["case_id"] = decoded["case_id"].astype(str)
    return decoded


def deliverable_name(request: DeliverableWriteRequest | EvaluationReadRequest) -> str:
    """Module-level deliverable name: this task names its own artifact."""
    return DELIVERABLE_FILE
