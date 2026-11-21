"""Standalone baseline for the pitching-airfoil wake task (ORG-MAE-2).

Two parts in one PyTorch module (``WakeBaseline``):

* forecast: a small convolutional encoder-decoder maps the ten history frames
  plus the kinematics/phase channels directly to the next 20 frames (residual
  on the last history frame), trained with MSE on all strided training windows;
* statistics and regime: inverse-distance k-nearest-neighbour interpolation
  (3 neighbours) in the normalised (k, theta0) plane over the training cases,
  stored as buffers. The dominant frequency is interpolated as a multiple of
  the forcing frequency; the regime as a weighted vote.

Run with the SIDERIUS interpreter (the task's plugins import the framework)::

    $PY run_baseline.py init     --seed S --output init_model.pt
    $PY run_baseline.py train    --data_dir D --seed S --output_dir O --init_model init_model.pt
    $PY run_baseline.py evaluate --data_dir D --model final_model.pt --output_dir O
    $PY run_baseline.py predict  --data_dir D --split_file F --model M --output_dir O   (helper)

A CUDA GPU is used when available (one GPU).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "task" / "task.yaml"

HISTORY, HORIZON, NY, NX, N_BINS = 10, 20, 48, 96, 50
FIELD = NY * NX
OUT_DIM = HORIZON * FIELD + FIELD + N_BINS + 1 + 3
MAX_CASES = 1200
MEAN_MODES = 32  # mean fields of the training cases are stored as a PCA basis (keeps artifacts < 5 MiB)
WIDTH = 64
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VORT_SCALE = 4.0
KNN = 3
WINDOW_SUBSAMPLE = 3  # every 3rd of the data path's strided training windows (start stride 15 frames = 1.5 c/U); keeps the baseline under 30 min on an RTX 5090
K_RANGE, TH_RANGE = (2.0, 7.0), (math.radians(5.0), math.radians(35.0))


def setup_torch() -> torch.device:
    torch.set_num_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())


class WakeBaseline(nn.Module):
    """[B, 14, 48, 96] -> [B, 96822]."""

    def __init__(self) -> None:
        super().__init__()
        w = WIDTH
        self.e1 = block(14, w)
        self.e2 = block(w, 2 * w)
        self.e3 = block(2 * w, 4 * w)
        self.d2 = block(4 * w + 2 * w, 2 * w)
        self.d1 = block(2 * w + w, w)
        self.head = nn.Conv2d(w, HORIZON, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.register_buffer("n_cases", torch.zeros((), dtype=torch.long))
        self.register_buffer("kin", torch.zeros(MAX_CASES, 2))
        self.register_buffer("mean_avg", torch.zeros(FIELD))
        self.register_buffer("mean_basis", torch.zeros(MEAN_MODES, FIELD))
        self.register_buffer("mean_coef", torch.zeros(MAX_CASES, MEAN_MODES))
        self.register_buffer("log_psd", torch.zeros(MAX_CASES, N_BINS))
        self.register_buffer("st_ratio", torch.zeros(MAX_CASES))
        self.register_buffer("regime", torch.zeros(MAX_CASES, 3))

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x[:, :HISTORY] / VORT_SCALE, x[:, HISTORY:]], dim=1)
        h1 = self.e1(z)
        h2 = self.e2(self.pool(h1))
        h3 = self.e3(self.pool(h2))
        u2 = self.d2(torch.cat([self.up(h3), h2], dim=1))
        u1 = self.d1(torch.cat([self.up(u2), h1], dim=1))
        return x[:, HISTORY - 1 : HISTORY] + VORT_SCALE * self.head(u1)

    @staticmethod
    def normalised_kinematics(k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.stack([(k - K_RANGE[0]) / (K_RANGE[1] - K_RANGE[0]),
                            (theta - TH_RANGE[0]) / (TH_RANGE[1] - TH_RANGE[0])], dim=-1)

    def statistics(self, x: torch.Tensor) -> torch.Tensor:
        n = int(self.n_cases)
        k = x[:, HISTORY, 0, 0] * 2 * math.pi
        theta = x[:, HISTORY + 1, 0, 0]
        q = self.normalised_kinematics(k, theta)
        d = torch.cdist(q, self.kin[:n])
        dist, idx = d.topk(min(KNN, n), dim=1, largest=False)
        w = 1.0 / (dist + 1e-3)
        w = w / w.sum(dim=1, keepdim=True)
        mean = self.mean_avg + (w[..., None] * self.mean_coef[idx]).sum(1) @ self.mean_basis
        psd = (w[..., None] * self.log_psd[idx]).sum(1)
        st = (w * self.st_ratio[idx]).sum(1) * k / (2 * math.pi)
        prob = (w[..., None] * self.regime[idx]).sum(1)
        return torch.cat([mean, psd, st[:, None], torch.log(prob + 1e-3)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.forecast(x).flatten(1), self.statistics(x)], dim=1)


def compose_task():
    from workflows.task_composition import compose_run_task_bindings

    return compose_run_task_bindings(str(MANIFEST))


def full_scope(data_path, *, training: bool):
    from execute_tools.task_data_path import ScopeBuildRequest

    request = ScopeBuildRequest(round_kind="formal", selection_strategy="snapshot", portion=1.0)
    return data_path.build_training_scope(request) if training else data_path.build_eval_scope(request)


def half_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict stored in float16 (keeps each artifact under the 5 MiB file limit)."""
    return {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _read_request(directory: Path):
    from execute_tools.task_data_path import EvaluationReadRequest

    return EvaluationReadRequest(deliverable_dir=str(directory), exp_id="baseline", run_name="baseline",
                                 model_type="wake_unet_knn_baseline")


def score(composition, scope, data_dir: Path, deliverable: Path) -> dict[str, float]:
    from execute_tools.task_data_path import bind_task_data_path

    data_path = composition.task_data_path
    payload = data_path.read_evaluation_payload(_read_request(deliverable.parent))
    metrics: dict[str, float] = {}
    with bind_task_data_path(data_path):
        for metric in [composition.metric, *composition.secondary_metrics]:
            outcome = metric.evaluate({0: str(deliverable)}, evaluation_payload=payload, task_scope=scope,
                                      data_dir=str(data_dir))
            scalar = getattr(outcome, "scalar", None)
            if scalar is None:
                raise RuntimeError(f"metric {metric.spec.id} could not score: {outcome}")
            metrics[metric.spec.id] = float(scalar)
    return metrics


@torch.no_grad()
def write_predictions(model: nn.Module, composition, scope, data_dir: Path, output_dir: Path, device) -> Path:
    data_path = composition.task_data_path
    table, pairs = data_path.windows(scope, str(data_dir))
    model.eval()
    outs = []
    for s in range(0, len(pairs), 16):
        x = torch.from_numpy(np.stack([table.features(c, t) for c, t in pairs[s : s + 16]])).to(device)
        outs.append(model(x).float().cpu().numpy())
    flat = np.concatenate(outs)
    module = importlib.import_module(type(data_path).__module__)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / module.DELIVERABLE_FILE
    module.save_deliverable(str(path), case_ids=[table.case_ids[c] for c, _ in pairs], starts=[t for _, t in pairs],
                            flat=flat)
    return path


def evaluate_model(model, composition, data_dir: Path, output_dir: Path, device) -> dict[str, float]:
    scope = full_scope(composition.task_data_path, training=False)
    deliverable = write_predictions(model, composition, scope, data_dir, output_dir, device)
    metrics = score(composition, scope, data_dir, deliverable)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def fill_statistics(model: WakeBaseline, table) -> None:
    n = len(table.case_ids)
    if n > MAX_CASES:
        raise ValueError(f"{n} training cases exceed the buffer size {MAX_CASES}")
    k = torch.from_numpy(table.k.astype(np.float32))
    th = torch.from_numpy(np.radians(table.theta0_deg).astype(np.float32))
    model.n_cases.fill_(n)
    model.kin[:n] = model.normalised_kinematics(k, th)
    fields = torch.from_numpy(table.mean_vort.reshape(n, -1)).double()
    avg = fields.mean(0)
    _, _, vh = torch.linalg.svd(fields - avg, full_matrices=False)
    basis = vh[:MEAN_MODES]
    model.mean_avg.copy_(avg.float())
    model.mean_basis[: basis.shape[0]] = basis.float()
    model.mean_coef[:n, : basis.shape[0]] = ((fields - avg) @ basis.T).float()
    model.log_psd[:n] = torch.from_numpy(table.log_psd)
    model.st_ratio[:n] = torch.from_numpy(table.st_dom) / (k / (2 * math.pi))
    model.regime[:n] = torch.nn.functional.one_hot(torch.from_numpy(table.regime), 3).float()


def command_init(args) -> int:
    setup_torch()
    torch.manual_seed(args.seed)
    model = WakeBaseline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(half_state(model), args.output)
    print(f"wrote {args.output} ({sum(p.numel() for p in model.parameters())} parameters)")
    return 0


def command_train(args) -> int:
    device = setup_torch()
    torch.use_deterministic_algorithms(True, warn_only=True)
    composition = compose_task()
    data_path = composition.task_data_path
    table, pairs = data_path.windows(full_scope(data_path, training=True), str(args.data_dir))
    pairs = pairs[::WINDOW_SUBSAMPLE]
    model = WakeBaseline()
    if args.init_model is None:
        raise SystemExit("this PyTorch baseline needs --init_model (baseline/init_model.pt)")
    model.load_state_dict(load_state(args.init_model))
    fill_statistics(model, table)
    model.to(device)
    vort = torch.from_numpy(np.ascontiguousarray(table.vort)).to(device)  # float16 [n, F, 48, 96]
    kin = torch.tensor([[k / (2 * math.pi), math.radians(t)] for k, t in zip(table.k, table.theta0_deg)],
                       dtype=torch.float32, device=device)
    kk = torch.from_numpy(table.k.astype(np.float32)).to(device)
    tf = torch.from_numpy(table.t_frames).to(device)
    pc = torch.tensor([c for c, _ in pairs], device=device)
    ps = torch.tensor([s for _, s in pairs], device=device)
    off_in = torch.arange(-HISTORY, 0, device=device)
    off_out = torch.arange(0, HORIZON, device=device)

    def batch(ix):
        c, s = pc[ix], ps[ix]
        hist = vort[c[:, None], s[:, None] + off_in].float()
        fut = vort[c[:, None], s[:, None] + off_out].float()
        ph = kk[c] * tf[s]
        const = torch.stack([kin[c, 0], kin[c, 1], torch.sin(ph), torch.cos(ph)], 1)[:, :, None, None]
        x = torch.cat([hist, const.expand(-1, -1, NY, NX)], 1)
        return x, fut

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * math.ceil(len(pairs) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=max(steps, 1), pct_start=0.1)
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(pairs), generator=generator).to(device)
        running = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            x, y = batch(order[start : start + BATCH_SIZE])
            optimizer.zero_grad(set_to_none=True)
            loss = ((model.forecast(x) - y) ** 2).mean() / VORT_SCALE**2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(x)
        print(f"epoch {epoch:3d} loss {running / len(pairs):.5f} {time.perf_counter() - started:.0f}s", flush=True)
    train_seconds = time.perf_counter() - started
    del vort
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(half_state(model), args.output_dir / "final_model.pt")
    model.load_state_dict(load_state(args.output_dir / "final_model.pt"))  # score exactly the stored weights
    started = time.perf_counter()
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir, device)
    report = {"metrics": metrics, "train_seconds": round(train_seconds, 1),
              "evaluate_seconds": round(time.perf_counter() - started, 1), "device": str(device),
              "train_windows": len(pairs), "train_cases": len(table.case_ids)}
    if device.type == "cuda":
        report["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, sort_keys=True))
    return 0


def _load_model(path: Path, device) -> WakeBaseline:
    model = WakeBaseline()
    model.load_state_dict(load_state(path))
    return model.to(device)


def command_evaluate(args) -> int:
    device = setup_torch()
    composition = compose_task()
    model = _load_model(args.model, device)
    started = time.perf_counter()
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir, device)
    print(json.dumps({"metrics": metrics, "evaluate_seconds": round(time.perf_counter() - started, 1)}, sort_keys=True))
    return 0


def command_predict(args) -> int:
    """Write the deliverable for an arbitrary split file (e.g. the organizers' hidden test)."""
    from execute_tools.task_data_path import ScopeBuildRequest

    device = setup_torch()
    composition = compose_task()
    model = _load_model(args.model, device)
    scope = composition.task_data_path._selection(
        args.split_file, ScopeBuildRequest(round_kind="formal", selection_strategy="snapshot", portion=1.0), False)
    path = write_predictions(model, composition, scope, args.data_dir, args.output_dir, device)
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="pitching-airfoil wake baseline")
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init"); p.add_argument("--seed", type=int, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--data_dir", type=Path, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", type=Path, required=True); p.add_argument("--init_model", type=Path)
    p = sub.add_parser("evaluate")
    p.add_argument("--data_dir", type=Path, required=True); p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data_dir", type=Path, required=True); p.add_argument("--split_file", required=True)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--output_dir", type=Path, required=True)
    args = cli.parse_args(argv)
    return {"init": command_init, "train": command_train, "evaluate": command_evaluate, "predict": command_predict}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
