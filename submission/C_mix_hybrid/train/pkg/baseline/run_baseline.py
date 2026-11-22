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
WIDTH = 96
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VORT_SCALE = 4.0
KNN = 3
WINDOW_SUBSAMPLE = 3  # every 3rd strided training window; every 3rd of the data path's strided training windows (start stride 15 frames = 1.5 c/U); keeps the baseline under 30 min on an RTX 5090
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


def mirror_x(x: torch.Tensor) -> torch.Tensor:
    """Reflection y -> -y of a model input: vorticity row-flip + sign flip; forcing phase + pi."""
    out = x.clone()
    out[:, :HISTORY] = -x[:, :HISTORY].flip(-2)
    out[:, HISTORY + 2:HISTORY + 4] = -x[:, HISTORY + 2:HISTORY + 4]
    return out


def mirror_field(z: torch.Tensor) -> torch.Tensor:
    """Reflection of vorticity fields [..., 48, 96]."""
    return -z.flip(-2)


class WakeBaseline(nn.Module):
    """[B, 14, 48, 96] -> [B, 96822]."""

    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        w = width
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


class WakeMirrorAvg(WakeBaseline):
    """Config C: reflection-averaged forecast; statistics head unchanged; same state_dict as WakeBaseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forecast(x)
        reflected = mirror_field(self.forecast(mirror_x(x)))
        future = 0.5 * (f + reflected)
        return torch.cat([future.flatten(1), self.statistics(x)], dim=1)



# Training-set kinematics range (train.npz, 810 cases): k in [2.0016, 6.9976], theta0 in [5.009, 34.995] deg.
K_TRAIN_RANGE = (2.0016000270843506, 6.997600078582764)
THETA0_DEG_RANGE = (5.008999824523926, 34.994998931884766)
# Out-of-range shrinkage strength reached at the last lead (linear ramp from 0 at lead 1). Chosen on a proxy
# extrapolation test (train k <= 5.49, score held-out training cases k 6.0-7.0): NMSE 0.99 -> 0.78 at 0.7.
OOD_SHRINK_K = 0.7
OOD_SHRINK_THETA0 = 0.0


class WakeMirrorAvgOOD(WakeMirrorAvg):
    """C, plus shrinkage of the forecast toward the mean of the 10 history frames when the kinematics lie
    outside the training range. Inside the range it is exactly WakeMirrorAvg. The gate uses k and theta0,
    which reflection leaves unchanged, and the history mean reflects with the fields, so reflection
    equivariance is preserved. Statistics head unchanged. Same state_dict as WakeBaseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        k = x[:, HISTORY, 0, 0] * (2 * math.pi)          # constant maps: read one pixel
        th = torch.rad2deg(x[:, HISTORY + 1, 0, 0])
        a = torch.zeros_like(k)
        # strength grows with distance outside the k range (proxy: clip(0.6 + 0.6*(dk-0.5), 0.6, 1.0) -> 0.743 vs 0.781 flat 0.7)
        dk = torch.clamp(torch.maximum(k - K_TRAIN_RANGE[1], K_TRAIN_RANGE[0] - k), min=0.0)
        a = torch.where(dk > 0, torch.clamp(0.6 + 0.6 * (dk - 0.5), 0.6, 1.0), a)
        th_out = (th > THETA0_DEG_RANGE[1]) | (th < THETA0_DEG_RANGE[0])
        a = torch.where(th_out, torch.maximum(a, torch.full_like(a, OOD_SHRINK_THETA0)), a)
        if not bool((a > 0).any()):
            return out
        n = HORIZON * FIELD
        future = out[:, :n].reshape(-1, HORIZON, NY, NX)
        hist_mean = x[:, :HISTORY].mean(dim=1, keepdim=True)
        ramp = torch.linspace(0, 1, HORIZON, device=x.device, dtype=x.dtype).view(1, HORIZON, 1, 1)
        w = a.view(-1, 1, 1, 1) * ramp
        future = (1 - w) * future + w * hist_mean
        return torch.cat([future.flatten(1), out[:, n:]], dim=1)


FRAME_DT = 0.1             # convective time between stored frames (train.npz t_frames step)
PERIODIC_MAX_RESIDUAL = 0.3          # in-range k: hard gate (CNN is good there)
OOD_BLEND_FULL, OOD_BLEND_NONE = 0.5, 1.2  # k above the training range: copy weight 1 below 0.5, 0 above 1.2


class WakeHybrid(WakeMirrorAvgOOD):
    """WakeMirrorAvgOOD, except that when one forcing period T = 2*pi/k fits inside the 10-frame history
    (T <= 0.9, i.e. k >= 6.98) and the history itself repeats with that period (normalised residual
    < PERIODIC_MAX_RESIDUAL), the forecast is the history copied forward by whole periods (linear time
    interpolation). Parameter-free; on training cases with k >= 6.98 the copy scores NMSE 0.003 when the
    residual is < 0.02. Validation (k <= 6.5) never qualifies, so it is unchanged. Copying commutes with
    the y-reflection and the residual is reflection-invariant, so equivariance is preserved."""

    @staticmethod
    def _sample(hist: torch.Tensor, t: float) -> torch.Tensor:
        """hist [10, NY, NX] at times 0, dt, ..., 9 dt; linear interpolation at time t in [0, 9 dt]."""
        u = min(max(t / FRAME_DT, 0.0), HISTORY - 1.0)
        i = min(int(math.floor(u)), HISTORY - 2)
        w = u - i
        return (1 - w) * hist[i] + w * hist[i + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        span = (HISTORY - 1) * FRAME_DT
        k = (x[:, HISTORY, 0, 0] * (2 * math.pi)).tolist()
        for b, kb in enumerate(k):
            if kb <= 0:
                continue
            T = 2 * math.pi / kb
            if T > span:
                continue
            hist = x[b, :HISTORY].double()
            errs = [torch.mean((hist[i] - self._sample(hist, i * FRAME_DT - T)) ** 2)
                    for i in range(HISTORY) if i * FRAME_DT - T >= 0]
            if not errs:
                continue
            resid = float(torch.stack(errs).mean() / torch.clamp(torch.mean((hist - hist.mean(0)) ** 2), min=1e-12))
            if kb > K_TRAIN_RANGE[1]:
                # proxy (CNN trained k<=5.5, scored k 6-7): CNN+pull 0.689, hard gate 0.3 0.270, this blend 0.1905
                wcopy = min(max((OOD_BLEND_NONE - resid) / (OOD_BLEND_NONE - OOD_BLEND_FULL), 0.0), 1.0)
            else:
                wcopy = 1.0 if resid < PERIODIC_MAX_RESIDUAL else 0.0
            if wcopy <= 0.0:
                continue
            frames = []
            for j in range(HORIZON):
                tau = span + (j + 1) * FRAME_DT                 # future time on the history clock
                m = math.ceil((tau - span) / T - 1e-9)
                frames.append(self._sample(hist, tau - m * T))
            copy = torch.stack(frames).flatten().to(out.dtype)
            out[b, : HORIZON * FIELD] = wcopy * copy + (1.0 - wcopy) * out[b, : HORIZON * FIELD]
        return out


MIX_WEIGHTS = (0.5, 0.5, 0.0)   # (this width-96 network, m64a metric-weighted width-64, m64b MSE width-64); set from validation


class WakeMixHybrid(WakeHybrid):
    """WakeHybrid on a weighted mixture of three forecast networks: this one (width WIDTH=96, metric-weighted loss,
    30 epochs, 1/3 windows), m64a (width 64, metric-weighted loss, 40 epochs, all windows) and m64b (width 64, MSE,
    40 epochs, all windows). Mixing is linear, so mirror averaging, out-of-range shrinkage and the period copy apply to
    the mixture. Statistics head from this network (filled identically from the training data in every network)."""

    def __init__(self) -> None:
        super().__init__()
        self.m64a = WakeBaseline(width=64)
        self.m64b = WakeBaseline(width=64)

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = MIX_WEIGHTS
        out = a * super().forecast(x) + b * self.m64a.forecast(x)
        return out + c * self.m64b.forecast(x) if c else out

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
    model = WakeBaseline(width=args.width)
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
    pairs = pairs[::args.subsample]
    model = WakeBaseline(width=args.width)
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
    mvt = torch.from_numpy(np.ascontiguousarray(table.mean_vort)).float().to(device)
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
    steps = args.epochs * math.ceil(len(pairs) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=max(steps, 1), pct_start=0.1)
    generator = torch.Generator().manual_seed(args.seed)
    aug_gen = torch.Generator().manual_seed(args.seed + 1_000_003)  # independent: window order identical to A
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(pairs), generator=generator).to(device)
        running = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            ix = order[start : start + BATCH_SIZE]
            x, y = batch(ix)
            if args.loss == "nmse":  # per-window variance of the truth about the case mean (mirror-invariant)
                wvar = ((y - mvt[pc[ix]][:, None]) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-6)
            if args.mirror:
                flip = (torch.rand(len(x), generator=aug_gen) < 0.5).to(device)
                if bool(flip.any()):
                    x[flip] = mirror_x(x[flip])
                    y[flip] = mirror_field(y[flip])
            if args.k_mode == "none":
                x[:, HISTORY] = 0.0
            elif args.k_mode == "drop":
                hide = torch.rand(len(x), device=device) < args.k_param
                x[hide, HISTORY] = 0.0
            elif args.k_mode == "noise":
                x[:, HISTORY] += (torch.randn(len(x), device=device) * args.k_param / (2 * math.pi))[:, None, None]
            optimizer.zero_grad(set_to_none=True)
            if args.loss == "nmse":
                loss = (((model.forecast(x) - y) ** 2).mean(dim=(1, 2, 3)) / wvar).mean()
            else:
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
    metrics = {} if args.train_only else evaluate_model(model, composition, args.data_dir, args.output_dir, device)
    report = {"metrics": metrics, "train_seconds": round(train_seconds, 1),
              "evaluate_seconds": round(time.perf_counter() - started, 1), "device": str(device),
              "train_windows": len(pairs), "train_cases": len(table.case_ids)}
    if device.type == "cuda":
        report["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, sort_keys=True))
    return 0


def _load_model(path: Path, device, cls=None) -> WakeBaseline:
    model = (cls or WakeBaseline)()
    model.load_state_dict(load_state(path))
    return model.to(device)


def command_evaluate(args) -> int:
    device = setup_torch()
    composition = compose_task()
    model = _load_model(args.model, device, WakeMirrorAvg if getattr(args, "mirror_avg", False) else None)
    started = time.perf_counter()
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir, device)
    print(json.dumps({"metrics": metrics, "evaluate_seconds": round(time.perf_counter() - started, 1)}, sort_keys=True))
    return 0


def command_predict(args) -> int:
    """Write the deliverable for an arbitrary split file (e.g. the organizers' hidden test)."""
    from execute_tools.task_data_path import ScopeBuildRequest

    device = setup_torch()
    composition = compose_task()
    model = _load_model(args.model, device, WakeMirrorAvg if getattr(args, "mirror_avg", False) else None)
    scope = composition.task_data_path._selection(
        args.split_file, ScopeBuildRequest(round_kind="formal", selection_strategy="snapshot", portion=1.0), False)
    path = write_predictions(model, composition, scope, args.data_dir, args.output_dir, device)
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="pitching-airfoil wake baseline")
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init"); p.add_argument("--seed", type=int, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--width", type=int, default=WIDTH)
    p = sub.add_parser("train")
    p.add_argument("--width", type=int, default=WIDTH); p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--subsample", type=int, default=WINDOW_SUBSAMPLE)
    p.add_argument("--train-only", action="store_true", help="save trained artifact without post-training validation")
    p.add_argument("--data_dir", type=Path, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", type=Path, required=True); p.add_argument("--init_model", type=Path)
    p.add_argument("--mirror", action="store_true", help="random y-reflection augmentation, p=0.5 per example")
    p.add_argument("--k_mode", default="true", choices=("true", "drop", "noise", "none"))
    p.add_argument("--k_param", type=float, default=0.0)
    p.add_argument("--loss", default="mse", choices=("mse", "nmse"))
    p = sub.add_parser("evaluate")
    p.add_argument("--data_dir", type=Path, required=True); p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--mirror_avg", action="store_true", help="evaluate as WakeMirrorAvg (config C)")
    p = sub.add_parser("predict")
    p.add_argument("--data_dir", type=Path, required=True); p.add_argument("--split_file", required=True)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--mirror_avg", action="store_true", help="evaluate as WakeMirrorAvg (config C)")
    args = cli.parse_args(argv)
    return {"init": command_init, "train": command_train, "evaluate": command_evaluate, "predict": command_predict}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
