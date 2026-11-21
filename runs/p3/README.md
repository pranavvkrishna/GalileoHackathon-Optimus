# P3 results: A_official / C_on_A / B_s0 / C_from_B_s0

Validation split only (12 cases, 48 windows). **These are local validation numbers, not leaderboard scores.**
Everything here was copied off the server from `~/person3/runs` (checksums in `SHA256SUMS`, verified
against the server). `*.npz` and `*.pt` files are gitignored and exist only on the machine that copied them.

## Results ([final/results_table.md](final/results_table.md))

| config | provenance | forecast_nmse | final-lead | st_abs_error | utility | reflection defect | gates | latency (H100) |
|---|---|---|---|---|---|---|---|---|
| A_official | organizer checkpoint (reference, **not a local run**) | 0.2571 | 0.4105 | 0.0701 | 0.7476 | 0.164 | pass | 5.6 ms |
| C_on_A | organizer checkpoint + mirror averaging | 0.2562 | 0.3696 | 0.0701 | 0.7483 | 0 | pass | 6.3 ms |
| B_s0 | local train, seed 0, reflection aug p=0.5 (P2) | 0.2094 | 0.3144 | 0.0701 | 0.7857 | 0.051 | pass | 5.6 ms |
| C_from_B_s0 | B_s0 weights + mirror averaging | **0.1856** | **0.2754** | 0.0701 | **0.8048** | 0 | pass | 6.3 ms |

Gates = output shape, finiteness, NMSE <= 1, the official `forecast_not_frozen_in_time` health check (real task view
and framework check code), and median batch-1 latency <= 100 ms. Latency was measured while another training job
(`Bplus_s0`) was using the GPU.

## Paired case-level differences in forecast_nmse ([final/paired.json](final/paired.json))
2000-resample case bootstrap; the interval covers case-sampling variability only (not seed noise).

| comparison | mean change | 95% interval | cases improved |
|---|---|---|---|
| C_from_B_s0 − B_s0 (matched: same weights) | −0.0238 | [−0.0273, −0.0204] | 12/12 |
| B_s0 − A_official (**not matched**) | −0.0477 | [−0.1098, −0.0065] | 7/12 |
| C_from_B_s0 − A_official (not matched) | −0.0715 | [−0.1333, −0.0299] | 12/12 |
| C_on_A − A_official (matched: same weights) | −0.0010 | [−0.0717, +0.0585] | 5/12 |

Figures: [final/fig1.pdf](final/fig1.pdf) (error by lead; per-case change vs kθ0) and [final/fig2.pdf](final/fig2.pdf)
(truth vs each model; case h1_008 was chosen by a fixed rule: nearest the median A_official error).

## Caveats
- **There is no matched local A.** The A_s0 run stopped at epoch 3, with no exit line and its tmux session gone,
  after `Bplus_s0` was started on the GPU at 22:16 UTC. So B − A compares against the organizer checkpoint. That
  conflates augmentation with any local-vs-organizer training difference. The organizers report that their retrain
  reproduced every metric exactly, but we did not verify this locally.
- There is one training seed and 12 validation cases, and the model was selected on validation.
- The exact reflection convention on the sampled grid is unverified. The organizers permit the transform, and the
  numerical involution and equivariance tests pass.
- C costs two forward passes; it measured 6.3 ms vs 5.6 ms.
- B was trained with the server's `run_baseline.py` (sha256 `48fdc8dc…`). It differs from the committed variant
  only in `torch.set_num_threads(1)`, which affects speed, not results. A copy is in `B_s0/from_p2/`.
- The statistics outputs (st_abs_error and the others) are identical for B_s0 and C_from_B_s0, as expected.

## Recommended submission candidate: C_from_B_s0
It has the highest measured utility among valid candidates. It passes every gate. Its improvement over B is the
cleanest result we have: same weights, all 12 cases better, and an interval that excludes 0.

Packaging (P1): `symbol: WakeMirrorAvg`; `train_command` = the official command plus `--mirror` after `train`;
`final_model.pt` = B_s0 weights (sha256 `ae4ef332…`, copy in `B_s0/from_p2/`). The spec requires a fresh
train-only reproduction (about 18 min on the H100), and the GPU is currently shared with `Bplus_s0`.

The fallback, already packaged, is `submission/C_on_A`. It is valid but shows no measurable gain over A_official.
