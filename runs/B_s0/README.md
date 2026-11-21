# B seed-0 training result

B completed all 40 epochs on the shared H100: 810 training cases, 22,680 windows, 1,068 seconds. Local validation forecast NMSE is 0.2093956555 and final-lead NMSE is 0.3144083686. These are validation scores, not leaderboard scores or a matched A/B comparison.

The completed run was already active when Person 2's agent connected; the agent monitored it and backed up its artifacts. It did not use the new `shared/run_person2.py` launcher. `train_log.txt` records the actual run and evaluation.

## Artifacts on the shared server

- Weights: `/home/team/wake/runs/B_s0/final_model.pt`
- Initializer: `/home/team/wake/variants/B/pitching_airfoil_wake/baseline/init_model.pt`
- Training source: `/home/team/wake/variants/B/pitching_airfoil_wake/baseline/run_baseline.py`
- Data: `/data/galileo/pitching-wing-wake/data` (read-only)

`checksums.json` records the SHA-256 of the source and weights copied from the completed run. Weights and datasets are excluded from Git. The checkpoint, initializer, source, metrics and log have also been backed up locally by Person 2. Copy any needed artifacts off the allocation before 16:00 PST.

## Actual run

From the B baseline directory:

```sh
python run_baseline.py train --mirror \
  --data_dir /data/galileo/pitching-wing-wake/data \
  --seed 0 --output_dir /home/team/wake/runs/B_s0 \
  --init_model ./init_model.pt
```

Width 64, batch 32, 40 epochs, AdamW lr 0.001 and weight decay 0.0001, OneCycleLR with 10% warm-up, forecast MSE divided by VORT_SCALE squared, gradient clip 1.0, WINDOW_SUBSAMPLE=3. Reflection probability is 0.5 with a separate RNG seeded 1000003. The official initial state checksum matched; a regenerated seed-0 state also matched its tensors on the H100.

## Person 3 handoff

Evaluate C-from-B via `evaluate --mirror_avg` using B's checkpoint and a fresh output directory. Check existing GPU jobs first. Compare to matched local A when available; distinguish the organizer checkpoint and C-on-A. Run official validity/health and batch-1 GPU latency checks before selection. Use the team's P3 scripts where appropriate.

## Physical audit and verification

The organizer task card explicitly permits up-down reflection: vorticity changes sign, and forcing phase shifts by pi. Both phase channels negate; frequency and positive amplitude remain unchanged. The stated observation domain is symmetric. Exact sample coordinates and top/bottom generator boundary settings have not been independently verified: the NPZ has no coordinate array. Report this limitation rather than presenting algebraic involution as physical-grid proof.

Existing mirror checks passed (synthetic forecast equivariance maximum absolute error 0.0). Additional local checks confirmed mixed-batch history/target alignment, untouched unselected examples, independent window RNG, and finite backward gradients. Both local and remote datasets passed official size and MD5 checks.

## Optional future launcher

`shared/run_person2.py` checks data and matching A/B constants, and can start a fresh CUDA B run while recording source, settings, initialization, logs and checksums. Its preflight was run successfully; its full training path has not been exercised. Do not repeat B just to use it.

The local source correction in this PR restores B's CPU thread count to A's value of one. This was also present in the source backed up from the completed B run. It does not change the model architecture.
