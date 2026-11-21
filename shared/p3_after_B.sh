#!/bin/bash
# P3 post-training evaluation. Run on the GPU box AFTER after_B.sh logs AFTER_DONE-style completion
# AND A_s0 training has finished (TRAIN_EXIT=0 in ~/wake/runs/A_s0/train_log.txt).
# Reads teammates' runs in ~/wake/runs (read-only); writes only to ~/person3/runs.
#   bash shared/p3_after_B.sh            (from ~/person3/repo)
set -euo pipefail
W=/home/team/wake/runs
P=/home/team/person3/runs
D=/data/galileo/pitching-wing-wake/data
BV=variants/B/pitching_airfoil_wake
source /opt/siderius/.venv/bin/activate

for r in A_s0 B_s0; do
  grep -q '^TRAIN_EXIT=0' $W/$r/train_log.txt || { echo "$r training not finished OK"; exit 1; }
  test -f $W/$r/final_model.pt || { echo "$r final_model.pt missing"; exit 1; }
done
test -f $W/C_from_B_s0/metrics.json || { echo "C_from_B_s0 official evaluate missing (after_B.sh step 6)"; exit 1; }

declare -A MODEL=([A_s0]=$W/A_s0/final_model.pt [B_s0]=$W/B_s0/final_model.pt [C_from_B_s0]=$W/B_s0/final_model.pt)
for r in A_s0 B_s0 C_from_B_s0; do
  f=""; [ $r = C_from_B_s0 ] && f=--mirror_avg
  mkdir -p $P/$r
  # official metrics: written by the train-time evaluation (A_s0/B_s0) or by `evaluate` (C)
  cp $W/$r/metrics.json $P/$r/metrics.json
  python shared/local_eval.py $f --baseline_dir $BV/baseline --data_dir $D --model ${MODEL[$r]} --out $P/$r/local_eval
  python shared/health_gate.py $f --baseline_dir $BV/baseline --plugins_dir $BV/task/plugins --data_dir $D \
    --model ${MODEL[$r]} --out $P/$r/gates --device cuda || echo "WARNING: $r failed a gate (see $P/$r/gates/gates.json)"
  python shared/test_clip_order.py --plugins_dir $BV/task/plugins --diag $P/$r/local_eval/diag.npz --metrics $P/$r/metrics.json
done

python shared/compare_runs.py --data_dir $D --out_dir $P/abc \
  --run A_s0=$P/A_s0 --run B_s0=$P/B_s0 --run C_from_B_s0=$P/C_from_B_s0 \
  --pred A_s0=$W/A_s0/pitching_wake_predictions.npz --pred B_s0=$W/B_s0/pitching_wake_predictions.npz \
  --pred C_from_B_s0=$W/C_from_B_s0/pitching_wake_predictions.npz \
  --pair B_s0-A_s0 --pair C_from_B_s0-B_s0 --pair C_from_B_s0-A_s0

python shared/results_table.py --out $P/abc/results_table \
  --row "A_official|0|organizer weights (not retrained locally)|$P/A_official" \
  --row "C_on_A|0|organizer weights + mirror averaging|$P/C_on_A" \
  --row "A_s0|0|local train, seed 0|$P/A_s0|$W/A_s0/train_log.txt" \
  --row "B_s0|0|local train, seed 0, reflection augmentation|$P/B_s0|$W/B_s0/train_log.txt" \
  --row "C_from_B_s0|0|B_s0 weights + mirror averaging (no training)|$P/C_from_B_s0|$W/B_s0/train_log.txt" \
  --metrics C_on_A=$P/C_on_A/official/metrics.json

# spec A8: the statistics head is filled deterministically from train data (fill_statistics) and is never
# augmented, so st_abs_error etc. must be identical for A_s0, B_s0 and C_from_B_s0; a mismatch is a bug.
python - <<EOF
import json
b = json.load(open("$P/B_s0/local_eval/metrics.json"))["metrics"]
c = json.load(open("$P/C_from_B_s0/local_eval/metrics.json"))["metrics"]
a = json.load(open("$P/A_s0/local_eval/metrics.json"))["metrics"]
for k in ("st_abs_error", "mean_field_rel_l2", "psd_log_rmse", "regime_balanced_accuracy"):
    print(k, a[k], b[k], c[k])
    assert b[k] == c[k], f"{k} differs between B_s0 and C_from_B_s0: statistics head changed (bug)"
    if abs(a[k] - b[k]) > 1e-6:
        print(f"WARNING: {k} differs between A_s0 and B_s0 ({a[k]} vs {b[k]}); spec A8 says investigate")
print("STATISTICS HEAD CHECK DONE")
EOF
echo "P3_AFTER_B_DONE -> $P/abc"
