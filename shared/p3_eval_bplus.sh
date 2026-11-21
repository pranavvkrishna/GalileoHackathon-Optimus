#!/bin/bash
# P3: wait for the teammate's Bplus_s0 training (TRAIN_EXIT line), then evaluate Bplus_s0 and C_from_Bplus_s0
# (mirror averaging), run gates, and compare against B_s0 / C_from_B_s0. Reads ~/wake (read-only); writes ~/person3.
#   tmux new-session -d -s p3bplus "bash ~/person3/repo/shared/p3_eval_bplus.sh"
set -uo pipefail
W=/home/team/wake/runs
P=/home/team/person3/runs
D=/data/galileo/pitching-wing-wake/data
BP=/home/team/wake/variants/Bplus/pitching_airfoil_wake
LOG=$P/bplus_eval.log
cd /home/team/person3/repo
source /opt/siderius/.venv/bin/activate
mkdir -p $P
echo "$(date -u +%T) waiting for Bplus_s0" >> $LOG
until grep -q '^TRAIN_EXIT=' $W/Bplus_s0/train_log.txt 2>/dev/null; do sleep 20; done
rc=$(grep '^TRAIN_EXIT=' $W/Bplus_s0/train_log.txt | tail -1 | cut -d= -f2)
echo "$(date -u +%T) Bplus finished, exit $rc" >> $LOG
if [ "$rc" != "0" ] || [ ! -f $W/Bplus_s0/final_model.pt ]; then echo "Bplus FAILED; nothing to evaluate" >> $LOG; exit 1; fi
M=$W/Bplus_s0/final_model.pt
sha256sum $M $BP/baseline/run_baseline.py >> $LOG

mkdir -p $P/Bplus_s0 $P/C_from_Bplus_s0
cp $W/Bplus_s0/metrics.json $P/Bplus_s0/metrics.json
(cd $BP/baseline && python run_baseline.py evaluate --mirror_avg --data_dir $D --model $M \
   --output_dir $P/C_from_Bplus_s0/official > $P/C_from_Bplus_s0/official_eval_log.txt 2>&1)
echo "C_from_Bplus official evaluate exit $?" >> $LOG
cp $P/C_from_Bplus_s0/official/metrics.json $P/C_from_Bplus_s0/metrics.json
for r in Bplus_s0 C_from_Bplus_s0; do
  f=""; [ $r = C_from_Bplus_s0 ] && f=--mirror_avg
  python shared/local_eval.py $f --baseline_dir $BP/baseline --data_dir $D --model $M --out $P/$r/local_eval \
    > $P/$r/local_eval_log.txt 2>&1; echo "$r local_eval exit $?" >> $LOG
  python shared/health_gate.py $f --baseline_dir $BP/baseline --plugins_dir $BP/task/plugins --data_dir $D \
    --model $M --out $P/$r/gates --device cuda > $P/$r/gates_log.txt 2>&1; echo "$r gates exit $?" >> $LOG
  python shared/test_clip_order.py --plugins_dir $BP/task/plugins --diag $P/$r/local_eval/diag.npz \
    --metrics $P/$r/metrics.json >> $LOG 2>&1
done

python shared/compare_runs.py --no_fig2 --out_dir $P/bplus \
  --run B_s0=$P/B_s0 --run C_from_B_s0=$P/C_from_B_s0 --run Bplus_s0=$P/Bplus_s0 --run C_from_Bplus_s0=$P/C_from_Bplus_s0 \
  --pair Bplus_s0-B_s0 --pair C_from_Bplus_s0-C_from_B_s0 --pair C_from_Bplus_s0-Bplus_s0 >> $LOG 2>&1
python shared/results_table.py --out $P/bplus/results_table \
  --row "A_official|0|organizer checkpoint (reference; not a local run)|$P/A_official" \
  --row "B_s0|0|local train, seed 0, reflection aug. p=0.5|$P/B_s0|$W/B_s0/train_log.txt" \
  --row "C_from_B_s0|0|B_s0 weights + mirror averaging, no training|$P/C_from_B_s0" \
  --row "Bplus_s0|0|as B_s0 with all training windows (subsample 1)|$P/Bplus_s0|$W/Bplus_s0/train_log.txt" \
  --row "C_from_Bplus_s0|0|Bplus_s0 weights + mirror averaging, no training|$P/C_from_Bplus_s0" >> $LOG 2>&1
echo "$(date -u +%T) P3_BPLUS_DONE" >> $LOG
