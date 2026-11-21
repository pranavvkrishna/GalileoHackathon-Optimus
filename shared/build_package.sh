#!/bin/bash
# Build a mirror-averaged submission package from original/official/ (run from the repo root).
#   bash shared/build_package.sh <name> <variant> <final_model.pt> <expected forecast_nmse>
# e.g. bash shared/build_package.sh C_from_Bplus Bplus runs/Bplus_s0/final_model.pt 0.1750
# Plugin = variants/<variant> run_baseline.py in both locations; symbol WakeMirrorAvg;
# train_command = official command with --mirror after "train"; seed 0; official initial_model.pt.
set -euo pipefail
NAME=$1; VAR=$2; MODEL=$3; EXPECT=$4
PY=${PY:-$HOME/.venvs/wake/bin/python}
S=submission/$NAME
SRC=variants/$VAR/pitching_airfoil_wake/baseline/run_baseline.py
grep -q '^    torch.set_num_threads(1)$' "$SRC" || { echo "threads != 1 in $SRC"; exit 1; }
rm -rf "$S" && cp -R original/official "$S"
cp "$SRC" "$S/run_baseline.py"
cp "$SRC" "$S/train/pkg/baseline/run_baseline.py"
cp "$MODEL" "$S/final_model.pt"
$PY -I - "$S/submission.json" <<'EOF'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); off = d["train_command"]
d["symbol"] = "WakeMirrorAvg"
d["train_command"] = off.replace("run_baseline.py train --train-only", "run_baseline.py train --mirror --train-only", 1)
assert d["train_command"] != off and d["train_command"].count("--mirror") == 1 and d["seed"] == 0
json.dump(d, open(p, "w"), indent=2); open(p, "a").write("\n")
print(json.dumps(d, indent=2))
EOF
grep -n "^WINDOW_SUBSAMPLE" "$S/run_baseline.py" "$S/train/pkg/baseline/run_baseline.py"
shasum -a 256 "$S/final_model.pt" "$MODEL" "$S/initial_model.pt" "$S/train/pkg/baseline/init_model.pt"
$PY -I shared/check_submission.py --pkg "$S" --data_dir data --expect_nmse "$EXPECT"
cd "$S"
find . -name __pycache__ -prune -exec rm -rf {} +
find . -name .DS_Store -delete
if grep -rIn -i "OPENAI\|sk-\|pd6ze\|api_key\|/Users/" .; then echo "SECRET/PATH SCAN FAILED"; exit 1; fi
if [ -n "$(find . -type l)" ]; then echo "SYMLINKS PRESENT"; exit 1; fi
rm -f "../$NAME.zip" && zip -qrX "../$NAME.zip" .
cd ..
unzip -l "$NAME.zip" | grep -E " (submission.json|run_baseline.py|final_model.pt|initial_model.pt)$|files$"
du -h "$NAME.zip"
shasum -a 256 "$NAME.zip"
