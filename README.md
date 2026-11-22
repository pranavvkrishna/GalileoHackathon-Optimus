# Validation can't see extrapolation — Team Optimus (G2-12)

Galileo Hackathon 2026, Track 2 · Task: **pitching-wing-wake** (forecast the vortex wake of a pitching NACA0012).

📄 **Report:** [`report/final/main.pdf`](report/final/main.pdf)

## Summary
Forecasting the wake has two separate limits: learning in-range dynamics, and extrapolating beyond the training range. The public validation set measures only the first.
- **In-range:** reflection augmentation + mirror averaging, all training windows, a variance-normalised loss and a 2-network ensemble cut validation NMSE **0.2571 → 0.1181 (−54%)**.
- **Extrapolation:** a parameter-free *period copy* (the wake locks onto the known forcing period) left validation unchanged but gave our largest hidden-score drop, **0.2876 → 0.1789**.
- **Overall:** final portal score **0.1538** (#129, leaderboard; 3rd place) vs organizer baseline **0.3937 (−61%)**.

## Results
| # | Model | Change | Val. NMSE | Portal |
|---|---|---|---|---|
| 8 | C_on_A | mirror averaging on organizer weights | 0.2562 | 0.4133 |
| 19 | C_from_B | + reflection-augmented training | 0.1856 | 0.3620 |
| 52 | C_from_Bplus_ood | + all windows, out-of-range shrink | 0.1440 | 0.2876 |
| 95 | C_from_Bplus_hybrid | + period copy | 0.1440 | 0.1789 |
| 110 | C_from_Bplus_hybrid2 | + soft copy blend | 0.1440 | 0.1680 |
| 120 | C_nmse_hybrid | + variance-normalised loss | 0.1261 | 0.1593 |
| 129 | C_ens_hybrid (final) | + 0.6/0.4 ensemble | 0.1181 | 0.1538 |

## Repository
- `report/final/` — report source, figures, `NUMBER_SOURCES.md` (every quoted number → source file)
- `variants/` — modified baseline (`run_baseline.py`: `--mirror`, `--loss nmse`, mirror-averaging / hybrid classes)
- `submission/` — packaged submissions (`submission.json` + plugin code; weights not tracked)
- `shared/` — evaluation, paired bootstrap, packaging and test scripts
- `runs/` — metrics and training logs
- `spec.md` — original experiment plan

## Reproduce
Requires SIDERIUS v1.1.0 and the released data (Zenodo 10.5281/zenodo.24386760; `train.npz` only for fitting).
```bash
python run_baseline.py train --mirror [--loss nmse] --train-only --data_dir $DATA_DIR --seed 0 \
  --output_dir out --init_model init_model.pt
python run_baseline.py evaluate --mirror_avg --data_dir $DATA_DIR --model out/final_model.pt --output_dir eval
```
The exact command for each submission is in its `submission/<name>/submission.json`.

## Team
Pranav Vijay Krishna · Raghav Vijayakannan · Hrithik Yash

## AI use
Claude Code assisted with environment setup, code, evaluation, packaging and report drafting; SIDERIUS provided the official scorer. Authors verified all results; see the report's Section 6.
