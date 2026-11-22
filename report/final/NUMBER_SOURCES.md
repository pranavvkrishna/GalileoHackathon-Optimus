# Number sources for report/final/main.tex

Build: `python prepare_update.py --snapshot <wake_remote_backup_20261121_1845> --data_dir ../../data --ens_zip <snapshot>/person3/C_ens_hybrid.zip`
(writes `plot_data_final.npz`, `report_numbers.json`), then `python build_figures.py`, then `tectonic main.tex`.
Scoring rule everywhere: per-lead NMSE clipped at 4, then averaged (official order); paired intervals: 12 case means,
2,000 resamples, `numpy.random.default_rng(0)`, 2.5/97.5% quantiles (same as `build_figures.py`).

## Recomputed here (report_numbers.json, plot_data_final.npz)

| Quantity in report | Value | Source data |
|---|---|---|
| A_ref NMSE / final lead | 0.2571 / 0.4105 | plot_data.npz `lead_A_ref` (organizer checkpoint predictions) |
| A_avg NMSE / final | 0.2562 / 0.3696 | runs/C_on_A/local_eval/diag.npz |
| B NMSE / final | 0.2094 / 0.3144 | plot_data.npz `lead_B` |
| C NMSE / final | 0.1856 / 0.2754 | plot_data.npz `lead_C` |
| B+ NMSE / final / defect | 0.1670 / 0.2373 / 0.0493 | runs/Bplus_s0/local_eval/diag.npz |
| C+ NMSE / final | 0.1440 / 0.2121 | plot_data.npz `lead_C+` |
| B_nmse NMSE / final / defect | 0.1526 / 0.2206 / 0.0469 | snapshot person3/runs/nmse_s0/local_eval/diag.npz |
| C_nmse NMSE / final | 0.1261 / 0.1837 | snapshot person3/runs/C_from_nmse_s0/local_eval/diag.npz |
| C_ens NMSE / final / defect | 0.1181 / 0.1747 / 0 | C_ens_hybrid.zip (sha256 853773018fb4…0c80c9106) on CPU; official scorer = local = 0.118142 |
| E NMSE / final | 0.1897 / 0.2855 | plot_data.npz `lead_E` |
| C_w96 / C_w96b NMSE; worse than C_nmse 12/12 | 0.1463 / 0.1382 | snapshot person3/runs/C_from_w96{,b}_s0/local_eval/diag.npz |
| 3-network mix best; gain over C_ens | 0.1161; 0.0020 | saved official predictions person3/runs/*/official/pitching_wake_predictions.npz ("0.1165" in commit 269cd38 does not reproduce) |
| C − B | −0.0238 [−0.0273, −0.0204], 12/12 | plot_data.npz |
| C+ − C | −0.0416 [−0.0475, −0.0362], 12/12 | plot_data.npz |
| C_nmse − C+ | −0.0179 [−0.0304, −0.0044], 8/12 | diag + plot_data |
| C_ens − C_nmse | −0.0080 [−0.0123, −0.0043], 10/12 | package run + diag |
| A_avg − A_ref | −0.0010 [−0.0717, 0.0585] | diag + plot_data |
| C − E | in Fig. 2b only | plot_data.npz |
| C_ens − A_ref | −0.1390 [−0.1982, −0.0985], 12/12 | package run + plot_data |
| C_ens checks, latency | health pass; 88.2 ms (1 laptop CPU thread) | prepare_update.py run |
| K_RANGE effect | st_abs_error 0.070109 → 0.070132 (2.3e-5), forecasts identical | C_ens package run vs shared/local_eval.py |
| Overall gains | portal −60% (0.3937→0.1593; −59.5%), validation −54% (0.2571→0.1181; −54.1%) | portal_scores.json + report_numbers.json (computed in build_figures.py, stored in figure_provenance.json) |

## Team-supplied or recorded (not recomputable here)

| Quantity | Value | Source |
|---|---|---|
| Portal scores #8 0.41326, #19 0.36196, #52 0.28762, #95 0.17894, #110 0.16798, #120 0.15930; #129 under review (leaderboard 0.1538 unconfirmed); baseline 0.3937 | | team portal record → portal_scores.json |
| "C's portal score 0.3620", "0.2876 → 0.1789", "largest single portal drop" | −0.1087 vs next −0.0743 (#19→#52) | portal_scores.json |
| "three of seven submissions share 0.1440, portal 0.2876–0.1680" | #52, #95, #110 | portal_scores.json + report_numbers.json |
| Hidden tiers: T2 kθ0 2.3–3.3 rad; T3a θ0 38–45°; T3f k 7.5–9; validation block | | original/package/pitching_airfoil_wake/TASK_CARD.md (Hidden test section) |
| Fit times B 1,068 s; B+ 3,882 s; B_nmse 4,468 s | | runs/B_s0/train_log.txt; runs/Bplus_s0/train_log.txt; snapshot person3/exp/full_nmse/train_log.txt |
| A_ref / B defect | 0.1638 / 0.0505 | metrics_table.json (earlier verified bundle) |
| Windows 22,680 / 68,040; 810 cases; 1,890,708 params | | train logs; run_baseline.py init output |
| Frequency proxy: CNN 0.99, clamp 1.04, shrink 0.78 | 0.9925 / 1.0446 / 0.7807 | runs/p3/proxy/proxy_results.json; runs/p3/README.md |
| Hiding k in training: 1.03–1.05 | 1.0295–1.0468 | runs/p3/exp/results.json (R1_drop50, R2_nok) |
| Period copy: 1,944/2,100 windows, 523 cases, 0.059; 0.044–0.058 per k bin; k ≥ 6.98 | | runs/p3/periodic/README.md |
| Fig. 3a: CNN + shrink 0.689, hard gate 0.270, soft blend 0.1905; 132 cases | | commit 741de6a message (no result file) → portal_scores.json |
| Fig. 3b: 21 windows, copy on 16 (NMSE 0.025), hybrid 0.119 vs CNN 0.158 | | runs/p3/periodic/README.md → portal_scores.json |
| Fresh reproduction of B/C matched exactly | | reproduction_manifest.json, REVIEW.txt (earlier report bundle, P2) |
| Smoke test epoch-0 loss 0.44678 | | remote run 22:38 UTC; equals runs/B_s0/train_log.txt epoch 0 |
| SIDERIUS v1.1.0 5ccb436f8, master e3163175 identical | | runs/*/metrics.json vs runs/*/siderius_master/metrics.json |
| Init sha256 2bdf7cfd…; `init --seed 0` identical tensors | | sha256 of released init_model.pt; this session's check |
| 0.6/0.4 weight: no recorded search | | submission/C_ens_hybrid/run_baseline.py:240; commit 469b3ce; no sweep log found |
| C_ens source 469b3ce unchanged at 269cd38 | | `git diff 469b3ce origin/main -- submission/C_ens_hybrid` empty |

## Not done
`period_copy_eval.py` (per-window copy NMSE vs residual r on training windows, plus a k ≥ 6.98 example field) is written
but was not run: the GPU server went offline and train.npz is not on the Mac. Fig. 3 therefore shows recorded results.
Running it later (CPU only, needs train.npz) would allow the requested scatter figure.
