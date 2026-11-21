| configuration | seed | provenance | split | forecast_nmse | forecast_nmse_final | st_abs_error | utility | reflection_defect | health_status | latency_ms | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_official | 0 | organizer checkpoint (reference; not a local run) | validation (12 cases, 48 windows) | 0.2571 | 0.4105 | 0.0701 | 0.7476 | 0.1638 | pass | 5.6 | - |
| C_on_A | 0 | organizer checkpoint + mirror averaging, no training | validation (12 cases, 48 windows) | 0.2562 | 0.3696 | 0.0701 | 0.7483 | 0.0000 | pass | 6.3 | - |
| B_s0 | 0 | local train, seed 0, reflection aug. p=0.5 | validation (12 cases, 48 windows) | 0.2094 | 0.3144 | 0.0701 | 0.7857 | 0.0505 | pass | 5.6 | 1068.0 |
| C_from_B_s0 | 0 | B_s0 weights + mirror averaging, no training | validation (12 cases, 48 windows) | 0.1856 | 0.2754 | 0.0701 | 0.8048 | 0.0000 | pass | 6.3 | - |
