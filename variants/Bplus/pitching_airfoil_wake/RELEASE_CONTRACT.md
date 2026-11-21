# Final event submission contract — 20261121-v3

This release notice overrides older seed and submission-format examples in TASK_CARD/README files. Scientific checks and thresholds remain as declared, including the explicit aneurysm rank-only diagnostic amendment.

Task: `pitching-wing-wake`. This official baseline uses seed **0**, `kind=model`. Its authoritative manifest is `submission.json` in the sibling `official-submission.zip`.

- Initial artifact: `initial_model.pt`.
- Final artifact: `final_model.pt`.
- Inference entry point: `run_baseline.py`.
- Complete training source: `train/`.

The initial artifact records the real untrained model before fitting; the final artifact contains trained parameters. Do not substitute trained weights for initialization. The evaluator treats submitted initialization as audit evidence, not a warm-start permission.

Keep submission.json at the ZIP root, together with all referenced source, initial and final artifacts. Include complete training and inference code and the actual random seed; do not upload only weights or only the final best model. Use the seed you actually used for your own submission; the value above describes this official baseline, not a requirement that every team use the same seed.

Official reproducible training command (set DATA_DIR to the public data directory described by the task):

```sh
python train/pkg/baseline/run_baseline.py train --train-only --data_dir $DATA_DIR --seed 0 --output_dir out --init_model "${PACKAGE_DIR:-train/pkg}/baseline/init_model.pt" && cp out/final_model.pt final_model.pt
```

Submit the ZIP through the event submission portal. The limit is 10 model submissions per team across all tasks, with models due at 16:00 PST. Upload the separate final PDF report in the same portal and link it to the final model submission ID by 16:30 PST; report updates do not consume model submission attempts.
