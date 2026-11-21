# Person 2 verified evidence and limitations

Fresh 40-epoch train-only reproduction completed from seed-driven initialization. Every trained tensor and every prediction array matches the original B/C run exactly. C-from-B validation NMSE is **0.18558599596016098**, final-lead NMSE **0.27543407048315954**, with recorded health/validity checks passing. The packaged class is WakeMirrorAvg. Data: https://doi.org/10.5281/zenodo.24386760 .

Verified artifact on the shared server: `/home/team/person2/evidence-2238/C_from_B_fresh_verified_v2.zip`. ZIP SHA-256: `0382f3bd83c37a5400b0cc21188df41e30ff6e885b1432aaca43bea52f32bd38`. Final weights SHA-256: `ae4ef332f84ca36ce1aecddea1f1363a997f1e18919086c6b4b69dcf0f2c1027`. ZIP, original/new weights, predictions, and a full source/log archive are backed up locally by Person 2; large artifacts are excluded from Git. No portal upload was made by this agent; P1 owns selection and submission.

## What the evidence supports

- C improves B (NMSE 0.2094 to 0.1856) on all 12 cases and all 20 leads in the independent audit. Case-bootstrap intervals describe sampling variability only, not seed uncertainty or validation-selection effects.
- C-from-Bplus achieves 0.14398255 and improves all 12 cases versus C-from-B. It has 3x training windows/optimizer steps at 40 epochs. This is an extra-exposure comparison, not an augmentation-only comparison. Full fresh Bplus reproduction has not been completed here.
- An exploratory equal-weight A_official+B forecast ensemble scores 0.18968019. C-minus-ensemble mean is -0.00409419, with case-bootstrap interval [-0.02370704, +0.01239693]; C is better in only 5/12 cases. These cases do not clearly establish an accuracy advantage specific to symmetry rather than generic averaging. The reference ensemble uses two trained models, is not a matched independent-seed control, and has no deployment/health/latency claim.
- The defined C forecast map has exact measured equivariance; this does not force each wake to be symmetric. Organizer guidance supports the transform, but exact sampled-grid coordinates and generator boundary settings remain independently unverified.

## Incomplete control and timing

A full matched local A was not completed. The existing four-epoch A log is a pilot, not a control. Do not claim causal augmentation gains from the organizer reference alone. Waiting/shared GPU contention and the verifier recovery consumed the remaining control window. Clean independent-seed ensemble training remains future work.

The fresh fit took 1251.7 seconds, including contention. Its post-fit verifier initially imported an archived `queue.py` instead of Python's standard library. This was my helper naming error. Verification was recovered with isolated imports, with no retraining, model change, file deletion, or alteration of the original logs. The submitted model ZIP excludes all these archival scheduling helpers.

The human explicitly authorized temporary SIGSTOP of five overlapping P3 ablations until reproduction checks finished. State/files were preserved. Identity-checked SIGCONT restored the live processes; one had already ended. Paused durations must be disclosed in those jobs' wall-clock timing. Recorded C latency is 1.9591 ms on H100 with resident stopped contexts: compute-uncontended, not an empty-memory GPU. Preserve the raw gate note and the action-ledger explanation.

## Coordination and provenance

Read issue https://github.com/pranavvkrishna/GalileoHackathon-Optimus/issues/2 for reservations, the pause authorization, artifact handoff, and mechanism-control findings. Publishing a note does not prove another agent acknowledged it. All changes were additive; no existing files were deleted. No further Person 2 GPU training is queued.

The JSON/CSV evidence, research rationale and action ledgers here support the human-authored report. Codex assisted with environment setup, literature/source inspection, verification code, metric audits, packaging and handoff; the original B implementation/run and P3 evaluations were already team work. Humans retain scientific judgments and submission decisions.
