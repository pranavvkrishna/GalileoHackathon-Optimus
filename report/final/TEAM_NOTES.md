# TEAM placeholders in main.tex — evidence and candidate wording

Write these in your own words (organizer AI policy). The same bullets sit as `% TEAM` comments under each placeholder in main.tex.

## Abstract, final sentence
Evidence: validation 0.2571 → 0.1181 (−54%) from training changes; portal 0.3937 → 0.1593 (−60%, best confirmed);
period copy left validation at 0.1440 but moved portal 0.2876 → 0.1789 (largest single drop).
Candidates:
- Validation-only model selection would have missed our largest hidden-tier gain.
- Known physics (the forcing period) did more for extrapolation than any training change.
- Reflection symmetry helped only once the network was trained on reflected data.

## Key insight 1 — "Symmetry helps once it is learned"
Evidence: A_avg − A_ref −0.0010 [−0.0717, 0.0585] with zero defect; C − B −0.0238 [−0.0273, −0.0204], 12/12.
Candidates:
- Enforcing consistency at test time is not enough; the network must be trained on the symmetry.
- Test-time symmetry acts like a free two-member ensemble only when both passes are in-distribution.

## Key insight 2 — "In-range accuracy came from training choices"
Evidence: more windows −0.0416 (12/12); variance-normalised loss −0.0179 (8/12); ensemble −0.0080 (10/12);
C_ens − A_ref −0.1390 [−0.1982, −0.0985], 12/12; width 96 worse than C_nmse in 12/12; 3-network mix only 0.0020 better.
Candidates:
- Matching the loss to the metric mattered more than model size.
- Data exposure gave the largest single training gain.

## Key insight 3 — "Extrapolation needs known physics, and validation cannot see it"
Evidence: Fig. 1b vertical drop #52 → #95 at fixed validation; Fig. 3 (proxy 0.689 → 0.1905; k ≥ 6.98 windows 0.158 → 0.119);
clamping k worse (1.04 vs 0.99); hiding k no help (1.03–1.05).
Candidates:
- Use a physical invariant (the forcing period) where the network has no data.
- An in-range public validation set cannot select extrapolation methods; proxies built from training data can.

## Takeaway for others (end of Sec. 5)
Candidates:
- Build an extrapolation proxy from your own training data before trusting an in-range validation set.
- Encode known physics (periods, symmetries) where the network has no data; learn the rest.
- A symmetry only helps at test time if the model was trained on it.

## Human-review statement (Sec. 6)
State who read which sections, which numbers were spot-checked against NUMBER_SOURCES.md, and which
"Interpretation:" sentences the team accepts, edits or rejects.

## Contributions (Sec. 6) — drafted from git history, please confirm
- Pranav Vijay Krishna (11 commits): setup, mirror augmentation/averaging code, SIDERIUS scoring, B/B+ training on H100, packages #8–#19.
- Raghav Vijaykannan (22 commits; git name "Vijayakannan"): gates, paired comparisons, proxies, shrinkage, period copy/blend,
  variance-normalised loss, ensemble, width-96, packages #52–#129, report drafts.
- Hrithik Yash (no commits): report review/editing per the earlier report bundle (REVIEW.txt). Add any other roles
  (e.g. physics sign-off, P2 reproduction/A_s0) if they belong to Hrithik.
