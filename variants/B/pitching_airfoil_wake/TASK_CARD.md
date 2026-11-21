> **Event release 20261121-v3:** The root RELEASE_CONTRACT.md overrides historical seed and submission-format examples below. Use the actual seed and separate initial/final artifacts from the official submission.json.

# Task card: Forecast a pitching-airfoil wake under unseen kinematics

## Scientific question

A NACA0012 airfoil pitching sinusoidally about its quarter chord sheds a
vortex wake whose character depends on the reduced frequency k = omega c / U
and the pitch amplitude theta0. Simulations of a pitching NACA0012 at Re 500
found periodic force histories at small k theta_max, quasi-periodic ones from
k theta_max of about 2.2 rad and chaotic ones at about 3.1 rad and above
(non-sinusoidal pitch with the effective angle of attack of a plunging foil;
Zaman, Lai,
Young and Ashraf, 18th Australasian Fluid Mechanics Conference, 2012); PIV of
a flapping foil showed the wake passing from a von Karman street to a reverse
(thrust) street and then to a deflected, asymmetric street as frequency and
amplitude grow (Godoy-Diana, Aider and Wesfreid, Phys. Rev. E 77, 016308,
2008); and immersed-boundary simulations of a pitching-plunging elliptic foil at
Re 300 mapped the periodic, quasi-periodic and chaotic wake regimes in terms of the maximum
effective angle of attack and a leading-edge Strouhal number (Majumdar, Bose
and Sarkar, J. Fluid Mech. 942, A40, 2022). Flapping-wing and flow-control design needs fast models that
predict this wake for kinematics that were never simulated. The task asks: for
unseen (k, theta0), given only the kinematics and a one-convective-time
history of the vorticity field, can a model (a) forecast the next two
convective times of the vorticity field, (b) predict the long-time wake
statistics (mean field, wake spectrum, dominant frequency), and (c) say which
regime the wake is in? Flow fields only: forces are not part of the task.

## Task design

Each simulated case is one pitching kinematics theta(t) = theta0 sin(k t) at
chord Reynolds number 500, run for 60 convective times c/U; the first 15 are
discarded as transient. One sample is one forecast window of one case.

- Model input `[B, 14, 48, 96]` float32: ten history frames of spanwise
  vorticity omega_z c/U (0.1 c/U apart) on a 48 x 96 grid covering
  x in [-0.5, 5.5] c from the leading edge and y in [-1.5, 1.5] c (16 cells per
  chord), plus four constant maps: k/(2 pi), theta0 (rad), and the forcing
  phase sin(k t_s), cos(k t_s) at the first forecast frame.
- Model output `[B, 96822]` float32: the next 20 frames (2 c/U), the case's
  long-time mean vorticity field, the log10 normalised wake spectrum in 50
  bins of f c/U over [0, 2.5], the dominant wake frequency, and three regime
  logits. The layout is in `task/declared/task_config.yaml` and
  `task/plugins/_wake_task.py` (`unpack`).
- Splits are by kinematics (a case is the independent unit). Training
  (`train.npz`, 810 cases): 110 cases spread with a low-discrepancy design
  over k in [2, 7], theta0 in [5, 35] deg, minus a held-out block, plus 700
  cases from a second low-discrepancy draw over the same region that also
  excludes the hidden boundary band (k theta0 in [2.3, 3.3] rad) and stays at
  least 0.02 (normalised k, theta0 distance) from every validation and hidden
  case. Validation
  (`validation.npz`): every case inside the block k in [5.5, 6.5],
  theta0 in [24, 32] deg, which straddles the periodic-to-aperiodic transition
  of this flow (k theta0 about 2.3 to 3.6 rad); four declared windows per
  case (`win_case`, `win_start`, forecast starts at t = 21.1, 31.1, 41.1,
  51.1). Training uses every fifth window start of every training case.
- Private hidden test (kept by the organizers, scored per tier, never
  public): T1 in-range kinematics outside the block; T2 kinematics dense
  near the regime boundary (k theta0 in [2.3, 3.3] rad); T3a amplitude
  extrapolation (theta0 38 to 45 deg); T3f frequency extrapolation
  (k 7.5 to 9). The hidden cases are new simulations that exist nowhere
  else.

**Training-data rule (binding).** Models may be trained only on the released
(Zenodo) files of this task, `train.npz` and `validation.npz`. Augmentation of
the released data is allowed, for example the exact up-down mirror symmetry of
the flow (y to -y with omega_z sign flipped and theta0 sign or forcing phase
shifted by pi), extra windows or frame strides cut from the released sequences,
cropping, noise, re-sampling and mixing of released samples. No additional data
of any kind may be used: no new simulations with WaterLily.jl or any other flow
solver (including more cases, longer runs or other resolutions), no other
downloaded or database flow data, and no labels from external models or codes.
Hidden-test scores count only for code that the organizers can retrain on the
released files. The rule that no flow solver may run at inference (Evaluation)
still applies.

`task.yaml` binds `WakeTaskDataPath` with `train_file: train.npz` and
`validation_file: validation.npz`.

## Data

- DOI: 10.5281/zenodo.24386760
- Size: 3,436,578,184 bytes in 2 files: `train.npz` (3,386,399,684 bytes, 810 cases) and `validation.npz` (50,178,500 bytes, 12 cases, 48 scored windows)
- License: cc-by-4.0
- Access: public
- Processing: `fetch_data.py` verifies size and md5 of both files against
  `data_source.json` and copies them unchanged into the data folder. Files are
  NumPy archives read with `allow_pickle=False`.

Source: organizer-generated 2D incompressible Navier-Stokes simulations with
WaterLily.jl (MIT license; boundary data immersion method,
Cartesian grid, FP32, convective exit; Weymouth and Font, Comput. Phys.
Commun. 315, 109748, 2025), NACA0012 pivoting at c/4, Re = 500,
domain 12c x 6c, 64 cells per chord; the stored vorticity is block-averaged to
16 cells per chord. The 700 added training cases were run with the same
generator script, Julia/WaterLily versions and settings on NERSC Perlmutter
A100 GPUs (the first 110 on the organizers' RTX GPUs and CPUs). QC: two
existing training cases re-run on an A100 reproduce the regime, the dominant
frequency (within 2e-4), the mean field (relative L2 0.004 and 0.026) and
the first frame (NMSE 0.001 to 0.004). Each file also holds the cross-stream velocity at five
wake probes (x = 2, 3, 4 c from the leading edge on the axis, and
x = 3 c, y = +-0.5 c), sampled every 0.05 c/U from t = 0, and the per-case
targets. Labels are computed from the probe records over t in [15, 60]:
spectrum = Hann-windowed periodogram averaged over probes (each normalised to
unit power) and summed into 0.05-wide bins; dominant frequency = spectral peak
(parabolic refinement); regime from the periodicity defect
D(m) = rms(v(t + m T) - v(t)) / (sqrt(2) std v), T = 2 pi / k, m = 1..4:
periodic if min D < 0.10, aperiodic if min D >= 0.30, quasi-periodic
otherwise. Known limits: the target is this solver's physics at this
resolution (see Known shortcuts and leakage risks and Difficulty).

## Evaluation

Primary metric `forecast_nmse` (lower is better): for each declared window,
the squared error of the 20 forecast frames divided by the variance of the
true frames about the case's long-time mean field, averaged over the leads;
per-window values are clipped at 4; mean over windows. 1.0 equals predicting
the mean wake; persistence of the last frame scores 2.06 on validation.

Secondary metrics, each a component of the `evaluation:` section, with window
predictions pooled per case so that a case counts once: `st_abs_error`
(absolute error of the dominant frequency in f c/U) is scored;
`forecast_nmse_final` (lead 20 only), `mean_field_rel_l2` (relative L2 error
of the time-mean vorticity), `psd_log_rmse` (RMS error of the log10 wake
spectrum over 50 bins) and `regime_balanced_accuracy` (argmax of
window-averaged probabilities; chance 1/3) are report-only.

Selection is a weighted sum of forecast (weight 2, bounds 0 to 1) and
dominant frequency (0.5, bounds 0 to 0.3 f c/U); the utility is
(2 (1 - forecast_nmse) + 0.5 (1 - st_abs_error / 0.3)) / 2.5, each term
clipped to [0, 1]. The regime balanced accuracy is excluded because regime
labels near the transition change with resolution (9 of 12 validation cases
agree between 32 and 64 cells per chord) and a two-threshold rule on
k theta0 already scores 0.667 on validation, above the baseline's 0.611.
The mean-field and spectrum errors are excluded because their labels are not
resolution-converged: re-running all 185 cases at 32 cells per chord changes
the mean field by a median relative L2 of 0.32 (0.93 in the
validation block) and the log-spectrum by a median of 0.28 decades
(0.57 in the block), comparable to the baseline's errors.
Constraints: `forecast_nmse <= 1` (worse than the mean wake is unusable) and
median forward latency `inference_latency_ms <= 100` per window on the
training device. Rule: the model must be a learned surrogate; running a flow
solver (WaterLily.jl or any CFD code) at inference is not allowed. Advancing
the solver at 64 cells per chord needs 130 to 730 time steps per c/U,
each with a multigrid pressure solve (3.8 to 14 s per c/U measured on our
shared GPUs, 1.5 s or more on 2 CPU threads), and the state must first be spun up from rest over about 20 c/U, so
the latency bound also blocks it in practice; the baseline needs 1.1 ms per
window on an RTX 5080. Health: `sample_dispersion_floor`
on the temporal RMS of the 20 predicted frames of each window, computed from
the forecast alone, blocks forecasts frozen in time (persistence, steady or
blurred static fields) or whose unsteadiness does not vary with the
kinematics.

References for reading the scores:
- Twin experiments (same case, velocity perturbed by 1e-4 U at t = 20, six
  cases, 64 cells per chord): forecast NMSE at 2 c/U is at most 1.9e-5. Over
  10 c/U one case (k = 5.85, theta0 = 34 deg) reaches NMSE 0.1 at 5.5 c/U;
  the others stay below 0.01. The ideal 2 c/U forecast score is therefore
  essentially 0.
- Statistics noise floor from two halves of a 105 c/U run (three cases): 0.20
  to 0.33 decades in `psd_log_rmse`, 0.05 to 0.21 in `mean_field_rel_l2`, St
  within 0.007, regime agreeing in all three.
- Lock-in prior (St = k/2 pi, always periodic): 0.256 in `st_abs_error` and
  0.333 in balanced accuracy on validation.
- Two thresholds on k theta0 fitted on the training labels (2.60 and
  3.05 rad; training BA 0.98): regime balanced accuracy 0.556 on validation
  (baseline 0.611) and 0.944 on the boundary tier (baseline 0.744). Fitted on
  the first 110 training cases only (2.60 / 2.80 rad) the rule scored 0.667 on
  validation. A learned model should at least match this rule.

Validation has only 12 cases (regimes 2 / 4 / 6), so its scores are noisy:
case-bootstrap 95% intervals for the baseline are 0.21 to 0.32 for
`forecast_nmse` and 0.36 to 0.67 for balanced accuracy. The organizers report
hidden tiers with bootstrap intervals.

## Baseline

PyTorch (`pytorch`, `final_model.pt`, `init_model.pt`; no extra packages).
One module with two parts: a convolutional encoder-decoder (3 levels,
width 64, 1.89 M parameters) that maps the 14 input channels directly to the
20 future frames as a residual on the last history frame (MSE, AdamW 1e-3,
one-cycle schedule, 40 epochs, batch 32, 22,680 training windows: every
third window of the data path's stride, i.e. forecast starts every 1.5 c/U,
over all 810 training cases);
and inverse-distance 3-nearest-neighbour interpolation in the normalised
(k, theta0) plane over the training cases for the mean field, spectrum,
dominant frequency (as a multiple of the forcing frequency) and regime vote,
stored as buffers (the training mean fields as a 32-mode PCA basis with per-case coefficients). Weights are stored in float16 (each artifact stays under
5 MiB) and evaluation uses exactly the stored weights. Validation scores (seed 0, RTX 5080 run): forecast_nmse 0.257, forecast_nmse_final 0.410, mean_field_rel_l2 0.794, psd_log_rmse 0.973, st_abs_error 0.070, regime_balanced_accuracy 0.611. Compute: one GPU. The full train plus validation evaluation took 839 s on one RTX 5090 (train 780 s; forecast_nmse 0.253 there) and 1,337 s on one RTX 5080 (train 1,283 s), with peak GPU memory 4.1 GiB. The checker's retrain on an RTX 5080 reproduced every metric exactly. Median forward latency is about 1.1 ms per window (same network as before).

Hidden tiers (baseline; case-bootstrap 95% intervals; k theta0 rule = reference):

| tier | cases (P/Q/A) | forecast_nmse | st_abs_error | regime BA | k theta0 rule BA |
|---|---|---|---|---|---|
| T1 in-range | 23 (21/1/1) | 0.052 [0.036, 0.079] | 0.028 [0.000, 0.077] | 0.67 [0.50, 1.00] | 1.00 |
| T2 boundary | 20 (10/4/6) | 0.201 [0.144, 0.288] | 0.079 [0.022, 0.156] | 0.74 [0.52, 0.96] | 0.94 |
| T3a amplitude | 10 (3/1/6) | 0.604 [0.465, 0.724] | 0.248 [0.124, 0.377] | 0.56 [0.39, 0.94] | 0.50 |
| T3f frequency | 10 (6/1/3) | 1.354 [1.092, 1.573] | 0.121 [0.000, 0.307] | 0.56 [0.33, 1.00] | 1.00 |

Selection utility (forecast 2, St 0.5; higher is better), validation / T1 /
T2 / T3a / T3f: baseline 0.748 / 0.940 / 0.787 / 0.351 / 0.119 (T3f infeasible: forecast_nmse > 1);
with only the first 110 training cases the same baseline scored 0.614 / 0.897
/ 0.714 / 0.267 / 0.119. Persistence scores 0.153 / 0.181 / 0.147 / 0.035 / 0.119 and the lock-in
prior 0.624 / 0.937 / 0.695 / 0.353 / 0.119.

The baseline passes the Health gate on every tier and the latency constraint;
it violates `forecast_nmse <= 1` on T3f. Persistence scores 1.96 to 2.06 on
every tier.

## Novelty

All data are new simulations generated by the organizers for this task; the
hidden test cases exist nowhere else. Public benchmarks of pitching or
flapping foils give forces or a handful of cases; here a model must cover a
two-parameter kinematics space with regime changes and is scored on
short-term field forecasts, long-time spectra and regime classification at
the same time, per tier, including dense sampling near the regime boundary
and amplitude / frequency extrapolation.

## Difficulty

The wake changes qualitatively across the design space: locked periodic
streets at small k theta0, modulated and deflected wakes and strongly
aperiodic wakes above about k theta0 = 2.5 rad in this setup. The validation
block and hidden tier T2 sit on this transition, and only 46 of the 810
training cases are non-periodic (4 quasi-periodic), so interpolating the statistics from
neighbouring kinematics is not enough. The forecast must advect and generate
vortices for a kinematics the model has not seen. The baseline's forecast
error grows from 0.05 (hidden in-range tier) to 0.20 (boundary tier), 0.26
(validation block) and 0.60 to 1.35 (extrapolation tiers); its (report-only)
regime balanced accuracy is below a two-threshold rule on k theta0 on
the boundary tier (0.74 vs 0.94). The baseline is data-limited: with the
first 110 training cases, retraining on 25 %, 50 % and 100 % of them gave
validation forecast_nmse 0.909, 0.457 and 0.357 (two seeds each); with all
810 released training cases it reaches 0.257 (dominant-frequency error 0.149
to 0.070). The 185 original cases also exist at 32 cells per chord
(organizer disk), but they are not released: their labels differ from the
64-cell labels near the transition (table below). Twin experiments show the flow is deterministic over the
2 c/U horizon (twin NMSE below 1e-4), so the gap between the baseline and 0
is model error, not chaos.

Limits, stated honestly. The target is this solver's physics at 64 cells per
chord in 2D, not experiment. Resolution check: each of the original 185 cases
(110 training, validation, hidden) was also run at 32 cells per chord (the 700
added training cases were not); the agreement between the two resolutions is:

| split | n | regime agrees | St within 0.01 | mean-field rel. L2 (median / max) | log-spectrum RMS (median) |
|---|---|---|---|---|---|
| train | 110 | 104 | 105 | 0.27 / 1.43 | 0.20 |
| validation | 12 | 9 | 9 | 0.93 / 1.48 | 0.57 |
| T1 in-range | 23 | 23 | 22 | 0.29 / 1.52 | 0.18 |
| T2 boundary | 20 | 15 | 16 | 0.59 / 1.47 | 0.66 |
| T3a amplitude | 10 | 9 | 4 | 1.02 / 1.45 | 0.68 |
| T3f frequency | 10 | 9 | 7 | 0.46 / 1.43 | 0.47 |
| all | 185 | 169 | 163 | 0.32 | 0.28 |

Disagreements are periodic versus quasi-periodic calls with min D near the
0.10 threshold, and the low-frequency modulation peak of aperiodic wakes.
Mean-field differences are largest for deflected wakes, where the deflection
side differs between runs. Near the transition the labels carry resolution
uncertainty: in the validation block the two resolutions disagree on the
regime and the dominant frequency in 3 of 12 cases, and a third resolution
(128 cells per chord) was not affordable, so convergence of the 64-cell labels
there is not demonstrated. The regime thresholds (min D 0.10 and 0.30) were
set by the organizers, not taken from the literature. In this setup the
periodic-to-aperiodic transition sits at larger k theta0 than in Zaman et al.
(2012). At k = 4 their theta_max = 30.9 deg is quasi-periodic and 45 deg is
chaotic; our 30.9 deg case is periodic at both 32 and 64 cells per chord and
our 45 deg case is quasi-periodic at both. The setups differ: Zaman et al.
prescribe a non-sinusoidal pitch, theta(t) = arctan(k h cos(omega t)), that
reproduces the effective angle of attack of a plunging foil; they use a
body-fitted ANSYS Fluent grid and classify periodicity from the thrust
history. Here the pitch is sinusoidal, the body is immersed, and the
classification uses wake velocity probes.

## Known shortcuts and leakage risks

- Collapse to the mean wake or to persistence: scored by the normalised
  forecast error (mean wake = 1, persistence = 2.06) and blocked by
  the Health gate on the forecast's own unsteadiness and the
  `forecast_nmse <= 1` constraint.
- Running the solver at test time: k and theta0 are inputs, the initial
  condition (rest) is deterministic and WaterLily.jl is public, so re-simulating
  each case would reproduce the labels. This is forbidden by rule and blocked
  in practice by the 100 ms median latency constraint per window. Simulating
  extra training cases offline is likewise forbidden by the training-data rule.
- Fitting a k theta0 threshold for the regime: reported as a reference
  (balanced accuracy 0.667 on validation, 0.917 on the boundary tier); the
  regime score is report-only, so this shortcut cannot raise the selection
  utility.
- Lock-in prior for statistics (St = k/2 pi, always periodic): right for most
  training cases, wrong on the aperiodic cases that dominate the validation
  block; balanced accuracy (not accuracy) prevents the majority class from
  scoring well.
- Memorising public cases: validation and hidden kinematics are disjoint from
  training; hidden cases are new simulations not on Zenodo. Validation truth
  is public by design (it is the public development split); the hidden tiers
  are the reported test.
- Leakage via names or order: case ids carry no parameters beyond what the
  input gives (k, theta0 are inputs anyway); windows are scored by
  (case id, start), not by position.
- Using future frames: the input holds only the ten frames before the
  forecast start; the training file holds full sequences of training cases
  only.
- Metric edge cases: per-window NMSE is clipped at 4 so a diverged window
  cannot dominate but still scores worst; statistics are pooled per case so
  many windows of one case cannot outweigh another case; non-finite outputs
  make the deliverable unscoreable.
- Selection on a small validation set: 12 cases; bootstrap intervals are wide
  (forecast_nmse 0.21 to 0.32; report-only regime balanced accuracy 0.36 to
  0.67), so differences of a few hundredths
  are not meaningful.

## Contact

SIDERIUS organizers, UC San Diego (Galileo Hackathon Track 1 organizer task
ORG-MAE-2). Reviewer area: wakes, spectral analysis and modal decomposition
(the scientific reviewer's own research area overlaps with this task; this is
disclosed to the committee).
