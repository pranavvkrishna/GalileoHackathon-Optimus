# spec.md — Pitching-wing wake: final three-hour execution plan

## 1. Decision and scope
Question: Does teaching or enforcing the pitching problem’s reflection symmetry improve forecasts at unseen pitching conditions?

Chosen comparison:

Configuration	Forecast model	Training	Purpose
A	Released width-64 CNN, unmodified	One local seed-0 run	Controlled reference
B	Same CNN with physically consistent reflection augmentation	One local seed-0 run, matched to A	Test learning symmetry from examples
C	B with reflection averaging of its future-field predictions	No additional fitting	Test exact forecast consistency at extra inference cost
Evaluate the shipped official checkpoint immediately as a separately labeled reference and fallback. Two training runs total; no second B seed before establishing A. C is an inference intervention, not a third independently trained architecture.

Hypothesis: B reduces reflection inconsistency; C enforces it algebraically and may improve forecast accuracy. Accuracy is an empirical question. Evidence against the accuracy hypothesis is unchanged or worse paired case-level forecast error despite better consistency.

This is a completed research recommendation and execution specification. Training, hardware timing, data-array validation, health outcomes and forecast improvements have not been executed or measured here. Code below is proposed until the team runs its acceptance checks. The human team authors the final scientific judgments.


### Changes from the supplied runbook
Restore a local matched A/B control before spending time on extra B seeds.
Remove half-period self-symmetry as a validation requirement.
Correct clipping order to match the actual metric code.
Use serial GPU jobs and measured runtime; remove assumed concurrency savings and assumed GPU capacity.
Separate organizer provenance from local reproduction.
Use one consistent initialization, class-loading and packaging procedure.
A rejected physical transform invalidates both augmentation and averaging; C-on-A is only a fallback for failed training, not failed symmetry.

## 2. Evidence and why this comparison
Inspected sources: final release, downloaded task and official-submission ZIP text/code, dataset metadata, participant guide, report rules and main.tex. Task files inspected include RELEASE_CONTRACT, TASK_CARD, baseline training/inference, input construction, metric computation and health views. Full training arrays were not downloaded for this planning work.

The previous focused literature review screened 19 abstracts and inspected five full texts. A subsequent full-text check directly addressed group averaging:

Evidence	Consequence for this plan
Bose et al. 2021, pitching–heaving wake transitions, methods/results including deflected wakes	Individual asymmetric states are physically meaningful; do not force each forecast to be symmetric. Different motion/Re means thresholds do not transfer.
Eivazi et al. 2020, nonlinear ROM, AE-LSTM methods and pitching-airfoil comparisons	A recurrent ROM adds representation fitting and rollout integration; it is not a drop-in improvement to this direct forecast.
Li et al. 2020, FNO, operator construction and Navier–Stokes benchmarks	FNO is a credible alternative, but introduces mode selection and nonperiodic moving-boundary integration. No claim that it cannot fit three hours or would lose.
Fromme et al. 2025, equivariant fluid surrogates, symmetry/architecture/results	Symmetry must preserve the entire physical problem; do not transfer its D4 group or 3D architecture.
Li et al. 2026, neural operator splitting, split supervision and transformed predictors	Required solver intermediate states are absent; reject a full reproduction. Its stochastic rollout strategy is not this forecast average.
Elesedy & Zaidi 2021, equivariant generalisation, Section 6/Lemma 6	Group averaging gives an equivariant predictor. Population squared-loss benefits require assumptions including invariant input measure and equivariant target; no automatic improvement of this fixed validation set’s clipped NMSE follows.
Decision: retain symmetry because it tests one mechanism with a small change to the supplied training code and one short inference wrapper. A POD-plus-regression comparator is also plausible and need not be intrusive; however, basis fitting and choosing a conditional temporal predictor add implementation and comparison choices. An FNO adds a new architecture and tuning problem. Neither alternative has been timed here. Reject them for this sprint’s controlled question and integration budget, not presumed inferiority. No further broad literature survey is required.


## 3. Authoritative task facts
Data: 810 train cases; 12 validation cases and 48 declared validation windows, about 3.44 GB of released files. Fit weights and statistics buffers on train only; validation is for evaluation/selection. Current guide and retraining environment require train-only fitting despite broader wording in the task card’s released-files paragraph.
Physics: 2D pitching NACA0012, Re500. Inputs: float32 [batch,14,48,96]; ten vorticity frames, then constant maps of k/(2π), positive amplitude in radians, sine and cosine of the first forecast-frame phase.
Outputs: float32 [batch,96822]: future fields92160, mean field4608, log spectrum50, frequency1, regime logits3.
Baseline: direct 20-frame forecast; three-level width64 encoder-decoder; residual on last observed frame; 40 epochs, batch32, AdamW lr1e-3/wd1e-4, OneCycleLR, window subsample3. Training targets only the forecast. Keep its training-only 3-NN statistics component unchanged.
Utility: [2*clip(1-NMSE,0,1)+0.5*clip(1-StError/0.3,0,1)]/2.5. Check all official health/validity rules, finite outputs, NMSE≤1 and median sample-forward latency≤100ms. Statistics outputs remain mandatory.
No new CFD simulations, external flow data, externally generated labels or inference solver. The task card explicitly permits consistent reflection augmentation.
Organizer reference: validation NMSE0.257, frequency error0.070; training roughly13–21min on measured RTX GPUs. These are organizer measurements, not predictions of H100 runtime or local results.
Organizer perturbation tests show this two-convective-time horizon remains predictable. Do not attribute model errors to an established fundamental chaos barrier.

## 4. Implementation: one source of truth

### Shared layout and initialization
Keep immutable copies of the task and official example. Each variant retains the original package layout, including its sibling task files, so the baseline’s relative MANIFEST resolves:

```
wake/
  original/package/pitching_airfoil_wake/
  original/official/
  variants/A/pitching_airfoil_wake/{baseline/,task/,...}
  variants/B/pitching_airfoil_wake/{baseline/,task/,...}
  runs/{A_s0,B_s0,C_from_B_s0,A_official}/
  shared/{diagnose.py,plot_results.py}
  submission/
  report/
```
Use environment variables or parsed arguments inside Python; literal "$PKG" in a Python string is not expanded. Create run directories before redirecting logs. Do not overwrite the organizer copy or a teammate’s active source.

Use the exact released seed0 initializer for both A/B, with a recorded checksum; each run saves the actual untrained state before any fitting. The selected package must regenerate legitimate initialization from the recorded seed using included source, then train without depending on an uploaded trained artifact. Audit official initializer equivalence before adapting it. If initialization or training exposure differs from the official setup, label the local control accordingly. A later seed1 experiment must generate its own initialization.

Same cases/windows, normalization, architecture, initialization, training-window order, epochs, optimizer and schedule for A/B. Use an independent augmentation RNG. C loads B’s parameters. This matches training exposure, not guaranteed wall-clock compute.


### B: transform the complete conditioned example
Let R reverse y rows. Confirm row order and the discrete coordinate reflection using source/metadata; domain endpoints alone are insufficient to establish exact sampled centering. If metadata cannot resolve sampling, record that limitation and do not claim an exact physical-grid proof from an involution test alone.

Proposed helpers, after importing torch and the task HISTORY constant:

```python
def mirror_x(x):
    out = x.clone()
    out[:, :HISTORY] = -x[:, :HISTORY].flip(-2)
    out[:, HISTORY + 2:HISTORY + 4] *= -1
    return out

def mirror_field(z):
    return -z.flip(-2)
```
Channels10/11 stay unchanged; sine/cosine both negate, representing phase+π with positive amplitude. Vorticity changes sign under reflection. For each training example independently, transform history and all20 future targets with probability0.5. Keep batch size, optimizer steps and epochs fixed. Use a separate augmentation RNG that does not change the baseline window permutation. Statistics buffers are never augmented or refitted on validation.

Validation checks: explicit channels and phase interpretation, G(G(x))=x, S(S(y))=y, axis/shape checks, and consistency with the full geometry/motion/boundaries. Do not require a trajectory half a period later to equal its own mirror: a deflected or irregular state can satisfy map equivariance without that self-symmetry.


### C: forecast averaging, not symmetric states
```python
class WakeMirrorAvg(WakeBaseline):
    def forward(self, x):
        f = self.forecast(x)
        reflected = mirror_field(self.forecast(mirror_x(x)))
        future = 0.5 * (f + reflected)
        return torch.cat([future.flatten(1), self.statistics(x)], dim=1)
```
This averages complete forecasts, including residual references. It yields F_C(Gx)=S(F_C(x)) algebraically when G/S are the valid involutions; it does not impose F_C(x)=S(F_C(x)). It preserves the state_dict structure. Numerically verify in eval mode with measured absolute/relative tolerances; do not infer tolerances solely from storage dtype. Measure extra inference cost. Keep statistics unchanged and limit exact-equivariance claims to forecasts.

Ensure the evaluation loader actually instantiates WakeMirrorAvg for C. Its reproducible training procedure trains B from the recorded seed and produces weights loadable by that no-argument class; it must not require a pre-existing external B checkpoint.


## 5. Evaluation and acceptance
Use official evaluation code unchanged. Its denominator is each window’s true future variance about the case’s long-time mean. _wake_metrics.score clips per-lead NMSE at4 before averaging:

```python
clipped = np.minimum(lead_nmse, 4.0)  # [windows,20]
window_nmse = clipped.mean(axis=1)
official_check = window_nmse.mean()
```
Do not use minimum(lead_nmse.mean(1),4). Verify with an artificial row containing both values above4 and below4, then compare actual diagnostic outputs to official scores. Plot clipped lead curves if labeling them official-style; retain unclipped errors in logs.

Two additional diagnostics:

NMSE by forecast lead, averaging windows within each case and then cases.
Forecast reflection defect: ||F(Gx)-S(F(x))||² / max(||F(x)||²+||F(Gx)||²,1e-12), averaged by case. This measures consistency, not accuracy.
Compute paired case-level B−A and C−B changes over the same12 cases. A cheap2000-resample paired case bootstrap gives a descriptive95% interval for case-sampling variability; it does not estimate training-seed noise, remove validation-selection bias or prove a population gain. Two B seeds alone would not define a universal noise floor. One matched seed is acceptable here if disclosed.

Table columns: configuration, seed, provenance, split, forecast_nmse, forecast_nmse_final, st_abs_error, utility, reflection_defect, health_status, latency_ms, train_seconds.

Export official predictions plus diag.npz containing case IDs/window starts, [48,20] lead errors and [48] defects; metrics.json, configuration, complete logs and code/checkpoint checksums. Read actual window counts from the package. Check statistics predictions agree where the underlying buffers/data match.

Figure1: lead-error curves and paired per-case changes. Figure2: truth versus A/B/C at early/late lead for the first window of the case nearest median A error. Shared color scale and equal aspect; no cherry-picking the biggest gain.

Measure latency in eval/no_grad mode after warmup, batch1, CUDA synchronization around timed forward execution. Exclude CPU copying/double conversion unless the official checker includes it; use the checker’s methodology when available. Record device. Run the full official health gate, not only finite-value/NMSE checks.

Select the highest locally measured official utility among valid candidates. For negligible gains relative to a predeclared practical tolerance, prefer simpler inference; choose that tolerance before examining results and report it. Do not use two-seed spread as a significance threshold. Selection for submission does not establish a scientific claim.


## 6. Teammate cards and schedule
Use actual remaining event time; offsets below assume the full180min is available. Model uploads close16:00 PST November21; target15:45; reports close16:30. If less time remains, cut optional work first. Inspect the actual GPU and memory; serial training is the default.

Slot	Owner	Finishable task	Output and acceptance
0–15min	P1	Set environment, data access and official-checkpoint inference	Labeled A_official metrics; correct contract; diagnose unexpected differences without arbitrary±0.05 acceptance
0–20min	P2	Implement B transform and checks	Validated channel/phase/axis logic; physical assumptions recorded
0–25min	P3	Implement C wrapper, metric export, report skeleton	Class/shape checks and synthetic clipping-order check
15–50min target	P1	Time full loading/epoch; train local A	Initial/final states, logs; project remaining training+reproduction time
50–85min target	P2	Train B serially	Same settings/seed as A except augmentation
85–105min	P3	Evaluate A/B/C, diagnostic and all gates	Comparable table, paired changes and plots
105–115min	All	Freeze candidate and research decisions	Logged choice; no open-ended sweeps
115–150min	P1	Complete fresh train-only reproduction	Correct selected class, regenerated init/final artifacts and evaluated results
150–165min	P1/P3	Check ZIP and upload	Submission ID, saved ZIP and checksum
Throughout;165–180min	All	Assemble report evidence and author judgments	Four-page body draft; final30min after model deadline only finalizes report
After the initial timing pilot, commit a common epoch budget that fits two training jobs plus one selected-model reproduction. If40 epochs cannot fit, reduce A/B together before the final runs and label them budget-limited local variants. Retain the official checkpoint as a separately labeled reference. Do not assume two concurrent jobs save time because they fit in memory.


### Fallbacks

If physical reflection is invalid, stop B and C; retain baseline, do not apply the same invalid averaging to it. Document the audit and any actual bounded baseline experiment.
If reflection is valid but B training fails, C-on-A is a permitted relabeled diagnostic/fallback without retraining.
If C fails health/latency, submit a valid A/B. If only the organizer checkpoint is available, describe provenance honestly; copying it is not a new trained improvement.
Secure a valid baseline submission early if permitted and practical. No extra seeds/window-count experiments until submission/reproduction time is protected.

## 7. Copy-ready coding-agent assignments

### P1 — Control, reproduction and submission
Read the released contract, task card, official manifest and baseline source. Own local A and final packaging. Preserve immutable organizer files. Establish official-checkpoint inference and report its provenance; do not claim local retraining. Time loading/one epoch and commit a common A/B budget. Train A from actual seed0 initialization, save initial/final artifacts and full source/logs. Keep train-only statistics. Coordinate serial GPU use. Reproduce the selected model with its exact submitted command in a fresh unique directory exposing train data only; verify correct class and all artifacts. Upload early enough to meet15:45 target. No architecture changes or sweeps. Observe failure→hypothesis→one fix→measure→decision→log. If schedule slips, prioritize the valid baseline and completed reproduction.


### P2 — Training intervention
Read input construction and baseline batch/forecast/initializer code. Own B. Implement G/S exactly as §4, confirm physical/discrete assumptions and run involution/channel checks; do not test validity by half-period self-symmetry. Use independent p0.5 augmentation randomness, transforming complete histories and future targets. Keep architecture, seed/init, data/windows, training exposure and statistics identical to A. Train in the agreed serial slot and export artifacts/logs/metrics. Do not run a second B seed or increase windows before matched control and submission reserves exist. Any change beyond augmentation is separately labeled and requires a team decision. Stop both symmetry methods if their physical assumption fails.


### P3 — Inference intervention, evaluation and report
Own C, shared evaluation/plots and official report skeleton. Apply averaging only to B’s complete future-field forecasts and leave statistics unchanged. Verify class-loading, state_dict compatibility and numerical forecast equivariance; do not force individual wakes to be symmetric. Match official per-lead clipping order and test a synthetic counterexample. Export case-aligned measurements, paired bootstrap and the two predeclared figures. Measure latency without host-copy diagnostic overhead and execute full health checks. No independent training or hidden-test tuning. Supply measurements and limitations; humans author insights. If C fails gates, retain a valid simpler candidate and log the reason.


## 8. Submission and reproduction checklist
Start from the official submission layout: root submission.json, self-contained inference plugin, actual untrained initial and final trained state_dict, complete nonempty train/ source. Keep no-argument class/symbol, seed, train_command and selected artifact consistent. Do not use plugin_config for this task. Implement seed-driven initialization in included code, recording equivalence to official initialization where claimed; submitted checkpoints are audit artifacts, not trusted warm starts.

Run the exact final manifest command in a fresh unique directory with train-only data and no old final weights; complete the run and verify initial/final outputs. Evaluate C with its actual class, not the default baseline loader. Report local reproduction differences and investigate material changes. Do not confuse a startup check with reproduction.

Use actual checker limits (guide approximately100MB zipped,1GB unpacked,2000 files). No secrets, symlinks, private paths or network dependencies. Avoid destructive shared temporary-path cleanup. Build a fresh archive so old entries cannot remain. Save accepted submission ID, source revision, ZIP and checksum. Report validity/health outcomes honestly.


## 9. Four-page report scaffold
Use official LaTeX formatting, with no margin/font/spacing compression. All body material including AI statement fits4 pages; references begin separately, aiming for1 page. No narrative appendices.

Required material	Approximate pages	Owner and fill-in evidence
Title/abstract	0.25	All: actual question, comparison, result and key limitation
1 Goal/hypothesis	0.45	P2: physical reflection relation, support/contradiction criteria
2 Research trajectory	0.55	All: real run/why/change/result/next-decision table
3 Final model/comparisons	1.45	P1/P3: A/B/C, table, Fig1, costs/gates, exact submission ID/code/artifact
4 At most3 insights	0.65	All: claim/evidence/limitation; Fig2; human-authored judgments
5 Limits/next experiment	0.30	All:12 cases, one seed, validation selection, two-pass averaging; next control
6 Reproducibility/contributions/AI	0.35	P1: commands/env/seeds/splits/evidence links/roles and verified AI uses
If A wins: distinguish improved consistency from absent accuracy improvement. If B wins: augmentation helped in tested cases but exact projection did not. If C wins: report accuracy and extra inference cost; generic averaging remains a possible contributor. If inconclusive: report measured paired changes and uncertainty without inventing a winner. No private-tier extrapolation claim from local validation alone.

The theory guarantees equivariance for the defined averaged map, not improved clipped NMSE on these12 cases. Ordinary squared-loss population guarantees require additional assumptions; the finite10-frame input may also omit dynamical state. Physical symmetry therefore motivates the test rather than proving its outcome. A same-cost nonsymmetry ensemble or another matched training seed is a concrete future control, not a mandatory new hackathon branch.


## 10. Immediate actions
P1 validates official-checkpoint inference and measures loading/epoch cost.
P2 validates the reflection implementation and prepares the single training change.
P3 verifies metric/class correctness and starts exports, figures and report skeleton.
Freeze this plan after feasibility checks. Further literature searching is optional only if an unresolved physical assumption blocks execution; do not replace experiments with another broad survey.

---

# Appendix — exact implementation (for Claude Code)

Verified against release 20261121-v3: `pitching_airfoil_wake/baseline/run_baseline.py`, `task/plugins/_wake_task.py`, `task/plugins/_wake_metrics.py`, and the official-submission ZIP. Code below is proposed and untested; run every acceptance check.

## A0. CPU-ONLY MODE (use this if we get no GPU) — overrides A5/A7/A12 timings

**Why:** the baseline trained 40 epochs in 13–21 min on an RTX GPU. On a CPU that is likely many hours. `setup_torch()` also calls `torch.set_num_threads(1)`, so CPU training is single-threaded unless changed. Full local training of A and B is **not feasible** on CPU.

**CPU plan (everything here is inference-only and runs in minutes):**

| Config | What | Training |
|---|---|---|
| A | official organizer weights (`original/official/final_model.pt`) | none (the organizers report their retrain reproduced every metric exactly) |
| C_on_A | the same official weights wrapped in `WakeMirrorAvg` (A6 step 2) | none |
| B (optional) | short matched pilot: A_short vs B_short with identical reduced budget | only if a timed pilot shows it fits |

1. Patch run_baseline.py as in A6 (all steps). For CPU runs, change `torch.set_num_threads(1)` in `setup_torch()` to `torch.set_num_threads(os.cpu_count())` **in the variant copy only**. Record this; it does not change results, only speed.
2. Evaluate A (CPU):
   `python run_baseline.py evaluate --data_dir $DATA_DIR --model $WAKE/original/official/final_model.pt --output_dir $WAKE/runs/A_official`
3. Evaluate C_on_A (CPU):
   `python run_baseline.py evaluate --mirror_avg --data_dir $DATA_DIR --model $WAKE/original/official/final_model.pt --output_dir $WAKE/runs/C_on_A`
4. Run diagnose.py for both (add `--mirror_avg` for C_on_A). Paired bootstrap over the 12 cases: C_on_A − A.
5. **Optional B pilot:** time ONE epoch with a reduced budget (set `EPOCHS = 3`, `WINDOW_SUBSAMPLE = 15` in both copies). If one epoch takes under about 10 min, train A_short and B_short with identical settings and compare B_short − A_short. Label them "budget-limited". If it's too slow, skip B and say so in the report (limitations + next experiment).
6. **Submission on CPU:** the C_on_A package = the official package plus the patched run_baseline.py (both copies), with `symbol: WakeMirrorAvg`. Leave `train_command` **unchanged** (no `--mirror`; the organizers retrain the official baseline, then our class averages at inference). Keep the official `initial_model.pt` and `final_model.pt`.
   - Reproduction check on CPU: you cannot rerun 40 epochs. Instead, rely on the organizer statement that the official baseline retrain reproduces exactly, and verify that our class loads the official weights and passes the latency check (2 forward passes; must stay under 100 ms median on the evaluator's device). State this honestly in report §6.
7. Data: `validation.npz` (50 MB) alone is enough for steps 2–4. The full 3.4 GB `train.npz` is needed only for training and for `fill_statistics`. The official weights already contain the statistics buffers, so evaluation does NOT need train.npz. Check that `evaluate` runs with only validation.npz present; if it insists on train.npz, download it.

**What changes in the story:** the question becomes "does enforcing reflection consistency at inference improve the official CNN's forecasts at unseen kinematics, and at what cost?" B (teaching symmetry through training) becomes a budget-limited pilot or the stated next experiment.

**CPU clock:**

| Time | Task |
|---|---|
| 13:10–13:40 | setup, patch, test_mirror.py, get validation.npz |
| 13:40–14:00 | eval A and C_on_A, diagnostics |
| 14:00–14:30 | optional B pilot (time one epoch first) |
| 14:30–14:45 | figures, freeze the choice |
| 14:45–15:30 | package, latency check, ZIP |
| 15:30–15:45 | upload |
| 15:45–16:30 | report |

## A1. Roles (actual team)
| Role | Person | Owns |
|---|---|---|
| P1 + code for P2/P3 | CS teammate (uses Claude Code) | environment, all code edits, GPU runs (A, B, C eval), reproduction, ZIP + upload |
| Physics + results | Mechanical teammate | sign-off on the reflection transform before B trains; runs diagnose/plot scripts; physical interpretation; report §1 and §4 |
| Report + log + clock | Business teammate | LaTeX template, decision/AI-use log, results table, report §2, §5, §6, abstract; calls the 14:35 freeze and the 15:45 upload |

## A2. Facts the code relies on
- `x` is `[B,14,48,96]`: channels 0–9 vorticity history; 10 is k/2π; 11 is θ0 (rad, positive); 12 is sin(k·t_s); 13 is cos(k·t_s).
- Grid rows run from y = −1.5c (row 0) to +1.5c (row 47); columns are x. Reflection is `flip(dims=[-2])`.
- `command_train` builds batches in the inner `batch(ix)`, which returns `x [B,14,48,96]` and `y [B,20,48,96]`. The loss is `((model.forecast(x) - y)**2).mean() / VORT_SCALE**2`. The statistics head is filled by `fill_statistics(model, table)` from train only.
- Official forecast metric (`_wake_metrics.score`): `np.minimum(window_nmse(...), 4.0).mean()`, which **clips each lead, then averages**. `window_nmse` returns `[W,20]` unclipped values, with the per-window variance of the truth about `table.mean_vort[c]`.
- Official manifest (`submission.json`): `plugin: run_baseline.py`, `symbol: WakeBaseline`, `weights: final_model.pt`, `initial_weights: initial_model.pt`, `train_dir: train/`, `seed: 0`, `kind: model`, plus `train_command`.

## A3. Setup (on the GPU machine)
```bash
export WAKE=$HOME/wake; cd $WAKE           # = cloned team repo
mkdir -p original data runs/{A_official,A_s0,B_s0,C_from_B_s0} shared report submission
cd original
wget -q https://github.com/yuema137/siderius-hackathon/releases/download/track2-2026-11-21/pitching-wing-wake.zip
wget -q https://github.com/yuema137/siderius-hackathon/releases/download/track2-2026-11-21/pitching-wing-wake-official-submission.zip
unzip -q pitching-wing-wake.zip -d package && unzip -q pitching-wing-wake-official-submission.zip -d official
cd $WAKE
mkdir -p variants/A variants/B
cp -r original/package/pitching_airfoil_wake variants/A/ && cp -r original/package/pitching_airfoil_wake variants/B/
(cd original/package/pitching_airfoil_wake && python fetch_data.py --output_dir $WAKE/data)
export DATA_DIR=$WAKE/data
sha256sum original/official/initial_model.pt original/package/pitching_airfoil_wake/baseline/init_model.pt   # record; expect identical
nvidia-smi
```
`.gitignore`: `data/`, `original/`, `runs/**/*.pt`, `*.npz`, `*.zip`, `__pycache__/`.

## A4. A_official reference (≈2 min)
```bash
cd $WAKE/variants/A/pitching_airfoil_wake/baseline
python run_baseline.py evaluate --data_dir $DATA_DIR --model $WAKE/original/official/final_model.pt --output_dir $WAKE/runs/A_official
```
Expected near the organizer's 0.257 / 0.070. If it differs a lot, diagnose before training anything.

## A5. Local A (seed 0, unchanged code, serial)
```bash
cd $WAKE/variants/A/pitching_airfoil_wake/baseline
python run_baseline.py train --data_dir $DATA_DIR --seed 0 --output_dir $WAKE/runs/A_s0 \
  --init_model $WAKE/variants/A/pitching_airfoil_wake/baseline/init_model.pt 2>&1 | tee $WAKE/runs/A_s0/train_log.txt
```
- Without `--train-only`, the script evaluates validation at the end and prints metrics JSON.
- After 2 epochs, project the total time. If 40 epochs × the epoch time for A + B + one reproduction does not fit before about 15:15, stop. Lower `EPOCHS` for **both** A and B, and label them budget-limited.

## A6. Patch for B and C — edit only `variants/B/pitching_airfoil_wake/baseline/run_baseline.py`

**(1) After `def block(...)`:**
```python
def mirror_x(x: torch.Tensor) -> torch.Tensor:
    """Reflection y -> -y of a model input: vorticity row-flip + sign flip; forcing phase + pi."""
    out = x.clone()
    out[:, :HISTORY] = -x[:, :HISTORY].flip(-2)
    out[:, HISTORY + 2:HISTORY + 4] = -x[:, HISTORY + 2:HISTORY + 4]
    return out


def mirror_field(z: torch.Tensor) -> torch.Tensor:
    """Reflection of vorticity fields [..., 48, 96]."""
    return -z.flip(-2)
```

**(2) After `class WakeBaseline`:**
```python
class WakeMirrorAvg(WakeBaseline):
    """Config C: reflection-averaged forecast; statistics head unchanged; same state_dict as WakeBaseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forecast(x)
        reflected = mirror_field(self.forecast(mirror_x(x)))
        future = 0.5 * (f + reflected)
        return torch.cat([future.flatten(1), self.statistics(x)], dim=1)
```

**(3) In `command_train`, directly after `generator = torch.Generator().manual_seed(args.seed)`:**
```python
    aug_gen = torch.Generator().manual_seed(args.seed + 1_000_003)  # independent: window order identical to A
```

**(4) In the inner training loop, replace `x, y = batch(order[start : start + BATCH_SIZE])` with:**
```python
            x, y = batch(order[start : start + BATCH_SIZE])
            if args.mirror:
                flip = (torch.rand(len(x), generator=aug_gen) < 0.5).to(device)
                if bool(flip.any()):
                    x[flip] = mirror_x(x[flip])
                    y[flip] = mirror_field(y[flip])
```
Note: `aug_gen` is drawn every batch only when `--mirror` is set, so A's random stream is untouched.

**(5) CLI and loader:**
```python
# in main(), train parser:
p.add_argument("--mirror", action="store_true", help="random y-reflection augmentation, p=0.5 per example")
# in main(), evaluate parser AND predict parser:
p.add_argument("--mirror_avg", action="store_true", help="evaluate as WakeMirrorAvg (config C)")

def _load_model(path: Path, device, cls=None) -> WakeBaseline:
    model = (cls or WakeBaseline)()
    model.load_state_dict(load_state(path))
    return model.to(device)

# in command_evaluate and command_predict:
model = _load_model(args.model, device, WakeMirrorAvg if getattr(args, "mirror_avg", False) else None)
```

**(6) Unit checks (CPU, before any GPU time) — `shared/test_mirror.py`:**
```python
import sys, torch
sys.path.insert(0, sys.argv[1])          # path to variants/B/pitching_airfoil_wake/baseline
import run_baseline as rb
torch.manual_seed(0)
x = torch.randn(3, 14, 48, 96); y = torch.randn(3, 20, 48, 96)
assert torch.equal(rb.mirror_x(rb.mirror_x(x)), x)
assert torch.equal(rb.mirror_field(rb.mirror_field(y)), y)
mx = rb.mirror_x(x)
assert torch.equal(mx[:, 10:12], x[:, 10:12])                       # k, theta0 unchanged
assert torch.equal(mx[:, 12:14], -x[:, 12:14])                      # phase + pi
assert torch.equal(mx[:, :10, 0], -x[:, :10, 47])                   # row 0 <-> row 47, sign flipped
m = rb.WakeMirrorAvg().eval()
with torch.no_grad():
    out = m(x); assert out.shape == (3, 96822)
    F  = out[:, :20*48*96].reshape(3, 20, 48, 96)
    Fg = m(rb.mirror_x(x))[:, :20*48*96].reshape(3, 20, 48, 96)
    err = (Fg - rb.mirror_field(F)).abs().max().item()
print("equivariance max abs err", err); assert err < 1e-4
sd = rb.WakeBaseline().state_dict(); m.load_state_dict(sd)          # same state_dict
print("ALL MIRROR CHECKS PASSED")
```
Mechanical teammate signs off on the physics, i.e. that reflecting the geometry, motion and boundaries maps the flow onto a valid flow, **before** B trains.

## A7. Train B (seed 0, serial after A)
```bash
cd $WAKE/variants/B/pitching_airfoil_wake/baseline
python run_baseline.py train --mirror --data_dir $DATA_DIR --seed 0 --output_dir $WAKE/runs/B_s0 \
  --init_model $WAKE/variants/B/pitching_airfoil_wake/baseline/init_model.pt 2>&1 | tee $WAKE/runs/B_s0/train_log.txt
```
Use the same `EPOCHS` as A.

## A8. Evaluate C (no training)
```bash
cd $WAKE/variants/B/pitching_airfoil_wake/baseline
python run_baseline.py evaluate --mirror_avg --data_dir $DATA_DIR --model $WAKE/runs/B_s0/final_model.pt --output_dir $WAKE/runs/C_from_B_s0
```
st_abs_error must be identical for A_s0, B_s0 and C (the statistics head is unchanged); a mismatch means a bug.

## A9. Diagnostics — `shared/diagnose.py` (paths via arguments; correct clip order)
```python
"""python shared/diagnose.py --baseline_dir <variants/B/.../baseline> --plugins_dir <.../task/plugins>
       --data_dir $DATA_DIR --model <final_model.pt> --out <runs/X> [--mirror_avg]"""
import sys, json, time, argparse, numpy as np, torch
ap = argparse.ArgumentParser()
for a in ("--baseline_dir", "--plugins_dir", "--data_dir", "--model", "--out"): ap.add_argument(a, required=True)
ap.add_argument("--mirror_avg", action="store_true"); A = ap.parse_args()
sys.path[:0] = [A.plugins_dir, A.baseline_dir]
import run_baseline as rb
from _wake_task import read_table
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = (rb.WakeMirrorAvg if A.mirror_avg else rb.WakeBaseline)(); m.load_state_dict(rb.load_state(A.model)); m.to(dev).eval()
t = read_table(f"{A.data_dir}/validation.npz"); pairs = t.windows(False)
X = torch.from_numpy(np.stack([t.features(c, s) for c, s in pairs])).to(dev)
H, N = rb.HORIZON, rb.FIELD
@torch.no_grad()
def fc(x): return m(x)[:, :H*N].reshape(-1, H, 48, 96)
F = fc(X).double().cpu().numpy(); Fg = fc(rb.mirror_x(X)).double().cpu().numpy()
lead = np.zeros((len(pairs), H))                      # unclipped, same as _wake_metrics.window_nmse
for w, (c, s) in enumerate(pairs):
    tr = t.vort[c, s:s+H].astype(np.float64); var = np.mean((tr - t.mean_vort[c].astype(np.float64))**2)
    lead[w] = np.mean((F[w] - tr)**2, axis=(1, 2)) / max(var, 1e-12)
clipped = np.minimum(lead, 4.0)                        # clip per lead FIRST (official order)
win = clipped.mean(1)
SF = -F[:, :, ::-1, :]
defect = ((Fg - SF)**2).sum((1,2,3)) / np.maximum((F**2).sum((1,2,3)) + (Fg**2).sum((1,2,3)), 1e-12)
case = np.array([c for c, _ in pairs]); start = np.array([s for _, s in pairs])
cases = np.unique(case)
case_nmse = {str(t.case_ids[c]): float(win[case == c].mean()) for c in cases}
case_defect = {str(t.case_ids[c]): float(defect[case == c].mean()) for c in cases}
x1 = X[:1]
for _ in range(5): fc(x1)
ts = []
for _ in range(20):
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); fc(x1)
    if dev == "cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
np.savez(f"{A.out}/diag.npz", case=case, start=start, lead_nmse=lead, defect=defect,
         k=t.k[case], theta0_deg=t.theta0_deg[case], regime=t.regime[case])
json.dump({"forecast_nmse_check": float(clipped.mean()), "forecast_nmse_final_check": float(clipped[:, -1].mean()),
           "reflection_defect": float(np.mean(list(case_defect.values()))), "latency_ms_median": float(np.median(ts)),
           "device": dev, "case_nmse": case_nmse, "case_defect": case_defect},
          open(f"{A.out}/metrics_extra.json", "w"), indent=1)
print("done", A.out, float(clipped.mean()))
```
**Acceptance:**
- `forecast_nmse_check` equals the official `forecast_nmse` in that run's `metrics.json` to about 1e-6. If not, fix the script before plotting.
- Synthetic check: `np.minimum([[5,0]],4).mean() == 2.0`, whereas averaging first and then clipping would give 2.5.
- C's `reflection_defect` should be near 0. A and B will be larger.

Run it for each run:
```bash
B=$WAKE/variants/B/pitching_airfoil_wake
for r in "A_official $WAKE/original/official/final_model.pt" "A_s0 $WAKE/runs/A_s0/final_model.pt" "B_s0 $WAKE/runs/B_s0/final_model.pt"; do set -- $r
  python shared/diagnose.py --baseline_dir $B/baseline --plugins_dir $B/task/plugins --data_dir $DATA_DIR --model $2 --out $WAKE/runs/$1; done
python shared/diagnose.py --mirror_avg --baseline_dir $B/baseline --plugins_dir $B/task/plugins --data_dir $DATA_DIR \
  --model $WAKE/runs/B_s0/final_model.pt --out $WAKE/runs/C_from_B_s0
```

## A10. Paired bootstrap + figures — `shared/plot_results.py`
```python
import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
R = ["A_s0", "B_s0", "C_from_B_s0"]
cn = {r: json.load(open(f"{RUNS}/{r}/metrics_extra.json"))["case_nmse"] for r in R}
cases = sorted(cn["A_s0"])
def paired(r1, r0, n=2000, seed=0):
    d = np.array([cn[r1][c] - cn[r0][c] for c in cases]); g = np.random.default_rng(seed)
    b = d[g.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), np.percentile(b, [2.5, 97.5]).tolist(), d
res = {}
for r1, r0 in [("B_s0", "A_s0"), ("C_from_B_s0", "B_s0"), ("C_from_B_s0", "A_s0")]:
    mu, ci, _ = paired(r1, r0); res[f"{r1}-{r0}"] = {"mean": mu, "ci95": ci}; print(r1, "-", r0, f"{mu:+.4f}", ci)
json.dump(res, open(f"{RUNS}/paired_bootstrap.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for r in R:
    d = np.load(f"{RUNS}/{r}/diag.npz"); L = np.minimum(d["lead_nmse"], 4.0)
    per_case = np.array([L[d["case"] == c].mean(0) for c in np.unique(d["case"])])
    ax[0].plot(np.arange(1, 21) * 0.1, per_case.mean(0), label=r)
ax[0].set_xlabel("lead time (c/U)"); ax[0].set_ylabel("NMSE (clipped per lead)"); ax[0].legend(fontsize=7)
d = np.load(f"{RUNS}/A_s0/diag.npz")
order = sorted(np.unique(d["case"]))
kth = [float(d["k"][d["case"] == c][0] * np.radians(d["theta0_deg"][d["case"] == c][0])) for c in order]
for (r1, r0), mk in [(("B_s0", "A_s0"), "o"), (("C_from_B_s0", "B_s0"), "s")]:
    _, _, dd = paired(r1, r0); ax[1].scatter(kth, dd, marker=mk, label=f"{r1} − {r0}")
ax[1].axhline(0, c="k", lw=.5); ax[1].set_xlabel("k·θ0 (rad)"); ax[1].set_ylabel("Δ case NMSE"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("report/fig1.pdf"); print("wrote report/fig1.pdf")
```
- Check that `cases` (sorted case-id strings) and `order` (sorted integer indices) line up. Map with `t.case_ids` if they don't.
- **Figure 2:** truth vs. A/B/C at lead 1 and lead 20 for the first window of the case whose A_s0 case NMSE is nearest the median. Use `cmap="RdBu_r"`, a shared ±vmax and `aspect="equal"`.

## A11. Submission (selected config) — start from `original/official/`
```bash
rm -rf $WAKE/submission/pkg && cp -r $WAKE/original/official $WAKE/submission/pkg && cd $WAKE/submission/pkg
cp $WAKE/variants/B/pitching_airfoil_wake/baseline/run_baseline.py run_baseline.py
cp $WAKE/variants/B/pitching_airfoil_wake/baseline/run_baseline.py train/pkg/baseline/run_baseline.py
```
Edit `submission.json`:

| Selected | `symbol` | `train_command` |
|---|---|---|
| C | `WakeMirrorAvg` | official command + ` --mirror` after `train` |
| B | `WakeBaseline` | official command + ` --mirror` |
| A | official package unchanged | official command |

The official command is:
```
python train/pkg/baseline/run_baseline.py train --train-only --data_dir $DATA_DIR --seed 0 --output_dir out --init_model "${PACKAGE_DIR:-train/pkg}/baseline/init_model.pt" && cp out/final_model.pt final_model.pt
```
`initial_model.pt` stays the official seed-0 untrained state (same architecture and seed); confirm its sha256 matches `train/pkg/baseline/init_model.pt`.

**Fresh reproduction (mandatory):**
```bash
R=$(mktemp -d) && cp -r $WAKE/submission/pkg/. $R/ && cd $R && rm -f final_model.pt
mkdir -p $R/trainonly && ln -s $DATA_DIR/train.npz $R/trainonly/train.npz
DATA_DIR=$R/trainonly bash -c "$(python -c 'import json;print(json.load(open("submission.json"))["train_command"])')"
cd $WAKE/variants/B/pitching_airfoil_wake/baseline
python run_baseline.py evaluate [--mirror_avg if C] --data_dir $DATA_DIR --model $R/final_model.pt --output_dir $R/eval
```
- Compare with the runs/ result, then copy `$R/final_model.pt` into `submission/pkg/`.
- If `train --train-only` fails because `validation.npz` is missing, record that, and run with the full data folder instead. The organizers' sandbox defines the real rule.

**ZIP and upload:**
```bash
cd $WAKE/submission/pkg && find . -name __pycache__ -prune -exec rm -rf {} + ; grep -rIn "sk-\|api_key\|OPENAI" . || true
python -c "import json;print(json.load(open('submission.json')))"
rm -f ../final.zip && zip -qr ../final.zip . && unzip -l ../final.zip | head -20 && du -h ../final.zip && sha256sum ../final.zip
```
Upload at submit.siderius.scitra.org/track2 (team ID + access code). Save the submission ID; the report PDF links to it.

## A12. Clock (start 13:00; adjust to real time)
| Time | GPU | Others |
|---|---|---|
| 13:00–13:20 | setup, data, A_official eval | Mech: review the reflection math; Biz: report template + log |
| 13:20–13:55 | train A_s0 | CS: B/C patch + `test_mirror.py` on CPU |
| 13:55–14:30 | train B_s0 (after Mech sign-off) | CS: diagnose.py/plot_results.py dry run on A_official |
| 14:30–14:45 | eval C, diagnostics ×4, figures | Biz: results table |
| 14:45 | **freeze: pick submission** | all |
| 14:45–15:30 | fresh reproduction of the selected model | Mech + Biz: write report |
| 15:30–15:45 | ZIP + upload | — |
| 15:45–16:30 | — | finish and upload report PDF |
