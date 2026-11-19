# BrakeEnvelope-4

Reproducible experiments for **Calibrating Continuous-Speed Braking**, by Lei Chen, Cemil Emre Yavas, Yue Zhang, and Nancy R. Mead.

This project evaluates a two-endpoint conformal envelope for the stopping law

`D(v) = tau * v + beta * v**2`, with `beta = 1 / (2 * deceleration)` and `0 <= v <= 3 m/s`.

The synthetic benchmark has four parameter populations: correlated Gaussian, negatively correlated heteroscedastic Gaussian, heavy-tailed, and a mixture with mutually exclusive delay/braking faults. A complete episode specifies a context and both physical coefficients. The same episode defines the complete continuous-speed stopping curve.

## Run

Use Python 3.12 and a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/experiment.py --workers 12 --seeds 40 --test-size 10000
python src/analyze.py
python src/measurement.py --workers 12 --seeds 40 --test-size 10000
python src/measurement.py --benchmark-only
```

The original experiment and measurement experiment each create 160 independent training/calibration/test splits and 1,600,000 unique test episodes. Together they contain 320 independent splits and 3,200,000 unique test episodes. The measurement experiment reuses its latent episodes across nine measurement configurations; these repeated evaluations are not counted as new test episodes. Set `--workers` to the desired number of CPU processes; each worker uses one numerical thread. A GPU is not required. The complete study is small enough for a workstation with 18 GB RAM. The recorded run used twelve workers. Runtime is available in `results/metrics_runtime.json` and depends on the software environment and processor.

A development check is available with `python src/experiment.py --pilot --workers 4 --test-size 1000`. It uses seed 7 and writes `results/pilot.csv`. Confirmatory runs use seeds 1000 onward; the released analysis uses seeds 1000–1039.

## Project files

| Path | Purpose |
|---|---|
| `src/experiment.py` | Synthetic data generator, coefficient and scale learners, conformal ranks, nine methods, including a jointly calibrated functional-grid equivalence control, clearance changes, and dynamics stress tests. |
| `src/analyze.py` | Macro and population summaries, repeated-split 95% intervals, and paired speed contrasts. |
| `tests/test_invariants.py` | Eight checks of the endpoint identity, speed inversion, safety implication, ranks, numerical braking integration, policy inversion, convex grid bounds, and seeded generation. |
| `src/measurement.py` | Paired-distance reconstruction, exact bounded-error scoring, independent noisy-label experiments, paired analyses, and score microbenchmark. |
| `tests/test_measurement.py` | Six checks of reconstruction, error-box extrema, latent-score domination, zero-error equality, and joint-grid equivalence. |
| `protocol.txt` | Original experimental design and documented comparator clarification. |
| `revision_protocol.txt` | Fixed design for bounded-error identification and the stronger comparator. |
| `results/measurement_metrics.csv` | Per-split results for every measurement bound, profile, layout, and method. |
| `results/measurement_summary.csv` and `measurement_scenarios.csv` | Macro and population-specific noisy-label results with intervals. |
| `results/measurement_contrasts.csv` | Paired speed costs and curve-risk differences. |
| `results/measurement_runtime.json` | Extension run settings and fit timings. |
| `results/measurement_example.npz` | Preserved paired-distance example at seed 2000, heteroscedastic population, 2.5-cm under-reading. |
| `results/score_benchmark.json` | Score-computation timings and equality check across all shared Endpoint/Joint-grid settings. |
| `results/metrics.csv` | Per-split results for all methods, nominal levels, calibration sizes, and conditions. |
| `results/summary_macro.csv` | Population-averaged estimates and 95% intervals across seed blocks. |
| `results/summary_scenarios.csv` | Separate results for each parameter population. |
| `results/paired_contrasts.csv` | Paired differences and relative gains in mean speed. |
| `results/analysis_details.json` | Experiment counts and calibration-set diagnostics. |
| `results/metrics_runtime.json` | Run settings, software platform, and timing records. |
| `results/example_*.npz` | Complete seed-1000 test arrays and calibration predictions for each population. |
| `requirements.txt` | Exact numerical-library versions. |
| `CITATION.cff` | Machine-readable attribution. |
| `LICENSE` | MIT license for the code and generated benchmark data. |
| `SHA256SUMS` | File integrity checksums. |

## Data generation and splitting

Each run uses independent `numpy.random.SeedSequence` streams for center fitting (4,000 episodes), scale fitting (2,000), calibration (2,047), testing (10,000), calibration decisions, and test clearances. The primary calibration size is 511; smaller settings use prefixes of the independent calibration stream. Test episodes are reused for paired comparisons within a run.

Contexts contain five independent uniform `[0,1]` covariates. Their coordinate order is fixed. Parameter centers are

- `mu_tau = 0.08 + 0.20*x[0] + 0.10*x[1]**2` seconds;
- `mu_beta = 0.18 + 0.20*x[2] + 0.12*x[3]**2 + 0.05*sin(pi*x[4])` seconds squared per meter.

Heterogeneous scales are `0.025 + 0.08*x[0]**2` and `0.03 + 0.12*x[2]**2`. The generator specifies the covariance, heavy-tail multiplier, and mixture fault probabilities in `population()`. Final parameters are bounded below by 0.015 seconds and 0.04 seconds squared per meter. Values specify a controlled engineering benchmark; they are synthetic parameters rather than sensor measurements.

## Methods and conditions

`nominal` uses the predicted coefficients without a margin. `pooled` calibrates one randomly recorded speed per episode. `endpoint` calibrates the maximum of two endpoint scores. `joint_grid` jointly calibrates the maximum across 201 normalized residual evaluations including the zero-speed limit; it is statistically identical to Endpoint. `policy` calibrates the inflation required by the standard clearance-based speed rule. `coefficient` uses two coefficient-wise Bonferroni bounds. `grid_bonferroni` calibrates twenty positive speed knots and uses convex secants with an exact zero origin. `constant_scale` replaces learned scales with fixed positive constants. `tolerance` uses a binomial tolerance rank at delta 0.05.

`standard`, `low`, and `high` use log-uniform clearances on `[0.08,4.5]`, `[0.03,0.4]`, and `[0.8,4.5]` meters. The same calibration is reused for these deployment conditions. `brake_shift` multiplies test beta by 1.25. `cubic` adds `0.04*v**3` to true test stopping distance. `cubic_corrected` also adds that known term to the endpoint envelope. The last two stressors intentionally depart from the nominal law or exchangeability assumptions.

## Bounded-error identification

Independent seeds 2000–2039 generate folds with 4,000 training, 2,000 scale, 511 calibration, and 10,000 test episodes. Two distances are observed at 0.75 and 2.5 m/s with the same latent coefficients within the pair. Absolute distance-error bounds are 0, 1, 2.5, and 5 cm. Positive bounds have symmetric uniform and under-reading uniform profiles. Additional fixed comparisons change the first speed to 0.3 and 1.5 m/s at 2.5 cm under-reading. There are nine configurations per split. Training labels are reconstructed from the noisy distances and floored for fitting; calibration reconstructions are not clipped.

`oracle_label` uses latent calibration coefficients with the noisy-trained fits. `naive_label` uses reconstructed coefficients without an allowance. `robust_label` takes the worst score over the distance-error rectangle, evaluated exactly at speed 0 and V. It protects the latent curve when the stated bounds hold. Its allowance does not use the sign or distribution of the noise. All true test parameters are used only for evaluation. The same latent episodes and noise uniforms are reused across conditions to support paired comparisons.

Measurement result identifiers add `eta` (meters), `noise`, `u1`, and `u2` (m/s); `q` is the calibrated normalized threshold. The example NPZ records `cal_measured`, `cal_reconstructed`, `cal_y`, calibration predictions/scales, test coefficients/predictions/scales, clearances, and each method's decisions. The robust formula assumes stable parameters across the measurement pair and valid absolute error bounds; measurement-speed or clearance error is a different modeling problem.

## Result columns

Identifiers are `scenario`, `seed`, `n`, `alpha`, `condition`, and `method`. `curve_failure` is the fraction of complete stopping curves exceeding their envelopes. `stop_failure` is the fraction of selected stops exceeding available clearance. `mean_speed` is measured in m/s. `speed_ratio` averages selected speed divided by the clairvoyant feasible speed; it can exceed one for unsafe decisions. `conservative_loss` averages the nonnegative fractional speed deficit relative to that oracle. `mean_margin` is the distance correction at 1.5 m/s; it can be infinite when a finite-sample rank is unresolved. `abstain` is the zero-speed fraction.

Summary suffixes `_lo` and `_hi` are 95% Student-t intervals. Macro summaries first average the four independent populations within each seed and then use the 40 seed blocks. Paired relative-speed intervals use the delta method. These repeated-experiment intervals are distinct from a guarantee conditional on one deployed calibration set.

Saved NPZ arrays contain `x` (N by 5), `parameters`, `prediction`, and `scale` (N by 2, ordered tau then beta), `clearance` (N), and per-method `_speed` and `_failure` arrays. Calibration arrays are `cal_y`, `cal_prediction`, and `cal_scale`; `vlog` and `dcal` contain the recorded speeds and calibration clearances.

## Reproduction

The released metrics correspond to the pinned dependencies. Random seeds, fixed hyperparameters, and independent streams determine the numerical results. Eight-worker and twelve-worker executions are checked for identical original and measurement metric values. Timing fields are expected to vary. Independent software or hardware versions can introduce small floating-point differences near decision thresholds.

Fourteen scientific tests compare endpoint and bounded-error suprema with 10,001 candidate speeds, check all measurement-error corners, and independently integrate the delayed constant-deceleration dynamics. The isolated timing benchmark uses 511 calibration episodes and 1,000 repetitions after 20 warmups. It measures score computations only; timing values naturally vary by system and load. All benchmark scenarios, including unfavorable stress-test results and the more efficient fixed-policy comparator, are retained.

## Attribution and license

The project and generated synthetic data are distributed under the MIT license. Cite the authors and project using `CITATION.cff`. Correspondence: Lei Chen, lchen@georgiasouthern.edu.
