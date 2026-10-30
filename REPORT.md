# Lane-drift monitor: report

Aamna Khan · 2026-10-29 · code, configs and raw per-clip scores in this repository

## 1. Problem and use case
A fleet-safety or driver-coaching product needs to know *when and how far a vehicle drifted out of its lane*, from an ordinary
forward dashcam, cheaply enough to run on every trip. A classical single-image detector (colour mask, Canny, ROI, Hough) solves the first
half of that. A real system also needs continuity over video, a lateral distance in metres, and an alert rule whose error
rate is known. This project adds those, and measures them.

Success criteria set before evaluation: (1) a lateral-offset estimate on nearly every frame, (2) offset error well under 0.2 m,
(3) departure events found with high recall and few false alarms, (4) real-time on a CPU.

## 2. Design relative to a plain per-frame detector
| Per-frame baseline | This project |
|---|---|
| single images, per-pixel loops | vectorised package, tested functions, ~10 ms/frame |
| least-squares line fit | outlier-robust fit |
| fixed triangular ROI | ROI follows the tracked lanes and the estimated horizon |
| two lines required in every frame | tracker infers a missing line from the lane width, coasts through gaps |
| pixels | metres, via an assumed lane width |
| no alert logic | departure events with hysteresis, duration and merging rules |
| visual checks | simulator with exact ground truth, held-out test seeds, bootstrap intervals |

## 3. Evaluation design
- **Data.** Simulated dashcam clips (`docs/DATASET.md`): exact lateral offset for every frame; three difficulty levels.
  Validation seeds (8 clips) were used for tuning; 36 test clips (12 per level, 20 s, 30 fps) were used once for the tables below.
- **Variants (ablation).** `baseline` = a standard per-frame pipeline (HSV mask, Canny, triangular ROI, Hough, regression; own tuned parameters); `+robust` = robust fit +
  adaptive mask; `+tracking` = + temporal tracker; `full` = + lane-guided ROI.
- **Metrics.** Frame coverage; offset MAE/P95/bias; event precision/recall/F1 and false alarms per hour. Intervals resample clips.
- **Ground truth for events** applies the detector's own rule to the true offsets, so the metric measures perception only.

## 4. Results
Offset (mean over clips; 95% bootstrap CI over clips):

| difficulty | variant | coverage | MAE (m) | P95 (m) |
|---|---|---|---|---|
| easy | baseline | 0.25 [0.19, 0.32] | 0.081 [0.074, 0.088] | 0.212 |
| easy | +robust | 0.28 [0.24, 0.33] | 0.077 [0.069, 0.086] | 0.200 |
| easy | +tracking | 0.99 [0.96, 1.00] | 0.055 [0.045, 0.065] | 0.152 |
| easy | full | 0.99 [0.98, 1.00] | 0.051 [0.041, 0.062] | 0.130 |
| medium | baseline | 0.22 [0.17, 0.28] | 0.083 [0.073, 0.095] | 0.205 |
| medium | +robust | 0.25 [0.21, 0.30] | 0.081 [0.073, 0.089] | 0.202 |
| medium | +tracking | 0.95 [0.93, 0.98] | 0.061 [0.053, 0.069] | 0.156 |
| medium | full | 1.00 [1.00, 1.00] | 0.053 [0.046, 0.061] | 0.136 |
| hard | baseline | 0.09 [0.03, 0.15] | 0.104 [0.078, 0.140] | 0.187 |
| hard | +robust | 0.07 [0.02, 0.15] | 0.117 [0.085, 0.158] | 0.209 |
| hard | +tracking | 0.49 [0.25, 0.72] | 0.072 [0.058, 0.088] | 0.197 |
| hard | full | 0.55 [0.32, 0.79] | 0.065 [0.054, 0.077] | 0.173 |

Departure events (pooled over clips):

| difficulty | variant | true | TP | FP | FN | recall | F1 [95% CI] | false alarms/h |
|---|---|---|---|---|---|---|---|---|
| easy | baseline / +robust | 17 | 0 | 0 | 17 | 0.00 | 0.00 | 0 |
| easy | +tracking | 17 | 14 | 0 | 3 | 0.82 | 0.90 [0.67, 1.00] | 0 |
| easy | full | 17 | 16 | 0 | 1 | 0.94 | 0.97 [0.91, 1.00] | 0 |
| medium | baseline / +robust | 24 | 0 | 0 | 24 | 0.00 | 0.00 | 0 |
| medium | +tracking | 24 | 18 | 1 | 6 | 0.75 | 0.84 [0.73, 0.92] | 15 |
| medium | full | 24 | 24 | 1 | 0 | 1.00 | 0.98 [0.93, 1.00] | 15 |
| hard | baseline / +robust | 22 | 0 | 0 | 22 | 0.00 | 0.00 | 0 |
| hard | +tracking | 22 | 8 | 0 | 14 | 0.36 | 0.53 [0.13, 0.79] | 0 |
| hard | full | 22 | 12 | 0 | 10 | 0.55 | 0.71 [0.44, 0.88] | 0 |

Mean start delay of matched events is within about 0.1 s of the truth (the detector is slightly early, because the true
event is defined on the same threshold and the estimate is noisy around it). The single medium false alarm is 1 event in
4 minutes of video, so "15 per hour" has a very wide interval and should not be quoted as a rate.

Figures: `outputs/figures/ablation_bars.png`, `trace_{easy,medium,hard}.png` (truth vs estimate with departures shaded),
`lane_width_sensitivity.png`. A 14 s annotated demo is in `outputs/demo/demo_medium.mp4`.

## 5. What the ablation says
1. **Tracking is where the gain comes from.** Coverage goes from about 0.25 to about 0.99 on easy and medium clips. The reason is
   structural: the right line is dashed (3 m paint, 9 m gap) and, during a drift, the left line leaves the image. A per-frame
   method that needs both lines therefore has almost no output exactly when a departure happens. Inferring the missing line
   from the tracked lane width and coasting through gaps fixes this.
2. **The robust fit and adaptive mask alone do little** (coverage 0.25 → 0.28, MAE 0.081 → 0.077, within the intervals). They
   matter only once the tracker is on, where bad measurements would otherwise be blended into the track.
3. **The baseline's zero event recall is partly an artefact of its design** (no continuity), not proof that Hough-based detection is
   poor. A fairer "per-frame with single-line fallback" baseline is not included. Read the baseline column as "a plain per-frame pipeline".
4. **Guided ROI helps modestly but consistently** (event F1 0.84 → 0.98 on medium, 0.53 → 0.71 on hard; MAE 0.061 → 0.053 on medium).
   The hard-clip intervals overlap, so that gain is suggestive, not established.

## 6. Limitations (read these)
1. **No real labelled video.** All metrics are on simulated clips drawn by a renderer I wrote. Results show the method works when
   the world looks like the simulator; they do not estimate real-world accuracy.
2. **The tuned detector did not transfer to real photos.** On 25 real road photos (no labels, so this only counts "both lines found",
   not correctness) the baseline found both lines in 76% and the sim-tuned robust detector in 60% (`outputs/results/real_stills.csv`).
   The tuned thresholds (for example V >= 165 for white, top-hat >= level) are partly specific to simulator brightness. Visual inspection
   of the contact sheet also shows some baseline detections on the wrong line, so 76% is not 76% correct.
3. **Low light breaks it.** On hard clips 3 of 12 produced almost no estimates (coverage 0.00, 0.00, 0.03): the dimmest, most worn
   scenes fall below the absolute brightness thresholds, so the tracker never initialises. Per-clip coverage is bimodal, which is why the
   hard-level intervals are wide (0.32-0.79).
4. **Scale comes from an assumption.** Offset error from a wrong lane width is offset x (3.7 / true width - 1): at 1.0 m off-centre
   that is +12 cm for a 3.3 m lane and -5 cm for a 3.9 m lane. In the simulation the correlation between true width and the mean signed
   error is only -0.09 because most frames are near the centre. Real roads, camera roll and an off-centre camera add more.
5. **Event rule not validated against real incidents.** The 0.95 m threshold encodes a 1.8 m car in a 3.7 m lane. Lane changes, turn
   signals, intersections and low speed would all raise false alarms in practice; none are modelled.
6. **Small samples.** 22-24 true events per difficulty level and 36 clips in total, tuned on only 8 validation clips.

## 7. Next steps, in order of value
1. Validate on real labelled data (TuSimple / CULane / BDD100K lane labels for detection; a vehicle-side lateral reference for metres).
2. Make the thresholds scene-adaptive (per-frame V percentile or CLAHE) to remove the low-light failure and the sim-to-real gap; retune
   on a mix of simulated and real frames.
3. Add a fairer baseline (per-frame with single-line inference) so the tracking contribution is isolated from the dashed-line effect.
4. Gate alerts on speed and turn-signal state; calibrate lane width per road class.
5. Optionally replace the line detector with a small learned lane model and keep the tracker, event logic and metrics unchanged.

## 8. Reproducing
`make all` (tests, tuning on validation seeds, evaluation on test seeds, report), or `notebooks/run_on_colab.ipynb` on a free CPU
runtime. Tuned configs: `configs/baseline.json`, `configs/final.json`. Per-clip scores: `outputs/results/clip_scores.csv`.
