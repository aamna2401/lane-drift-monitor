# Data

## Simulated clips (used for all metrics)
`lanedrift/simulate.py` renders dashcam video from a pinhole camera model with exact ground truth: lateral offset, lane
width, departure events. Three presets (easy, medium, hard) add shadows, worn paint, glare, low light, neighbouring-lane
lines, a lead vehicle (half the time white), curvature, blur, noise and JPEG compression.

| split | seeds | used for |
|---|---|---|
| validation | 1000-1022 (8 clips, 10 s, 15 fps) | `scripts/tune.py` only |
| test | 0-11, 100-111, 200-211 (36 clips, 20 s, 30 fps) | `scripts/run_eval.py` |

Per clip: random lane width 3.3-3.9 m (the detector assumes 3.7 m), speed 15-28 m/s, an Ornstein-Uhlenbeck wander
(sigma 0.22 m), 0-3 true departures (peak 1.15-1.7 m) and 0-2 near-misses (peak 0.55-0.8 m).
A true departure is defined by the same rule the detector uses (|offset| >= 0.95 m for 0.4 s, exit below 0.75 m).

Not simulated: rain or dirt on the lens, camera roll, hood reflections, cones, lane changes, other markings (arrows, crosswalks).

## Real photos (qualitative check only)
25 real road photos (`data/real_stills/test1.jpg` ... `test25.jpg`) come from the Udacity
self-driving-car lane-finding project. They carry no labels and their licence is not stated here, so they are not
redistributed: place them in `data/real_stills/` yourself to run `scripts/eval_real_stills.py`.

## What real validation would need
Dashcam video with a measured lateral position or labelled lane lines, for example TuSimple, CULane or BDD100K lane
annotations (check each licence). Lane-line labels would give a pixel-level check of the detector; lateral position in
metres needs vehicle-side ground truth (RTK GPS with a lane map, or manual annotation against a known lane width).
