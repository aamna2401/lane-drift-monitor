# Method

## Detection (per frame)
1. **Colour mask.** HSV: yellow (H 12-40, S>=80, V>=120) or white (S<=65, V>=150) ANDed with a
   top-hat of V (kernel 31 px): paint must also be brighter than its surroundings, which suppresses bright asphalt and glare.
2. **Canny** on the blurred grey image, kept only near the colour mask and inside the ROI.
3. **ROI.** Static trapezoid when nothing is tracked; otherwise a band around the *predicted* lane lines, from just above
   the estimated horizon (intersection of the two tracked lines).
4. **Probabilistic Hough**, then drop segments with |dy/dx| < 0.4. Negative slope = left lane, positive = right (image y points down).
5. **Fit** x = a*y + b per side (stable for near-vertical lines), length-weighted, with three rounds of MAD-based outlier removal.

## Tracking
Per side: accept a measurement only if its position at the evaluation row is within `gate_px` of the track; blend with an EMA
(alpha tuned). If both lines are accepted, their spacing must stay within 0.6-1.5x of the tracked lane width. If only one line
is seen (dashed line gap, or a line out of frame during a drift) the other is placed one tracked lane width away. With no
measurement the track coasts for up to 15 frames with decaying confidence, then resets.

## Offset in metres
At row 0.85H: offset = (W/2 - lane_centre_px) / lane_width_px * lane_width_m. Positive = vehicle right of centre. Monocular
and uncalibrated: the scale is the *assumed* lane width (3.7 m default), and the camera is assumed centred in the vehicle
(`camera_offset_m` corrects a known mount offset). Yaw adds a small bias because the row sits a few metres ahead.

## Events
State machine on (time, offset, confidence): enter at |offset| >= 0.95 m (a 1.8 m car in a 3.7 m lane has a wheel on the
line), exit below 0.75 m (hysteresis), keep if it lasted >= 0.4 s, merge same-side events closer than 1 s, ignore frames
with confidence < 0.3. Same rule defines the ground truth, so the metric measures perception, not policy.

## Tuning
Random search (60 trials per pipeline) over Canny, Hough, white/yellow V thresholds, top-hat level, tracker alpha/gate and
ROI margin, on validation clips. Objective = (1 - event F1) + (1 - coverage) + offset MAE (m). The baseline gets its own search
over the shared detector parameters so the comparison is not against untuned values.

## Metrics
Frame coverage; offset MAE, P95 and bias (m, frames with an estimate); event precision/recall/F1 with one-to-one time-overlap
matching (same side, tolerance 0.5 s); start delay; false alarms per hour. Intervals: 2000 bootstrap resamples of clips.
