"""Scoring: offset error, event precision/recall, false alarms per hour."""
from dataclasses import dataclass

import numpy as np

from .events import Event


def match_events(truth: list[Event], pred: list[Event], min_iou: float = 0.1, tol_s: float = 0.5):
    """Greedy one-to-one matching of predicted to true events by time overlap (with a small tolerance).

    Returns (matches, unmatched_pred, unmatched_truth) where matches are (truth, pred) pairs.
    """
    used, matches = set(), []
    for t in truth:
        best, best_iou = None, 0.0
        for j, q in enumerate(pred):
            if j in used or (q.peak_m > 0) != (t.peak_m > 0):
                continue
            inter = min(t.end_s, q.end_s) - max(t.start_s, q.start_s) + tol_s
            union = max(t.end_s, q.end_s) - min(t.start_s, q.start_s) + tol_s
            iou = max(inter, 0) / union
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= min_iou:
            used.add(best)
            matches.append((t, pred[best]))
    unmatched_p = [q for j, q in enumerate(pred) if j not in used]
    matched_t = {id(t) for t, _ in matches}
    return matches, unmatched_p, [t for t in truth if id(t) not in matched_t]


@dataclass
class ClipScore:
    n_frames: int
    coverage: float            # fraction of frames with an offset estimate
    mae_m: float
    p95_m: float
    bias_m: float
    tp: int
    fp: int
    fn: int
    start_delay_s: list
    duration_s: float


def score_clip(truth_off, est_off, truth_events, pred_events, fps) -> ClipScore:
    est = np.asarray(est_off, float)
    ok = ~np.isnan(est)
    err = est[ok] - np.asarray(truth_off)[ok]
    matches, fp, fn = match_events(truth_events, pred_events)
    return ClipScore(
        n_frames=len(est), coverage=float(ok.mean()),
        mae_m=float(np.abs(err).mean()) if ok.any() else float("nan"),
        p95_m=float(np.percentile(np.abs(err), 95)) if ok.any() else float("nan"),
        bias_m=float(err.mean()) if ok.any() else float("nan"),
        tp=len(matches), fp=len(fp), fn=len(fn),
        start_delay_s=[q.start_s - t.start_s for t, q in matches], duration_s=len(est) / fps)
