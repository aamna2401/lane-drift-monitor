"""LaneMonitor: frame in, lane state + lateral offset + departure events out."""
from dataclasses import dataclass

import numpy as np

from .config import Config
from .detect import FrameDetection, detect_lines, guided_roi, static_roi
from .events import Event, EventDetector
from .geometry import lateral_offset_m
from .track import LaneTracker, TrackState


@dataclass
class FrameResult:
    index: int
    t: float
    offset_m: float | None
    lane_width_px: float | None
    confidence: float
    source: str
    state: TrackState
    detection: FrameDetection


class LaneMonitor:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._tracker: LaneTracker | None = None
        self._events = EventDetector(self.cfg.events)
        self._i = 0

    def _roi_for(self, shape, state: TrackState | None):
        c, (h, w) = self.cfg, shape[:2]
        if c.guided_roi and state is not None and state.left is not None and state.right is not None:
            xl, xr = state.left, state.right
            denom = (xl.a - xr.a)
            y_h = (xr.b - xl.b) / denom if abs(denom) > 1e-6 else c.roi.top_y * h   # lines meet at the horizon
            y_top = float(np.clip(y_h + c.roi.horizon_margin * h, 0.3 * h, 0.85 * h))
            return guided_roi(shape, xl, xr, y_top, c.roi.guided_margin_px)
        r = c.roi
        return static_roi(shape, r.top_y, r.top_half_w, r.bottom_half_w)

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        c, (h, w) = self.cfg, frame_bgr.shape[:2]
        if self._tracker is None:
            self._tracker = LaneTracker(c.track, h, w, c.geometry.eval_row * h)
        prev = TrackState(self._tracker.left, self._tracker.right, 0, self._tracker.width_px, "") if c.tracking else None
        det = detect_lines(frame_bgr, c, self._roi_for(frame_bgr.shape, prev))
        if c.tracking:
            state = self._tracker.update(det.left, det.right)
        else:                                            # per-frame: both lines or nothing
            ok = det.left is not None and det.right is not None
            state = TrackState(det.left, det.right, 1.0 if ok else 0.0, None, "both" if ok else "none")
        offset = width_px = None
        if state.left is not None and state.right is not None:
            offset, width_px = lateral_offset_m(state.left, state.right, w, h, c.geometry)
            if width_px < 0.15 * w:                      # lines crossed or collapsed: not a lane
                offset, width_px, state = None, None, TrackState(None, None, 0.0, None, "none")
        t = self._i / c.fps
        self._events.update(t, offset, state.confidence)
        res = FrameResult(self._i, t, offset, width_px, state.confidence, state.source, state, det)
        self._i += 1
        return res

    def finish(self) -> list[Event]:
        return self._events.finish()
