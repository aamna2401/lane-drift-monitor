"""Temporal tracking of the two lane lines: gating, smoothing, coasting and single-line inference."""
from dataclasses import dataclass

from .config import TrackParams
from .detect import Line


@dataclass
class TrackState:
    left: Line | None
    right: Line | None
    confidence: float
    width_px: float | None
    source: str          # "both" | "left" | "right" | "coast" | "none"


class LaneTracker:
    def __init__(self, p: TrackParams, height: int, width: int, y_eval: float):
        self.p, self.H, self.W, self.y = p, height, width, y_eval
        self.left: Line | None = None
        self.right: Line | None = None
        self.width_px: float | None = None
        self.misses = 0

    def _blend(self, old: Line | None, new: Line) -> Line:
        if old is None:
            return Line(new.a, new.b, new.support, new.n_seg)
        k = self.p.alpha
        return Line((1 - k) * old.a + k * new.a, (1 - k) * old.b + k * new.b, new.support, new.n_seg)

    def _gate(self, old: Line | None, new: Line | None) -> Line | None:
        if new is None:
            return None
        if old is not None and abs(new.x_at(self.y) - old.x_at(self.y)) > self.p.gate_px:
            return None
        return new

    def _shifted(self, ref: Line, other_old: Line | None, sign: int) -> Line:
        """Copy of the other line's last shape, translated so it sits one lane width from `ref` at the eval row."""
        a = other_old.a if other_old is not None else -ref.a
        target = ref.x_at(self.y) + sign * self.width_px
        return Line(a, target - a * self.y)

    def update(self, meas_l: Line | None, meas_r: Line | None) -> TrackState:
        p = self.p
        if self.left is None or self.right is None:          # not initialised: need two sane lines
            if meas_l is not None and meas_r is not None:
                w = meas_r.x_at(self.y) - meas_l.x_at(self.y)
                if 0.2 * self.W < w < 0.95 * self.W:
                    self.left, self.right, self.width_px, self.misses = self._blend(None, meas_l), self._blend(None, meas_r), w, 0
                    return TrackState(self.left, self.right, 1.0, w, "both")
            return TrackState(None, None, 0.0, None, "none")

        l, r = self._gate(self.left, meas_l), self._gate(self.right, meas_r)
        if l is not None and r is not None:                   # both lines: check the lane width is plausible
            w = r.x_at(self.y) - l.x_at(self.y)
            lo, hi = p.width_tol
            if not (lo * self.width_px <= w <= hi * self.width_px):
                dl = abs(l.x_at(self.y) - self.left.x_at(self.y))
                dr = abs(r.x_at(self.y) - self.right.x_at(self.y))
                l, r = (l, None) if dl <= dr else (None, r)

        if l is not None and r is not None:
            self.left, self.right = self._blend(self.left, l), self._blend(self.right, r)
            self.width_px = (1 - p.width_alpha) * self.width_px + p.width_alpha * (self.right.x_at(self.y) - self.left.x_at(self.y))
            self.misses, src, conf = 0, "both", 1.0
        elif l is not None:
            self.left = self._blend(self.left, l)
            self.right = self._shifted(self.left, self.right, +1)
            self.misses, src, conf = 0, "left", 0.6
        elif r is not None:
            self.right = self._blend(self.right, r)
            self.left = self._shifted(self.right, self.left, -1)
            self.misses, src, conf = 0, "right", 0.6
        else:
            self.misses += 1
            if self.misses > p.max_hold:
                self.left = self.right = None
                return TrackState(None, None, 0.0, None, "none")
            src, conf = "coast", 0.5 * (1 - self.misses / (p.max_hold + 1))
        return TrackState(self.left, self.right, conf, self.width_px, src)
