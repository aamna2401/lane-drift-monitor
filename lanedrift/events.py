"""Departure events from the offset stream: hysteresis, minimum duration, merging."""
from dataclasses import dataclass

from .config import EventParams


@dataclass
class Event:
    start_s: float
    end_s: float
    peak_m: float
    side: str            # "left" | "right"

    @property
    def duration_s(self):
        return self.end_s - self.start_s

    def to_dict(self):
        return {"start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3), "duration_s": round(self.duration_s, 3),
                "peak_offset_m": round(self.peak_m, 3), "side": self.side}


class EventDetector:
    """Streaming state machine. Feed (t, offset, confidence) per frame; call finish() at the end."""

    def __init__(self, p: EventParams):
        self.p = p
        self.events: list[Event] = []
        self._start = None
        self._peak = 0.0
        self._last_t = 0.0

    def update(self, t: float, offset: float | None, confidence: float):
        p = self.p
        self._last_t = t
        if offset is None or confidence < p.min_confidence:
            return                                  # no evidence: hold the current state
        mag = abs(offset)
        if self._start is None:
            if mag >= p.enter_m:
                self._start, self._peak = t, offset
        else:
            if abs(offset) > abs(self._peak):
                self._peak = offset
            if mag < p.exit_m:
                self._close(t)

    def _close(self, t):
        if t - self._start >= self.p.min_duration_s:
            self.events.append(Event(self._start, t, self._peak, "right" if self._peak > 0 else "left"))
        self._start = None

    def finish(self) -> list[Event]:
        if self._start is not None:
            self._close(self._last_t)
        merged: list[Event] = []
        for e in self.events:
            if merged and e.start_s - merged[-1].end_s <= self.p.merge_gap_s and (e.peak_m > 0) == (merged[-1].peak_m > 0):
                m = merged[-1]
                merged[-1] = Event(m.start_s, e.end_s, e.peak_m if abs(e.peak_m) > abs(m.peak_m) else m.peak_m, m.side)
            else:
                merged.append(e)
        return merged


def events_from_series(times, offsets, confidences, p: EventParams) -> list[Event]:
    det = EventDetector(p)
    for t, o, c in zip(times, offsets, confidences):
        det.update(float(t), None if o != o else float(o), float(c))   # NaN-safe
    return det.finish()
