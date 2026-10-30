import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lanedrift import Config, LaneMonitor                         # noqa: E402
from lanedrift.config import EventParams, TrackParams             # noqa: E402
from lanedrift.detect import Line, _fit                           # noqa: E402
from lanedrift.events import events_from_series                   # noqa: E402
from lanedrift.geometry import lateral_offset_m                   # noqa: E402
from lanedrift.metrics import match_events                        # noqa: E402
from lanedrift.simulate import Clip                               # noqa: E402
from lanedrift.track import LaneTracker                           # noqa: E402

H, W = 540, 960


# ---- geometry -------------------------------------------------------------------------
def test_offset_zero_when_centred_and_signed():
    g = Config().geometry
    left, right = Line(-0.5, 400.0), Line(0.5, 80.0)               # symmetric about x = 480... check at eval row
    y = g.eval_row * H
    cx = 0.5 * (left.x_at(y) + right.x_at(y))
    shift = W / 2 - cx
    left2, right2 = Line(left.a, left.b + shift), Line(right.a, right.b + shift)
    assert abs(lateral_offset_m(left2, right2, W, H, g)[0]) < 1e-6
    # lane centre appears left of the image centre -> vehicle is right of centre -> positive
    left3, right3 = Line(left2.a, left2.b - 50), Line(right2.a, right2.b - 50)
    assert lateral_offset_m(left3, right3, W, H, g)[0] > 0


def test_offset_scale_equals_lane_width():
    g = Config().geometry
    y = g.eval_row * H
    left, right = Line(0, 300.0), Line(0, 700.0)                    # 400 px lane; centre 500 vs image centre 480
    off, w = lateral_offset_m(left, right, W, H, g)
    assert w == pytest.approx(400) and off == pytest.approx(-20 / 400 * g.lane_width_m)


# ---- line fitting ---------------------------------------------------------------------
def test_robust_fit_rejects_outlier_segment():
    good = np.array([[150, 400, 200, 300], [220, 260, 250, 200], [110, 480, 130, 440]], float)   # on x = -0.5*y + 350
    outlier = np.array([[400, 450, 430, 460]], float)
    plain = _fit(np.vstack([good, outlier]), robust=False)
    robust = _fit(np.vstack([good, outlier]), robust=True)
    assert abs(robust.a - (-0.5)) < abs(plain.a - (-0.5))


# ---- tracker --------------------------------------------------------------------------
def test_tracker_gates_jumps_coasts_then_drops():
    p = TrackParams(max_hold=3)
    t = LaneTracker(p, H, W, 460)
    l, r = Line(-0.6, 500.0), Line(0.6, 150.0)
    assert t.update(l, r).source == "both"
    assert t.update(Line(-0.6, 900.0), Line(0.6, 700.0)).source == "coast"     # wild jump gated out
    for _ in range(2):
        t.update(None, None)
    assert t.update(None, None).source == "none"


def test_tracker_infers_missing_line_from_lane_width():
    t = LaneTracker(TrackParams(), H, W, 460)
    l, r = Line(-0.6, 476.0), Line(0.6, 484.0)        # 200 px and 760 px at the eval row
    t.update(l, r)
    s = t.update(l, None)
    assert s.source == "left"
    assert s.right.x_at(460) - s.left.x_at(460) == pytest.approx(t.width_px, rel=1e-3)


# ---- events ---------------------------------------------------------------------------
def _series(profile, fps=30):
    t = np.arange(len(profile)) / fps
    return t, np.asarray(profile, float), np.ones(len(profile))


def test_event_requires_min_duration_and_hysteresis():
    p = EventParams()
    blip = [0] * 30 + [1.2] * 3 + [0] * 30                          # 0.1 s above threshold
    assert events_from_series(*_series(blip), p) == []
    real = [0] * 30 + [1.2] * 20 + [0.85] * 10 + [1.2] * 10 + [0] * 30   # dips to 0.85 (> exit 0.75): stays one event
    ev = events_from_series(*_series(real), p)
    assert len(ev) == 1 and ev[0].side == "right"
    neg = [0] * 30 + [-1.3] * 20 + [0] * 30
    assert events_from_series(*_series(neg), p)[0].side == "left"


def test_event_ignores_low_confidence_frames():
    t, o, _ = _series([0] * 20 + [1.3] * 20 + [0] * 20)
    assert events_from_series(t, o, np.zeros(len(o)), EventParams()) == []


def test_matching_is_one_to_one():
    from lanedrift.events import Event
    truth = [Event(1, 3, 1.2, "right")]
    pred = [Event(1.1, 2.5, 1.1, "right"), Event(2.6, 3.0, 1.0, "right")]
    m, fp, fn = match_events(truth, pred)
    assert len(m) == 1 and len(fp) == 1 and fn == []


# ---- simulator + end to end -----------------------------------------------------------
def test_simulator_is_deterministic_and_has_events():
    a, b = Clip(5, "medium", 8, n_departures=1, near_misses=0), Clip(5, "medium", 8, n_departures=1, near_misses=0)
    assert np.array_equal(a.frame(10), b.frame(10))
    assert len(a.ground_truth_events(EventParams())) == 1


def test_pipeline_tracks_centred_clean_clip():
    clip = Clip(2, "easy", 3, n_departures=0, near_misses=0)
    mon = LaneMonitor(Config())
    res = [mon.process(f) for f in list(clip.frames())[::2]]
    est = np.array([np.nan if r.offset_m is None else r.offset_m for r in res])
    assert np.mean(~np.isnan(est)) > 0.9
    assert np.nanmean(np.abs(est - clip.offset_m[::2][:len(est)])) < 0.2
