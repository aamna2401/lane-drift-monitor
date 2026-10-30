"""All tunable parameters in one place. Defaults are the tuned values from scripts/tune.py."""
from dataclasses import asdict, dataclass, field, replace


@dataclass
class DetectParams:
    # colour mask (OpenCV HSV: H 0-179, S/V 0-255). Yellow and white paint.
    yellow_h: tuple = (12, 40)
    yellow_s_min: int = 80
    yellow_v_min: int = 120
    white_s_max: int = 65
    white_v_min: int = 150
    # local-contrast term: a pixel must also be brighter than its surroundings (top-hat), so
    # bright asphalt, glare and washed-out frames do not pass the absolute white threshold.
    tophat_kernel: int = 31
    tophat_min: int = 25
    blur_ksize: int = 5
    canny: tuple = (50, 150)
    mask_dilate: int = 5
    hough_rho: float = 2.0
    hough_threshold: int = 25
    hough_min_len: int = 25
    hough_max_gap: int = 30
    min_abs_slope: float = 0.4      # |dy/dx| in image coordinates; rejects near-horizontal segments


@dataclass
class RoiParams:
    # static trapezoid as fractions of (width, height): top y, top half-width, bottom half-width
    top_y: float = 0.60
    top_half_w: float = 0.08
    bottom_half_w: float = 0.55
    guided_margin_px: int = 45      # half-width of the band around predicted lane lines
    horizon_margin: float = 0.03    # extra height above the estimated horizon, fraction of H


@dataclass
class TrackParams:
    alpha: float = 0.35             # EMA weight of a new measurement
    gate_px: float = 70.0           # max jump of a line's bottom position between frames
    max_hold: int = 15              # frames to coast on the last estimate with no measurement
    width_tol: tuple = (0.6, 1.5)   # accepted lane width relative to the tracked width
    width_alpha: float = 0.05


@dataclass
class GeometryParams:
    lane_width_m: float = 3.7       # assumed; real lanes vary (see docs/METHOD.md, sensitivity)
    camera_offset_m: float = 0.0    # camera lateral position relative to vehicle centre, + = right
    eval_row: float = 0.95          # row (fraction of H) where the offset is measured


@dataclass
class EventParams:
    enter_m: float = 0.95           # centre-offset where a 1.8 m wide car's wheel reaches the line
    exit_m: float = 0.75            # hysteresis
    min_duration_s: float = 0.4
    merge_gap_s: float = 1.0
    min_confidence: float = 0.3


@dataclass
class Config:
    detect: DetectParams = field(default_factory=DetectParams)
    roi: RoiParams = field(default_factory=RoiParams)
    track: TrackParams = field(default_factory=TrackParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    events: EventParams = field(default_factory=EventParams)
    fps: float = 30.0
    # ablation switches
    robust_fit: bool = True         # length-weighted fit with outlier rejection
    tracking: bool = True
    guided_roi: bool = True
    adaptive_mask: bool = True      # top-hat term in the colour mask

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        def tup(x):
            return tuple(x) if isinstance(x, list) else x
        sub = {"detect": DetectParams, "roi": RoiParams, "track": TrackParams, "geometry": GeometryParams, "events": EventParams}
        kw = {k: v for k, v in d.items() if k not in sub}
        for k, klass in sub.items():
            if k in d:
                kw[k] = klass(**{a: tup(b) for a, b in d[k].items()})
        return cls(**kw)

    @classmethod
    def load(cls, path) -> "Config":
        import json
        return cls.from_dict(json.loads(open(path).read()))

    def with_(self, **kw):
        return replace(self, **kw)


def baseline_config() -> Config:
    """Plain per-frame pipeline: HSV mask + Canny + triangle ROI + Hough + regression, per frame."""
    c = Config(robust_fit=False, tracking=False, guided_roi=False, adaptive_mask=False)
    c.detect.yellow_h, c.detect.yellow_s_min, c.detect.yellow_v_min = (10, 100), 0, 190   # H in (10,100), V>=190
    c.detect.white_s_max, c.detect.white_v_min = 65, 140
    c.detect.canny, c.detect.hough_threshold = (100, 300), 100
    c.detect.hough_min_len, c.detect.hough_max_gap, c.detect.blur_ksize = 40, 40, 5
    c.roi.top_y, c.roi.top_half_w, c.roi.bottom_half_w = 0.5, 0.0, 0.5   # triangle (0,H),(W/2,H/2),(W,H)
    return c
