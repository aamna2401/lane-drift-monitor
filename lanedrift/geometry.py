"""Pixels to metres: lateral offset of the vehicle from the lane centre.

Monocular and uncalibrated, so the scale comes from the assumed lane width at the evaluation row.
Offset > 0 means the vehicle is right of the lane centre.
"""
from .config import GeometryParams
from .detect import Line


def lateral_offset_m(left: Line, right: Line, width: int, height: int, g: GeometryParams) -> tuple[float, float]:
    """Returns (offset_m, lane_width_px) at the evaluation row."""
    y = g.eval_row * height
    xl, xr = left.x_at(y), right.x_at(y)
    w_px = xr - xl
    centre = 0.5 * (xl + xr)
    offset = (width / 2.0 - centre) / w_px * g.lane_width_m + g.camera_offset_m
    return float(offset), float(w_px)
