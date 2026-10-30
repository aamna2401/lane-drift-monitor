"""Per-frame lane line detection: colour mask + Canny + ROI + Hough + slope filter + line fit.

Steps 1-8 of the course brief, vectorised (the original used per-pixel Python loops) and with the
parts the assignment left open (line fitting, automatic ROI) completed.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config, DetectParams


@dataclass
class Line:
    """x = a*y + b in image coordinates (stable for near-vertical lane lines)."""
    a: float
    b: float
    support: float = 0.0     # total length in px of the segments behind the fit
    n_seg: int = 0

    def x_at(self, y):
        return self.a * y + self.b


@dataclass
class FrameDetection:
    left: Line | None
    right: Line | None
    mask: np.ndarray
    edges: np.ndarray
    roi: np.ndarray             # uint8 polygon mask
    segments: np.ndarray        # (N, 4) x1,y1,x2,y2 after slope filtering


def colour_mask(bgr: np.ndarray, p: DetectParams, adaptive: bool) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (p.blur_ksize, p.blur_ksize), 0), cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (p.yellow_h[0], p.yellow_s_min, p.yellow_v_min), (p.yellow_h[1], 255, 255))
    whi = cv2.inRange(hsv, (0, 0, p.white_v_min), (179, p.white_s_max, 255))
    mask = cv2.bitwise_or(yel, whi)
    if adaptive:
        v = hsv[..., 2]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.tophat_kernel, p.tophat_kernel))
        local = cv2.threshold(cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k), p.tophat_min, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(mask, local)
    return mask


def static_roi(shape, top_y, top_half_w, bottom_half_w) -> np.ndarray:
    h, w = shape[:2]
    cx = w / 2
    poly = np.array([[cx - bottom_half_w * w, h], [cx - top_half_w * w, top_y * h],
                     [cx + top_half_w * w, top_y * h], [cx + bottom_half_w * w, h]], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def guided_roi(shape, left: Line, right: Line, y_top: float, margin: float) -> np.ndarray:
    """Band around the predicted lane lines: keeps the true lines, drops neighbouring-lane and kerb clutter."""
    h, w = shape[:2]
    y0, y1 = int(max(0, y_top)), h - 1
    m = np.zeros((h, w), np.uint8)
    for ln in (left, right):
        # margin shrinks towards the horizon where the lane is narrower
        poly = np.array([[ln.x_at(y0) - margin * 0.4, y0], [ln.x_at(y0) + margin * 0.4, y0],
                         [ln.x_at(y1) + margin, y1], [ln.x_at(y1) - margin, y1]], np.int32)
        cv2.fillPoly(m, [poly], 255)
    return m


def _fit(segs: np.ndarray, robust: bool) -> Line | None:
    if len(segs) == 0:
        return None
    x1, y1, x2, y2 = segs.T
    length = np.hypot(x2 - x1, y2 - y1)
    ys = np.concatenate([y1, y2])
    xs = np.concatenate([x1, x2])
    wt = np.concatenate([length, length]) if robust else np.ones_like(ys)
    keep = np.ones(len(ys), bool)
    for _ in range(3 if robust else 1):
        if keep.sum() < 2 or np.ptp(ys[keep]) < 1:
            return None
        a, b = np.polyfit(ys[keep], xs[keep], 1, w=wt[keep])
        if not robust:
            break
        res = np.abs(xs - (a * ys + b))
        mad = np.median(res[keep]) + 1e-6
        new = res < max(4.0, 3.0 * 1.4826 * mad)
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    return Line(float(a), float(b), float(length.sum()), int(len(segs)))


def detect_lines(bgr: np.ndarray, cfg: Config, roi: np.ndarray | None = None) -> FrameDetection:
    p = cfg.detect
    h, w = bgr.shape[:2]
    if roi is None:
        r = cfg.roi
        roi = static_roi(bgr.shape, r.top_y, r.top_half_w, r.bottom_half_w)
    mask = colour_mask(bgr, p, cfg.adaptive_mask)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (p.blur_ksize, p.blur_ksize), 0)
    edges = cv2.Canny(gray, *p.canny)
    near_mask = cv2.dilate(mask, np.ones((p.mask_dilate, p.mask_dilate), np.uint8))
    edges = cv2.bitwise_and(cv2.bitwise_and(edges, near_mask), roi)   # step 4: drop edges that are not lane coloured
    lines = cv2.HoughLinesP(edges, p.hough_rho, np.pi / 180, p.hough_threshold,
                            minLineLength=p.hough_min_len, maxLineGap=p.hough_max_gap)
    segs = np.zeros((0, 4), np.float64) if lines is None else lines.reshape(-1, 4).astype(np.float64)
    if len(segs):
        dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(dx != 0, dy / dx, np.inf * np.sign(dy + 1e-9))
        segs = segs[np.abs(slope) >= p.min_abs_slope]
        slope = slope[np.abs(slope) >= p.min_abs_slope]
    else:
        slope = np.zeros(0)
    # image y grows downwards: left lane has negative slope, right lane positive (brief, step 7)
    left, right = _fit(segs[slope < 0], cfg.robust_fit), _fit(segs[slope > 0], cfg.robust_fit)
    return FrameDetection(left, right, mask, edges, roi, segs)
