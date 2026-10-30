"""Synthetic dashcam clips with exact ground truth.

No labelled dashcam video with lateral-position ground truth is freely available, so the evaluation
uses a pinhole-camera road renderer: the true lateral offset, lane width and drift events are known
for every frame. The renderer is independent of the detector (it draws ground-plane polygons through
a camera model; the detector never sees the lane geometry).

Difficulty presets add nuisances that break lane detectors: shadows, worn paint, glare, low light,
neighbouring-lane markings, a lead vehicle (sometimes white), curvature, blur and noise.
Not simulated: rain on the lens, camera roll, hood reflections, cones, lane changes.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import EventParams
from .events import Event, events_from_series

W, H = 960, 540
FOCAL, CAM_H, CY = 600.0, 1.35, 270.0
HORIZON_Y = 0.42 * H
PITCH0 = float(np.arctan((CY - HORIZON_Y) / FOCAL))
Z_NEAR, Z_FAR = 1.5, 100.0

PRESETS = {
    "easy":   dict(bright=(0.9, 1.05), shadows=0, wear=(0.9, 1.0), noise=2, blur=0, curv=0.0, neighbour=False, lead=False, glare=0.0),
    "medium": dict(bright=(0.7, 1.0), shadows=2, wear=(0.7, 0.95), noise=5, blur=3, curv=1 / 900, neighbour=True, lead=True, glare=0.0),
    "hard":   dict(bright=(0.4, 0.75), shadows=4, wear=(0.4, 0.75), noise=9, blur=5, curv=1 / 450, neighbour=True, lead=True, glare=0.35),
}
PAINT_WHITE, PAINT_YELLOW = (235, 235, 235), (40, 200, 235)   # BGR


def project(X, Y, Z, cam_x, yaw, pitch):
    """World (x right, y up, z forward) -> pixel, for a camera at (cam_x, CAM_H, 0)."""
    x, y, z = X - cam_x, Y - CAM_H, Z
    xr = x * np.cos(yaw) - z * np.sin(yaw)
    zr = x * np.sin(yaw) + z * np.cos(yaw)
    zc = np.maximum(-y * np.sin(pitch) + zr * np.cos(pitch), 1e-3)
    yc = y * np.cos(pitch) + zr * np.sin(pitch)
    return np.stack([W / 2 + FOCAL * xr / zc, CY - FOCAL * yc / zc], -1)


def _poly(img, pts, colour, alpha=1.0):
    pts = np.round(np.asarray(pts) * 16).astype(np.int32)
    if alpha >= 0.999:
        cv2.fillPoly(img, [pts], colour, cv2.LINE_AA, shift=4)
        return
    layer = img.copy()
    cv2.fillPoly(layer, [pts], colour, cv2.LINE_AA, shift=4)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


@dataclass
class Clip:
    seed: int
    difficulty: str = "medium"
    duration_s: float = 20.0
    fps: float = 30.0
    n_departures: int | None = None      # None: random 0-3
    near_misses: int | None = None       # drifts that stay inside the line (must NOT raise an alarm)
    times: np.ndarray = field(init=False)
    offset_m: np.ndarray = field(init=False)
    lane_width_m: float = field(init=False)

    def __post_init__(self):
        self.p = PRESETS[self.difficulty]
        rng = np.random.default_rng(self.seed)
        n = int(self.duration_s * self.fps)
        self.times = np.arange(n) / self.fps
        self.lane_width_m = float(rng.uniform(3.3, 3.9))
        self.speed = float(rng.uniform(15, 28))
        phi = np.exp(-1 / (self.fps * 3.0))
        eps = rng.standard_normal(n)
        x = np.zeros(n)
        for k in range(1, n):
            x[k] = phi * x[k - 1] + 0.22 * np.sqrt(1 - phi ** 2) * eps[k]
        nd = int(rng.integers(0, 4)) if self.n_departures is None else self.n_departures
        nm = int(rng.integers(0, 3)) if self.near_misses is None else self.near_misses
        centres = self._spaced(rng, nd + nm, self.duration_s)
        kinds = ["dep"] * nd + ["near"] * nm
        rng.shuffle(kinds)
        for c, kind in zip(centres, kinds):
            amp = rng.uniform(1.15, 1.7) if kind == "dep" else rng.uniform(0.55, 0.8)
            x += rng.choice([-1, 1]) * amp * np.exp(-((self.times - c) ** 2) / (2 * rng.uniform(0.5, 1.0) ** 2))
        self.offset_m = x
        self.curv = float(rng.choice([-1, 1]) * self.p["curv"] * rng.uniform(0.3, 1.0))
        self.bright = float(rng.uniform(*self.p["bright"]))
        self.wear = rng.uniform(*self.p["wear"], size=3)
        self.shadow_z = rng.uniform(0, 60, self.p["shadows"])
        self.lead_z = float(rng.uniform(14, 30)) if self.p["lead"] else None
        self.lead_white = bool(rng.random() < 0.5)
        self.pitch_phase = float(rng.uniform(0, 2 * np.pi))
        self.sky = np.array([200, 170, 140], float) * self.bright + 20
        self.grass = np.array([60, 110, 80], float) * self.bright
        self.road = float(rng.uniform(85, 125)) * self.bright

    @staticmethod
    def _spaced(rng, n, dur, gap=4.5):
        lo = min(3.0, dur / 4)
        for _ in range(200):
            c = np.sort(rng.uniform(lo, dur - lo, n))
            if n < 2 or np.all(np.diff(c) > gap):
                return c
        return np.linspace(lo, dur - lo, n)

    def ground_truth_events(self, p: EventParams) -> list[Event]:
        return events_from_series(self.times, self.offset_m, np.ones_like(self.times), p)

    # -- rendering -----------------------------------------------------------------------
    def _strip(self, img, base, width, z0, z1, colour, alpha, cam):
        z = np.geomspace(max(z0, Z_NEAR), z1, 30)
        xc = base + 0.5 * self.curv * z ** 2
        left = project(xc - width / 2, 0 * z, z, *cam)
        right = project(xc + width / 2, 0 * z, z, *cam)
        _poly(img, np.concatenate([left, right[::-1]]), colour, alpha)

    def frame(self, k: int) -> np.ndarray:
        p, rng = self.p, np.random.default_rng((self.seed, k))
        t, cam_x = self.times[k], self.offset_m[k]
        dxdt = (self.offset_m[min(k + 1, len(self.times) - 1)] - self.offset_m[max(k - 1, 0)]) * self.fps / 2
        yaw = float(np.arctan2(dxdt, self.speed)) + np.deg2rad(0.15)
        pitch = PITCH0 + np.deg2rad(0.35) * np.sin(0.9 * t + self.pitch_phase)
        cam = (cam_x, yaw, pitch)
        s, lw = self.speed * t, self.lane_width_m

        img = np.empty((H, W, 3), np.float32)
        img[:] = self.grass
        hy = int(project(np.array([0.0]), np.array([0.0]), np.array([4000.0]), *cam)[0, 1])
        img[:max(hy, 0)] = self.sky
        road = tuple(float(v) for v in self.road * np.array([1.0, 1.0, 1.02]))
        shoulder = tuple(v * 1.12 for v in road)
        self._strip(img, 0.0, 3 * lw + 2, Z_NEAR, Z_FAR, shoulder, 1.0, cam)
        self._strip(img, 0.0, 2 * lw, Z_NEAR, Z_FAR, road, 1.0, cam)
        a_l, a_r, a_n = self.wear
        self._strip(img, -lw / 2, 0.15, Z_NEAR, Z_FAR, PAINT_YELLOW, a_l, cam)          # left: solid yellow
        j = int(np.floor((Z_NEAR + s) / 12.0)) - 1                                       # right: dashed white (3 m / 9 m)
        while j * 12.0 - s < Z_FAR:
            d0, d1 = j * 12.0 - s, j * 12.0 - s + 3.0
            if d1 > Z_NEAR:
                self._strip(img, lw / 2, 0.15, max(d0, Z_NEAR), min(d1, Z_FAR), PAINT_WHITE, a_r, cam)
            j += 1
        if p["neighbour"]:
            for side in (-1, 1):
                self._strip(img, side * lw * 1.5, 0.15, Z_NEAR, Z_FAR, PAINT_WHITE, a_n, cam)
        if self.lead_z is not None:
            zl = self.lead_z
            xl = 0.1 + 0.5 * self.curv * zl ** 2
            box = project(np.array([xl - 0.9, xl + 0.9, xl + 0.9, xl - 0.9]), np.array([0.3, 0.3, 1.5, 1.5]), np.full(4, zl), *cam)
            _poly(img, box, PAINT_WHITE if self.lead_white else (50, 40, 160))
        gain = np.ones((H, W), np.float32)                                               # shadows move with the road
        for z0 in self.shadow_z:
            zz = ((z0 - s) % 60) + 3
            quad = project(np.array([-8.0, 8.0, 8.0, -8.0]), np.zeros(4), np.array([zz, zz, zz + 10, zz + 10]), *cam)
            cv2.fillPoly(gain, [np.round(quad).astype(np.int32)], 0.55, cv2.LINE_AA)
        img *= gain[..., None]
        if p["glare"]:
            yy, xx = np.mgrid[0:H, 0:W]
            img += p["glare"] * 120 * np.exp(-(((xx - W / 2) / 260) ** 2 + ((yy - (hy + 60)) / 70) ** 2))[..., None]
        out = np.clip(img, 0, 255)
        if p["blur"]:
            out = cv2.GaussianBlur(out, (0, 0), p["blur"] / 3.0)
        if p["noise"]:
            out = out + rng.normal(0, p["noise"], out.shape)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], cv2.IMREAD_COLOR)

    def frames(self):
        for k in range(len(self.times)):
            yield self.frame(k)
