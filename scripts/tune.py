"""Random-search the detector parameters on VALIDATION clips only (seeds 1000+), for the baseline and the full pipeline.

The test clips (seeds 0-299) are never touched here. Writes configs/baseline.json and configs/final.json.
Usage: python scripts/tune.py [--trials 60]
"""
import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lanedrift import Config                                    # noqa: E402
from lanedrift.config import baseline_config                    # noqa: E402
from lanedrift.runner import evaluate_variants                  # noqa: E402
from lanedrift.simulate import Clip                             # noqa: E402

VAL_SEEDS = {"easy": [1000, 1001], "medium": [1010, 1011, 1012], "hard": [1020, 1021, 1022]}
FPS, DUR = 15.0, 10.0
_FRAMES = []


def _init():
    import cv2
    cv2.setNumThreads(1)


def _render(args):
    diff, seed = args
    c = Clip(seed, diff, DUR, FPS, n_departures=2, near_misses=1)
    return c, list(c.frames())


def sample(rng, base: Config, kind: str) -> Config:
    c = copy.deepcopy(base)
    d = c.detect
    lo = int(rng.integers(30, 110))
    d.canny = (lo, int(lo * rng.uniform(2, 3.5)))
    d.hough_threshold = int(rng.integers(15, 60) if kind == "final" else rng.integers(40, 140))
    d.hough_min_len = int(rng.integers(15, 50))
    d.hough_max_gap = int(rng.integers(15, 50))
    d.white_v_min = int(rng.integers(110, 200))
    d.yellow_v_min = int(rng.integers(100, 200)) if kind == "final" else int(rng.integers(120, 220))
    if kind == "final":
        d.tophat_min = int(rng.integers(12, 45))
        c.track.alpha = float(rng.uniform(0.2, 0.6))
        c.track.gate_px = float(rng.uniform(40, 110))
        c.roi.guided_margin_px = int(rng.integers(30, 80))
    return c


def objective(scores):
    cov = np.mean([s.coverage for s in scores])
    mae = np.nanmean([s.mae_m for s in scores])
    tp, fp, fn = (sum(getattr(s, k) for s in scores) for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return (1 - f1) + (1 - cov) + (0 if np.isnan(mae) else mae)       # lower is better


def _score(args):
    cfg_dict, idx = args
    cfg = Config.from_dict(cfg_dict)
    out = []
    for clip, frames in _FRAMES:
        out.append(evaluate_variants(clip, {"v": cfg}, frames)["v"][0])
    return objective(out)


def _set_frames(frames):
    global _FRAMES
    _FRAMES = frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    jobs = [(d, s) for d, seeds in VAL_SEEDS.items() for s in seeds]
    with Pool(8, initializer=_init) as pool:
        rendered = pool.map(_render, jobs)
    print(f"rendered {len(rendered)} validation clips")
    rng = np.random.default_rng(0)
    for kind, base in [("baseline", baseline_config()), ("final", Config())]:
        cands = [base] + [sample(rng, base, kind) for _ in range(args.trials)]
        _set_frames(rendered)                                   # forked workers inherit the frames
        with Pool(8, initializer=_init) as pool:
            scores = pool.map(_score, [(c.to_dict(), i) for i, c in enumerate(cands)])
        best = int(np.argmin(scores))
        print(f"{kind}: default objective {scores[0]:.3f} -> best {scores[best]:.3f} (trial {best})")
        (ROOT / "configs" / f"{kind}.json").write_text(json.dumps(cands[best].to_dict(), indent=1))
