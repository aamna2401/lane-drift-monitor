"""Test-set evaluation: ablation variants x difficulty on held-out simulated clips (seeds 0..N-1).

Variants (all share the tuned detector parameters of the full pipeline except `baseline`):
  baseline   course pipeline, own tuned parameters, per frame
  +robust    length-weighted fit with outlier rejection + adaptive (top-hat) colour mask
  +tracking  + temporal tracker (gating, EMA, coasting, single-line inference)
  full       + lane-guided ROI
Usage: python scripts/run_eval.py [--clips 12] [--duration 20] [--workers 8]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lanedrift import Config                              # noqa: E402
from lanedrift.runner import evaluate_variants            # noqa: E402
from lanedrift.simulate import Clip                       # noqa: E402

OUT = ROOT / "outputs" / "results"


def variants() -> dict[str, Config]:
    final = Config.load(ROOT / "configs" / "final.json")
    return {
        "baseline": Config.load(ROOT / "configs" / "baseline.json"),
        "+robust": final.with_(tracking=False, guided_roi=False),
        "+tracking": final.with_(guided_roi=False),
        "full": final,
    }


def work(job):
    import cv2
    cv2.setNumThreads(1)
    diff, seed, dur = job
    clip = Clip(seed, diff, dur)
    res = evaluate_variants(clip, variants())
    rows, traces = [], {}
    for name, (s, df, events) in res.items():
        rows.append(dict(difficulty=diff, seed=seed, variant=name, lane_width_m=clip.lane_width_m, n_true_events=len(clip.ground_truth_events(Config().events)),
                         n_frames=s.n_frames, coverage=s.coverage, mae_m=s.mae_m, p95_m=s.p95_m, bias_m=s.bias_m,
                         tp=s.tp, fp=s.fp, fn=s.fn, mean_start_delay_s=(sum(s.start_delay_s) / len(s.start_delay_s)) if s.start_delay_s else float("nan"),
                         duration_s=s.duration_s, events=json.dumps([e.to_dict() for e in events])))
        if seed % 100 == 0:
            traces[name] = df
    if seed % 100 == 0:
        pd.concat(traces, names=["variant"]).reset_index(level=0).to_csv(OUT / f"trace_{diff}.csv", index=False)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=12, help="clips per difficulty")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--traces-only", action="store_true", help="only re-render the seed-0/100/200 clips for the trace figures")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, i + k * 100, a.duration) for k, d in enumerate(["easy", "medium", "hard"]) for i in range(a.clips)]
    if a.traces_only:
        jobs = [j for j in jobs if j[1] % 100 == 0]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(work, jobs) for r in part]
    if a.traces_only:
        sys.exit(0)
    df = pd.DataFrame(rows).sort_values(["difficulty", "seed", "variant"])
    df.to_csv(OUT / "clip_scores.csv", index=False)
    print(df.groupby(["difficulty", "variant"])[["coverage", "mae_m", "p95_m", "tp", "fp", "fn"]].mean().round(3).to_string())
