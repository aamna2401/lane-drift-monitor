"""python -m lanedrift analyse <video> --out outputs/run1 [--config configs/final.json]"""
import argparse

from .config import Config
from .video import analyse_video


def main():
    ap = argparse.ArgumentParser(prog="lanedrift")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse", help="analyse a dashcam video")
    a.add_argument("video")
    a.add_argument("--out", default="outputs/run")
    a.add_argument("--config", default=None, help="JSON config (e.g. configs/final.json)")
    a.add_argument("--lane-width", type=float, default=None, help="assumed lane width in metres")
    a.add_argument("--no-video", action="store_true")
    a.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    if args.lane_width:
        cfg.geometry.lane_width_m = args.lane_width
    events = analyse_video(args.video, args.out, cfg, not args.no_video, args.max_frames)
    print(f"{len(events)} departure event(s)")
    for e in events:
        print(f"  {e.side:5s} {e.start_s:6.2f}s - {e.end_s:6.2f}s  peak {e.peak_m:+.2f} m")


if __name__ == "__main__":
    main()
