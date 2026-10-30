"""Run the monitor over a video file: per-frame CSV, events JSON, optional annotated video."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .pipeline import FrameResult, LaneMonitor


def annotate(frame: np.ndarray, r: FrameResult, alerting: bool) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    y0 = int(0.6 * h)
    for ln, col in ((r.state.left, (0, 220, 0)), (r.state.right, (0, 0, 255))):
        if ln is not None:
            cv2.line(out, (int(ln.x_at(h - 1)), h - 1), (int(ln.x_at(y0)), y0), col, 4)
    txt = "no lane" if r.offset_m is None else f"offset {r.offset_m:+.2f} m ({r.source})"
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 180) if alerting else (40, 40, 40), -1)
    cv2.putText(out, ("LANE DEPARTURE  " if alerting else "") + txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def analyse_video(path: str, out_dir: str, cfg: Config | None = None, write_video: bool = True, max_frames: int | None = None):
    cfg = cfg or Config()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    mon = LaneMonitor(cfg.with_(fps=fps))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer, rows, i = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        r = mon.process(frame)
        rows.append((r.index, r.t, r.offset_m, r.confidence, r.source))
        if write_video:
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            offs = [x[2] for x in rows[-1:] if x[2] is not None]
            writer.write(annotate(frame, r, bool(offs) and abs(offs[0]) >= cfg.events.enter_m))
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    events = mon.finish()
    pd.DataFrame(rows, columns=["frame", "t", "offset_m", "confidence", "source"]).to_csv(out_dir / "frames.csv", index=False)
    (out_dir / "events.json").write_text(json.dumps({"fps": fps, "frames": i, "events": [e.to_dict() for e in events]}, indent=1))
    return events
