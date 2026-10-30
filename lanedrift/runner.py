"""Run a monitor over a clip or a video and collect per-frame results."""
import numpy as np
import pandas as pd

from .config import Config
from .metrics import ClipScore, score_clip
from .pipeline import LaneMonitor
from .simulate import Clip


def run_frames(frames, cfg: Config, keep=None):
    """Yield-free helper: returns (DataFrame of per-frame output, events). `keep(i, result)` may capture frames."""
    mon = LaneMonitor(cfg)
    rows = []
    for i, f in enumerate(frames):
        r = mon.process(f)
        if keep is not None:
            keep(i, f, r)
        rows.append((r.t, np.nan if r.offset_m is None else r.offset_m, r.confidence, r.source,
                     np.nan if r.lane_width_px is None else r.lane_width_px))
    df = pd.DataFrame(rows, columns=["t", "offset_m", "confidence", "source", "lane_width_px"])
    return df, mon.finish()


def evaluate_clip(clip: Clip, cfg: Config) -> tuple[ClipScore, pd.DataFrame, list]:
    df, events = run_frames(clip.frames(), cfg.with_(fps=clip.fps))
    df["truth_m"] = clip.offset_m
    truth_events = clip.ground_truth_events(cfg.events)
    return score_clip(clip.offset_m, df.offset_m.to_numpy(), truth_events, events, clip.fps), df, events


def evaluate_variants(clip: Clip, cfgs: dict[str, Config], frames=None):
    """Render the clip once and run every named config over the same frames."""
    frames = frames if frames is not None else list(clip.frames())
    out = {}
    for name, cfg in cfgs.items():
        df, events = run_frames(frames, cfg.with_(fps=clip.fps))
        df["truth_m"] = clip.offset_m
        out[name] = (score_clip(clip.offset_m, df.offset_m.to_numpy(), clip.ground_truth_events(cfg.events), events, clip.fps), df, events)
    return out
