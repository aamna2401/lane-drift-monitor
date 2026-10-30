"""Tables (with bootstrap CIs over clips) and figures from outputs/results/clip_scores.csv and the seed-0 traces."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lanedrift import Config                              # noqa: E402
from lanedrift.events import events_from_series           # noqa: E402

RES, FIG = ROOT / "outputs" / "results", ROOT / "outputs" / "figures"
ORDER = ["baseline", "+robust", "+tracking", "full"]
DIFFS = ["easy", "medium", "hard"]
rng = np.random.default_rng(0)


def boot(values, n=2000):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m = rng.choice(v, (n, len(v))).mean(1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


def ci_str(vals, fmt="{:.3f}"):
    m, lo, hi = boot(vals)
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]" if m == m else "n/a"


def f1_boot(d, n=2000):
    """Micro F1 resampling clips."""
    tp, fp, fn = d.tp.to_numpy(), d.fp.to_numpy(), d.fn.to_numpy()
    idx = rng.integers(0, len(d), (n, len(d)))
    T, P, N = tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1)
    f = 2 * T / np.maximum(2 * T + P + N, 1)
    pt = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return pt, *np.percentile(f, [2.5, 97.5])


def tables(df):
    lines = []
    lines.append("### Offset accuracy (metres, mean over clips, 95% bootstrap CI over clips)\n")
    lines.append("| difficulty | variant | frame coverage | MAE | P95 abs error |\n|---|---|---|---|---|")
    for d in DIFFS:
        for v in ORDER:
            s = df[(df.difficulty == d) & (df.variant == v)]
            lines.append(f"| {d} | {v} | {ci_str(s.coverage, '{:.2f}')} | {ci_str(s.mae_m)} | {ci_str(s.p95_m)} |")
    lines.append("\n### Departure events (all clips pooled; F1 CI by resampling clips)\n")
    lines.append("| difficulty | variant | true events | TP | FP | FN | precision | recall | F1 [95% CI] | mean start delay (s) | false alarms / hour |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for d in DIFFS:
        for v in ORDER:
            s = df[(df.difficulty == d) & (df.variant == v)]
            tp, fp, fn = int(s.tp.sum()), int(s.fp.sum()), int(s.fn.sum())
            f, lo, hi = f1_boot(s)
            hours = s.duration_s.sum() / 3600
            lines.append(f"| {d} | {v} | {tp + fn} | {tp} | {fp} | {fn} | {tp / max(tp + fp, 1):.2f} | {tp / max(tp + fn, 1):.2f} | "
                         f"{f:.2f} [{lo:.2f}, {hi:.2f}] | {s.mean_start_delay_s.mean():+.2f} | {fp / hours:.1f} |")
    return "\n".join(lines)


def fig_bars(df):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (col, title) in zip(ax, [("coverage", "Frame coverage"), ("mae_m", "Offset MAE (m)"), ("fn", "Missed departures per clip")]):
        for i, v in enumerate(ORDER):
            vals = [df[(df.difficulty == d) & (df.variant == v)][col].mean() for d in DIFFS]
            a.bar(np.arange(3) + i * 0.2 - 0.3, vals, 0.2, label=v)
        a.set_xticks(range(3), DIFFS)
        a.set_title(title)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_bars.png", dpi=150)
    plt.close(fig)


def fig_trace(diff="hard"):
    f = RES / f"trace_{diff}.csv"
    if not f.exists():
        return
    tr = pd.read_csv(f)
    ep = Config().events
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a, v in zip(ax, ["baseline", "full"]):
        d = tr[tr.variant == v]
        a.plot(d.t, d.truth_m, "k", lw=1.6, label="truth")
        a.plot(d.t, d.offset_m, ".", ms=2.5, color="tab:red" if v == "baseline" else "tab:blue", label=f"{v} estimate")
        for e in events_from_series(d.t, d.truth_m, np.ones(len(d)), ep):
            a.axvspan(e.start_s, e.end_s, color="orange", alpha=0.25)
        a.axhline(ep.enter_m, ls="--", c="gray", lw=0.8)
        a.axhline(-ep.enter_m, ls="--", c="gray", lw=0.8)
        a.set_ylabel("offset (m)")
        a.legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"{diff} example clip: lateral offset (orange = true departure, dashed = alert threshold)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(FIG / f"trace_{diff}.png", dpi=150)
    plt.close(fig)


def fig_width_sensitivity(df):
    s = df[(df.variant == "full") & df.bias_m.notna()]
    fig, a = plt.subplots(figsize=(5, 3.5))
    a.scatter(s.lane_width_m, s.bias_m, c=s.difficulty.map({"easy": "tab:green", "medium": "tab:orange", "hard": "tab:red"}), s=18)
    a.axvline(Config().geometry.lane_width_m, c="gray", ls="--")
    a.set_xlabel("true lane width (m); dashed = assumed 3.7 m")
    a.set_ylabel("mean signed offset error (m)")
    fig.tight_layout()
    fig.savefig(FIG / "lane_width_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "clip_scores.csv")
    md = tables(df)
    corr = df[df.variant == "full"][["lane_width_m", "bias_m"]].dropna().corr().iloc[0, 1]
    md += f"\n\nCorrelation of true lane width with signed offset error, full pipeline: {corr:.2f}\n"
    (RES / "summary.md").write_text(md)
    fig_bars(df)
    for d in DIFFS:
        fig_trace(d)
    fig_width_sensitivity(df)
    print(md)
