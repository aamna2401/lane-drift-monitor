"""Sanity check on real photos (the 25 course test images): does the detector find both lane lines?

No ground truth, so this is a qualitative check, not a metric of accuracy. Images are not redistributed
(see docs/DATASET.md); put them in data/real_stills/. Writes outputs/results/real_stills.csv and
outputs/figures/real_stills_sheet.jpg (git-ignored).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lanedrift import Config                                     # noqa: E402
from lanedrift.detect import detect_lines                        # noqa: E402

CFGS = {"baseline": Config.load(ROOT / "configs" / "baseline.json"),
        "robust": Config.load(ROOT / "configs" / "final.json").with_(tracking=False, guided_roi=False)}


def draw(img, det, y_top):
    out = img.copy()
    for ln, col in ((det.left, (0, 255, 0)), (det.right, (0, 0, 255))):
        if ln is not None:
            cv2.line(out, (int(ln.x_at(out.shape[0] - 1)), out.shape[0] - 1), (int(ln.x_at(y_top)), int(y_top)), col, 4)
    return out


if __name__ == "__main__":
    files = sorted((ROOT / "data" / "real_stills").glob("*.jpg"), key=lambda p: int(p.stem[4:]))
    if not files:
        sys.exit("no images in data/real_stills")
    rows, tiles = [], []
    for f in files:
        img = cv2.imread(str(f))
        img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        row = {"image": f.name}
        panel = []
        for name, cfg in CFGS.items():
            det = detect_lines(img, cfg)
            row[f"{name}_left"], row[f"{name}_right"] = det.left is not None, det.right is not None
            row[f"{name}_both"] = det.left is not None and det.right is not None
            panel.append(draw(img, det, 0.6 * img.shape[0]))
        rows.append(row)
        tiles.append(cv2.resize(np.hstack(panel), None, fx=0.25, fy=0.25))
    df = pd.DataFrame(rows)
    (ROOT / "outputs" / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "outputs" / "results" / "real_stills.csv", index=False)
    print(df.drop(columns="image").mean().round(2).to_string())
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    (ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "outputs" / "figures" / "real_stills_sheet.jpg"), sheet)
