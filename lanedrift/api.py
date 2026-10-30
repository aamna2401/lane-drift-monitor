"""Minimal FastAPI service: POST a frame, get lane lines and the lateral offset. Stateless (no tracking)."""
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import Config
from .detect import detect_lines
from .geometry import lateral_offset_m

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "final.json"
CFG = (Config.load(CFG_PATH) if CFG_PATH.exists() else Config()).with_(tracking=False, guided_roi=False)
app = FastAPI(title="lane-drift-monitor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/frame")
async def frame(image: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    h, w = img.shape[:2]
    det = detect_lines(img, CFG)
    out = {"left": None, "right": None, "offset_m": None}
    for k in ("left", "right"):
        ln = getattr(det, k)
        if ln is not None:
            out[k] = {"x_bottom": ln.x_at(h - 1), "x_at_0.6h": ln.x_at(0.6 * h), "support_px": ln.support}
    if det.left is not None and det.right is not None:
        out["offset_m"], out["lane_width_px"] = lateral_offset_m(det.left, det.right, w, h, CFG.geometry)
    return out
