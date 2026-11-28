"""Moving-camera test on held-out DUT clips, with the camera motion simulated.

Each frame is cropped to a window two-thirds its size that pans, tilts and shakes over time, so the
drone moves on screen even when it doesn't move in the air. The true box is shifted the same way.
Compares ByteTrack vs BoT-SORT (camera-motion compensation), and the orange trail raw vs corrected:
trail error = how far each of the last 30 trail points sits from where the drone really was, drawn
in the current frame (pixels).

Run: .venv/bin/python scripts/evaluate/test_moving_camera.py --model runs/train/dut_v3_n960/weights/best.pt
"""
import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from camera_motion import CameraMotion, apply

CROP = 2 / 3  # window size as a share of the frame
TRAIL = 30


def offset(i, fps, mx, my):
    t = i / fps  # slow pan + tilt + hand shake, kept inside the margin (mx, my)
    ox = mx * (0.5 + 0.4 * math.sin(2 * math.pi * t / 6.0) + 0.05 * math.sin(2 * math.pi * t * 2.3))
    oy = my * (0.5 + 0.4 * math.sin(2 * math.pi * t / 4.5 + 1.0) + 0.05 * math.sin(2 * math.pi * t * 3.1))
    return int(round(ox)), int(round(oy))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def run(model, clip, tracker, conf, imgsz, stride):
    gt = json.loads((ROOT / "runs/review/gt" / f"dut__{clip}.json").read_text())
    cap = cv2.VideoCapture(str(ROOT / "footage/dut" / f"{clip}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    model.predictor = None
    cm = CameraMotion()
    raw, fixed, true_world = deque(maxlen=TRAIL), deque(maxlen=TRAIL), deque(maxlen=TRAIL)
    vis = hit = no_motion = frames = 0
    ids = []
    err_raw, err_fix = [], []
    i = -1
    while True:
        ok, f = cap.read()
        i += 1
        if not ok:
            break
        if i % stride:
            continue
        H, W = f.shape[:2]
        CW, CH = int(W * CROP), int(H * CROP)
        ox, oy = offset(i, fps, W - CW, H - CH)
        crop = f[oy:oy + CH, ox:ox + CW]
        r = model.track(crop, persist=True, conf=conf, imgsz=imgsz, device="mps", tracker=tracker, verbose=False)[0]
        boxes = r.boxes.xyxy.tolist()
        m = cm.update(crop, boxes)
        no_motion += m is None
        frames += 1
        fixed_pts = apply(m, list(fixed)) if m is not None else list(fixed)
        fixed.clear(); fixed.extend(fixed_pts)
        g = gt["gt"][i] if i < len(gt["gt"]) else None
        gbox = [g[0] * W - ox, g[1] * H - oy, g[2] * W - ox, g[3] * H - oy] if g else None
        if gbox and (gbox[2] < 0 or gbox[3] < 0 or gbox[0] > CW or gbox[1] > CH):
            gbox = None
        best = max(range(len(boxes)), key=lambda k: float(r.boxes.conf[k])) if boxes else None
        if gbox:
            vis += 1
            on = [k for k, b in enumerate(boxes) if iou(b, gbox) >= 0.3]
            if on:
                hit += 1
                k = on[0]
                ids.append(int(r.boxes.id[k]) if r.boxes.id is not None else None)
        if best is not None:
            b = boxes[best]
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            raw.append(c); fixed.append(c)
            # only score trail points that were really on the drone; a box on a bird says nothing about the correction
            on_drone = gbox is not None and iou(b, gbox) >= 0.3
            true_world.append(((g[0] + g[2]) / 2 * W, (g[1] + g[3]) / 2 * H) if on_drone else None)
            # where each remembered point SHOULD be drawn now: the true drone position then, in this frame
            for pr, pf, tw in zip(raw, fixed, true_world):
                if tw is None:
                    continue
                tx, ty = tw[0] - ox, tw[1] - oy
                err_raw.append(math.hypot(pr[0] - tx, pr[1] - ty))
                err_fix.append(math.hypot(pf[0] - tx, pf[1] - ty))
        else:
            raw.clear(); fixed.clear(); true_world.clear()
    cap.release()
    switches = sum(1 for a, b in zip(ids, ids[1:]) if a is not None and b is not None and a != b)
    med = lambda v: round(sorted(v)[len(v) // 2], 1) if v else None
    return {"clip": clip, "tracker": Path(tracker).stem, "visible": vis, "on_target_pct": round(100 * hit / max(vis, 1), 1),
            "id_switches": switches, "trail_err_raw_px": med(err_raw), "trail_err_fixed_px": med(err_fix), "no_motion_estimate_pct": round(100 * no_motion / max(frames, 1), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs/train/dut_v3_n960/weights/best.pt"))
    ap.add_argument("--clips", default="video01,video07,video10,video17")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "runs/moving_camera_test.json"))
    a = ap.parse_args()
    model = YOLO(a.model)
    rows = []
    for clip in a.clips.split(","):
        for tr in ("bytetrack.yaml", str(ROOT / "botsort_motion.yaml")):
            row = run(model, clip, tr, a.conf, a.imgsz, a.stride)
            rows.append(row)
            print(row, flush=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
