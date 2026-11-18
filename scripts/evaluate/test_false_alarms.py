"""Test 1: false alarms. Run the drone model on clips with NO drones (birds, kites)
and count how often it draws a 'drone' box anyway.

Usage: python scripts/evaluate/test_false_alarms.py [--model ...] [--imgsz 960] [--confs 0.25,0.35,0.5]
Clips: footage/negatives/*  (Wikimedia Commons, see README)
Writes runs/false_alarms/report.json and a contact sheet of the worst frames.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/train/dut_n960/weights/best.pt")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--confs", default="0.25,0.35,0.5")
ap.add_argument("--out", default="runs/false_alarms")
ap.add_argument("--every", type=int, default=2, help="test every Nth frame")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(a.model)
confs = [float(c) for c in a.confs.split(",")]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows, worst = [], []
tot = {c: 0 for c in confs}; tot_frames = 0
for clip in sorted(Path("footage/negatives").iterdir()):
    cap = cv2.VideoCapture(str(clip)); n = 0; hits = {c: 0 for c in confs}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        i += 1
        if i % a.every: continue
        r = model.predict(fr, imgsz=a.imgsz, conf=min(confs), device=dev, verbose=False)[0]
        cs = [float(c) for c in r.boxes.conf]
        n += 1
        for c in confs:
            if any(x >= c for x in cs): hits[c] += 1
        if cs and max(cs) >= 0.35:
            worst.append((max(cs), clip.name, r.plot()))
            worst = sorted(worst, key=lambda t: -t[0])[:8]
    tot_frames += n
    for c in confs: tot[c] += hits[c]
    row = {"clip": clip.name, "frames_tested": n, **{f"false_alarm_pct@{c}": round(100*hits[c]/n, 1) if n else None for c in confs}}
    rows.append(row); print(json.dumps(row), flush=True)
summary = {f"false_alarm_pct@{c}": round(100*tot[c]/tot_frames, 1) for c in confs}
summary["frames_tested"] = tot_frames
print("ALL", json.dumps(summary))
(out / "report.json").write_text(json.dumps({"model": a.model, "imgsz": a.imgsz, "clips": rows, "overall": summary}, indent=2))
if worst:
    tiles = [cv2.resize(w[2], (480, 270)) for w in worst]
    while len(tiles) % 4: tiles.append(np.zeros((270, 480, 3), np.uint8))
    cv2.imwrite(str(out / "worst_false_alarms.jpg"), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
