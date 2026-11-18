"""Add cluttered-background drone frames to the training set (for model v3).

The 20 DUT tracking videos have the drone marked in every frame, including the hard ones
where it flies in front of trees and buildings. We split them once, by difficulty:
  TRAIN_VIDEOS  -> every 4th frame becomes a training image (with its drone box)
  HELD_OUT      -> never trained on; the only videos v3's tracking score may be judged on
Run:  .venv/bin/python scripts/data/prepare_tracking_split.py
"""
import json, shutil
from pathlib import Path
import cv2

TRAIN_VIDEOS = ["video16", "video13", "video04", "video05", "video19", "video15", "video20", "video02", "video06", "video11"]
HELD_OUT = ["video17", "video01", "video12", "video07", "video08", "video18", "video10", "video03", "video09", "video14"]
EVERY = 4
D = Path.home() / "yolo-datasets/dut-anti-uav"
FR, GT = D / "tracking/frames/Anti-UAV-Tracking-V0", D / "tracking/gt/Anti-UAV-Tracking-V0GT"
IM, LB = D / "yolo/images/train_trk", D / "yolo/labels/train_trk"
for p in (IM, LB):
    shutil.rmtree(p, ignore_errors=True); p.mkdir(parents=True)
n = boxes = empty = 0
for v in TRAIN_VIDEOS:
    gt = [list(map(float, l.split())) for l in (GT / f"{v}_gt.txt").read_text().split("\n") if l.strip()]
    imgs = sorted((FR / v).glob("*.jpg"))
    h, w = cv2.imread(str(imgs[0])).shape[:2]
    for k in range(0, min(len(imgs), len(gt)), EVERY):
        x, y, bw, bh = gt[k][:4]
        name = f"{v}_{imgs[k].stem}"
        shutil.copy(imgs[k], IM / f"{name}.jpg")
        if bw > 0 and bh > 0:
            (LB / f"{name}.txt").write_text(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n"); boxes += 1
        else:
            (LB / f"{name}.txt").write_text(""); empty += 1
        n += 1
(D / "dut_v3.yaml").write_text(
    f"path: {D / 'yolo'}\ntrain: [images/train, images/train_neg, images/train_trk]\n"
    "val: [images/val, images/val_neg]\ntest: images/test\nnames:\n  0: drone\n")
(D / "yolo/tracking_split.json").write_text(json.dumps({"train": TRAIN_VIDEOS, "held_out": HELD_OUT, "every": EVERY}))
print(f"{n} frames from {len(TRAIN_VIDEOS)} videos ({boxes} with a drone, {empty} without); held out: {HELD_OUT}")
