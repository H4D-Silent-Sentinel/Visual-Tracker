"""Drone-type test: score a model on the held-out ratio1 test images, one result per type.

Drone types (propeller, fixed_wing): share of drones found (a box overlapping the true box, IoU >= 0.3) and
share of boxes that were wrong. Non-drone types (plane, helicopter, bird): share of images with any "drone" box.
Run: .venv/bin/python scripts/evaluate/test_drone_types.py --model runs/train/dut_v3_n960/weights/best.pt --out runs/drone_types_v3.json
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

TEST = Path.home() / "yolo-datasets" / "ratio1_test"
DRONE_TYPES = ("propeller", "fixed_wing")
OTHER_TYPES = ("plane", "helicopter", "bird")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def truth(label, w, h):
    out = []
    for line in label.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()
    model = YOLO(a.model)
    res = {"model": a.model, "conf": a.conf, "imgsz": a.imgsz, "types": {}}
    for kind in DRONE_TYPES + OTHER_TYPES:
        imgs = sorted((TEST / kind / "images").glob("*.jpg"))
        n_true = found = boxes = wrong = img_with_box = 0
        for img in imgs:
            r = model.predict(str(img), conf=a.conf, imgsz=a.imgsz, device="mps", verbose=False)[0]
            h, w = r.orig_shape
            pred = r.boxes.xyxy.tolist()
            gt = truth(TEST / kind / "labels" / f"{img.stem}.txt", w, h)
            boxes += len(pred); img_with_box += bool(pred)
            n_true += len(gt)
            found += sum(any(iou(g, p) >= 0.3 for p in pred) for g in gt)
            wrong += sum(not any(iou(p, g) >= 0.3 for g in gt) for p in pred)
        entry = {"images": len(imgs)}
        if kind in DRONE_TYPES:
            entry.update(drones=n_true, found=found, found_pct=round(100 * found / max(n_true, 1), 1),
                         wrong_boxes=wrong, wrong_box_pct=round(100 * wrong / max(boxes, 1), 1))
        else:
            entry.update(images_with_drone_box=img_with_box, false_alarm_pct=round(100 * img_with_box / max(len(imgs), 1), 1))
        res["types"][kind] = entry
        print(kind, entry, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
