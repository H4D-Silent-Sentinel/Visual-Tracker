"""Convert the REAL part of ratio1/Drone_Detection_Survaillance (CC BY-NC 4.0, Hugging Face) to YOLO format for v4.

Propeller and fixed-wing drones become class 0 (drone). Planes, helicopters and birds get no box, so those
images teach the model what is NOT a drone. Their dev split becomes extra validation; their test split is kept
out of training as the drone-type test set, one folder per type. A source video found in the test split is
dropped from training and validation so no test video leaks in.

Run: .venv/bin/python scripts/data/prepare_ratio1.py   (after downloading to ~/yolo-datasets/ratio1)
"""
import collections
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path.home() / "yolo-datasets" / "ratio1"
DUT = Path.home() / "yolo-datasets" / "dut-anti-uav" / "yolo"
TEST_OUT = Path.home() / "yolo-datasets" / "ratio1_test"
DRONE_NAMES = {"propeller", "fixed_wing", "fixed-wing", "drone"}
TYPES = {"1.Propeller": "propeller", "2.Fixed_wing": "fixed_wing", "3.Plane": "plane",
         "4.Helicopter": "helicopter", "5.Bird": "bird"}


def items(split):
    for cls_dir, kind in TYPES.items():
        for xml in sorted((SRC / split / "REAL" / cls_dir).rglob("*.xml")):
            img = xml.with_suffix(".jpg")
            if img.exists():
                yield kind, xml.parent.name, xml, img


def yolo_lines(xml, seen):
    root = ET.parse(xml).getroot()
    w = float(root.findtext("size/width")); h = float(root.findtext("size/height"))
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        seen[name] += 1
        if name not in DRONE_NAMES:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, x1), min(w, x2); y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def write(img, lines, img_dir, lab_dir, name):
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_dir / f"{name}.jpg")
    (lab_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    seen = collections.Counter()
    test_videos = {(k, v) for k, v, _, _ in items("test")}
    report = {"source": "huggingface.co/datasets/ratio1/Drone_Detection_Survaillance (REAL only)",
              "license": "CC BY-NC 4.0", "dropped_for_leak": collections.Counter(), "counts": {}}
    for split, dest in (("train", "train_r1"), ("dev", "val_r1")):
        for d in ("images", "labels"):
            shutil.rmtree(DUT / d / dest, ignore_errors=True)
        c = collections.Counter()
        for kind, video, xml, img in items(split):
            if (kind, video) in test_videos:
                report["dropped_for_leak"][f"{split}:{kind}"] += 1
                continue
            write(img, yolo_lines(xml, seen), DUT / "images" / dest, DUT / "labels" / dest, f"r1_{kind}_{video}_{img.stem}")
            c[kind] += 1
        report["counts"][dest] = dict(c)
    shutil.rmtree(TEST_OUT, ignore_errors=True)
    c = collections.Counter()
    for kind, video, xml, img in items("test"):
        write(img, yolo_lines(xml, seen), TEST_OUT / kind / "images", TEST_OUT / kind / "labels", f"{video}_{img.stem}")
        c[kind] += 1
    report["counts"]["test"] = dict(c)
    for kind in TYPES.values():
        (TEST_OUT / f"{kind}.yaml").write_text(f"path: {TEST_OUT / kind}\ntrain: images\nval: images\ntest: images\nnames:\n  0: drone\n")
    report["object_names"] = dict(seen)
    report["dropped_for_leak"] = dict(report["dropped_for_leak"])
    (TEST_OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
