"""Make motion-blurred copies of a random quarter of the v4 training images (train_blur), same labels.

A moving camera (our chaser drone) smears the target. Random blur length 7-21 px, random direction.
Run after prepare_ratio1.py: .venv/bin/python scripts/data/prepare_blur.py
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

DUT = Path.home() / "yolo-datasets" / "dut-anti-uav" / "yolo"
SOURCES = ["train", "train_trk", "train_r1"]
FRACTION = 0.25
random.seed(4)


def kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    return k / max(k.sum(), 1e-6)


def main():
    for d in ("images", "labels"):
        shutil.rmtree(DUT / d / "train_blur", ignore_errors=True)
        (DUT / d / "train_blur").mkdir(parents=True)
    n = 0
    for src in SOURCES:
        imgs = sorted(p for p in (DUT / "images" / src).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for img in random.sample(imgs, int(len(imgs) * FRACTION)):
            lab = DUT / "labels" / src / f"{img.stem}.txt"
            if not lab.exists():
                continue
            im = cv2.imread(str(img))
            if im is None:
                continue
            out = cv2.filter2D(im, -1, kernel(random.choice(range(7, 22, 2)), random.uniform(0, 180)))
            name = f"blur_{src}_{img.stem}"
            cv2.imwrite(str(DUT / "images" / "train_blur" / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shutil.copy2(lab, DUT / "labels" / "train_blur" / f"{name}.txt")
            n += 1
    print("blurred copies:", n)


if __name__ == "__main__":
    main()
