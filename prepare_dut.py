"""Convert DUT Anti-UAV detection (Pascal VOC XML) to YOLO format.

Expects ~/yolo-datasets/dut-anti-uav/zips/{train,val,test}.zip
(from huggingface.co/datasets/aegean-ai/dut-anti-uav, folder detection/).
Writes ~/yolo-datasets/dut-anti-uav/yolo/{images,labels}/{train,val,test}
and dut.yaml for Ultralytics.
"""
import shutil, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path.home() / "yolo-datasets" / "dut-anti-uav"
OUT = ROOT / "yolo"


def main():
    stats = {}
    for split in ["train", "val", "test"]:
        raw = ROOT / "raw"
        if not (raw / split).exists():
            with zipfile.ZipFile(ROOT / "zips" / f"{split}.zip") as z:
                z.extractall(raw)
        img_dir = next(p for p in (raw / split).rglob("img") if p.is_dir())
        xml_dir = img_dir.parent / "xml"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for xml in sorted(xml_dir.glob("*.xml")):
            root = ET.parse(xml).getroot()
            W = float(root.find("size/width").text)
            H = float(root.find("size/height").text)
            lines = []
            for obj in root.findall("object"):
                b = obj.find("bndbox")
                x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            img = img_dir / (xml.stem + ".jpg")
            if not img.exists():
                continue
            dst = OUT / "images" / split / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            (OUT / "labels" / split / (xml.stem + ".txt")).write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
        stats[split] = (n_img, n_box)
        print(split, n_img, "images", n_box, "drone boxes")
    (ROOT / "dut.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: drone\n")
    print("wrote", ROOT / "dut.yaml")


if __name__ == "__main__":
    main()
