"""Add bird and kite photos (no drones) to the training set so the model learns they are NOT drones.

Source: COCO 2017 train set (cocodataset.org, CC BY 4.0 annotations; images from Flickr).
We pick photos labeled bird or kite, favoring ones where the birds/kites are SMALL in the frame
(distant birds in the sky are what fooled the model in Test 1). Each photo is added with an
empty label file, which YOLO treats as "nothing to find here".

The bird/kite videos in footage/negatives/ are NOT used here; they stay a clean test set.

Run:  .venv/bin/python scripts/data/prepare_negatives.py
"""
import json, random, statistics, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path.home() / "yolo-datasets"
COCO = D / "coco-negatives"
YOLO_DIR = D / "dut-anti-uav" / "yolo"
ANN_ZIP = COCO / "annotations_trainval2017.zip"
UA = {"User-Agent": "H4D-student-research/1.0"}
N_TRAIN = {"bird": 1300, "kite": 800}
N_VAL = {"bird": 200, "kite": 100}
CAT = {"bird": 16, "kite": 38}
BAD = {5}  # skip photos that also contain an airplane (category 5): too close to a drone label


def get(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(str(dest) + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    Path(str(dest) + ".part").rename(dest)


def main():
    COCO.mkdir(parents=True, exist_ok=True)
    print("downloading COCO annotations (241 MB)...", flush=True)
    get("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ANN_ZIP)
    with zipfile.ZipFile(ANN_ZIP) as z:
        data = json.loads(z.read("annotations/instances_train2017.json"))
    imgs = {i["id"]: i for i in data["images"]}
    rel = {}  # image id -> {cat: [relative box areas]}
    cats_in = {}
    for a in data["annotations"]:
        im = imgs[a["image_id"]]
        cats_in.setdefault(a["image_id"], set()).add(a["category_id"])
        for name, cid in CAT.items():
            if a["category_id"] == cid:
                rel.setdefault(a["image_id"], {}).setdefault(name, []).append(a["area"] / (im["width"] * im["height"]))
    random.seed(0)
    chosen = {"train": [], "val": []}
    used = set()
    for name in CAT:
        cand = [(statistics.median(v[name]), iid) for iid, v in rel.items()
                if name in v and not (cats_in[iid] & BAD) and iid not in used]
        cand.sort()  # smallest birds/kites first
        pool = [iid for _, iid in cand[: int((N_TRAIN[name] + N_VAL[name]) * 1.5)]]
        random.shuffle(pool)
        tr, va = pool[: N_TRAIN[name]], pool[N_TRAIN[name]: N_TRAIN[name] + N_VAL[name]]
        chosen["train"] += [(iid, name) for iid in tr]
        chosen["val"] += [(iid, name) for iid in va]
        used.update(tr + va)
        print(f"{name}: {len(tr)} train, {len(va)} val (median size {cand[0][0]*100:.2f}%-{cand[len(pool)][0]*100:.2f}% of frame)", flush=True)

    jobs = []
    for split, items in chosen.items():
        (YOLO_DIR / "images" / f"{split}_neg").mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / f"{split}_neg").mkdir(parents=True, exist_ok=True)
        for iid, name in items:
            fn = imgs[iid]["file_name"]
            jobs.append((imgs[iid]["coco_url"], YOLO_DIR / "images" / f"{split}_neg" / f"{name}_{fn}",
                         YOLO_DIR / "labels" / f"{split}_neg" / f"{name}_{Path(fn).stem}.txt"))
    done = 0

    def fetch(job):
        url, img, lab = job
        for _ in range(3):
            try:
                get(url, img)
                lab.write_text("")  # empty label = background image
                return True
            except Exception:
                pass
        return False

    with ThreadPoolExecutor(16) as ex:
        for ok in ex.map(fetch, jobs):
            done += 1
            if done % 250 == 0:
                print(f"  downloaded {done}/{len(jobs)}", flush=True)
    (D / "dut-anti-uav" / "dut_neg.yaml").write_text(
        f"path: {YOLO_DIR}\n"
        "train: [images/train, images/train_neg]\n"
        "val: [images/val, images/val_neg]\n"
        "test: images/test\n"
        "names:\n  0: drone\n")
    manifest = {s: [{"file": f"{n}_{imgs[i]['file_name']}", "kind": n, "coco_id": i,
                     "flickr_url": imgs[i].get("flickr_url")} for i, n in items] for s, items in chosen.items()}
    (YOLO_DIR / "negatives_manifest.json").write_text(json.dumps(manifest))
    print("done; wrote", D / "dut-anti-uav" / "dut_neg.yaml", flush=True)


if __name__ == "__main__":
    main()
