"""Generate the review files the dashboard draws from. Nothing here trains or changes a model.

  runs/review/gt/<clip>.json            true drone box per frame (tracking test clips only)
  runs/review/<output>/<clip>.json      the model's boxes for every frame of a clip
  runs/review/<output>/_summary.json    per-clip totals + the worst moment to jump to
  runs/review/<model>/photos_<split>.json   boxes on the held-out test photos (Data Pool overlay)
  runs/review/photo_test.json           precision / recall on the 2,200 labeled test photos

Outputs:  v1, v2, v3 (default tracker, cutoff 0.35, same as Test 2) and v2_hold, v3_hold
(bytetrack_hold.yaml, detector cutoff 0.08). Tracking-test clips are run on the original frames,
exactly like Test 2; other clips are run on the browser copy, so frame N always means the same picture.

Run:  .venv/bin/python scripts/review/make_review.py --gt-only     (no model, safe any time)
      .venv/bin/python scripts/review/make_review.py               (waits for training jobs to finish first)
"""
import argparse, json, subprocess, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
DS = Path.home() / "yolo-datasets/dut-anti-uav"
FR, GT = DS / "tracking/frames/Anti-UAV-Tracking-V0", DS / "tracking/gt/Anti-UAV-Tracking-V0GT"
WEB, REVIEW = ROOT / "footage/web", ROOT / "runs/review"
RUNS = {"v1": "dut_n960", "v2": "dut_neg_n960", "v3": "dut_v3_n960"}
OUTPUTS = [("v1", "v1", "default"), ("v2", "v2", "default"), ("v3", "v3", "default"), ("v2_hold", "v2", "hold"), ("v3_hold", "v3", "hold")]
SIZES = [("under 16 px", 0, 16), ("16-32 px", 16, 32), ("32-64 px", 32, 64), ("over 64 px", 64, 1e9)]
IMGSZ = 960


def running(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def iou(b, g):
    x1, y1, x2, y2 = max(b[0], g[0]), max(b[1], g[1]), min(b[2], g[2]), min(b[3], g[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    return inter / u if u > 0 else 0


def write_gt(index):
    (REVIEW / "gt").mkdir(parents=True, exist_ok=True)
    for key, c in index.items():
        if c["group"] != "dut":
            continue
        imgs = sorted((FR / c["name"]).glob("*.jpg"))
        h, w = cv2.imread(str(imgs[0])).shape[:2]
        rows = [list(map(float, l.split())) for l in (GT / f"{c['name']}_gt.txt").read_text().split("\n") if l.strip()]
        gt = [[round(x / w, 5), round(y / h, 5), round((x + bw) / w, 5), round((y + bh) / h, 5)] if bw > 0 and bh > 0 else None
              for x, y, bw, bh in (r[:4] for r in rows)]
        ok = len(gt) == c["frames"] == len(imgs)
        (REVIEW / "gt" / f"{key}.json").write_text(json.dumps({"clip": key, "frames": len(gt), "src_w": w, "src_h": h,
                                                                "timebase_ok": ok, "gt": gt}))
        print("gt", key, len(gt), "frames", "OK" if ok else "FRAME COUNT MISMATCH", flush=True)


def frames_of(c):
    if c["group"] == "dut":
        for p in sorted((FR / c["name"]).glob("*.jpg")):
            yield cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(ROOT / c["file"]))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
        cap.release()


def predict_clip(model, dev, c, tracker):
    model.predictor = None
    kw = dict(conf=0.35, tracker="bytetrack.yaml") if tracker == "default" else dict(conf=0.08, tracker=str(ROOT / "bytetrack_hold.yaml"))
    out = []
    for fr in frames_of(c):
        h, w = fr.shape[:2]
        r = model.track(fr, persist=True, imgsz=IMGSZ, device=dev, verbose=False, **kw)[0]
        out.append([[round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5), round(float(b.conf), 3),
                     int(b.id) if b.id is not None else None]
                    for b in r.boxes for (x1, y1, x2, y2) in [b.xyxy[0].tolist()]])
    return out


def summarize(c, boxes, gt):
    n = len(boxes)
    s = {"frames": n, "box_frames": sum(1 for b in boxes if b)}
    if c["group"] == "dut" and gt:
        g, W, H = gt["gt"], gt["src_w"], gt["src_h"]
        vis = on = fa = idsw = 0
        last_id, run_start, best = None, None, (0, None)
        by = {name: [0, 0] for name, _, _ in SIZES}
        for k in range(min(n, len(g))):
            bs, gb = boxes[k], g[k]
            hit = None
            if gb:
                vis += 1
                side = max((gb[2] - gb[0]) * W, (gb[3] - gb[1]) * H)
                bucket = next(name for name, lo, hi in SIZES if lo <= side < hi)
                by[bucket][0] += 1
                ok = [b for b in bs if iou(b, gb) >= 0.3]
                if ok:
                    hit = max(ok, key=lambda b: b[4])
                    on += 1
                    by[bucket][1] += 1
                    if last_id is not None and hit[5] is not None and hit[5] != last_id:
                        idsw += 1
                    if hit[5] is not None:
                        last_id = hit[5]
            if gb and not hit:
                if run_start is None:
                    run_start = k
                if k - run_start + 1 > best[0]:
                    best = (k - run_start + 1, run_start)
            else:
                run_start = None
            if bs and (not gb or not any(iou(b, gb) > 0.1 for b in bs)):
                fa += 1
        s.update(visible=vis, on_target=on, on_target_pct=round(100 * on / vis, 1) if vis else None, false_alarm_frames=fa,
                 id_switches=idsw, longest_miss_frames=best[0], worst=best[1], by_size=by)
    elif c["group"] == "negatives":
        top = max(((max(b[4] for b in bs), k) for k, bs in enumerate(boxes) if bs), default=(0, None))
        s.update(false_alarm_frames=s["box_frames"], false_alarm_pct=round(100 * s["box_frames"] / n, 1) if n else None, worst=top[1])
    else:
        first_gap = next((k for k in range(1, n) if boxes[k - 1] and not boxes[k]), None)
        s.update(worst=first_gap)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--only", default="", help="comma list of outputs, e.g. v1,v2")
    a = ap.parse_args()
    index = json.loads((WEB / "index.json").read_text())
    write_gt(index)
    if a.gt_only:
        return
    if not a.no_wait:
        while running("detect train") or running("run_v3.sh") or running("retest_v2.sh"):
            time.sleep(30)
    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    demo = [d["clip"] for d in json.loads((ROOT / "dashboard/demo_clips.json").read_text())] if (ROOT / "dashboard/demo_clips.json").exists() else []
    order = sorted(index, key=lambda k: (k not in demo, {"dut": 0, "negatives": 1}.get(index[k]["group"], 2), k))
    only = set(x for x in a.only.split(",") if x)
    models = {}
    for oid, mid, tracker in OUTPUTS:
        if only and oid not in only:
            continue
        w = ROOT / "runs/train" / RUNS[mid] / "weights/best.pt"
        if not w.exists() or running(f"name={RUNS[mid]}"):
            print("skip", oid, "(model not finished)", flush=True)
            continue
        model = models.setdefault(mid, YOLO(str(w)))
        d = REVIEW / oid
        d.mkdir(parents=True, exist_ok=True)
        summ = json.loads((d / "_summary.json").read_text()) if (d / "_summary.json").exists() else {}
        for key in order:
            c = index[key]
            p = d / f"{key}.json"
            if p.exists() and p.stat().st_mtime > w.stat().st_mtime and key in summ:
                continue
            t0 = time.time()
            boxes = predict_clip(model, dev, c, tracker)
            ok = len(boxes) == c["frames"]
            gt = json.loads((REVIEW / "gt" / f"{key}.json").read_text()) if c["group"] == "dut" else None
            p.write_text(json.dumps({"clip": key, "output": oid, "model": mid, "tracker": tracker, "imgsz": IMGSZ,
                                     "frames": len(boxes), "timebase_ok": ok, "boxes": boxes}, separators=(",", ":")))
            summ[key] = summarize(c, boxes, gt)
            (d / "_summary.json").write_text(json.dumps(summ))
            print(oid, key, len(boxes), "frames", "OK" if ok else "FRAME COUNT MISMATCH", f"{time.time()-t0:.0f}s", flush=True)

    # held-out photos: boxes for the Data Pool overlay + labeled precision/recall
    photo = json.loads((REVIEW / "photo_test.json").read_text()) if (REVIEW / "photo_test.json").exists() else {}
    for mid, model in models.items():
        w = ROOT / "runs/train" / RUNS[mid] / "weights/best.pt"
        for split in ("test", "val_neg"):
            p = REVIEW / mid / f"photos_{split}.json"
            if p.exists() and p.stat().st_mtime > w.stat().st_mtime:
                continue
            (REVIEW / mid).mkdir(parents=True, exist_ok=True)
            out = {}
            files = sorted((DS / "yolo/images" / split).glob("*.jpg"))
            for i in range(0, len(files), 16):
                for f, r in zip(files[i:i + 16], model.predict([str(x) for x in files[i:i + 16]], imgsz=IMGSZ, conf=0.35, device=dev, verbose=False)):
                    out[f.name] = [[round(v, 5) for v in xy] + [round(float(c), 3)] for xy, c in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist())]
            p.write_text(json.dumps(out, separators=(",", ":")))
            print("photos", mid, split, len(out), flush=True)
        if mid not in photo or photo[mid].get("weights_mtime") != w.stat().st_mtime:
            m = YOLO(str(w)).val(data=str(DS / "dut.yaml"), split="test", imgsz=IMGSZ, device=dev, verbose=False, plots=False,
                                 project=str(ROOT / "runs/val"), name=f"review_{mid}", exist_ok=True)
            photo[mid] = {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4), "map50": round(float(m.box.map50), 4),
                          "map50_95": round(float(m.box.map), 4), "images": 2200, "split": "test", "weights_mtime": w.stat().st_mtime}
            (REVIEW / "photo_test.json").write_text(json.dumps(photo, indent=1))
            print("photo test", mid, photo[mid], flush=True)
    print("ALL DONE", time.strftime("%H:%M"), flush=True)


if __name__ == "__main__":
    main()
