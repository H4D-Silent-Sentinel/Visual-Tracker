"""CENTCOM-4 Drone Hide and Seek: local research dashboard.

Run:   .venv/bin/python dashboard/app.py      then open http://localhost:8000
Stop:  Ctrl-C in the Terminal window.

Tabs
  Clip Review  recorded clips in a normal video player; boxes are drawn from saved per-frame
               prediction files (runs/review/<output>/<clip>.json), so you can scrub, step frames
               and switch model outputs at the same moment
  Results      v1 / v2 / v3 compared on the same held-out clips, training status from the logs
  Data Pool    every photo and video we train and test on
  Collect      live camera, recording, labeling our own footage, flagged hard examples

The server itself never runs a model unless you ask it to (labeling proposals, camera overlay),
and it refuses to while a training job is running. Standard library + OpenCV only.
"""
import collections
import csv
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import classification_data

HERE = Path(__file__).resolve().parent
PORT = int(__import__('os').environ.get('PORT', '8000'))
DS = Path.home() / "yolo-datasets" / "dut-anti-uav"
DATA = DS / "yolo"
WEB = ROOT / "footage" / "web"
REVIEW = ROOT / "runs" / "review"
THUMBS = ROOT / "runs" / "thumbs"
OURS = ROOT / "our_data"
TEAL, ORANGE = (197, 224, 63), (61, 107, 255)

MODELS = {  # id -> training run folder, label, what it was trained on
    "v1": {"run": "dut_n960", "label": "v1", "data": "5,200 drone photos"},
    "v2": {"run": "dut_neg_n960", "label": "v2", "data": "v1 + 2,100 bird/kite photos marked no drone"},
    "v3": {"run": "dut_v3_n960", "label": "v3", "data": "v2 + 2,393 frames from 10 tracking videos (cluttered backgrounds)"},
}
OUTPUTS = [  # id, model, tracker, label, Test 1 report dir, Test 2 report dir
    {"id": "v1", "model": "v1", "tracker": "default", "label": "v1", "t1": "runs/false_alarms", "t2": "runs/tracking_test"},
    {"id": "v2", "model": "v2", "tracker": "default", "label": "v2", "t1": "runs/false_alarms_v2", "t2": "runs/tracking_test_v2"},
    {"id": "v3", "model": "v3", "tracker": "default", "label": "v3", "t1": "runs/false_alarms_v3", "t2": "runs/tracking_test_v3"},
    {"id": "v2_hold", "model": "v2", "tracker": "hold", "label": "v2 + hold", "t1": None, "t2": "runs/tracking_test_v2_hold"},
    {"id": "v3_hold", "model": "v3", "tracker": "hold", "label": "v3 + hold", "t1": None, "t2": "runs/tracking_test_v3_hold"},
]
OUT = {o["id"]: o for o in OUTPUTS}
GROUP_INFO = {
    "dut": {"label": "Tracking test", "source": "DUT Anti-UAV tracking set", "truth": "drone box marked in every frame"},
    "negatives": {"label": "No drone (birds/kites)", "source": "Wikimedia Commons", "truth": "no drone anywhere in the clip"},
    "demo": {"label": "Demo flights", "source": "Wikimedia Commons", "truth": "none"},
    "ours": {"label": "Our footage", "source": "Recorded by us", "truth": "our own labels, if any"},
}


def weights(model):
    return ROOT / "runs" / "train" / MODELS[model]["run"] / "weights" / "best.pt"


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


_pg = {}


def pgrep(pattern):
    """True if a process matching the pattern is running (cached for 5 s)."""
    t, v = _pg.get(pattern, (0, False))
    if time.time() - t > 5:
        try:
            v = subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
        except Exception:
            v = False
        _pg[pattern] = (time.time(), v)
    return v


def training_active():
    return pgrep("detect train")


def iou(b, g):
    x1, y1, x2, y2 = max(b[0], g[0]), max(b[1], g[1]), min(b[2], g[2]), min(b[3], g[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    return inter / u if u > 0 else 0


def brackets(img, x1, y1, x2, y2, color, t):
    L = max(6, int(min(x2 - x1, y2 - y1) * 0.28))
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x2, y2, -1, -1), (x1, y2, 1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)


# ------------------------------------------------------------------ training status

def run_status(model):
    """What the training logs and files actually say about one training run."""
    run = ROOT / "runs" / "train" / MODELS[model]["run"]
    out = {"model": model, "label": MODELS[model]["label"], "data": MODELS[model]["data"], "state": "not started",
           "epochs_done": 0, "epochs_planned": None, "usable": False}
    if not run.exists():
        if model == "v3" and pgrep("run_v3.sh"):
            out["state"] = "queued"
        return out
    args = run / "args.yaml"
    if args.exists():
        m = re.search(r"^epochs:\s*(\d+)", args.read_text(), re.M)
        out["epochs_planned"] = int(m.group(1)) if m else None
        out["started"] = args.stat().st_mtime
    rows = []
    rc = run / "results.csv"
    if rc.exists():
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(rc.open())]
        out["last_update"] = rc.stat().st_mtime
    out["epochs_done"] = len(rows)
    running = pgrep(f"name={MODELS[model]['run']}")
    best = run / "weights" / "best.pt"
    if running:
        out["state"] = "training"
        if rows and out["epochs_planned"]:
            per = float(rows[-1]["time"]) / len(rows)
            left = out["epochs_planned"] - len(rows)
            # the log is the freshest signal of progress inside the current pass
            out["eta_s"] = max(0, int(per * left - (time.time() - out["last_update"])))
            out["sec_per_epoch"] = int(per)
    elif best.exists():
        out["usable"] = True
        full = out["epochs_planned"] and len(rows) >= out["epochs_planned"]
        out["state"] = "finished" if full else f"stopped at pass {len(rows)} of {out['epochs_planned']}"
    else:
        out["state"] = "no weights"
    if rows:
        b = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
        out["val"] = {"precision": float(b["metrics/precision(B)"]), "recall": float(b["metrics/recall(B)"]),
                      "map50": float(b["metrics/mAP50(B)"]), "best_epoch": int(float(b["epoch"]))}
    return out


# ------------------------------------------------------------------ clips, outputs, results

def trained_videos():
    """Which tracking videos v3 trained on, verified from the actual training files."""
    d = DATA / "images" / "train_trk"
    if not d.exists():
        return None
    return sorted({p.name.split("_")[0] for p in d.glob("*.jpg")})


def clip_index():
    return read_json(WEB / "index.json", {})


def test_rows():
    """Per-clip numbers from the completed test reports, keyed by output then clip key."""
    out = {}
    for o in OUTPUTS:
        r2 = read_json(ROOT / o["t2"] / "report.json") if o["t2"] else None
        r1 = read_json(ROOT / o["t1"] / "report.json") if o["t1"] else None
        rows = {}
        if r2:
            for v in r2["videos"]:
                rows[f"dut__{v['video']}"] = {"kind": "tracking", **v}
        if r1:
            for c in r1["clips"]:
                rows[f"negatives__{Path(c['clip']).stem}"] = {"kind": "false_alarm", "frames_tested": c["frames_tested"],
                                                              "fa25": c["false_alarm_pct@0.25"], "fa35": c["false_alarm_pct@0.35"],
                                                              "fa50": c["false_alarm_pct@0.5"]}
        out[o["id"]] = {"rows": rows, "t1": r1["overall"] if r1 else None, "t2": bool(r2)}
    return out


def output_state():
    st = {m: run_status(m) for m in MODELS}
    res = []
    for o in OUTPUTS:
        s = st[o["model"]]
        d = REVIEW / o["id"]
        n = len(list(d.glob("*__*.json"))) if d.exists() else 0
        res.append({"id": o["id"], "label": o["label"], "model": o["model"], "tracker": o["tracker"],
                    "model_state": s["state"], "model_usable": s["usable"], "clips_with_predictions": n})
    return res


def clips_payload():
    idx = clip_index()
    tv = trained_videos()
    rows = test_rows()
    demo = read_json(HERE / "demo_clips.json", [])
    demo_by = {d["clip"]: d for d in demo}
    summ = {o["id"]: read_json(REVIEW / o["id"] / "_summary.json", {}) for o in OUTPUTS}
    clips = []
    for key, c in sorted(idx.items(), key=lambda kv: ({"dut": 0, "negatives": 1, "demo": 2, "ours": 3}.get(kv[1]["group"], 9), kv[0])):
        g = c["group"]
        split = None
        if g == "dut" and tv is not None:
            split = "v3 trained on it" if c["name"] in tv else "held out"
        outs = {}
        for o in OUTPUTS:
            p = REVIEW / o["id"] / f"{key}.json"
            outs[o["id"]] = {"available": p.exists(), "summary": rows[o["id"]]["rows"].get(key),
                             "worst": (summ[o["id"]].get(key) or {}).get("worst")}
        clips.append({"key": key, "group": g, "group_label": GROUP_INFO.get(g, {}).get("label", g),
                      "source": GROUP_INFO.get(g, {}).get("source", ""), "name": c["name"],
                      "url": f"/media/{key}.mp4", "frames": c["frames"], "fps": c["fps"], "w": c["w"], "h": c["h"],
                      "timebase_ok": c.get("timebase_ok", True), "has_gt": (REVIEW / "gt" / f"{key}.json").exists(),
                      "truth": "none" if g not in ("dut", "negatives") else ("boxes" if g == "dut" else "no_drone"),
                      "split": split, "meta": read_json(ROOT / "footage" / "ours" / f"{c['name']}.json") if g == "ours" else None,
                      "outputs": outs, "demo": demo_by.get(key)})
    return {"clips": clips, "outputs": output_state(), "demo": demo, "training_active": training_active()}


def agg_tracking(rows, keys):
    """Combine per-clip Test 2 rows into one score over the given clips (frame-weighted)."""
    use = [rows[k] for k in keys if k in rows]
    if len(use) != len(keys) or not use:
        return None
    vis = sum(r["drone_visible_frames"] for r in use)
    frames = sum(r["frames"] for r in use)
    on = sum(r["on_target_pct"] * r["drone_visible_frames"] / 100 for r in use)
    fa = sum(r["false_alarm_pct"] * r["frames"] / 100 for r in use)
    gaps = sum(r["losses_recovered"] for r in use)
    gapf = sum((r["avg_reacquire_s"] or 0) * r["losses_recovered"] for r in use)
    return {"clips": len(use), "frames": frames, "visible_frames": vis, "on_target_pct": round(100 * on / vis, 1),
            "false_alarm_pct": round(100 * fa / frames, 1), "id_switches": sum(r["id_switches"] for r in use),
            "avg_reacquire_s": round(gapf / gaps, 2) if gaps else None}


def results_payload():
    idx = clip_index()
    tv = trained_videos()
    rows = test_rows()
    dut = sorted(k for k, c in idx.items() if c["group"] == "dut")
    held = [k for k in dut if tv is not None and idx[k]["name"] not in tv]
    neg = sorted(k for k, c in idx.items() if c["group"] == "negatives")
    status = {m: run_status(m) for m in MODELS}
    summ = {o["id"]: read_json(REVIEW / o["id"] / "_summary.json", {}) for o in OUTPUTS}
    photo = read_json(REVIEW / "photo_test.json", {})

    def per_clip(keys):
        out = []
        for k in keys:
            out.append({"key": k, "name": idx[k]["name"],
                        "by": {o["id"]: rows[o["id"]]["rows"].get(k) for o in OUTPUTS},
                        "worst": {o["id"]: (summ[o["id"]].get(k) or {}).get("worst") for o in OUTPUTS}})
        return out

    sizes = {}
    for o in OUTPUTS:  # on-target by drone size, from the per-frame review files (held-out clips only)
        b = collections.OrderedDict((n, [0, 0]) for n in ("under 16 px", "16-32 px", "32-64 px", "over 64 px"))
        ok = bool(held)
        for k in held:
            s = (summ[o["id"]].get(k) or {}).get("by_size")
            if not s:
                ok = False
                break
            for n, (v, h) in s.items():
                b[n][0] += v
                b[n][1] += h
        if ok:
            sizes[o["id"]] = {n: {"visible": v, "on_target_pct": round(100 * h / v, 1) if v else None} for n, (v, h) in b.items()}

    return {
        "split": {"verified": tv is not None, "how": "read from the file names in the v3 training folder (images/train_trk)",
                  "trained": [f"dut__{v}" for v in (tv or [])], "held_out": held},
        "outputs": [{"id": o["id"], "label": o["label"], "model": o["model"], "tracker": o["tracker"]} for o in OUTPUTS],
        "held_out": {o["id"]: agg_tracking(rows[o["id"]]["rows"], held) if held else None for o in OUTPUTS},
        "all20": {o["id"]: agg_tracking(rows[o["id"]]["rows"], dut) for o in OUTPUTS if o["model"] != "v3"},
        "held_out_clips": per_clip(held),
        "false_alarms": {o["id"]: rows[o["id"]]["t1"] for o in OUTPUTS if o["t1"]},
        "false_alarm_clips": per_clip(neg),
        "photo_test": photo,
        "sizes": sizes,
        "training": [status[m] for m in MODELS],
        "queue": {"retest_v2": pgrep("retest_v2.sh"), "run_v3": pgrep("run_v3.sh"), "make_review": pgrep("make_review")},
        "now": time.time(),
    }


# ------------------------------------------------------------------ data pool

IMG_SPLITS = {"train": "DUT Anti-UAV", "val": "DUT Anti-UAV", "test": "DUT Anti-UAV", "train_neg": "COCO",
              "val_neg": "COCO", "train_trk": "DUT Anti-UAV tracking"}
SIZE_BUCKETS = [("tiny", 0, 0.02), ("small", 0.02, 0.05), ("medium", 0.05, 0.15), ("large", 0.15, 9)]
POOL = {"ready": False}
POOL_LOCK = threading.Lock()


def build_pool():
    splits, files = {}, {}
    for sp in IMG_SPLITS:
        d = DATA / "images" / sp
        names = sorted(p.name for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if d.exists() else []
        info, boxes = {}, 0
        for n in names:
            lab = DATA / "labels" / sp / (Path(n).stem + ".txt")
            bs = []
            if lab.exists():
                for line in lab.read_text().splitlines():
                    p = line.split()
                    if len(p) == 5:
                        bs.append([float(v) for v in p[1:]])
            boxes += len(bs)
            info[n] = (len(bs), max((b[2] for b in bs), default=0))
        random.Random(7).shuffle(names)
        kinds = collections.Counter(n.split("_")[0] for n in names) if sp.endswith("_neg") else {}
        splits[sp] = {"images": len(names), "boxes": boxes, "kinds": dict(kinds), "source": IMG_SPLITS[sp]}
        files[sp] = (names, info)
    with POOL_LOCK:
        POOL.update({"ready": True, "splits": splits, "files": files})


_photo_cache = {}


def photo_status(model, split):
    """Per-photo verdict from saved photo predictions: missed drones / false alarm boxes."""
    p = REVIEW / model / f"photos_{split}.json"
    if not p.exists():
        return None
    key = (model, split, p.stat().st_mtime)
    if key not in _photo_cache:
        preds = json.loads(p.read_text())
        out = {}
        for name, pb in preds.items():
            lab = DATA / "labels" / split / (Path(name).stem + ".txt")
            gts = []
            if lab.exists():
                for line in lab.read_text().splitlines():
                    q = line.split()
                    if len(q) == 5:
                        cx, cy, w, h = (float(v) for v in q[1:])
                        gts.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            missed = sum(1 for g in gts if not any(iou(b, g) >= 0.3 for b in pb))
            fa = sum(1 for b in pb if not any(iou(b, g) > 0.1 for g in gts))
            out[name] = {"missed": missed, "false_alarm": fa, "boxes": pb}
        _photo_cache.clear()
        _photo_cache[key] = out
    return _photo_cache[key]


def image_page(q):
    split = q.get("split", "train")
    page, per = int(q.get("page", 0)), min(120, int(q.get("per", 48)))
    with POOL_LOCK:
        names, info = POOL.get("files", {}).get(split, ([], {}))
    if q.get("kind"):
        names = [n for n in names if n.startswith(q["kind"] + "_")]
    if q.get("size"):
        lo, hi = next((a, b) for n, a, b in SIZE_BUCKETS if n == q["size"])
        names = [n for n in names if info[n][0] and lo <= info[n][1] < hi]
    st = photo_status(q["pred"], split) if q.get("pred") else None
    if q.get("result") and st is not None:
        names = [n for n in names if st.get(n, {}).get(q["result"], 0) > 0]
    items = []
    for n in names[page * per:(page + 1) * per]:
        it = {"name": n, "boxes": info[n][0]}
        if st is not None and n in st:
            it.update(missed=st[n]["missed"], false_alarm=st[n]["false_alarm"])
        items.append(it)
    return {"split": split, "total": len(names), "page": page, "items": items, "pred_available": st is not None}


def pool_payload():
    with POOL_LOCK:
        if not POOL["ready"]:
            return {"ready": False}
        splits = POOL["splits"]
    preds = {m: [s for s in ("test", "val_neg") if (REVIEW / m / f"photos_{s}.json").exists()] for m in MODELS}
    return {"ready": True, "splits": splits, "photo_predictions": preds,
            "sizes": [{"id": n, "label": f"{n} ({int(a*100)}-{int(b*100)}% of frame width)" if b < 9 else f"{n} (over {int(a*100)}%)"} for n, a, b in SIZE_BUCKETS]}


def read_frame(key, f):
    c = clip_index().get(key)
    if not c:
        return None
    cap = cv2.VideoCapture(str(ROOT / c["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(f), c["frames"] - 1)))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def make_thumb(q):
    kind, w = q.get("kind", ""), max(80, min(1600, int(q.get("w", "360"))))
    pred = q.get("pred") if q.get("pred") in MODELS else None
    if kind == "img":
        split, name = q.get("split", ""), q.get("name", "")
        if split not in IMG_SPLITS or "/" in name or "\\" in name or name.startswith("."):
            return None
        src = DATA / "images" / split / name
        key = f"img_{split}_{Path(name).stem}_{w}_{pred or ''}.jpg"
    elif kind == "clip":
        ck = q.get("clip", "")
        if ck not in clip_index():
            return None
        f = int(q.get("f", -1))
        if f < 0:
            f = int(clip_index()[ck]["frames"] * 0.35)
        key = f"clip_{ck}_{f}_{w}.jpg"
    else:
        return None
    cache = THUMBS / key
    if cache.exists() and not pred:
        return cache.read_bytes()
    if kind == "img":
        img = cv2.imread(str(src)) if src.exists() else None
        if img is None:
            return None
        H, W = img.shape[:2]
        sc = w / W
        img = cv2.resize(img, (w, int(H * sc)), interpolation=cv2.INTER_AREA)
        h = img.shape[0]
        lab = DATA / "labels" / split / (src.stem + ".txt")
        if lab.exists():
            for line in lab.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:])
                    x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    pad = 4 if (x2 - x1) < 20 else 0
                    brackets(img, x1 - pad, y1 - pad, x2 + pad, y2 + pad, TEAL, 2)
        if pred:
            st = photo_status(pred, split) or {}
            for b in st.get(name, {}).get("boxes", []):
                x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
                cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), ORANGE, 2, cv2.LINE_AA)
                if w >= 600:
                    cv2.putText(img, f"{pred} {b[4]*100:.0f}%", (x1 - 3, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2, cv2.LINE_AA)
    else:
        img = read_frame(ck, f)
        if img is None:
            return None
        H, W = img.shape[:2]
        img = cv2.resize(img, (w, int(H * w / W)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    if not pred:
        THUMBS.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(buf.tobytes())
    return buf.tobytes()


# ------------------------------------------------------------------ collect: flags, labels, camera

FILE_LOCK = threading.Lock()


def flags_load():
    return read_json(OURS / "flags.json", [])


def flags_save(fl):
    OURS.mkdir(exist_ok=True)
    (OURS / "flags.json").write_text(json.dumps(fl, indent=1))


def flag_add(d):
    if d.get("clip") not in clip_index() or d.get("kind") not in ("miss", "false_alarm"):
        return {"ok": False, "error": "bad flag"}
    with FILE_LOCK:
        fl = flags_load()
        f = int(d["frame"])
        old = next((x for x in fl if x["clip"] == d["clip"] and x["frame"] == f and x["kind"] == d["kind"]), None)
        if old:
            return {"ok": True, "count": len(fl), "existing": True, "flag": old}
        new = {"id": f"{int(time.time()*1000)}", "clip": d["clip"], "frame": f, "kind": d["kind"],
               "output": d.get("output"), "time": time.time()}
        fl.append(new)
        flags_save(fl)
    return {"ok": True, "count": len(fl), "existing": False, "flag": new}


def flag_undo_last():
    """Remove the most recently added flag (kept on disk, so it works after a reload too)."""
    with FILE_LOCK:
        fl = flags_load()
        if not fl:
            return {"ok": False, "error": "nothing to undo", "count": 0}
        last = max(fl, key=lambda x: x.get("time", 0))
        fl = [x for x in fl if x["id"] != last["id"]]
        flags_save(fl)
    return {"ok": True, "removed": last, "count": len(fl), "next": max(fl, key=lambda x: x.get("time", 0)) if fl else None}


def flag_remove(fid):
    with FILE_LOCK:
        fl = [x for x in flags_load() if x["id"] != fid]
        flags_save(fl)
    return {"ok": True, "count": len(fl)}


def gt_for(key, f):
    g = read_json(REVIEW / "gt" / f"{key}.json")
    if not g or f >= len(g["gt"]):
        return None
    return g["gt"][f]  # [x1,y1,x2,y2] normalized, or None when the drone is out of view


def yolo_line(b):
    return f"0 {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}"


def flags_export():
    """Write flagged frames to our_data/hard_examples (images + YOLO labels where the truth is known)."""
    out_i, out_l = OURS / "hard_examples" / "images", OURS / "hard_examples" / "labels"
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    idx = clip_index()
    n = labeled = 0
    rows = []
    for x in flags_load():
        img = read_frame(x["clip"], x["frame"])
        if img is None:
            continue
        name = f"{x['clip']}_{x['frame']:06d}"
        g = idx[x["clip"]]["group"]
        label = None
        own = labels_get(x["clip"]).get(str(x["frame"]))
        if own is not None:
            label = "\n".join(yolo_line(b) for b in own)
        elif g == "dut" and (REVIEW / "gt" / f"{x['clip']}.json").exists():
            b = gt_for(x["clip"], x["frame"])
            label = yolo_line(b) if b else ""
        elif g == "negatives":
            label = ""
        if label is not None:  # only frames whose truth we know; an unlabeled miss saved as "no drone" would teach the opposite
            cv2.imwrite(str(out_i / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_l / f"{name}.txt").write_text(label + ("\n" if label else ""))
            labeled += 1
        rows.append({"image": f"{name}.jpg", "clip": x["clip"], "frame": x["frame"], "kind": x["kind"], "labeled": label is not None})
        n += 1
    (OURS / "hard_examples" / "manifest.json").write_text(json.dumps(rows, indent=1))
    return {"ok": True, "exported": labeled, "need_labels": n - labeled, "folder": "our_data/hard_examples",
            "need": [r for r in rows if not r["labeled"]]}


def labels_dir(key):
    return OURS / "labels" / key


def labels_get(key):
    d = labels_dir(key)
    out = {}
    if d.exists():
        for p in d.glob("*.txt"):
            bs = []
            for line in p.read_text().splitlines():
                q = line.split()
                if len(q) == 5:
                    cx, cy, w, h = (float(v) for v in q[1:])
                    bs.append([round(cx - w / 2, 5), round(cy - h / 2, 5), round(cx + w / 2, 5), round(cy + h / 2, 5)])
            out[str(int(p.stem))] = bs
    return out


def label_save(d):
    key, f = d.get("clip"), int(d.get("frame", -1))
    if key not in clip_index() or f < 0:
        return {"ok": False, "error": "bad clip or frame"}
    ld, idir = labels_dir(key), OURS / "images" / key
    lp, ip = ld / f"{f:06d}.txt", idir / f"{f:06d}.jpg"
    with FILE_LOCK:
        if d.get("delete"):
            lp.unlink(missing_ok=True)
            ip.unlink(missing_ok=True)
        else:
            boxes = [[max(0.0, min(1.0, float(v))) for v in b[:4]] for b in d.get("boxes", [])]
            boxes = [b for b in boxes if b[2] - b[0] > 0.001 and b[3] - b[1] > 0.001]
            ld.mkdir(parents=True, exist_ok=True)
            idir.mkdir(parents=True, exist_ok=True)
            lp.write_text("".join(yolo_line(b) + "\n" for b in boxes))  # empty file = "no drone in this frame"
            if not ip.exists():
                img = read_frame(key, f)
                if img is not None:
                    cv2.imwrite(str(ip), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"ok": True, "labeled_frames": len(list(ld.glob("*.txt"))) if ld.exists() else 0}


def classification_label_save(d):
    if not isinstance(d, dict):
        return {"ok": False, "error": "annotation must be a JSON object"}
    key, frame = d.get("clip"), d.get("frame")
    clip = clip_index().get(key) if isinstance(key, str) else None
    if not clip or type(frame) is not int or not 0 <= frame < clip["frames"]:
        return {"ok": False, "error": "bad clip or frame"}
    try:
        with FILE_LOCK:
            image_bytes = None
            if not d.get("delete"):
                image = read_frame(key, frame)
                if image is None:
                    return {"ok": False, "error": "Could not read this frame; nothing was saved."}
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    return {"ok": False, "error": "Could not encode this frame; nothing was saved."}
                image_bytes = encoded.tobytes()
            annotation = classification_data.save_annotation(
                OURS, key, frame, d.get("boxes", []), d.get("object_label"),
                d.get("split"), d.get("source_id"), image_bytes=image_bytes,
                no_drone=d.get("no_drone", False), delete=d.get("delete", False))
        return {"ok": True, "annotation": annotation}
    except (ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def collection_export(d):
    if not isinstance(d, dict):
        return {"ok": False, "error": "export must be a JSON object"}
    try:
        with FILE_LOCK:
            return {"ok": True, **classification_data.export_collection(OURS, d.get("name"))}
    except (ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


_models = {}


def load_model(model):
    if model not in _models:
        from ultralytics import YOLO
        _models[model] = YOLO(str(weights(model)))
    return _models[model]


def best_usable_model():
    for m in ("v3", "v2", "v1"):
        if run_status(m)["usable"]:
            return m
    return None


def propose(key, f):
    """Model's suggested boxes for one frame: from a saved prediction file if there is one,
    otherwise a single live inference (refused while a training job is running)."""
    for oid in ("v3", "v2", "v1"):
        p = REVIEW / oid / f"{key}.json"
        if p.exists():
            d = read_json(p)
            if d and f < len(d["boxes"]):
                return {"ok": True, "source": f"saved {oid} predictions", "boxes": [b[:5] for b in d["boxes"][f]]}
    if training_active():
        return {"ok": False, "reason": "A training job is running, so the model is not being run. Draw the box by hand, or try again when training ends."}
    m = best_usable_model()
    img = read_frame(key, f)
    if m is None or img is None:
        return {"ok": False, "reason": "No finished model or frame available."}
    r = load_model(m).predict(img, imgsz=960, conf=0.25, verbose=False)[0]
    h, w = img.shape[:2]
    boxes = [[round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5), round(float(c), 3)]
             for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    return {"ok": True, "source": f"live {m}", "boxes": boxes}


_live = {}


def live():
    """The real-time engine (model runs on every frame as the clip plays). Loaded on first use."""
    if "m" not in _live:
        import sys
        sys.path.insert(0, str(HERE))
        import live_engine
        live_engine.ENGINE.start()
        _live["m"] = live_engine
    return _live["m"]


_camnames = {}


def camera_list():
    """Real cameras macOS reports, numbered the way OpenCV opens them (Desk View and screen capture skipped)."""
    if "t" not in _camnames or time.time() - _camnames["t"] > 15:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                                 capture_output=True, text=True, timeout=10).stderr
            vid = out.split("video devices:")[1].split("audio devices:")[0]
            real = [n.strip() for _, n in re.findall(r"\[(\d+)\] (.+)", vid)
                    if "Desk View" not in n and not n.startswith("Capture screen")]
        except Exception:
            real = ["Camera 0", "Camera 1"]
        _camnames.update(t=time.time(), v=[{"index": i, "name": n, "builtin": "MacBook" in n or "FaceTime" in n} for i, n in enumerate(real)])
    return _camnames["v"]


def ours_dataset():
    """What our own labeled data adds up to (frames we labeled + exported hard examples)."""
    labs = list((OURS / "labels").glob("*/*.txt")) if (OURS / "labels").exists() else []
    boxes = empty = 0
    for p in labs:
        n = sum(1 for l in p.read_text().splitlines() if l.strip())
        boxes += n
        empty += (n == 0)
    hard = list((OURS / "hard_examples" / "labels").glob("*.txt")) if (OURS / "hard_examples" / "labels").exists() else []
    return {"frames": len(labs), "with_drone": len(labs) - empty, "no_drone": empty, "boxes": boxes,
            "clips": len({p.parent.name for p in labs}), "hard_examples": len(hard),
            "yaml": (OURS / "dataset.yaml").exists()}


def ours_dataset_write():
    s = ours_dataset()
    if not s["frames"] and not s["hard_examples"]:
        return {"ok": False, "error": "Nothing labeled yet."}
    train = ["images"] if s["frames"] else []
    if s["hard_examples"]:
        train.append("hard_examples/images")
    val = DATA / "images" / "val"
    (OURS / "dataset.yaml").write_text(
        f"# Our own labeled frames (made by the dashboard's Collect tab)\npath: {OURS}\ntrain: [{', '.join(train)}]\n"
        + (f"val: {val}\n" if val.exists() else "val: [images]\n") + "names:\n  0: drone\n")
    cmd = (".venv/bin/yolo detect train model=runs/train/dut_v3_n960/weights/best.pt data=our_data/dataset.yaml "
           "imgsz=960 epochs=10 batch=16 device=mps project=$PWD/runs/train name=ours_v4")
    return {"ok": True, **ours_dataset(), "file": "our_data/dataset.yaml", "command": cmd}


class Camera(threading.Thread):
    """Reads a camera only while someone is watching or recording. Records raw frames."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.viewers = 0
        self.index = 0
        self.jpeg = None
        self.error = None
        self.overlay = False
        self.rec = None  # dict while recording
        self.last = None

    def start_rec(self, meta):
        with self.lock:
            if self.rec:
                return {"ok": False, "error": "already recording"}
            name = time.strftime("ours_%Y%m%d_%H%M%S")
            self.rec = {"name": name, "meta": meta, "writer": None, "frames": 0, "t0": time.time()}
        return {"ok": True, "name": name}

    def stop_rec(self):
        with self.lock:
            rec, self.rec = self.rec, None
        if not rec:
            return {"ok": False, "error": "not recording"}
        if rec["writer"] is not None:
            rec["writer"].release()
        if not rec["frames"]:
            return {"ok": False, "error": self.error or "no frames were captured from the camera"}
        d = ROOT / "footage" / "ours"
        secs = time.time() - rec["t0"]
        meta = {**rec["meta"], "date": time.strftime("%Y-%m-%d %H:%M"), "frames": rec["frames"], "seconds": round(secs, 1),
                "camera_index": self.index}
        (d / f"{rec['name']}.json").write_text(json.dumps(meta, indent=1))
        r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/review/make_web_clips.py")], capture_output=True, text=True)
        return {"ok": True, "name": rec["name"], "clip": f"ours__{rec['name']}", "frames": rec["frames"],
                "seconds": round(secs, 1), "web_copy": (WEB / f"ours__{rec['name']}.mp4").exists(), "log": r.stdout[-300:]}

    def run(self):
        cap, cur = None, None
        while True:
            with self.lock:
                want = self.viewers > 0 or self.rec is not None
                idx = self.index
            if not want:
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.2)
                continue
            if cap is None or cur != idx:
                if cap is not None:
                    cap.release()
                cap, cur = cv2.VideoCapture(idx), idx
                if not cap.isOpened():
                    self.error = f"Camera {idx} could not be opened. On a Mac, allow camera access for the app that started the dashboard (System Settings > Privacy & Security > Camera)."
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                    continue
                self.error = None
            ok, frame = cap.read()
            if not ok:
                self.error = f"Camera {idx} opened but returned no picture (permission or the camera is in use)."
                time.sleep(0.3)
                continue
            self.error = None
            with self.lock:
                rec = self.rec
                if rec is not None:
                    if rec["writer"] is None:
                        d = ROOT / "footage" / "ours"
                        d.mkdir(parents=True, exist_ok=True)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        rec["writer"] = cv2.VideoWriter(str(d / f"{rec['name']}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                        fps if fps and 1 < fps <= 120 else 30.0, frame.shape[1::-1])
                    rec["writer"].write(frame)
                    rec["frames"] += 1
            out = frame
            if self.overlay and not rec and not training_active():
                m = best_usable_model()
                if m:
                    r = load_model(m).predict(frame, imgsz=960, conf=0.35, verbose=False)[0]
                    out = frame.copy()
                    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                        brackets(out, int(x1), int(y1), int(x2), int(y2), TEAL, 3)
                        cv2.putText(out, f"DRONE {c*100:.0f}%", (int(x1), max(18, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAL, 2, cv2.LINE_AA)
            h, w = out.shape[:2]
            if w > 1280:
                out = cv2.resize(out, (1280, int(h * 1280 / w)))
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.jpeg = buf.tobytes()

    def state(self):
        with self.lock:
            rec = self.rec
            return {"index": self.index, "error": self.error, "overlay": self.overlay, "training_active": training_active(),
                    "recording": bool(rec), "rec_frames": rec["frames"] if rec else 0,
                    "rec_seconds": round(time.time() - rec["t0"], 1) if rec else 0, "has_picture": self.jpeg is not None}


CAM = Camera()


def coverage():
    idx = clip_index()
    rows = collections.OrderedDict()
    for key, c in idx.items():
        g = c["group"]
        meta = read_json(ROOT / "footage" / "ours" / f"{c['name']}.json", {}) if g == "ours" else {}
        k = (GROUP_INFO.get(g, {}).get("source", g), GROUP_INFO.get(g, {}).get("label", g), meta.get("background") or "unlabeled")
        r = rows.setdefault(k, {"source": k[0], "kind": k[1], "background": k[2], "clips": 0, "seconds": 0.0, "labeled_frames": 0})
        r["clips"] += 1
        r["seconds"] += c["frames"] / c["fps"]
        if g == "dut":
            r["labeled_frames"] += c.get("gt_frames") or 0
        own_frames = set(labels_get(key)) | set(classification_data.load_annotations(OURS, key))
        r["labeled_frames"] += len(own_frames)
    return list(rows.values())


def collect_payload():
    idx = clip_index()
    annotations = {key: classification_data.load_annotations(OURS, key) for key in idx}
    labeled = {key: len(set(labels_get(key)) | set(annotations[key])) for key in idx}
    rows = [annotation for labels in annotations.values() for annotation in labels.values()]
    splits = collections.Counter(annotation["split"] for annotation in rows)
    classes = collections.Counter(annotation["label"] for annotation in rows if not annotation["no_drone"])
    classification = {"frames": len(rows), "clips": sum(bool(labels) for labels in annotations.values()),
                      "development": splits["development"], "test": splits["test"],
                      "no_drone": sum(annotation["no_drone"] for annotation in rows), "labels": dict(classes)}
    return {"flags": flags_load(), "labeled": labeled, "coverage": coverage(), "camera": CAM.state(),
            "cameras": camera_list(), "dataset": ours_dataset(), "class_labels": list(classification_data.LABELS),
            "classifications": {key: len(labels) for key, labels in annotations.items()}, "classification_dataset": classification,
            "ours": [k for k, c in idx.items() if c["group"] == "ours"],
            "recordings": [{"clip": k, "name": c["name"], "seconds": round(c["frames"] / c["fps"], 1), "labeled": labeled.get(k, 0), "classified": len(annotations[k]),
                            "meta": read_json(ROOT / "footage" / "ours" / f"{c['name']}.json", {})}
                           for k, c in sorted(idx.items(), reverse=True) if c["group"] == "ours"]}


def recording_delete(name):
    """Delete one of OUR recordings and everything made from it. Public footage cannot be deleted here."""
    import re as _re
    import shutil
    if not _re.fullmatch(r"ours_\d{8}_\d{6}", name or ""):
        return {"ok": False, "error": "only our own recordings can be deleted here"}
    key = f"ours__{name}"
    removed = []
    with FILE_LOCK:
        for p in [ROOT / "footage" / "ours" / f"{name}.mp4", ROOT / "footage" / "ours" / f"{name}.json", WEB / f"{key}.mp4"]:
            if p.exists():
                p.unlink()
                removed.append(str(p.relative_to(ROOT)))
        for d in [OURS / "labels" / key, OURS / "images" / key]:
            if d.exists():
                shutil.rmtree(d)
                removed.append(str(d.relative_to(ROOT)))
        classification_dirs = [OURS / "classification" / kind / key for kind in ("images", "labels")]
        removed.extend(str(directory.relative_to(ROOT)) for directory in classification_dirs if directory.exists())
        classification_data.remove_clip_annotations(OURS, key)
        for t in THUMBS.glob(f"clip_{key}_*.jpg"):
            t.unlink()
        idx = clip_index()
        if key in idx:
            del idx[key]
            (WEB / "index.json").write_text(json.dumps(idx, indent=1))
        fl = flags_load()
        flags_save([x for x in fl if x["clip"] != key])
    return {"ok": True, "removed": removed}


# ------------------------------------------------------------------ http

STATIC = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_body(self, body, ctype, cache="no-store", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send_body(json.dumps(obj).encode(), "application/json", code=code)

    def send_file_range(self, path, ctype):
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 18, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self.send_body((HERE / "index.html").read_bytes(), STATIC[".html"])
            if p == "/favicon.ico":
                return self.send_body(b"", "image/x-icon", "max-age=86400")
            if p.startswith("/static/"):
                f = HERE / Path(p).name
                if f.suffix in STATIC and f.exists():
                    return self.send_body(f.read_bytes(), STATIC[f.suffix])
                return self.send_error(404)
            if p.startswith("/media/"):
                f = WEB / Path(unquote(p)).name
                if f.suffix == ".mp4" and f.exists():
                    return self.send_file_range(f, "video/mp4")
                return self.send_error(404)
            if p == "/live/stats":
                L = live()
                with L.ENGINE.lock:
                    return self.send_json(L.ENGINE.stats)
            if p == "/live/sources":
                L = live()
                ok = {m: run_status(m)["usable"] for m in MODELS}
                return self.send_json({"sources": L.list_sources(), "training_active": training_active(),
                                       "camera_names": {f"camera:{i}": n for i, n in L.camera_names().items()},
                                       "models": [{"id": k, "label": v[1], "ok": v[0].exists() and ok.get(k, True)} for k, v in L.MODELS.items()]})
            if p == "/live/set":
                live().ENGINE.update(**q)
                return self.send_json({"ok": True})
            if p == "/live/stream.mjpg":
                return self.live_stream()
            if p == "/api/clips":
                return self.send_json(clips_payload())
            if p == "/api/pred":
                f = REVIEW / q.get("out", "") / f"{q.get('clip', '')}.json"
                if q.get("out") in OUT and q.get("clip") in clip_index() and f.exists():
                    return self.send_body(f.read_bytes(), "application/json")
                return self.send_json({"error": "no predictions"}, 404)
            if p == "/api/gt":
                f = REVIEW / "gt" / f"{q.get('clip', '')}.json"
                if q.get("clip") in clip_index() and f.exists():
                    return self.send_body(f.read_bytes(), "application/json")
                return self.send_json({"error": "no ground truth"}, 404)
            if p == "/api/results":
                return self.send_json(results_payload())
            if p == "/api/pool":
                return self.send_json(pool_payload())
            if p == "/api/images":
                if q.get("split", "train") not in IMG_SPLITS:
                    return self.send_error(400)
                return self.send_json(image_page(q))
            if p == "/thumb":
                body = make_thumb(q)
                if body is None:
                    return self.send_error(404)
                return self.send_body(body, "image/jpeg", "no-store" if q.get("pred") else "max-age=86400")
            if p == "/api/collect":
                return self.send_json(collect_payload())
            if p == "/api/labels":
                key = q.get("clip", "")
                if key not in clip_index():
                    return self.send_json({"labels": {}, "classifications": {}})
                return self.send_json({"labels": labels_get(key), "classifications": classification_data.load_annotations(OURS, key)})
            if p == "/api/propose":
                return self.send_json(propose(q.get("clip", ""), int(q.get("frame", 0))))
            if p == "/api/cam":
                return self.send_json(CAM.state())
            if p == "/cam/stream.mjpg":
                return self.cam_stream()
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # keep the server alive and say what went wrong
            try:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:
                pass

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self.send_json({"ok": False, "error": "bad json"}, 400)
        try:
            if u.path == "/api/flag":
                if d.get("undo_last"):
                    return self.send_json(flag_undo_last())
                return self.send_json(flag_remove(d["remove"]) if d.get("remove") else flag_add(d))
            if u.path == "/api/flags/export":
                return self.send_json(flags_export())
            if u.path == "/api/ours/dataset":
                return self.send_json(ours_dataset_write())
            if u.path == "/api/recording/delete":
                return self.send_json(recording_delete(d.get("name")))
            if u.path == "/api/label":
                return self.send_json(label_save(d))
            if u.path == "/api/classification/label":
                return self.send_json(classification_label_save(d))
            if u.path == "/api/collections/export":
                return self.send_json(collection_export(d))
            if u.path == "/api/cam":
                with CAM.lock:
                    if "index" in d:
                        CAM.index = int(d["index"])
                        CAM.jpeg = None
                    if "overlay" in d:
                        CAM.overlay = bool(d["overlay"])
                if d.get("action") == "start":
                    return self.send_json(CAM.start_rec({k: str(d.get(k, ""))[:80] for k in ("location", "drone_type", "background", "notes", "source_id")}))
                if d.get("action") == "stop":
                    return self.send_json(CAM.stop_rec())
                return self.send_json({"ok": True, **CAM.state()})
            self.send_error(404)
        except Exception as e:
            self.send_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def live_stream(self):
        E = live().ENGINE
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with E.lock:
            E.viewers += 1
        last = None
        try:
            while True:
                jpg = E.jpeg
                if jpg is not None and jpg is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
                time.sleep(1 / 60)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with E.lock:
                E.viewers -= 1

    def cam_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with CAM.lock:
            CAM.viewers += 1
        last = None
        try:
            while True:
                jpg = CAM.jpeg
                if jpg is not None and jpg is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with CAM.lock:
                CAM.viewers -= 1
                if CAM.viewers == 0:
                    CAM.jpeg = None


if __name__ == "__main__":
    threading.Thread(target=build_pool, daemon=True).start()
    CAM.start()
    print(f"Dashboard: open http://localhost:{PORT}  (Ctrl-C to stop)")
    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):  # a browser closing a video request is not an error
            pass

    Server(("127.0.0.1", PORT), Handler).serve_forever()
