"""Live YOLO tracking dashboard for the H4D drone tracker.

Run:   .venv/bin/python dashboard/app.py      then open http://localhost:8000
Stop:  Ctrl-C in the Terminal window.

Three tabs:
  Live tracker  YOLO + ByteTrack on a video or camera, streamed to the browser (MJPEG) with live stats
  Data pool     every image and video we train and test on, with thumbnails and per-video test results
  Results       old model vs. bird/kite-trained model on Test 1 (false alarms) and Test 2 (tracking)
Standard library only, besides the packages track.py already uses.
"""
import collections
import csv
import json
import math
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from drone_verifier import DroneVerifier

HERE = Path(__file__).resolve().parent
PORT = 8000
DATA = Path.home() / "yolo-datasets" / "dut-anti-uav" / "yolo"
THUMBS = ROOT / "runs" / "thumbs"
MODELS = {  # id: (weights, label)
    "v3": (ROOT / "runs/train/dut_v3_n960/weights/best.pt", "v3"),
    "v2": (ROOT / "runs/train/dut_neg_n960/weights/best.pt", "v2"),
    "v1": (ROOT / "runs/train/dut_n960/weights/best.pt", "v1"),
    "stock": (ROOT / "yolo11n.pt", "Stock"),
}
REPORTS = {  # model id -> (Test 1 dir, Test 2 dir, training run)
    "v1": ("runs/false_alarms", "runs/tracking_test", "runs/train/dut_n960"),
    "v2": ("runs/false_alarms_v2", "runs/tracking_test_v2", "runs/train/dut_neg_n960"),
    "v3": ("runs/false_alarms_v3", "runs/tracking_test_v3", "runs/train/dut_v3_n960"),
}
STOCK_DRONE_LIKE = [4, 14, 33]  # COCO airplane, bird, kite
VIDEO_EXT = {".webm", ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ogv"}
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TEAL, ORANGE, WHITE = (197, 224, 63), (61, 107, 255), (240, 240, 240)
GROUPS = {"dut": "Tracking test (DUT)", "negatives": "No-drone test (birds/kites)", "": "Demo clips"}
CLIP_NOTES = {
    "army_173rd_counter_uav.webm": "U.S. Army counter-UAV training",
    "army_1ad_counter_uas.webm": "U.S. Army 1st Armored Div. counter-UAS",
    "army_balikatan_fpv.webm": "FPV drone at Balikatan exercise (drone in flight 47–53 s)",
    "blm_drone_liftoff.webm": "BLM drone liftoff",
    "blm_drone_training.webm": "BLM drone pilot training",
    "makers_party_uav.webm": "Small UAV flying at a makers fair",
}


def model_ok(m):
    return m in MODELS and MODELS[m][0].exists()


def default_model():
    # newest model that has finished training and been tested
    for m in ("v3", "v2"):
        if model_ok(m) and read_json(REPORTS[m][0]) and read_json(REPORTS[m][1]):
            return m
    return next(m for m in ("v1", "stock", "v2") if model_ok(m))


def list_sources():
    vids = [str(p.relative_to(ROOT)) for p in (ROOT / "footage").rglob("*") if p.suffix.lower() in VIDEO_EXT and "web" not in p.relative_to(ROOT).parts]
    vids.sort(key=lambda v: (0 if "/dut/" in v else 2 if "/negatives/" in v else 1, v))
    return vids + [f"camera:{i}" for i in camera_names()]


_cams = {}


def camera_names():
    """Real cameras macOS reports, in OpenCV's order: {index: name}. Skips Desk View and screen capture."""
    if "t" not in _cams or time.time() - _cams["t"] > 15:
        import re, subprocess
        names = {}
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                                 capture_output=True, text=True, timeout=10).stderr
            vid = out.split("video devices:")[1].split("audio devices:")[0]
            # OpenCV numbers only real cameras (no Desk View, no screen capture), in macOS's order,
            # so its index is the position in this filtered list, not ffmpeg's number.
            real = [n.strip() for _, n in re.findall(r"\[(\d+)\] (.+)", vid)
                    if "Desk View" not in n and not n.startswith("Capture screen")]
            names = dict(enumerate(real))
        except Exception:
            names = {0: "Camera 0", 1: "Camera 1", 2: "Camera 2"}
        _cams.update(t=time.time(), names=names)
    return _cams["names"]


def group_of(rel):
    parts = Path(rel).parts
    return parts[1] if len(parts) > 2 else ""


def mil_range(box_w_px, img_w_px, size_m, hfov_deg):
    mils = box_w_px * math.radians(hfov_deg) * 1000 / img_w_px
    return size_m * 1000 / mils if mils > 0 else None


def brackets(img, x1, y1, x2, y2, color, t):
    L = max(6, int(min(x2 - x1, y2 - y1) * 0.28))
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x2, y2, -1, -1), (x1, y2, 1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)


def read_json(rel):
    p = ROOT / rel / "report.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------- data pool

IMG_SPLITS = {"train": "drone", "val": "drone", "test": "drone", "train_neg": "neg", "val_neg": "neg"}
POOL = {"ready": False}
POOL_LOCK = threading.Lock()


def build_pool():
    """Count images/boxes per split and read video metadata (runs once in the background)."""
    splits = {}
    files = {}
    for sp in IMG_SPLITS:
        d = DATA / "images" / sp
        names = sorted(p.name for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if d.exists() else []
        boxes = 0
        for n in names:
            lab = DATA / "labels" / sp / (Path(n).stem + ".txt")
            if lab.exists():
                boxes += sum(1 for line in lab.read_text().splitlines() if line.strip())
        random.Random(7).shuffle(names)
        files[sp] = names
        kinds = collections.Counter(n.split("_")[0] for n in names) if IMG_SPLITS[sp] == "neg" else {}
        splits[sp] = {"images": len(names), "boxes": boxes, "kinds": dict(kinds)}
    videos = []
    for rel in list_sources():
        if rel.startswith("camera:"):
            continue
        cap = cv2.VideoCapture(str(ROOT / rel))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fps = fps if fps and fps > 1 else 30.0
        g = group_of(rel)
        videos.append({"path": rel, "name": Path(rel).stem, "file": Path(rel).name, "group": g,
                       "group_label": GROUPS.get(g, g), "note": CLIP_NOTES.get(Path(rel).name, ""),
                       "w": w, "h": h, "fps": round(fps, 1),
                       "seconds": round(n / fps, 1) if n and n > 0 else None})
    with POOL_LOCK:
        POOL.update({"ready": True, "splits": splits, "files": files, "videos": videos})


def video_results():
    """Per-video numbers from Test 1 / Test 2 for each model that has been tested."""
    out = {}
    for m, (t1, t2, _) in REPORTS.items():
        r1, r2 = read_json(t1), read_json(t2)
        if r2:
            for v in r2["videos"]:
                out.setdefault(f"footage/dut/{v['video']}.mp4", {})[m] = {
                    "on_target_pct": v["on_target_pct"], "false_alarm_pct": v["false_alarm_pct"],
                    "id_switches": v["id_switches"], "longest_lock_s": v["longest_lock_s"]}
        if r1:
            for c in r1["clips"]:
                out.setdefault(f"footage/negatives/{c['clip']}", {})[m] = {
                    "fa35": c["false_alarm_pct@0.35"], "fa50": c["false_alarm_pct@0.5"]}
    return out


def pool_summary():
    with POOL_LOCK:
        if not POOL["ready"]:
            return {"ready": False}
        res = video_results()
        vids = [dict(v, results=res.get(v["path"], {})) for v in POOL["videos"]]
        return {"ready": True, "splits": POOL["splits"], "videos": vids}


def image_page(split, page, per, kind=None):
    with POOL_LOCK:
        names = POOL.get("files", {}).get(split, [])
    if kind:
        names = [n for n in names if n.startswith(kind + "_")]
    out = []
    for n in names[page * per:(page + 1) * per]:
        lab = DATA / "labels" / split / (Path(n).stem + ".txt")
        nb = sum(1 for l in lab.read_text().splitlines() if l.strip()) if lab.exists() else 0
        out.append({"name": n, "boxes": nb})
    return {"split": split, "total": len(names), "page": page, "items": out}


def make_thumb(q):
    """Thumbnail JPEG for a dataset image (with its drone boxes drawn) or a video frame. Cached on disk."""
    kind, w = q.get("kind", ""), max(80, min(1600, int(q.get("w", "360"))))
    if kind == "img":
        split, name = q.get("split", ""), q.get("name", "")
        if split not in IMG_SPLITS or "/" in name or "\\" in name or name.startswith("."):
            return None
        src = DATA / "images" / split / name
        key = f"img_{split}_{Path(name).stem}_{w}.jpg"
    elif kind == "vid":
        src = (ROOT / q.get("path", "")).resolve()
        if not str(src).startswith(str((ROOT / "footage").resolve())) or src.suffix.lower() not in VIDEO_EXT:
            return None
        key = f"vid_{src.parent.name}_{src.stem}_{w}.jpg"
    else:
        return None
    cache = THUMBS / key
    if cache.exists():
        return cache.read_bytes()
    if not src.exists():
        return None
    if kind == "img":
        img = cv2.imread(str(src))
        if img is None:
            return None
        H, W = img.shape[:2]
        lab = DATA / "labels" / split / (src.stem + ".txt")
        scale = w / W
        img = cv2.resize(img, (w, int(H * scale)), interpolation=cv2.INTER_AREA)
        if lab.exists():
            for line in lab.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:])
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * H * scale)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * H * scale)
                    pad = 4 if (x2 - x1) < 20 else 0  # make tiny drones visible
                    brackets(img, x1 - pad, y1 - pad, x2 + pad, y2 + pad, TEAL, 2)
    else:
        cap = cv2.VideoCapture(str(src))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if n and n > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.35))
        ok, img = cap.read()
        cap.release()
        if not ok:
            return None
        H, W = img.shape[:2]
        img = cv2.resize(img, (w, int(H * w / W)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    THUMBS.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(buf.tobytes())
    return buf.tobytes()


def results_summary():
    out = {}
    for m, (t1, t2, run) in REPORTS.items():
        row = {"exists": model_ok(m), "label": MODELS[m][1]}
        r1, r2 = read_json(t1), read_json(t2)
        if r1:
            row["test1"] = r1["overall"]
            row["test1_clips"] = r1["clips"]
        if r2:
            row["test2"] = r2["overall"]
        rc = ROOT / run / "results.csv"
        if rc.exists():
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(rc.open())]
            if rows:
                best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
                row["train"] = {"epochs_done": len(rows), "best_epoch": int(float(best["epoch"])),
                                "precision": float(best["metrics/precision(B)"]), "recall": float(best["metrics/recall(B)"]),
                                "map50": float(best["metrics/mAP50(B)"]), "map50_95": float(best["metrics/mAP50-95(B)"])}
                args = ROOT / run / "args.yaml"
                if args.exists():
                    for line in args.read_text().splitlines():
                        if line.startswith("epochs:"):
                            row["train"]["epochs_planned"] = int(line.split(":")[1])
        out[m] = row
    return out


# ---------------------------------------------------------------- live engine

class Engine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        srcs = list_sources()
        default = next((s for s in srcs if s.endswith("dut/video09.mp4")), srcs[0])
        self.settings = {"source": default, "model": default_model(), "conf": 0.35, "imgsz": 960,
                         "size_m": 0.35, "hfov": 60.0, "paused": False, "tracker": "default", "verify": False}
        self.restart = True
        self.models = {}
        self.jpeg = None
        self.stats = {}
        self.viewers = 0
        self.verifier = None
        self.primary_id = None
        self.reset_counters()

    def update(self, **kw):
        with self.lock:
            if "restart" in kw:
                self.restart = True
            for k, v in kw.items():
                if k not in self.settings:
                    continue
                cur = self.settings[k]
                if isinstance(cur, bool):
                    v = str(v).lower() in ("1", "true", "yes")
                elif isinstance(cur, int):
                    v = int(float(v))
                elif isinstance(cur, float):
                    v = float(v)
                if k == "model" and not model_ok(v):
                    continue
                if k == "tracker" and v not in ("default", "hold"):
                    continue
                if k in ("source", "model", "imgsz", "tracker", "verify") and v != cur:
                    self.restart = True
                self.settings[k] = v

    def get_model(self, name):
        if name not in self.models:
            self.models[name] = YOLO(str(MODELS[name][0]))
        return self.models[name]

    def open_source(self, src):
        cap = cv2.VideoCapture(int(src.split(":")[1]) if src.startswith("camera:") else str(ROOT / src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return cap, (fps if fps and fps > 1 else 30.0), src.startswith("camera:"), (int(n) if n and n > 0 else None)

    def reset_counters(self):
        self.frames = self.locked = self.streak = self.best_streak = self.reacq = 0
        self.ever_locked = False
        self.last_seen = None
        self.trail = collections.deque(maxlen=60)
        self.conf_hist = collections.deque(maxlen=150)
        self.fps = 0.0
        with self.lock:
            self.stats = {
                'error': None, 'status': 'IDLE', 'device': DEVICE, 'settings': dict(self.settings),
                'fps': 0.0, 'source_fps': 30.0, 'frames': 0, 'pos': 0, 'total': None,
                'locked_pct': 0.0, 'current_lock_s': 0.0, 'longest_lock_s': 0.0,
                'reacquisitions': 0, 'targets': [], 'primary': None, 'candidate': None,
                'verification': {'enabled': self.settings['verify']},
                'trail': [], 'conf_hist': [], 'frame_size': [0, 0],
            }

    def run(self):
        cap, total = None, None
        while True:
            with self.lock:
                s = dict(self.settings)
                restart, self.restart = self.restart, False
                idle = self.viewers == 0
            if idle and restart:
                if cap:
                    cap.release()
                    cap = None
                with self.lock:
                    self.restart = True
                    self.stats = {**self.stats, 'status': 'IDLE', 'settings': s}
                time.sleep(.2)
                continue
            if restart:
                if self.verifier:
                    self.verifier.reset()
                    if not s['verify']:
                        self.verifier.close()
                        self.verifier = None
                if cap:
                    cap.release()
                cap, fps_in, live, total = self.open_source(s["source"])
                model = self.get_model(s["model"])
                model.predictor = None  # fresh tracker
                self.reset_counters()
            if idle and not restart:  # nobody watching: don't burn the GPU (e.g. while training)
                if self.verifier:
                    self.verifier.close()
                    self.verifier = None
                if live and cap is not None and cap.isOpened():  # free the camera for the Collect tab
                    cap.release()
                    with self.lock:
                        self.restart = True
                with self.lock:
                    self.stats["status"] = "IDLE"
                time.sleep(0.2)
                continue
            if s["paused"]:
                time.sleep(0.05)
                continue
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if live:
                    name = camera_names().get(int(s["source"].split(":")[1]), s["source"])
                    why = ("could not be opened" if not cap.isOpened() else "is connected but sends no picture")
                    with self.lock:
                        self.stats = {**self.stats, "status": "NO_CAMERA", "settings": s,
                                      "error": f"{name} {why}. For an iPhone: lock it, hold it sideways, keep it near the Mac, "
                                               f"and make sure no other app (FaceTime, Zoom) is using it. If this keeps happening, "
                                               f"restart the dashboard from Terminal and allow camera access."}
                    time.sleep(0.5)
                    if not cap.isOpened():
                        cap.release()
                        cap, fps_in, live, total = self.open_source(s["source"])
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                model.predictor = None
                self.trail.clear()
                if self.verifier:
                    self.verifier.reset()
                continue
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            classes = STOCK_DRONE_LIKE if s["model"] == "stock" else None
            hold = s["tracker"] == "hold"  # keep a locked drone through weak detections (see bytetrack_hold.yaml)
            r = model.track(frame, persist=True, conf=min(s["conf"], 0.08) if hold else s["conf"], imgsz=s["imgsz"], classes=classes,
                            device=DEVICE, tracker=str(ROOT / "bytetrack_hold.yaml") if hold else "bytetrack.yaml", verbose=False)[0]
            h, w = frame.shape[:2]
            dets = []
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                rng = mil_range(x2 - x1, w, s["size_m"], s["hfov"])
                dets.append({"id": int(b.id) if b.id is not None else None, "conf": float(b.conf),
                             "label": r.names[int(b.cls)], "box": [x1, y1, x2, y2],
                             "range_m": round(rng, 1) if rng else None})
            dets.sort(key=lambda d: -d["conf"])
            if s['verify']:
                if self.verifier is None:
                    self.verifier = DroneVerifier()
                self.verifier.update(frame, dets)
                primary = next((d for d in dets if d['verification']['status'] == 'confirmed'), None)
            else:
                primary = dets[0] if dets else None
            candidate = primary or (dets[0] if dets else None)
            verification = {'enabled': s['verify'], **(candidate.get('verification', {}) if candidate else {})}

            self.frames += 1
            now = time.time()
            if primary:
                if s['verify'] and primary['id'] != self.primary_id:
                    self.trail.clear()
                self.primary_id = primary['id']
                if self.ever_locked and self.streak == 0:
                    self.reacq += 1
                self.locked += 1
                self.streak += 1
                self.best_streak = max(self.best_streak, self.streak)
                self.ever_locked = True
                self.last_seen = now
                x1, y1, x2, y2 = primary["box"]
                self.trail.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
            else:
                self.streak = 0
            self.conf_hist.append(round(primary["conf"], 3) if primary else 0)
            lost_for = (now - self.last_seen) if self.last_seen else None
            status = "LOCKED" if primary else ("COASTING" if lost_for is not None and lost_for < 1.0 else "SEARCHING")
            if s['verify'] and candidate and not primary:
                status = {'rejected': 'FALSE_REVIEW', 'pending': 'REVIEWING',
                          'error': 'VERIFY_ERROR'}.get(verification.get('status'), 'UNVERIFIED')

            out = frame.copy()
            pts = [(int(px * w), int(py * h)) for px, py in self.trail]
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], ORANGE, max(1, int(3 * i / len(pts))), cv2.LINE_AA)
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                main = d is primary
                col = TEAL if main else WHITE
                verdict = d.get('verification', {})
                rejected = verdict.get('status') == 'rejected'
                if s['verify']:
                    col = TEAL if verdict.get('status') == 'confirmed' else (70, 90, 235) if rejected else (70, 190, 235)
                if not rejected:
                    brackets(out, x1, y1, x2, y2, col, 4 if main else 2)
                if s['verify']:
                    tag = 'DRONE NOT DETECTED' if rejected else 'DRONE DETECTED' if verdict.get('status') == 'confirmed' else verdict.get('status', 'unverified').upper()
                    if 'label' in verdict:
                        tag += f" ({verdict['label']})"
                else:
                    tag = f"{d['label'].upper()} {d['conf']*100:.0f}%"
                if d["id"] is not None:
                    tag = f"ID {d['id']}  " + tag
                if d["range_m"]:
                    tag += f"  ~{d['range_m']:.0f} m"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = max(th + 10, y1 - 10)
                cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 12, ty + 6), (20, 14, 8), -1)
                cv2.putText(out, tag, (x1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            if w > 1280:
                out = cv2.resize(out, (1280, int(h * 1280 / w)))
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])

            dt = time.time() - t0
            inst = 1 / max(dt, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            with self.lock:
                if ok:
                    self.jpeg = buf.tobytes()
                self.stats = {
                    "error": None,
                    "status": status, "device": DEVICE, "settings": s, "fps": round(self.fps, 1),
                    "source_fps": round(fps_in, 1), "frames": self.frames, "pos": pos, "total": total,
                    "locked_pct": round(100 * self.locked / self.frames, 1),
                    "current_lock_s": round(self.streak / fps_in, 1),
                    "longest_lock_s": round(self.best_streak / fps_in, 1),
                    "reacquisitions": self.reacq, "targets": dets[:5], "primary": primary,
                    "candidate": candidate, "verification": verification,
                    "trail": [[round(x, 4), round(y, 4)] for x, y in self.trail],
                    "conf_hist": list(self.conf_hist), "frame_size": [w, h],
                }
            if not live:  # play files at real speed
                wait = 1 / fps_in - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)


ENGINE = Engine()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_body(self, body, ctype, cache="no-store"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj):
        self.send_body(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            self.send_body((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/stats":
            with ENGINE.lock:
                self.send_json(ENGINE.stats)
        elif u.path == "/sources":
            self.send_json({"sources": list_sources(),
                            "models": [{"id": k, "label": v[1], "ok": model_ok(k)} for k, v in MODELS.items()]})
        elif u.path == "/set":
            ENGINE.update(**q)
            self.send_json({"ok": True})
        elif u.path == "/api/pool":
            self.send_json(pool_summary())
        elif u.path == "/api/images":
            sp = q.get("split", "train")
            if sp not in IMG_SPLITS:
                return self.send_error(400)
            self.send_json(image_page(sp, int(q.get("page", 0)), min(120, int(q.get("per", 48))), q.get("kind")))
        elif u.path == "/api/results":
            self.send_json(results_summary())
        elif u.path == "/thumb":
            try:
                body = make_thumb(q)
            except Exception:
                body = None
            if body is None:
                return self.send_error(404)
            self.send_body(body, "image/jpeg", "max-age=86400")
        elif u.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with ENGINE.lock:
                ENGINE.viewers += 1
            last = None
            try:
                while True:
                    with ENGINE.lock:
                        jpg = ENGINE.jpeg
                    if jpg is not None and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(1 / 60)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with ENGINE.lock:
                    ENGINE.viewers -= 1
        else:
            self.send_error(404)


if __name__ == "__main__":
    threading.Thread(target=build_pool, daemon=True).start()
    ENGINE.start()
    print(f"Dashboard running on {DEVICE}: open http://localhost:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
