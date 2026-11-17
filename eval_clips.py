"""Run YOLO on recorded drone clips and summarize how often it finds the drone.

Usage:
  python eval_clips.py                       # default model and clips
  python eval_clips.py --model yolo11s.pt --imgsz 1280

There are no hand-drawn answer boxes for these clips, so "hit rate" means:
the share of frames where YOLO reported at least one drone-like object
(airplane, bird, kite, or any 'drone' class in a fine-tuned model).
Every frame in the chosen time windows shows a drone, so this is a rough
detection rate. It can overcount if YOLO boxes something else as a bird.
"""
import argparse, collections, json, time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# (file, start_s, end_s, description)
CLIPS = [
    ("footage/blm_drone_training.webm", 0, 30, "small drone against the sky (hard)"),
    ("footage/army_balikatan_fpv.webm", 47, 53, "medium drone in flight over water"),
    ("footage/makers_party_uav.webm", 0, 50, "close-up quadcopter (easy)"),
]
DRONE_LIKE = {"airplane", "bird", "kite", "drone", "uav"}


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    dev = device()
    model = YOLO(a.model)
    tag = f"{Path(a.model).stem}_{a.imgsz}"
    out_dir = Path("runs") / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model, "imgsz": a.imgsz, "conf": a.conf, "device": dev, "clips": []}

    for path, s0, s1, desc in CLIPS:
        cap = cv2.VideoCapture(path)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s0 * fps_in))
        n_frames = int((s1 - s0) * fps_in)
        writer = None
        hits, labels, confs, frames, t_infer = 0, collections.Counter(), [], 0, 0.0
        model.predictor = None  # reset tracker between clips
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time()
            r = model.track(frame, persist=True, imgsz=a.imgsz, conf=a.conf,
                            device=dev, tracker="bytetrack.yaml", verbose=False)[0]
            t_infer += time.time() - t
            frames += 1
            names = [r.names[int(c)] for c in r.boxes.cls]
            labels.update(names)
            drone_confs = [float(c) for n, c in zip(names, r.boxes.conf) if n.lower() in DRONE_LIKE]
            if drone_confs:
                hits += 1
                confs.append(max(drone_confs))
            plot = r.plot()
            cv2.putText(plot, f"{tag}  {'DRONE-LIKE FOUND' if drone_confs else 'no drone-like box'}",
                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (60, 220, 120) if drone_confs else (60, 60, 230), 3)
            if writer is None:
                h, w = plot.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / (Path(path).stem + ".mp4")),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(plot)
        if writer:
            writer.release()
        res = {
            "clip": Path(path).name, "what": desc, "window_s": [s0, s1], "frames": frames,
            "hit_rate_pct": round(100 * hits / frames, 1) if frames else 0,
            "avg_conf_when_hit": round(sum(confs) / len(confs), 2) if confs else None,
            "avg_fps": round(frames / t_infer, 1) if t_infer else 0,
            "labels_seen": dict(labels.most_common(6)),
        }
        report["clips"].append(res)
        print(json.dumps(res))

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"saved annotated videos + report.json to {out_dir}")


if __name__ == "__main__":
    main()
