"""YOLO detect + track for the H4D visual tracker (runs on a Mac or a Jetson).

Examples:
  python track.py                         # webcam, all object types
  python track.py --source clip.mp4 --save
  python track.py --drone-like --range    # only bird/kite/airplane, show mil-relation range
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from drone_verifier import DroneVerifier

# COCO has no "drone" class yet; these are the closest stand-ins until we fine-tune.
DRONE_LIKE = {"airplane": 4, "bird": 14, "kite": 33}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO detect + track")
    p.add_argument("--source", default="0", help="0 = webcam, or a video/image path or URL")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (downloads on first use)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence cutoff (0-1)")
    p.add_argument("--imgsz", type=int, default=640, help="size YOLO resizes frames to")
    p.add_argument("--drone-like", action="store_true", help="only keep bird, kite, airplane")
    p.add_argument("--range", action="store_true", help="show mil-relation range on each box")
    p.add_argument("--target-size", type=float, default=0.35, help="target width in meters")
    p.add_argument("--hfov", type=float, default=60.0, help="camera horizontal field of view, degrees")
    p.add_argument("--save", action="store_true", help="save annotated video to runs/")
    p.add_argument("--no-show", action="store_true", help="don't open a window")
    p.add_argument("--verify", action="store_true", help="review new flying-object tracks with the local VLM scorer")
    p.add_argument("--verify-confidence", type=float, default=0.85, help="minimum VLM confidence for a confirmed yes/no")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"          # Jetson / NVIDIA
    if torch.backends.mps.is_available():
        return "mps"           # Apple Silicon GPU
    return "cpu"


def mil_range(box_w_px, img_w_px, size_m, hfov_deg):
    """Range (m) = size (m) x 1000 / mils, with mils measured from pixels."""
    mils_per_px = math.radians(hfov_deg) * 1000 / img_w_px
    mils = box_w_px * mils_per_px
    return size_m * 1000 / mils if mils > 0 else float("inf")


def main():
    a = parse_args()
    source = int(a.source) if a.source.isdigit() else a.source
    device = pick_device()
    classes = list(DRONE_LIKE.values()) if a.drone_like else None
    model = YOLO(a.model)
    verifier = DroneVerifier(a.verify_confidence) if a.verify else None
    if a.verify and classes is None:
        classes = [cid for cid, name in model.names.items() if name.lower() in {'drone', 'uav', *DRONE_LIKE}]
        if not classes:
            raise ValueError('Verification requires drone, UAV, airplane, bird, or kite model classes')
    print(f"model={a.model} device={device} source={a.source}")

    writer, frames, t0, fps, last = None, 0, time.time(), 0.0, None
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    results = model.track(source=source, stream=True, persist=True, conf=a.conf,
                          imgsz=a.imgsz, classes=classes, device=device,
                          tracker="bytetrack.yaml", verbose=False)
    for r in results:
        now = time.time()
        if last is not None:
            inst = 1 / max(now - last, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        last = now
        targets = [{'id': int(b.id) if b.id is not None else None, 'box': b.xyxy[0].tolist(),
                    'conf': float(b.conf), 'label': r.names[int(b.cls)]} for b in r.boxes]
        if verifier:
            verifier.update(r.orig_img, targets)
        frame = r.orig_img.copy() if verifier else r.plot()
        h, w = frame.shape[:2]

        for target in targets:
            x1, y1, x2, y2 = target['box']
            tid, name = target['id'], target['label']
            line = f"#{tid} {name} {target['conf']:.2f}"
            if verifier:
                v = target['verification']
                status = v['status']
                color = (60, 220, 120) if status == 'confirmed' else (60, 80, 240) if status == 'rejected' else (60, 200, 240)
                line = 'FALSE REVIEW: NOT A DRONE' if status == 'rejected' else 'DRONE LOCK' if status == 'confirmed' else status.upper()
                line += f" #{tid}" + (f" VLM {v['confidence']:.0%}" if 'confidence' in v else '')
                if status != 'rejected':
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, line, (int(x1), max(20, int(y1)-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            if a.range:
                line += f"  ~{mil_range(x2 - x1, w, a.target_size, a.hfov):.0f} m"
                cv2.putText(frame, line, (int(x1), int(y2) + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 200), 2)

        frames += 1
        cv2.putText(frame, f"{device}  {fps:.1f} fps  {len(r.boxes)} objects", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if a.save:
            if writer is None:
                path = out_dir / f"track_{int(time.time())}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                print(f"saving to {path}")
            writer.write(frame)
        if not a.no_show:
            cv2.imshow("YOLO tracker (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t0
    if verifier:
        verifier.close()
    if writer:
        writer.release()
        if frames == 1:  # single image: also save a still
            cv2.imwrite(str(out_dir / "last_frame.jpg"), frame)
    cv2.destroyAllWindows()
    print(f"frames={frames} total_time={elapsed:.1f}s avg_fps={frames / elapsed if elapsed else 0:.1f}")


if __name__ == "__main__":
    main()
