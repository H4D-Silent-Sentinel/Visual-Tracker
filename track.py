"""YOLO detect + track for the H4D visual tracker (runs on a Mac or a Jetson).

Examples:
  python track.py                         # webcam, all object types
  python track.py --source clip.mp4 --save
  python track.py --drone-like --range    # only bird/kite/airplane, show mil-relation range
  python track.py --phone-standin --base  # no drone? a cell phone plays the drone (labelled as a stand-in)
  python track.py --model runs/train/dut_v3_n960/weights/best.pt --imgsz 960 \
      --base 192.168.1.20 --yaw 92 --pitch 5  # also send tracks to the Silent Sentinel base
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from base_link import PHONE_CLASS, PHONE_SIZE_M, BaseLink, phone_range
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
    p.add_argument("--phone-standin", action="store_true",
                   help="testing without a drone: track cell phones with the stock model and report them as the drone")
    p.add_argument("--verify", action="store_true", help="review new flying-object tracks with the local VLM scorer")
    p.add_argument("--base", nargs="?", const="", default=None, metavar="HOST[:PORT]",
                   help="send drone tracks to the Silent Sentinel base (default 127.0.0.1:47010, or $VT_BASE)")
    p.add_argument("--src", help="camera id in the base site file (default post-1, or $VT_SRC)")
    p.add_argument("--yaw", type=float, default=0.0, help="true bearing the camera faces, degrees")
    p.add_argument("--pitch", type=float, default=0.0, help="camera tilt, degrees up")
    p.add_argument("--cam-lat", type=float, default=0.0, help="camera latitude (omit for a fixed post in the site file)")
    p.add_argument("--cam-lon", type=float, default=0.0)
    p.add_argument("--cam-alt", type=float, default=0.0, help="camera height, meters")
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
    if a.phone_standin:
        a.model, a.verify, a.target_size = "yolo11n.pt", False, PHONE_SIZE_M  # stock COCO has "cell phone"
    model = YOLO(a.model)
    if a.phone_standin:
        classes = [cid for cid, name in model.names.items() if name == PHONE_CLASS]
    verifier = DroneVerifier() if a.verify else None
    if a.verify and classes is None:
        classes = [cid for cid, name in model.names.items() if name.lower() in {'drone', 'uav', *DRONE_LIKE}]
        if not classes:
            raise ValueError('Verification requires drone, UAV, airplane, bird, or kite model classes')
    link = BaseLink(a.base or None, a.src) if a.base is not None else None
    link_settings = {"hfov": a.hfov, "yaw": a.yaw, "pitch": a.pitch, "verify": a.verify, "standin": a.phone_standin,
                     "cam_lat": a.cam_lat, "cam_lon": a.cam_lon, "cam_alt": a.cam_alt}
    print(f"model={a.model} device={device} source={a.source}"
          + (f" base={link.addr[0]}:{link.addr[1]} src={link.src}" if link else ""))

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
        if link:
            for target in targets:
                x1, _, x2, _ = target['box']
                target['range_m'] = (phone_range(target['box'], w, a.hfov) if a.phone_standin
                                     else mil_range(x2 - x1, w, a.target_size, a.hfov))
            link.publish(targets, w, h, link_settings, now)
            cue = link.latest_cue(link_settings)
            if cue and "rel_az_deg" in cue:
                side = "R" if cue["rel_az_deg"] >= 0 else "L"
                cv2.putText(frame, f"BASE T{cue['track_id']}: {abs(cue['rel_az_deg']):.0f} deg {side}, "
                            f"{cue['dist_m']:.0f} m (+/-{cue['sigma_m']:.0f})", (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        for target in targets:
            x1, y1, x2, y2 = target['box']
            tid, name = target['id'], target['label']
            line = f"#{tid} {name} {target['conf']:.2f}"
            if verifier:
                v = target['verification']
                status = v['status']
                color = (60, 220, 120) if status == 'confirmed' else (60, 80, 240) if status == 'rejected' else (60, 200, 240)
                line = 'DRONE NOT DETECTED' if status == 'rejected' else 'DRONE DETECTED' if status == 'confirmed' else status.upper()
                line += f" #{tid}" + (f" ({v['label']})" if 'label' in v else '')
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
    if link:
        link.close()
    if writer:
        writer.release()
        if frames == 1:  # single image: also save a still
            cv2.imwrite(str(out_dir / "last_frame.jpg"), frame)
    cv2.destroyAllWindows()
    print(f"frames={frames} total_time={elapsed:.1f}s avg_fps={frames / elapsed if elapsed else 0:.1f}")


if __name__ == "__main__":
    main()
