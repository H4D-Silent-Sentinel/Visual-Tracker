"""Test 2: real tracking. Run detector + ByteTrack on the 20 DUT Anti-UAV tracking
videos, where the true drone box is marked in every frame, and measure:

  on-target %     share of frames with a visible drone where our main box overlaps it (IoU >= 0.3)
  longest lock    longest run of consecutive on-target frames
  reacquire time  average length of a gap (drone visible, we're off it) before we get it back
  false alarms    share of frames where we drew a box but none of them overlaps the drone
  ID switches     times the tracker's ID for the drone changed while on target

Also writes plain .mp4 copies of each video to footage/dut/ for the dashboard.
Seconds assume 30 frames per second.

Run: .venv/bin/python scripts/evaluate/test_tracking.py [--model ...] [--no-mp4]
"""
import argparse, json
from pathlib import Path
import cv2, torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/train/dut_n960/weights/best.pt")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--iou", type=float, default=0.3)
ap.add_argument("--out", default="runs/tracking_test")
ap.add_argument("--tracker", default="bytetrack.yaml")
ap.add_argument("--det-conf", type=float, default=None, help="detector cutoff fed to the tracker (default: same as --conf)")
ap.add_argument("--videos", default="", help="comma list like video01,video07 (default: all)")
ap.add_argument("--no-mp4", action="store_true")
a = ap.parse_args()
FPS = 30.0
ROOT = Path.home() / "yolo-datasets/dut-anti-uav/tracking"
FR, GT = ROOT / "frames/Anti-UAV-Tracking-V0", ROOT / "gt/Anti-UAV-Tracking-V0GT"
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(a.model)
out_mp4 = Path("footage/dut"); out_mp4.mkdir(parents=True, exist_ok=True)
rep_dir = Path(a.out); rep_dir.mkdir(parents=True, exist_ok=True)


def iou(b, g):
    x1, y1 = max(b[0], g[0]), max(b[1], g[1]); x2, y2 = min(b[2], g[2]), min(b[3], g[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter
    return inter / u if u > 0 else 0


rows, T = [], dict(vis=0, on=0, boxed=0, fa=0, gaps=0, gap_frames=0, idsw=0, frames=0)
ONLY = set(v for v in a.videos.split(",") if v)
for vdir in sorted(p for p in FR.iterdir() if p.is_dir() and (not ONLY or p.name in ONLY)):
    gt = [list(map(float, l.split())) for l in (GT / f"{vdir.name}_gt.txt").read_text().split("\n") if l.strip()]
    imgs = sorted(vdir.glob("*.jpg"))
    model.predictor = None
    writer = None
    vis = on = fa = boxed = streak = best = gaps = gap_frames = cur_gap = idsw = 0
    last_id = None
    for k, (img, g) in enumerate(zip(imgs, gt)):
        fr = cv2.imread(str(img))
        if not a.no_mp4:
            h, w = fr.shape[:2]; s = min(1.0, 1280 / w)
            small = cv2.resize(fr, (int(w*s), int(h*s))) if s < 1 else fr
            if writer is None:
                writer = cv2.VideoWriter(str(out_mp4 / f"{vdir.name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, small.shape[1::-1])
            writer.write(small)
        r = model.track(fr, persist=True, imgsz=a.imgsz, conf=a.det_conf or a.conf, device=dev, tracker=a.tracker, verbose=False)[0]
        boxes = [(b.xyxy[0].tolist(), float(b.conf), int(b.id) if b.id is not None else None) for b in r.boxes]
        visible = g[2] > 0 and g[3] > 0
        gbox = [g[0], g[1], g[0]+g[2], g[1]+g[3]] if visible else None
        if boxes: boxed += 1
        hit = None
        if visible:
            vis += 1
            ok = [(bx, c, i) for bx, c, i in boxes if iou(bx, gbox) >= a.iou]
            if ok:
                hit = max(ok, key=lambda t: t[1])
                on += 1; streak += 1; best = max(best, streak)
                if cur_gap: gaps += 1; gap_frames += cur_gap; cur_gap = 0
                if last_id is not None and hit[2] is not None and hit[2] != last_id: idsw += 1
                if hit[2] is not None: last_id = hit[2]
            else:
                streak = 0; cur_gap += 1
        if boxes and (not visible or not any(iou(bx, gbox) > 0.1 for bx, _, _ in boxes)):
            fa += 1
    if writer: writer.release()
    n = len(imgs)
    row = {"video": vdir.name, "frames": n, "drone_visible_frames": vis,
           "on_target_pct": round(100*on/vis, 1) if vis else None,
           "longest_lock_s": round(best/FPS, 1),
           "losses_recovered": gaps, "avg_reacquire_s": round(gap_frames/gaps/FPS, 2) if gaps else None,
           "false_alarm_pct": round(100*fa/n, 1), "id_switches": idsw}
    rows.append(row); print(json.dumps(row), flush=True)
    for k2, v in dict(vis=vis, on=on, boxed=boxed, fa=fa, gaps=gaps, gap_frames=gap_frames, idsw=idsw, frames=n).items(): T[k2] += v
overall = {"videos": len(rows), "frames": T["frames"], "on_target_pct": round(100*T["on"]/T["vis"], 1),
           "avg_reacquire_s": round(T["gap_frames"]/T["gaps"]/FPS, 2) if T["gaps"] else None,
           "false_alarm_pct": round(100*T["fa"]/T["frames"], 1), "id_switches": T["idsw"],
           "median_longest_lock_s": sorted(r["longest_lock_s"] for r in rows)[len(rows)//2]}
print("ALL", json.dumps(overall))
(rep_dir / "report.json").write_text(json.dumps({"model": a.model, "tracker": a.tracker, "det_conf": a.det_conf or a.conf, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "videos": rows, "overall": overall}, indent=2))
