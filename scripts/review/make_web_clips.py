"""Make browser-playable (H.264, constant frame rate) copies of every clip for Clip Review.

Writes footage/web/<key>.mp4 and footage/web/index.json with the verified frame count, fps and
size of each copy. Frame N of a copy is frame N of its source, which is what lets the dashboard
line up boxes, ground truth and the video scrubber.
Run:  .venv/bin/python scripts/review/make_web_clips.py        (CPU only; safe to run while training)
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC, OUT = ROOT / "footage", ROOT / "footage" / "web"
EXT = {".webm", ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ogv"}
GT = Path.home() / "yolo-datasets/dut-anti-uav/tracking/gt/Anti-UAV-Tracking-V0GT"


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", str(p)],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return {"frames": int(s["nb_read_frames"]), "fps": round(int(a) / int(b), 4), "w": s["width"], "h": s["height"]}


def key_of(p):
    rel = p.relative_to(SRC)
    group = rel.parts[0] if len(rel.parts) > 1 else "demo"
    return group, f"{group}__{p.stem}"


def main():
    OUT.mkdir(exist_ok=True)
    idx_path = OUT / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    srcs = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in EXT and "web" not in p.relative_to(SRC).parts)
    for p in srcs:
        group, key = key_of(p)
        dst = OUT / f"{key}.mp4"
        if key in index and dst.exists() and index[key].get("src_mtime") == p.stat().st_mtime:
            continue
        src = probe(p)
        vf = "scale='if(gte(iw,ih),min(1280,iw),-2)':'if(gte(iw,ih),-2,min(1280,ih))',format=yuv420p"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(p), "-an", "-vf", vf, "-fps_mode", "passthrough" if group == "dut" else "cfr",
               "-r" if group != "dut" else "-g", str(src["fps"]) if group != "dut" else "15",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "15", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
        web = probe(dst)
        gt_frames = None
        if group == "dut":
            g = GT / f"{p.stem}_gt.txt"
            gt_frames = sum(1 for l in g.read_text().split("\n") if l.strip()) if g.exists() else None
        index[key] = {"key": key, "group": group, "name": p.stem, "source": str(p.relative_to(ROOT)), "file": f"footage/web/{key}.mp4",
                      "src_mtime": p.stat().st_mtime, "src_frames": src["frames"], "gt_frames": gt_frames, **web,
                      "timebase_ok": web["frames"] == (gt_frames if gt_frames is not None else web["frames"])}
        idx_path.write_text(json.dumps(index, indent=1))
        print(key, web, "gt", gt_frames, "OK" if index[key]["timebase_ok"] else "FRAME COUNT MISMATCH", flush=True)
    print("done", len(index), "clips")


if __name__ == "__main__":
    main()
