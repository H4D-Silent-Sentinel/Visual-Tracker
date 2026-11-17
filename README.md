# Visual-Tracker

Computer vision to track drones: YOLO11 detection + ByteTrack tracking. Runs on the Mac's GPU (Apple "mps") and will use CUDA on the Jetson.


## Setup (fresh clone)

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_dut.py            # needs the DUT Anti-UAV zips in ~/yolo-datasets/dut-anti-uav/zips
.venv/bin/python prepare_negatives.py      # bird/kite photos from COCO
.venv/bin/python prepare_tracking_split.py # v3 training frames
.venv/bin/python make_web_clips.py         # browser copies of footage/ (footage is not in git)
.venv/bin/python dashboard/app.py          # http://localhost:8000
```
Trained weights for v1, v2, v3 are in `runs/train/*/weights/best.pt`, test reports in `runs/false_alarms*/` and `runs/tracking_test*/`, and per-frame review files in `runs/review/`. Footage and datasets are not in the repo (size and licenses); see Data below for sources.

## Run it

Open Terminal and paste:

```
cd ~/Documents/School/Fall\ 2026/Hacking\ 4\ Defense/yolo-tracker
source .venv/bin/activate
python track.py
```

Press **q** in the video window to quit. The first webcam run asks for camera permission for Terminal: allow it, then run again.

## Options

| Command | What it does |
|---|---|
| `python track.py` | Webcam, every object type |
| `python track.py --drone-like` | Only bird, kite, airplane (closest to "drone" until we fine-tune) |
| `python track.py --range --target-size 0.35 --hfov 70` | Show mil-relation range on each box |
| `python track.py --source clip.mp4 --save` | Run on a video file, save the annotated video to `runs/` |
| `python track.py --conf 0.5` | Stricter confidence cutoff (fewer false alarms, more misses) |

## Notes

- The stock model doesn't know "drone" yet. Next step: fine-tune on drone footage.
- `--hfov` is your camera's horizontal field of view in degrees. MacBook webcams are roughly 70–80°; check your camera's spec for accurate ranges.
- Measured on this Mac: ~68 frames per second on a still image.

## Test on recorded drone footage

```
python eval_clips.py --model yolo11n.pt --imgsz 1280
```

Runs YOLO on the clips in `footage/` (public-domain / CC-licensed videos from Wikimedia Commons), saves annotated videos plus `report.json` to `runs/eval_<model>_<size>/`.

### First results (Nov 15, stock COCO models, no drone training yet)

| Clip | yolo11n @640 | yolo11n @1280 | yolo11m @1280 |
|---|---|---|---|
| Medium drone in flight over water (Balikatan, 0:47–0:53) | 86% of frames | 96% | 98% |
| Small drone against the sky (BLM training, 0:00–0:30)* | 53% | 71% | 41% |
| Close-up quad on the ground (Makers Party) | 1% | 0% | 0% |
| Speed on this Mac (M4 Pro GPU) | 63–93 fps | 48–57 fps | 14–16 fps |

\* Inflated: the last few seconds are title cards, and some "hits" are false boxes (a hat labeled "kite").

What it shows:
- The stock model has no "drone" class. A drone in the sky gets labeled **airplane**, which works surprisingly well.
- A drone against clutter (ground, people) is missed or called "boat"/"bench".
- Bigger input (1280) finds small drones better; the bigger model (m) is 4x slower and no better on the small drone.
- Next step: fine-tune on labeled drone images so it learns a real "drone" class.

Footage sources (Wikimedia Commons): "Balikatan 2026- 25th Infantry Division flies training drones..." (U.S. Army, public domain); "BLM drone training (34216006294)" (BLM, public domain); "Demonstration of Unmanned aerial vehicle during Makers Party 2013, Bangalore" (CC BY-SA 3.0).

## Trained drone model (Nov 16)

Fine-tuned `yolo11n` on the DUT Anti-UAV detection set (5,200 training images, one class: `drone`) at 960 px on the Mac's GPU. Training stopped after 24 of 30 planned passes (the Mac slept); the saved model is from pass 24.

Weights: `runs/train/dut_n960/weights/best.pt`

**Held-out test set (2,200 DUT images the model never saw):** precision 96%, recall 91%, mAP50 0.95, mAP50-95 0.64.

**Recorded clips, stock model vs trained model** (share of frames with a drone box):

| Clip | Stock yolo11n @1280 | Trained @960 | Trained @1280 |
|---|---|---|---|
| Medium drone in flight over water | 96% (as "airplane") | 99% | 98% |
| Small drone against the sky* | 71% (incl. false "kite" boxes) | 79% | 79% |
| Close-up quad on the ground among people | 0% | 46% | 62% |
| Speed on this Mac | 48–57 fps | 53–80 fps | 48–55 fps |

\* The last ~2 s of that window are title cards, so ~93% is the most it could score.

What changed:
- It now outputs a real `drone` label instead of borrowing "airplane".
- The false boxes (a hat as "kite", people as "boat") are gone in these clips.
- It finds drones against clutter it used to miss completely, though the busy ground scene is still the weakest case.

Run it:
```
python track.py --model runs/train/dut_n960/weights/best.pt --imgsz 960 --source footage/army_balikatan_fpv.webm
python eval_clips.py --model runs/train/dut_n960/weights/best.pt --imgsz 960
```
(Don't add `--drone-like` with the trained model; that flag picks COCO classes.)

Reproduce: `python prepare_dut.py`, then
`yolo detect train model=yolo11n.pt data=~/yolo-datasets/dut-anti-uav/dut.yaml imgsz=960 epochs=30 batch=16 device=mps`

Dataset: DUT Anti-UAV (Zhao et al., Dalian University of Technology, arXiv:2205.10851), Apache-2.0, via huggingface.co/datasets/aegean-ai/dut-anti-uav.

## Dashboard

```
.venv/bin/python dashboard/app.py
```
Then open http://localhost:8000. Start it from Terminal (not from another app) so macOS can ask for camera permission. Ctrl-C to stop.

| Tab | What it does |
|---|---|
| **Clip Review** | Plays a recorded clip in a normal video player and draws the model's boxes from saved per-frame files, so you can scrub, step frame by frame, and switch model output (v1, v2, v3, with or without the "hold" tracker) at the same moment. The timeline marks misses and false alarms where the truth is known. Demo clips (trees, open sky, birds/kites) are at the top of the list. |
| **Results** | v1 / v2 / v3 compared on the 10 held-out tracking clips only, false alarms on the bird/kite clips, precision/recall on the 2,200 test photos, and training status read from the logs. Click a row to open that clip. |
| **Data Pool** | Every photo and video, with source and split. Filter by split, drone size, and (once generated) each model's misses and false alarms. |
| **Collect** | Live camera with a record button, a labeler (model proposes a box, you accept or redraw), the frames you flagged in Clip Review, and a coverage table. |

Keys in Clip Review: Space play/pause, arrows or `,` `.` step a frame, `[` `]` change clip, `R` restart, `W` worst moment, `1`-`5` model output, `G` truth box, `M` flag miss, `F` flag false alarm.

Files it reads (none are written by the dashboard except under `our_data/` and `footage/ours/`):

| File | Made by |
|---|---|
| `footage/web/*.mp4`, `index.json` | `make_web_clips.py`: browser-playable copies, frame-for-frame with the originals |
| `runs/review/<output>/<clip>.json` | `make_review.py`: the model's boxes for every frame (generated Nov 17 for all 34 clips and 5 outputs) |
| `runs/review/gt/*.json` | `make_review.py --gt-only`: true drone box per frame for the 20 tracking clips |
| `runs/false_alarms*/`, `runs/tracking_test*/` | `test_false_alarms.py`, `test_tracking.py` |
| `dashboard/demo_clips.json` | hand-picked demo clips |
| `our_data/labels/`, `our_data/flags.json`, `our_data/hard_examples/` | your own labels and flagged frames; never mixed into the training folders automatically |

Held-out rule: v3 trained on frames from 10 of the 20 tracking clips (see `prepare_tracking_split.py`), so v1/v2/v3 are only compared on the other 10. The dashboard reads the split from the actual training folder.

`dashboard/ui_test.py` re-runs the browser checks. The previous live-stream dashboard is kept in `dashboard/old_live_tracker/`.

## Test 1: false alarms on birds and kites (Nov 16)

`python test_false_alarms.py` runs the trained model on 8 clips with no drones (flocks, seagulls, terns, pigeons, kites; `footage/negatives/`, Wikimedia Commons, CC/public domain).

| Confidence cutoff | Frames with a false "drone" box |
|---|---|
| 0.25 | 42% |
| 0.35 (default) | 31% |
| 0.50 | 18% |

**The model calls birds and kites drones.** Expected: its training set had only drones. Worst: tern colony and a flock at Sylt (~54%), kites (51%). Least: pigeons (9%), seagulls up close (15%). Fix: retrain with bird/kite images labeled as background or as their own class. See `runs/false_alarms/worst_false_alarms.jpg`.

## Test 2: real tracking on 20 videos (Nov 16)

`python test_tracking.py` runs detection + ByteTrack on the 20 DUT Anti-UAV tracking videos (24,804 frames, true drone box marked in every frame). Seconds assume 30 fps.

| Metric | Result |
|---|---|
| On target (our box overlaps the drone, IoU ≥ 0.3) | **91.1%** of frames where the drone is visible |
| Average time to reacquire after losing it | **0.17 s** |
| Frames with a false box (nothing on the drone) | 1.6% |
| Median longest continuous lock per video | 16.7 s |
| Track ID switches | 189 |

- 9 of 20 videos at 97–100% on target; worst is video16 (55% on target, 20% false boxes).
- Losses are short (a few frames), but the tracker often gives the same drone a new ID after a loss. A fast tracker that holds the box between detections would fix most of this.
- Caveat: these videos come from the same lab as the training photos, so scenes may look similar. Treat 91% as optimistic until we test on our own footage.

The 20 videos are also saved as plain .mp4 in `footage/dut/` for the dashboard.

## v2: retrained with bird and kite photos (Nov 17)

v2 = v1 trained 10 more passes with 2,100 COCO bird/kite photos marked "no drone" (`prepare_negatives.py`). Weights: `runs/train/dut_neg_n960/weights/best.pt`.

| Test | v1 | v2 |
|---|---|---|
| Test 1: frames with a false "drone" box on 8 bird/kite clips, cutoff 0.35 | 30.7% | **7.1%** |
| Same, cutoff 0.25 / 0.50 | 41.6% / 17.5% | 11.1% / 3.1% |
| Labeled test photos (2,200): precision / recall / mAP50 | 96% / 91% / 0.95 | 96.2% / 91.2% / 0.956 |
| Test 2, all 20 tracking clips: on target | 91.1% | 91.7% |
| Test 2: frames with a wrong-object box | 1.6% | **5.2% (worse)** |
| Test 2: track ID switches | 189 | 202 |
| Test 2, 10 held-out clips only: on target / wrong-object box / ID switches | 93.9% / 0.8% / 134 | 94.7% / 4.9% / 137 |

- Bird and kite false alarms fell by about three quarters with no loss on the labeled drone photos. They are not gone: 7% of frames still get a box, and pigeons got slightly worse (8.5% to 10.8%).
- On the tracking clips v2 draws a box on something that is not the drone more often than v1 did. Not yet explained; check in Clip Review once per-frame files exist.
- Per clip it is mixed: video17 (trees) 75.8% to 84.7%, video04 86.8% to 73.0%.

### "Hold" tracker setting (`bytetrack_hold.yaml`, detector cutoff 0.08)

| v2, 10 held-out clips | default tracker | hold |
|---|---|---|
| On target | 94.7% | 96.0% |
| Frames with a wrong-object box | 4.9% | 11.9% |
| ID switches | 137 | 139 |

On all 20 clips: on target 91.7% to 94.1%, wrong-object box 5.2% to 10.5%. It holds the drone more, at the cost of roughly double the wrong-object boxes.

## v3: retrained with cluttered-background frames (Nov 17)

v3 = v2 trained 10 more passes with every 4th frame of 10 DUT tracking clips (2,393 frames, `prepare_tracking_split.py`), to help with drones in front of trees and buildings. Weights: `runs/train/dut_v3_n960/weights/best.pt`.

Because v3 trained on 10 tracking clips, **tracking is compared only on the other 10** (video01, 03, 07, 08, 09, 10, 12, 14, 17, 18; 15,245 frames). v3's score on all 20 (95.7%) is inflated and should not be quoted.

| Test | v1 | v2 | v3 | v3 + hold |
|---|---|---|---|---|
| Bird/kite clips: frames with a false "drone" box (cutoff 0.35) | 30.7% | 7.1% | **1.8%** | not run |
| Labeled test photos (2,200): precision / recall | 96% / 91% | 96.2% / 91.2% | 95.6% / 91.3% | |
| Held-out tracking: on target | 93.9% | **94.7%** | 93.3% | 95.0% |
| Held-out tracking: frames with a wrong-object box | 0.8% | 4.9% | 1.0% | 5.6% |
| Held-out tracking: ID switches | 134 | 137 | **120** | 128 |
| video17 (drone in front of trees): on target | 75.8% | 84.7% | 80.4% | 84.1% |

- Birds and kites are mostly solved: kites 0%, tern colony 0.6%, worst clip (Sylt flock) 4.1%.
- Trees are not: the cluttered-background frames did not carry over to new tree scenes. v3 is below v2 on video17, video01 and video12.
- v3 undid v2's rise in wrong-object boxes and has the fewest ID switches.
- The hold tracker trades about 1.5-2 points of on-target time for roughly 5x the wrong-object boxes, on every model.
- Default model in the dashboard is now v3. Next step for trees: our own rooftop footage, and a tracker that holds a drone without locking onto other objects.
