# Visual-Tracker

Computer vision to track drones: YOLO11 detection + ByteTrack tracking. Runs on the Mac's GPU (Apple "mps") and will use CUDA on the Jetson.


## Setup (fresh clone)

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/data/prepare_dut.py            # needs the DUT Anti-UAV zips in ~/yolo-datasets/dut-anti-uav/zips
.venv/bin/python scripts/data/prepare_negatives.py      # bird/kite photos from COCO
.venv/bin/python scripts/data/prepare_tracking_split.py # v3 training frames
.venv/bin/python scripts/review/make_web_clips.py       # browser copies of footage/ (footage is not in git)
.venv/bin/python dashboard/app.py          # http://localhost:8000
```
Trained weights for v1, v2, v3 are in `runs/train/*/weights/best.pt`, test reports in `runs/false_alarms*/` and `runs/tracking_test*/`, and per-frame review files in `runs/review/`. Footage and datasets are not in the repo (size and licenses); see Data below for sources.

## Project layout

Run the commands below from the project root. The main tracker and VLM service stay at the root, and the dashboard starts with `dashboard/app.py`. Helper scripts are grouped by their job:

| Folder | Scripts |
|---|---|
| `scripts/data/` | Dataset preparation: DUT conversion, bird/kite negatives, tracking split |
| `scripts/evaluate/` | Clip evaluation, false-alarm and tracking benchmarks, VLM calibration |
| `scripts/review/` | Browser clip conversion and per-frame review files |
| `scripts/train/` | `run_v3.sh` and `retest_v2.sh` training/evaluation sequences |
| `scripts/launch/`, `native/` | Build the native Mac camera-permission launcher |
| `tests/` | Unit tests; browser checks and screenshots are in `tests/browser/` |

## Local VLM verification

For a complete run, execute `./"Start Tracker.command"` from the project root, or double-click that file. It opens the native **Visual Tracker** Mac app, requests camera access on the main run loop, and starts Python only after permission is granted. Then it starts the cached VLM and dashboard and opens **http://localhost:8001/#live** with verification enabled. Choose a camera or clip in Live. Quit the native app, close its window, or press Ctrl-C in the launching Terminal to stop services started by that app.

The first run builds `.venv/Visual Tracker.app` with Apple's Swift command line tools (`xcode-select --install` if missing); subsequent runs reuse the signed local bundle. Safari receives an MJPEG stream and does not need camera permission for this dashboard. macOS grants access to **Visual Tracker**, which launches the camera-reading Python process. If access was previously denied, the native app offers **Open Camera Settings**; enable Visual Tracker there and relaunch. macOS does not repeat its native permission prompt after a denial.

This fixes OpenCV's `can not spin main run loop from other thread` error: authorization happens in the native app before workers start, and `OPENCV_AVFOUNDATION_SKIP_AUTH=1` is passed to those workers only after access is granted. The native dashboard uses port 8001 to avoid reusing an old Terminal-launched dashboard on port 8000. Startup messages are saved in `.venv/visual-tracker-native.log`.

YOLO and ByteTrack keep detecting and tracking. Optional VLM verification reviews a **new tracked flying-object candidate once**, after collecting its clearest crop for 0.5 seconds. Ordinary movement, box jitter, and elapsed time do not run the VLM again. A substantially larger/sharper view or reappearance after a lost identity is an update and can request a fresh review. Requests run in a background worker; completed verdicts are cached per track.

The default is **Qwen3.5-2B-4bit**, running locally through MLX. Its pinned checkpoint and runtime packages are in `requirements.txt`; the scorer downloads the roughly 1.72 GB of weights on its first startup. This is the conservative choice for an Apple Silicon Mac with 8 GB total RAM. Qwen3.8 starts above that memory budget; Gemma 4 E2B's standard MLX weights are roughly 3.55 GB before runtime overhead. Model comparisons and sources are in [docs/vlm-model-selection.md](docs/vlm-model-selection.md). Drone accuracy and total memory/FPS must be tested on the target laptop, not inferred from general benchmarks or weight size. The pinned MLX wheels require Apple Silicon and macOS 26 or newer; use Python 3.11 or newer.

Install `requirements.txt` as above, then start the scorer in one Terminal:

```sh
HF_HOME="$PWD/.venv/huggingface" .venv/bin/python vlm_server.py
```

Wait for `Scorer ready`, then use either the dashboard's **Live → Verify with local VLM** checkbox or the CLI in another Terminal:

```sh
.venv/bin/python track.py --verify --model runs/train/dut_v3_n960/weights/best.pt --imgsz 960
# Add --source path/to/video.mp4 for a recorded clip.
```

Verification is opt-in; Clip Review continues to display its saved YOLO predictions. The Live YOLO confidence cutoff is **0.35** (the CLI detector defaults to **0.25**, configurable with `--conf`). The VLM chooses one of 35 object labels and maps only `drone` to yes. There is **no additional VLM confidence cutoff** or percentage display; the former 0.85 gate and `--verify-confidence` option are removed.

Candidates are cropped from raw frames, padded 25%, and conventionally resized while preserving aspect ratio. The scorer bounds its actual image preprocessing to 384 pixels per side. It uses one selected crop, no generated chain of thought, and no generative super-resolution. Tiny candidates below 12 native pixels stay unverified. Up to 32 track records and eight queued reviews are retained to bound memory; expired/reset identities cannot inherit stale results.

**Drone detected** keeps the lock when the VLM labels the candidate `drone`. **Drone not detected** rejects other labels. While waiting, the panel says **Checking for drone**; failures say **Unable to check**. If YOLO never proposes a box, the panel explicitly says that no YOLO candidate was available rather than claiming the VLM rejected it. Restart/toggle verification after starting an unavailable scorer; failures do not cause timer-based retries.

The local `POST http://127.0.0.1:8002/verify` endpoint accepts `{"image":"<base64 JPEG crop>"}` and returns an allowed object label plus a binary drone decision, for example:

```json
{"is_drone":false,"label":"bird","model":"mlx-community/Qwen3.5-2B-4bit","prompt_hash":"...","model_revision":"..."}
```

The model reviews **YOLO-positive crops only**, once per new or meaningfully updated candidate. It classifies the central object among drone, bird, kite, airplane, helicopter, balloon, bat, insect, butterfly, cloud, leaf, branch, tree, flag, antenna, pole, person, car, truck, bus, bicycle, motorcycle, boat, ship, building, window, reflection, camera, satellite dish, paraglider, plastic bag, umbrella, wind turbine, smoke, and other. Python constructs JSON from the winning one-token menu code. The VLM does not generate percentages or an explanation. No second YOLO model is loaded.

[docs/vlm-prompts.md](docs/vlm-prompts.md) contains the **exact prompt**, class menu, and scoring policy. [docs/candidate-classification.md](docs/candidate-classification.md) describes the labeled classification test. The earlier confidence/calibration study and saved-score utility remain in the repo for research, but the categorical scorer does not apply that former binary calibration.

The supplied drone screenshot passed YOLO at both input sizes and the categorical VLM identified its crop as `drone`. A five-inference MPS timing check on the same screenshot measured **64.6 inference FPS at 640** and **57.3 at 960** on a 16 GiB Mac. These timings exclude capture, video encoding, and VLM reviews; they are not live-camera FPS or a dataset accuracy result. Lower input resolution can miss smaller targets. The categorical VLM's warm review took about **0.6 seconds** in a scorer-only check. Actual shared-GPU performance and false positives require a representative recorded collection; a synthetic blank crop was labeled `flag`, so category accuracy is still imperfect even though its binary result was non-drone.

Run the new unit/integration checks with `.venv/bin/python -m unittest discover -s tests -v`. Browser checks are in `tests/browser/`.

## Collect a dataset and test false positives

1. Launch `./"Start Tracker.command"` and open **Collect**. Record a clip, or select an existing one. Recording saves raw frames and pauses the preview model to reduce processing load.
2. Choose a **source/session ID** and **Development** or **Test** split. Keep clips from the same session in one split. Draw or correct a suggested box, choose its true label, and save with **S**. Press **N** only after checking that the entire frame contains no drones.
3. Enter a collection name and click **Export test collection**. This freezes images and manual annotations under `our_data/collections/<name>/`, with separate development/test manifests. Later edits or recording deletion leave that snapshot intact.
4. Keep the local VLM running and execute the command displayed after export, for example:

```sh
.venv/bin/python scripts/evaluate/test_classification.py \
  --manifest our_data/collections/flight-session-01/test.jsonl \
  --out runs/classification_flight-session-01.json
```

The report compares YOLO-only false positives with the final VLM drone confirmations, and records class confusion, missed YOLO proposals, VLM misses, and service errors separately. Include drones plus birds, kites, aircraft, and checked no-drone frames from independent sessions; tune prompts on development data and use the test split to compare changes. Manual class labels stay separate from legacy drone-training labels, and collection images remain local and ignored by git. See [docs/candidate-classification.md](docs/candidate-classification.md) for the format and metric definitions.

## How to Run:

Open Terminal and paste the following:

```
cd /path/to/Visual-Tracker
source .venv/bin/activate
python track.py
```

Press **q** in the video window to quit. The first webcam run asks for camera permission for Terminal: allow it, then run again.

## Options

| Command | Output |
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
python scripts/evaluate/eval_clips.py --model yolo11n.pt --imgsz 1280
```

Runs YOLO on the clips in `footage/` (public-domain / CC-licensed videos from Wikimedia Commons), saves annotated videos plus `report.json` to `runs/eval_<model>_<size>/`.

### First results (Nov 15, stock models, no drone training yet)

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
python scripts/evaluate/eval_clips.py --model runs/train/dut_n960/weights/best.pt --imgsz 960
```
(Don't add `--drone-like` with the trained model; that flag picks COCO classes.)

Reproduce: `python scripts/data/prepare_dut.py`, then
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
| `footage/web/*.mp4`, `index.json` | `scripts/review/make_web_clips.py`: browser-playable copies, frame-for-frame with the originals |
| `runs/review/<output>/<clip>.json` | `scripts/review/make_review.py`: the model's boxes for every frame (generated Nov 17 for all 34 clips and 5 outputs) |
| `runs/review/gt/*.json` | `scripts/review/make_review.py --gt-only`: true drone box per frame for the 20 tracking clips |
| `runs/false_alarms*/`, `runs/tracking_test*/` | `scripts/evaluate/test_false_alarms.py`, `scripts/evaluate/test_tracking.py` |
| `dashboard/demo_clips.json` | hand-picked demo clips |
| `our_data/labels/`, `our_data/flags.json`, `our_data/hard_examples/` | your own labels and flagged frames; never mixed into the training folders automatically |

Held-out rule: v3 trained on frames from 10 of the 20 tracking clips (see `scripts/data/prepare_tracking_split.py`), so v1/v2/v3 are only compared on the other 10. The dashboard reads the split from the actual training folder.

`tests/browser/ui_test.py` re-runs the browser checks. The previous live-stream dashboard is kept in `dashboard/old_live_tracker/`.

## Test 1: false alarms on birds and kites (Nov 16)

`python scripts/evaluate/test_false_alarms.py` runs the trained model on 8 clips with no drones (flocks, seagulls, terns, pigeons, kites; `footage/negatives/`, Wikimedia Commons, CC/public domain).

| Confidence cutoff | Frames with a false "drone" box |
|---|---|
| 0.25 | 42% |
| 0.35 (default) | 31% |
| 0.50 | 18% |

**The model calls birds and kites drones.** Expected: its training set had only drones. Worst: tern colony and a flock at Sylt (~54%), kites (51%). Least: pigeons (9%), seagulls up close (15%). Fix: retrain with bird/kite images labeled as background or as their own class. See `runs/false_alarms/worst_false_alarms.jpg`.

## Test 2: real tracking on 20 videos (Nov 16)

`python scripts/evaluate/test_tracking.py` runs detection + ByteTrack on the 20 DUT Anti-UAV tracking videos (24,804 frames, true drone box marked in every frame). Seconds assume 30 fps.

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

v2 = v1 trained 10 more passes with 2,100 COCO bird/kite photos marked "no drone" (`scripts/data/prepare_negatives.py`). Weights: `runs/train/dut_neg_n960/weights/best.pt`.

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

v3 = v2 trained 10 more passes with every 4th frame of 10 DUT tracking clips (2,393 frames, `scripts/data/prepare_tracking_split.py`), to help with drones in front of trees and buildings. Weights: `runs/train/dut_v3_n960/weights/best.pt`.

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

## v4: retrained with drone types, aircraft and motion blur (Nov 21)

v4 = v3 trained 10 more passes with the real-video part of [ratio1/Drone_Detection_Survaillance](https://huggingface.co/datasets/ratio1/Drone_Detection_Survaillance) (CC BY-NC 4.0): propeller and fixed-wing drones as "drone"; planes, helicopters and birds as "no drone"; plus motion-blurred copies of a quarter of the training images. Rebuild: `scripts/data/prepare_ratio1.py`, `scripts/data/prepare_blur.py`; full run: `scripts/train/run_v4.sh`. Weights: `runs/train/dut_v4_n960/weights/best.pt`.

| Test | v3 | v4 |
|---|---|---|
| Planes boxed as drone (110 images) | 41.8% | 19.1% |
| Helicopters boxed as drone (143) | 26.6% | 1.4% |
| Close-up birds boxed as drone (69) | 44.9% | 4.3% |
| Bird/kite clips, false alarms (0.35) | 1.8% | 1.9% |
| Held-out tracking, on target | 93.3% | 93.0% |
| Held-out tracking, wrong-object boxes | 1.0% | 2.2% |
| video17 (trees), on target | 80.4% | 74.6% |
| Photo test precision / recall | 95.6% / 91.3% | 94.6% / 90.1% |

v4 fixes manned aircraft but is slightly worse elsewhere, so v3 stays the default. Full breakdown (data, per-clip, per-size, pass/fail): [docs/v3-to-v4.md](docs/v3-to-v4.md). The drone-type test (`scripts/evaluate/test_drone_types.py`) is too easy to show fixed-wing misses: both models find 100% of its fixed-wing drones.

## Moving camera: BoT-SORT and trail correction (Nov 22)

A hand-held phone or a camera on a moving drone shifts the whole picture every frame. ByteTrack predicts each drone's next position assuming a still camera, and the orange trail recorded raw screen positions, so both followed the camera instead of the drone.

- **`botsort_motion.yaml`**: BoT-SORT with camera-motion compensation (sparse optical flow). Now the default tracker in Live ("Moving camera").
- **`camera_motion.py`**: estimates the frame-to-frame shift/rotation/scale from background corners (the drone's box is skipped, RANSAC rejects other movers) and moves the old trail points with it. Live: "Correct for camera motion" (on by default).
- **`scripts/evaluate/test_moving_camera.py`**: simulates a pan, tilt and hand shake by moving a crop window over held-out clips (true boxes shifted the same way) and compares trackers and trail error.

v3 on the 10 held-out clips, real footage (`scripts/evaluate/test_tracking.py --tracker botsort_motion.yaml`):

| | ByteTrack | BoT-SORT + camera correction |
|---|---|---|
| On target | 93.3% | 94.3% |
| ID switches | 120 | 63 |
| Wrong-object boxes | 1.0% | 1.1% |
| video07 (source camera moves) | 92.7% | 96.6% |

Simulated camera motion (`runs/moving_camera_test.json`):

| Clip | ByteTrack | BoT-SORT | Trail error, raw → corrected |
|---|---|---|---|
| video10 | 93.4%, 48 ID switches | 96.6%, 2 ID switches | 63 px → 2–5 px |
| video17 (trees) | 38.2%, 78 ID switches | 80.3%, 16 ID switches | 7–57 px → 2 px |

Trail error is only meaningful on clips whose original camera was still (video10, video17); video01 and video07 already pan in the source footage. Next: confirm on our own iPhone pan/walk clips.
