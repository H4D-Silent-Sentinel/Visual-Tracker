# v3 → v4: what changed and what it did

Trained Nov 21, 2026 on an Apple M4 Pro (MPS). Bottom line: **v4 nearly stops calling manned aircraft and close-up birds "drone", but is slightly worse than v3 on our original drone footage, especially small (distant) drones and trees. v3 stays the default.**

## 1. What changed

### Training data

| Folder | Source | Images | Label | In v3 | In v4 |
|---|---|---:|---|:-:|:-:|
| `train` | DUT Anti-UAV detection photos | 5,200 | drone | ✓ | ✓ |
| `train_neg` | COCO birds and kites | 2,100 | no drone | ✓ | ✓ |
| `train_trk` | Every 4th frame of 10 DUT tracking clips | 2,393 | drone | ✓ | ✓ |
| `train_r1` | ratio1 Drone Detection Surveillance, real video frames | **7,634** | see below | | ✓ |
| `train_blur` | Motion-blurred copies of 25% of `train`, `train_trk`, `train_r1` | **3,806** | same as original | | ✓ |
| **Total** | | **9,693** (v3) → **21,133** (v4) | | | |

`train_r1` by type (from the dataset's train split, real frames only; its AI-generated images were not used):

| Type | Images | Labeled as |
|---|---:|---|
| Propeller (multirotor) drones | 4,200 | drone |
| Fixed-wing drones | 528 | drone |
| Planes | 1,225 | no drone (image has no box) |
| Helicopters | 1,017 | no drone |
| Birds (close-up) | 664 | no drone |

- Source: [ratio1/Drone_Detection_Survaillance](https://huggingface.co/datasets/ratio1/Drone_Detection_Survaillance), CC BY-NC 4.0 (fine for coursework; not for commercial use). Annotations converted from Pascal VOC XML.
- Motion blur: straight-line blur, random length 7–21 px, random direction; stands in for a moving camera.
- Validation added `val_r1` (709 images from the dataset's dev split) to v3's `val` + `val_neg`.
- Leak check: any source video that appears in the dataset's test split is dropped from training and validation. None overlapped.

### Training settings (unchanged from v3)

Started from v3's weights. YOLO11n, 960 px, 10 passes, batch 16, mosaic off for the last 3 passes, patience 5, seed 0. Only the data changed.

| | v3 | v4 |
|---|---|---|
| Training images | 9,693 | 21,133 |
| Time per pass | ~8.8 min | ~17.3 min |
| Total training time | ~1.5 h | ~2.9 h |
| Best pass | 10 | 10 (still improving) |

v4's own validation scores per pass (not comparable to v3's: different validation images):

| Pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Precision | 88.8 | 88.3 | 91.4 | 92.0 | 92.1 | 91.2 | 93.0 | 92.2 | 92.9 | 93.8 |
| Recall | 71.6 | 76.7 | 76.7 | 77.8 | 77.4 | 79.2 | 81.1 | 81.5 | 81.8 | 83.0 |
| mAP50 | 79.1 | 82.3 | 83.6 | 85.0 | 83.5 | 85.0 | 86.8 | 86.2 | 87.3 | 87.7 |

Still rising at pass 10: more passes may help.

## 2. Results

All tests use data neither model trained on. Default settings: confidence cutoff 0.35, 960 px, ByteTrack.

### Drone-type test (new): ratio1 test split, real frames

| Type | Images | v3 | v4 |
|---|---:|---|---|
| Propeller drones: found | 304 | 100% | 98.7% (4 missed) |
| Fixed-wing drones: found | 61 | 100% | 100% |
| Planes: boxed as drone | 110 | 41.8% (46) | **19.1% (21)** |
| Helicopters: boxed as drone | 143 | 26.6% (38) | **1.4% (2)** |
| Birds, close-up: boxed as drone | 69 | 44.9% (31) | **4.3% (3)** |

Caveat: these drones are mostly large and close, so v3 already finds all of them. This test does not show the fixed-wing misses we saw on other footage.

### Bird and kite clips (8 clips, 2,916 frames, no drones): share of frames with a false "drone" box

| Clip | Frames | v3 | v4 |
|---|---:|---|---|
| cranes_flock | 204 | 3.4% | 1.0% |
| flock_padilla_bay | 660 | 3.9% | 0.2% |
| flock_padilla_bay_2 | 315 | 0.6% | 0.0% |
| flock_sylt | 315 | 4.1% | 3.8% |
| kites | 260 | 0.0% | 0.4% |
| pigeons_tokyo | 223 | 0.9% | 2.7% |
| seagulls_close | 442 | 0.0% | 4.8% |
| terns_midway | 497 | 0.6% | 2.4% |
| **All, cutoff 0.35** | 2,916 | **1.8%** | **1.9%** |
| All, cutoff 0.25 | | 3.0% | 3.4% |
| All, cutoff 0.50 | | 0.8% | 0.9% |

Same total, different mistakes: v4 is better on distant flocks, worse on close gulls, pigeons and terns.

### Tracking: 10 held-out DUT videos (15,245 frames, 12,907 with the drone visible)

| | v3 | v4 |
|---|---|---|
| On target | **93.3%** | 93.0% |
| Wrong-object boxes (frames) | **1.0%** | 2.2% |
| ID switches | **120** | 128 |
| Times the drone was lost and found again | 175 | 178 |
| Average time to find it again | 0.16 s | 0.17 s |

Per video:

| Video | Visible frames | On target v3 → v4 | Wrong-object v3 → v4 | ID switches v3 → v4 |
|---|---:|---|---|---|
| video01 | 1,050 | 78.7% → **84.2%** | 0.0% → 0.0% | 5 → 6 |
| video03 | 100 | 100% → 96.0% | 0.0% → 0.0% | 0 → 0 |
| video07 | 1,827 | 92.7% → 92.3% | 1.4% → **0.5%** | 41 → 45 |
| video08 | 2,072 | 95.8% → **97.1%** | 0.4% → 0.2% | 24 → 31 |
| video09 | 1,562 | 100% → 100% | 3.9% → **12.2%** | 2 → 1 |
| video10 | 2,307 | 98.7% → 99.1% | 0.3% → 0.4% | 0 → 1 |
| video12 | 1,338 | 83.1% → 78.4% | 0.1% → 0.0% | 7 → 6 |
| video14 | 590 | 100% → 100% | 0.0% → 0.0% | 0 → 0 |
| video17 (trees) | 780 | 80.4% → **74.6%** | 0.8% → 0.6% | 40 → 37 |
| video18 | 1,281 | 98.7% → 98.5% | 0.0% → 0.0% | 1 → 1 |

Most of v4's extra wrong-object boxes come from video09 (12.2%). What it boxes there has not been checked yet (Clip Review → video09 → v4).

### By drone size (held-out videos; size = longer side of the true box)

| Size | Visible frames | v3 | v4 |
|---|---:|---|---|
| 16–32 px | 47 | 95.7% | 87.2% |
| 32–64 px | 2,898 | **88.4%** | 84.8% |
| over 64 px | 9,962 | 94.7% | **95.5%** |

v4 got better on large drones and worse on small ones, matching its new data: ratio1 drones are mostly big and close. The 16–32 px row is only 47 frames.

### Labeled photo test (2,200 DUT test photos)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v3 | **95.6%** | **91.3%** | **94.9%** | **63.9%** |
| v4 | 94.6% | 90.1% | 94.0% | 60.9% |

### Speed

Same network (YOLO11n, 2.6M parameters), so live speed is unchanged.

## 3. Pass / fail against the targets set before training

| Target | Result |
|---|---|
| Trees: v4 beats v3 by a clear margin | **Fail**: 80.4% → 74.6% (no tree data was added) |
| Bird/kite false alarms ≤ 3% | **Pass**: 1.9% |
| Held-out on target ≥ 93.3%, wrong object ≤ 1.5% | **Fail (close)**: 93.0%, 2.2% |
| Photo precision and recall within 1 point of v3 | **Fail (close)**: −1.0 and −1.2 |
| Live speed not lower | **Pass** |
| Drone types: no type worse | Mixed: propeller −1.3 points; fixed-wing equal |
| Manned aircraft false alarms low | **Big improvement**: planes −23 points, helicopters −25 |

## 4. Why, and what next

- **Why aircraft improved:** v4 saw 2,242 plane and helicopter images labeled "no drone". v3 had never seen one.
- **Why small drones got worse:** most new drone images are large and close; v4 shifted toward that look. Distant drones are what matters for a base.
- **Why trees got worse:** no new tree footage; the new data pulled the model elsewhere.
- **Next:**
  1. Find what v4 boxes in video09.
  2. Our own footage: drones in front of trees and distant fixed-wing drones (test-only clips first).
  3. More passes (scores were still rising), or a smaller share of close-up drone images.
  4. A harder drone-type test with small, distant drones.

## 5. Reproduce

```sh
# download ratio1 REAL frames to ~/yolo-datasets/ratio1 (huggingface_hub snapshot_download, allow_patterns "*/REAL/*")
.venv/bin/python scripts/data/prepare_ratio1.py
.venv/bin/python scripts/data/prepare_blur.py
scripts/train/run_v4.sh        # baseline, train, every test
```

Reports: `runs/drone_types_v3.json`, `runs/drone_types_v4.json`, `runs/false_alarms_v4/report.json`, `runs/tracking_test_v4/report.json`, `runs/review/photo_test.json`, `runs/train/dut_v4_n960/results.csv`.
