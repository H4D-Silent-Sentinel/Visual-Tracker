# Drone VLM Verification Implementation Plan

> **For agentic workers:** Execute the independent ownership scopes below in parallel, then integrate on the requested branch. No agent stages or commits shared files. The controller runs combined checks.

**Goal:** Add local, event-driven drone verification without blocking the tracker.

**Architecture:** YOLO/ByteTrack provide track IDs and raw image crops. A background client calls a resident MLX scorer once per new candidate or meaningful appearance update, caches its binary verdict and probability, and gates visual locks.

**Tech Stack:** Existing Python/OpenCV/YOLO, stdlib threads/HTTP/unittest, optional MLX-VLM.

**Spec:** `docs/superpowers/specs/2026-11-17-vlm-drone-verification-design.md`

## Global constraints

- Apple Silicon, 8 GB total RAM, default 4-bit Qwen3.5-2B; weights alone are not runtime memory.
- No periodic reviews or per-frame VLM inference; 0.5-second candidate crop collection, single worker, bounded queue/caches.
- No generative upscaling, model-written confidence, silent failure-as-negative, or calibration/test leakage.
- Confidence threshold starts at 0.85 and is explicitly unvalidated.
- All verification changes are optional, on `feat/vlm-drone-verification`.

## Parallel ownership

| Task | Dependencies | Files | Interface |
|---|---|---|---|
| Confidence | shared spec | `vlm_confidence.py`, `scripts/evaluate/calibrate_vlm.py`, `tests/test_vlm_confidence.py`, `docs/vlm-confidence-research.md` | `score_margin(margin, calibration=None)` |
| Scorer | shared spec | `vlm_server.py`, `requirements-vlm.txt`, `tests/test_vlm_server.py` | loopback POST `/verify`, spec JSON |
| Tracker/controller | shared spec | `drone_verifier.py`, `track.py`, `dashboard/live_engine.py`, `tests/test_drone_verifier.py`, README | target `verification` object with `status`, `is_drone`, `confidence`, `p_drone`, `calibrated`, `error` |
| Live UI | shared spec | `dashboard/live.js`, `dashboard/index.html`, `dashboard/style.css` | settings `verify`, stats `verification`, each target `verification` |

Each task tests only owned files while working. Shared contracts are fixed above; integration and changes to these interfaces run through the controller.

## Steps

- [x] Confidence: read current literature, implement stable two-label scoring and calibration validation, fit on a calibration manifest and evaluate a separate test manifest, test numerical edge cases and split leakage.

```sh
python3 -m unittest discover -s tests -p 'test_vlm_confidence.py' -v
```

- [x] Scorer: verify released MLX API, lazy-load one model, validate bounded local JPEG requests, score yes/no probabilities without sampling bias, return validated JSON; mock MLX in tests and provide optional dependency pins.

```sh
python3 -m unittest discover -s tests -p 'test_vlm_server.py' -v
```

- [x] Controller: implement per-track crop selection, event triggers, cache/reset/expiry, bounded async HTTP worker, and confidence gating. Wire both live callers and preserve detector-only mode. Test without real weights.

```sh
python3 -m unittest discover -s tests -p 'test_drone_verifier.py' -v
```

- [x] UI: add verification toggle and show confirmed, rejected, pending, and failed/uncertain states without changing offline review.

```sh
node --check dashboard/live.js
```

- [x] Integrate all ownership scopes; run unit tests, Python parse/compile checks, JavaScript syntax checks, mock HTTP smoke test, and actual scorer smoke/peak-memory timing when available. Review crop identity, candidate updates, calibration correctness, and GPU contention. Fix integration findings once, then rerun affected checks.

```sh
python3 -m unittest discover -s tests -v
git diff --check
```

- [x] Update README with startup, event behavior, confidence meaning/calibration commands, measured validation and limitations; commit the complete change on the requested branch without merging or publishing.

## Validation outcome

46 unit/integration tests passed. JavaScript, Python, and shell syntax checks passed. Native pinned-model inference passed after correcting the processor read-only property; the model-selection note records synthetic scorer timing and memory on a 16 GiB Mac. CLI and dashboard HTTP smoke checks passed. Real flight accuracy, fitted drone calibration, concurrent detector FPS, and 8 GB hardware validation remain open measurements.
