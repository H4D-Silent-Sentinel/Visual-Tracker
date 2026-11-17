# Event-driven drone verification

User-authorized implementation on `feat/vlm-drone-verification`; target is an Apple Silicon Mac with 8 GB total RAM. The user approved cropping and requested parallel agents, confidence research, and implementation on a branch.

## Behavior

YOLO/ByteTrack continue on each frame. A new track gathers crops for 0.5 seconds of visible observations; the clearest crop is submitted once to a separate local VLM process. Ordinary motion, bounding-box jitter, and elapsed time never trigger another review. A substantially clearer/larger view or a reappearance after a lost identity triggers an update, with debounce. Cached verdicts stay attached to each track. Lost tracks expire; resets invalidate pending replies.

Tentative candidates show REVIEWING/UNVERIFIED until a confident binary review. A confident yes retains the visual lock; a confident no suppresses the lock and shows FALSE REVIEW: NOT A DRONE. Low confidence and service failures do not become confident negatives. Tracker IDs, epoch/session, and crop revision identify requests; stale results cannot confirm a different object. One request runs at a time, queue length is bounded, and no completed verdict is periodically polled by the VLM.

## Images and model

Crop the source frame before annotations, pad the target 25%, preserve aspect ratio, and resize conventionally to at most 384 pixels. Keep the original object pixel width and sharpness; reject invalid boxes and do not pretend interpolation restores detail. No generative super-resolution or full-frame VLM calls. Cache only a bounded number of crops. Default model: `mlx-community/Qwen3.5-2B-4bit` (Apache-2.0, about 1.72 GB of weights); MLX runs locally with no generated chain of thought. One selected image per request minimizes latency. Runtime memory/FPS and drone accuracy must be measured, not inferred from weight size or general benchmarks.

## Scorer interface and confidence

`vlm_server.py` serves loopback HTTP, POST `/verify` accepts `{"image": "<base64 JPEG>"}` and returns `{"is_drone": true, "confidence": 0.95, "p_drone": 0.95, "raw_margin": 2.94, "calibrated": false, "model": "..."}`. `confidence` is the probability of the selected label, `p_drone` is the probability of drone. Error responses use `{"error": "..."}` with a non-2xx status. The JSON is constructed and validated by Python; the VLM is prompted for exactly `yes` or `no` and their pre-sampling log probabilities are scored, without asking the model to invent a number.

`vlm_confidence.py` provides `score_margin(margin, calibration=None)` returning the four confidence/verdict fields above plus `raw_margin`; margin is log P(yes) minus log P(no). Optional calibration JSON contains positive `temperature`, finite `bias`, and model/prompt identity metadata. Platt-style sigmoid(margin / temperature + bias) handles overconfidence and class bias. Uncalibrated scores are explicitly marked and never advertised as measured correctness. Default decision confidence threshold is 0.85, provisional until validation.

Calibration fits only a dedicated calibration split, evaluates on a separate test split, and reports Brier score, negative log likelihood, reliability/ECE, confusion counts, and abstention coverage. Splits are by clip/source, not adjacent video frames. Existing held-out tracking ground truth and bird/kite negative clips remain test data; do not train or fit calibration on them unless a separate split is explicitly created. Invalid/mismatched calibration is rejected.

## Integration and scope

New verifier logic is shared by `track.py` and `dashboard/live_engine.py`; existing detector-only behavior remains available when verification is disabled. CLI opts in with `--verify`; dashboard Live has an explicit toggle and verification readout. Dashboard stats expose per-target verification plus primary candidate status; lock counters and trails count only confirmed targets when verification is on. Multiple candidates retain independent verdicts. The old dashboard, training scripts, and saved offline review outputs stay unchanged.

Tests use standard-library unittest and mock/local scorers to exercise identity isolation, stale replies, rate limits, confident negative suppression, uncertainty, cropping, calibration validation, and nonblocking tracking. A local scorer smoke test and laptop benchmark are attempted when dependencies/hardware permit. Document any unavailable real-world evaluation.
