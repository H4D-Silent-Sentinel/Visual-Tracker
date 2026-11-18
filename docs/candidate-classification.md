# Candidate object classification

The existing v3 YOLO detector proposes candidate boxes. Its selected crop is then classified by the resident local VLM into one of the 35 classes in [`candidate_labels.py`](../candidate_labels.py). The single `drone` class includes quadcopters, FPV drones, multirotors, and fixed-wing unmanned aircraft. Crewed airplanes and helicopters have separate labels; `other` covers uncertain or unlisted objects. There is no second YOLO model or agreement stage.

The scorer maps the ordered classes to single-token letter codes, reads the first answer token, and constructs JSON containing `label` and `is_drone`. `is_drone` is true exactly when `label` is `drone`. It does not ask the model to generate a confidence number. The true label used for evaluation is never sent to the VLM.

Classification runs once on a selected crop for each new or meaningfully updated candidate. It does not run on every video frame. Crop size remains at most 384 pixels per edge; selecting a broader label set does not establish the model's real accuracy or the detector's frame rate. Measure both on the actual Mac and footage.

## Record and label a collection

Use **Collect** to record raw camera footage or select an existing clip. Pick the development or test split and a source/session ID, then select a frame, draw or explicitly accept the object's box, and choose its true class. Save the classification annotation only after checking it yourself. A model's suggested box or live predicted class is not ground truth. Frames containing several labeled objects produce one test row per saved box; all boxes in one annotation share its chosen class.

For a frame you have checked contains **no drones anywhere**, save an explicit no-drone annotation with no boxes and no object class. This is binary background truth; it does not claim every leaf, bird, or cloud belongs to `other`. Leave ambiguous frames unlabeled until you can establish their truth.

Manual classification annotations and captured full-frame JPEGs are separate from the existing drone-training labels:

```
our_data/classification/images/<clip>/<frame>.jpg
our_data/classification/labels/<clip>/<frame>.json
```

Each annotation records its normalized xyxy boxes, class or explicit no-drone flag, source, clip, frame, split, and captured image dimensions. Saves validate the image and coordinates before writing annotation data. A source or clip cannot occur in both development and test, including across HTTP saves and CLI exports. These files do not alter raw recordings, legacy YOLO labels, or a training dataset automatically.

Freeze the currently annotated frames into a named collection using Collect's export action or:

```sh
.venv/bin/python scripts/data/export_classification_set.py --name flight-session-01
```

The export copies the images into `our_data/collections/flight-session-01/images/`, converts boxes to pixel coordinates, and writes `development.jsonl`, `test.jsonl`, and `metadata.json`. Image paths in a manifest are relative to that manifest, so the collection remains usable after copying its whole folder. Source and clip partitions are checked again. The completed collection appears atomically and an existing name is never overwritten. Later annotation changes or recording deletion leave frozen collections intact. Use a new name to freeze later revisions, and annotate at least one test frame before running the test manifest. Export prints the exact test command.

All collection data stays under the ignored `our_data/` folder; only pipeline code belongs in git.

## Classification test

Start the local VLM service, then provide JSONL with one labeled target per row. Use full original images and pixel-coordinate ground-truth boxes; paths relative to the manifest resolve beside that manifest. Example with fictional local filenames:

```json
{"image":"images/drone-001.jpg","expected_label":"drone","candidate_bbox":[100,80,220,160],"source_id":"session-001","clip_id":"clip-001"}
{"image":"images/bird-001.jpg","expected_label":"bird","candidate_bbox":[40,30,90,75],"source_id":"session-002","clip_id":"clip-002"}
```

For boxed objects, `expected_label` must belong to the shared taxonomy. Image, label, box, source, and clip fields are required; frozen exports also retain the frame number. Label each object and its box independently of the model's prediction, and retain source/clip IDs to identify correlated frames. One row evaluates that target; a row does not supply truth for unrelated objects elsewhere in the image.

An explicitly checked background frame instead has binary truth:

```json
{"image":"images/background-001.jpg","expected_label":null,"candidate_bbox":null,"expected_is_drone":false,"source_id":"session-003","clip_id":"clip-003"}
```

For these rows the script evaluates **every actual YOLO drone proposal** through the VLM. Any final `drone` label is a false confirmation. Non-drone labels are valid binary rejections without being judged against an invented object class. No YOLO proposals means correct background screening with no VLM call; it is recorded separately and never presented as a VLM-produced non-drone verdict. Partial service failures do not become correct negative frames.

```sh
.venv/bin/python scripts/evaluate/test_classification.py \
  --manifest candidates.jsonl --out runs/classification/report.json \
  --model runs/train/dut_v3_n960/weights/best.pt \
  --url http://127.0.0.1:8002/verify --imgsz 960 --conf 0.35
```

The script loads the primary YOLO checkpoint once and runs it on each image. It selects the highest-IoU drone proposal overlapping the row's labeled target by at least **0.3**, uses the same 25%-padded `crop_target` as tracking, and makes one actual `/verify` call for that selected proposal. It uses the detector's box for the crop; the ground-truth box only matches the proposal to the labeled target. Duplicate matching boxes are not counted as separate targets, and unrelated proposals without ground truth are not scored against this row's label.

The report contains label confusion and accuracy on successfully scored proposals, false drone confirmations, drones missed by the VLM, `other`/unknown results, service errors, and source/clip counts. Each successful review retains the scorer's model, checkpoint revision, and prompt hash for comparing runs. **No matching YOLO proposal is reported as `no_proposal`, with no VLM call and no fabricated non-drone verdict.** Missing proposals and errors are excluded from VLM class accuracy and VLM false-negative counts; the report records missing proposals separately by true label. A valid `other` result counts as unknown and, for a true drone, a VLM miss. Existing report files are preserved.

The report also compares the known false-positive proposals from YOLO alone against final drone confirmations. The baseline includes matched non-drone target proposals and all proposals on binary no-drone frames. `vlm_suppressed_false_positive_proposals` and `false_positive_reduction` are available only when every review succeeded; failures never masquerade as suppression. Reduction is null when there were no known false-positive proposals to compare. Background results have both frame counts and proposal counts; multiple objects and adjacent frames are correlated observations, so these counts are not independent statistical samples or a measure of overall field accuracy.

The HTTP endpoint must be on loopback, and redirects are rejected so candidate crops stay local. First-stage YOLO errors and invalid VLM response contracts appear as errors rather than negative classifications.

A useful test set needs labeled drone positives and bird/kite/aircraft/background negatives from independent sources, with results broken down by distance, object pixel size, blur, and scene. Use held-out clip/source splits. One screenshot is a smoke test, not an accuracy benchmark. No second-model weights were downloaded for this implementation.
