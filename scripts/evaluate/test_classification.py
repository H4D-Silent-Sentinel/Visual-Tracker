"""Evaluate VLM labels on real primary-YOLO proposals matched to labeled candidates.

Run: .venv/bin/python scripts/evaluate/test_classification.py --manifest candidates.jsonl --out runs/classification/report.json
"""
import argparse
import base64
import collections
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from candidate_labels import DRONE_LABEL, LABELS, OTHER_LABEL


def validate_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("candidate_bbox must contain four pixel coordinates [x1,y1,x2,y2]")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
        raise ValueError("candidate_bbox coordinates must be finite numbers")
    if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("candidate_bbox must have nonnegative coordinates and positive area")
    return box


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def validate_truth(record):
    if "expected_label" not in record or "candidate_bbox" not in record:
        raise ValueError("expected_label and candidate_bbox must be explicitly supplied")
    if record["expected_label"] is None:
        if record["candidate_bbox"] is not None or record.get("expected_is_drone") is not False:
            raise ValueError("background rows require label=null, bbox=null, and expected_is_drone=false")
        return True
    if record["expected_label"] not in LABELS:
        raise ValueError(f"expected_label must be one of {LABELS}")
    validate_box(record["candidate_bbox"])
    if "expected_is_drone" in record and (
            type(record["expected_is_drone"]) is not bool or record["expected_is_drone"] != (record["expected_label"] == DRONE_LABEL)):
        raise ValueError("expected_is_drone must agree with the ground-truth label")
    return False


def load_manifest(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("each row must be a JSON object")
                for key in ("image", "source_id", "clip_id"):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f"{key} must be a nonempty string")
                validate_truth(record)
                image = Path(record["image"])
                if not image.is_absolute():
                    image = Path(path).resolve().parent / image
                if not image.is_file():
                    raise ValueError(f"image does not exist: {image}")
                records.append({**record, "image": str(image.resolve())})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("manifest has no labeled candidates")
    return records


def _score_result(result, proposal, scorer):
    result = {**result, "proposal": proposal}
    try:
        response = scorer(result["image"], proposal)
        if not isinstance(response, dict) or response.get("label") not in LABELS:
            raise ValueError("VLM response must contain a label from the shared taxonomy")
        if type(response.get("is_drone")) is not bool or response["is_drone"] != (response["label"] == DRONE_LABEL):
            raise ValueError("VLM is_drone must agree with its label")
        identity = {key: response[key] for key in ("model", "model_revision", "prompt_hash") if key in response}
        return {**result, "status": "unknown" if response["label"] == OTHER_LABEL else "classified",
                "label": response["label"], "is_drone": response["is_drone"], **identity}
    except Exception as exc:
        return {**result, "error": str(exc)}


def evaluate_record(record, detector, scorer):
    """Boxed truth matches one target; background truth tests every actual proposal."""
    result = {**record, "status": "error", "label": None, "is_drone": None, "proposal": None}
    try:
        background = validate_truth(record)
        proposals = [validate_box(box) for box in detector(record["image"])]
        if background:
            if not proposals:
                return {**result, "status": "background_clear", "background_correct": True, "proposal_results": []}
            scored = [_score_result(result, proposal, scorer) for proposal in proposals]
            confirmed = any(score["is_drone"] is True for score in scored)
            complete = all(score["status"] != "error" for score in scored)
            return {**result, "status": "background", "proposal_results": scored,
                    "is_drone": confirmed if complete or confirmed else None,
                    "background_correct": not confirmed if complete or confirmed else None}
        matches = [(box_iou(box, record["candidate_bbox"]), box) for box in proposals]
        matches = [(overlap, box) for overlap, box in matches if overlap >= 0.3]
        if not matches:
            return {**result, "status": "no_proposal"}
        _, proposal = max(matches, key=lambda entry: entry[0])
        return _score_result(result, proposal, scorer)
    except Exception as exc:
        return {**result, "error": str(exc)}


def summarize(results):
    confusion = collections.defaultdict(collections.Counter)
    no_proposal = collections.Counter()
    statuses = collections.Counter(result["status"] for result in results)
    correct = false_confirmed = missed = scored = labeled_scored = unknown = errors = 0
    yolo_false_positives = reviewed_false_positives = 0
    background_correct = background_false_positive = background_false_proposals = 0
    for result in results:
        expected = result["expected_label"]
        if result["status"] == "no_proposal":
            no_proposal[expected] += 1
        if expected is None:
            background_correct += result.get("background_correct") is True
            background_false_positive += result.get("background_correct") is False
        proposal_results = result.get("proposal_results", [result])
        for prediction in proposal_results:
            if expected != DRONE_LABEL and prediction["proposal"] is not None:
                yolo_false_positives += 1
            if prediction["status"] == "error":
                errors += 1
                continue
            if prediction["status"] not in ("classified", "unknown"):
                continue
            scored += 1
            reviewed_false_positives += expected != DRONE_LABEL
            unknown += prediction["status"] == "unknown"
            label = prediction["label"]
            if expected is not None:
                labeled_scored += 1
                confusion[expected][label] += 1
                correct += label == expected
            false_confirmed += prediction["is_drone"] and expected != DRONE_LABEL
            background_false_proposals += prediction["is_drone"] and expected is None
            missed += not prediction["is_drone"] and expected == DRONE_LABEL
    comparison_complete = errors == 0 and yolo_false_positives == reviewed_false_positives
    suppressed = yolo_false_positives - false_confirmed if comparison_complete else None
    return {
        "candidate_count": len(results),
        "source_count": len({r["source_id"] for r in results}),
        "clip_count": len({(r["source_id"], r["clip_id"]) for r in results}),
        "scored_proposals": scored,
        "labeled_scored_proposals": labeled_scored,
        "no_proposal": statuses["no_proposal"],
        "no_proposal_by_label": dict(no_proposal),
        "unknown": unknown, "errors": errors,
        "label_confusion": {label: dict(counts) for label, counts in confusion.items()},
        "label_accuracy_on_scored_proposals": correct / labeled_scored if labeled_scored else None,
        "false_confirmed_drone": false_confirmed,
        "missed_drone_by_vlm": missed,
        "yolo_only_false_positive_proposals": yolo_false_positives,
        "reviewed_false_positive_proposals": reviewed_false_positives,
        "false_positive_comparison_complete": comparison_complete,
        "vlm_suppressed_false_positive_proposals": suppressed,
        "false_positive_reduction": suppressed / yolo_false_positives if comparison_complete and yolo_false_positives else None,
        "background_frames": sum(r["expected_label"] is None for r in results),
        "background_no_proposal": statuses["background_clear"],
        "background_correct": background_correct,
        "background_false_positive_frames": background_false_positive,
        "background_false_positive_proposals": background_false_proposals,
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def score_proposal(image_path, proposal, url):
    import cv2
    from drone_verifier import crop_target

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"image cannot be decoded: {image_path}")
    cropped = crop_target(frame, proposal)
    if cropped is None:
        raise ValueError("YOLO proposal does not produce a valid crop")
    ok, jpeg = cv2.imencode(".jpg", cropped[0])
    if not ok:
        raise ValueError("candidate JPEG encoding failed")
    body = json.dumps({"image": base64.b64encode(jpeg.tobytes()).decode("ascii")}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=120) as response:
        data = response.read((1 << 20) + 1)
        if len(data) > 1 << 20:
            raise ValueError("VLM response exceeds 1 MiB")
        return json.loads(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new JSON report; existing files are preserved")
    parser.add_argument("--model", type=Path, default=ROOT / "runs/train/dut_v3_n960/weights/best.pt")
    parser.add_argument("--url", default="http://127.0.0.1:8002/verify")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        if args.imgsz <= 0 or not math.isfinite(args.conf) or not 0 < args.conf <= 1:
            raise ValueError("imgsz must be positive and conf must be greater than 0 and at most 1")
        endpoint = urlsplit(args.url)
        if endpoint.scheme != "http" or endpoint.hostname not in ("localhost", "127.0.0.1", "::1") or endpoint.username:
            raise ValueError("VLM URL must be a local HTTP endpoint")
        records = load_manifest(args.manifest)
        if not args.model.is_file():
            raise ValueError(f"primary YOLO weights are missing: {args.model}")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".venv/ultralytics"))
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/matplotlib"))
        import torch
        from ultralytics import YOLO

        model = YOLO(str(args.model))
        if DRONE_LABEL not in model.names.values():
            raise ValueError("the primary YOLO checkpoint must contain the drone class")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        def detector(image):
            predictions = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False, save=False)
            if not predictions or predictions[0].boxes is None:
                return []
            result = predictions[0]
            return [box for box, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())
                    if result.names[int(class_id)] == DRONE_LABEL]

        results = []
        for record in records:
            started = time.monotonic()
            result = evaluate_record(record, detector, lambda image, box: score_proposal(image, box, args.url))
            result["elapsed_s"] = round(time.monotonic() - started, 3)
            results.append(result)
            print(json.dumps(result), flush=True)
        report = {"proposal_model": str(args.model.resolve()), "vlm_url": args.url,
                  "imgsz": args.imgsz, "conf": args.conf, "labels": LABELS,
                  "summary": summarize(results), "results": results}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report["summary"]), flush=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
