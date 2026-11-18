"""Manual classification ground truth and frozen collections, separate from YOLO training data."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile

from PIL import Image
from candidate_labels import DRONE_LABEL, LABELS

SPLITS = ("development", "test")


def _key(value, kind="clip"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError(f"{kind} must be a safe name containing letters, digits, underscores, dots, or hyphens")
    return value


def _fields(key, frame, boxes, label, split, source_id, no_drone):
    _key(key)
    if type(frame) is not int or frame < 0:
        raise ValueError("frame must be a nonnegative integer")
    if split not in SPLITS:
        raise ValueError("split must be development or test")
    if not isinstance(source_id, str) or not source_id.strip() or len(source_id.strip()) > 80:
        raise ValueError("source_id must contain 1 to 80 characters")
    if type(no_drone) is not bool or not isinstance(boxes, (list, tuple)):
        raise ValueError("boxes must be a list and no_drone must be a boolean")
    normalized = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError("each box must contain four finite normalized xyxy coordinates")
        clipped = [max(0.0, min(1.0, float(v))) for v in box]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError("boxes must have positive area after clamping to the image")
        normalized.append(clipped)
    if normalized:
        if no_drone or label not in LABELS:
            raise ValueError("boxed objects require a canonical label and cannot be marked no_drone")
    elif not no_drone or label is not None:
        raise ValueError("an empty frame requires no_drone=true and label=null")
    return {"clip_id": key, "frame": frame, "boxes": normalized, "label": label,
            "split": split, "source_id": source_id.strip(), "no_drone": no_drone,
            "expected_is_drone": label == DRONE_LABEL, "manual": True}


def _dimensions(image):
    with Image.open(io.BytesIO(image) if isinstance(image, bytes) else image) as decoded:
        if decoded.format != "JPEG" or decoded.mode != "RGB":
            raise ValueError("classification frames must be RGB JPEG images")
        decoded.load()
        return decoded.size


@contextmanager
def _lock(root):
    directory = Path(root) / "classification"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _records(root, key=None):
    directory = Path(root) / "classification/labels"
    paths = (directory / key).glob("*.json") if key is not None else directory.glob("*/*.json")
    records = []
    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("manual") is not True:
            raise ValueError(f"only manual classification annotations are accepted: {path}")
        normalized = _fields(data.get("clip_id"), data.get("frame"), data.get("boxes"), data.get("label"),
                             data.get("split"), data.get("source_id"), data.get("no_drone"))
        if path.parent.name != normalized["clip_id"] or path.stem != str(normalized["frame"]):
            raise ValueError(f"annotation identity does not match its filename: {path}")
        expected_image = f"classification/images/{normalized['clip_id']}/{normalized['frame']}.jpg"
        if data.get("image") != expected_image:
            raise ValueError(f"annotation image path does not match its identity: {path}")
        records.append({**data, **normalized})
    return records


def _check_splits(records):
    for field in ("source_id", "clip_id"):
        groups = {}
        for record in records:
            value, split = record[field], record["split"]
            if value in groups and groups[value] != split:
                raise ValueError(f"{field} {value!r} occurs in both development and test")
            groups[value] = split


def save_annotation(root, key, frame, boxes, label, split, source_id, *, image_bytes=None, no_drone=False, delete=False):
    """Save only a user's explicit label; image_bytes are the adapter's raw-frame JPEG."""
    root = Path(root)
    _key(key)
    if type(frame) is not int or frame < 0 or type(delete) is not bool:
        raise ValueError("frame must be a nonnegative integer and delete must be a boolean")
    image_path = root / "classification/images" / key / f"{frame}.jpg"
    label_path = root / "classification/labels" / key / f"{frame}.json"
    if delete:
        with _lock(root):
            label_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
        return {"deleted": True, "clip_id": key, "frame": frame}
    annotation = _fields(key, frame, boxes, label, split, source_id, no_drone)
    if image_bytes is not None and not isinstance(image_bytes, bytes):
        raise ValueError("image_bytes must contain encoded JPEG bytes")
    with _lock(root):
        _check_splits([*_records(root), annotation])
        if image_bytes is None and not image_path.is_file():
            raise ValueError("a raw-frame JPEG is required for a new annotation")
        width, height = _dimensions(image_bytes if image_bytes is not None else image_path)
        annotation.update(image=str(image_path.relative_to(root)), width=width, height=height)
        encoded = (json.dumps(annotation, allow_nan=False, indent=2) + "\n").encode()
        if image_bytes is not None:
            _atomic_write(image_path, image_bytes)
        _atomic_write(label_path, encoded)
    return annotation


def load_annotations(root, key):
    _key(key)
    if not (Path(root) / "classification").exists():
        return {}
    with _lock(root):
        return {str(record["frame"]): record for record in _records(root, key)}


def remove_clip_annotations(root, key):
    """Recording deletion may remove derived classification data, never frozen collections."""
    _key(key)
    if not (Path(root) / "classification").exists():
        return
    with _lock(root):
        for kind in ("images", "labels"):
            directory = Path(root) / "classification" / kind / key
            if directory.exists():
                shutil.rmtree(directory)


def export_collection(root, name):
    root = Path(root).resolve()
    _key(name, "collection")
    destination = root / "collections" / name
    with _lock(root):
        if destination.exists():
            raise FileExistsError(f"frozen collection already exists: {destination}")
        records = _records(root)
        if not records:
            raise ValueError("there are no manually labeled classification frames to export")
        _check_splits(records)
        for record in records:
            dimensions = _dimensions(root / record["image"])
            if dimensions != (record.get("width"), record.get("height")):
                raise ValueError("captured frame dimensions differ from its annotation")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination.parent))
        try:
            manifests = {split: [] for split in SPLITS}
            for record in records:
                image = Path("images") / record["clip_id"] / f"{record['frame']}.jpg"
                copied = temporary / image
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / record["image"], copied)
                common = {"image": image.as_posix(), "source_id": record["source_id"],
                          "clip_id": record["clip_id"], "frame": record["frame"],
                          "expected_is_drone": record["expected_is_drone"]}
                if record["no_drone"]:
                    manifests[record["split"]].append({**common, "expected_label": None, "candidate_bbox": None})
                else:
                    w, h = record["width"], record["height"]
                    for x1, y1, x2, y2 in record["boxes"]:
                        manifests[record["split"]].append({**common, "expected_label": record["label"],
                                                          "candidate_bbox": [x1*w, y1*h, x2*w, y2*h]})
            for split, rows in manifests.items():
                (temporary / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
            metadata = {"name": name, "frames": len(records), "created_at": datetime.now(timezone.utc).isoformat(),
                        "manual_ground_truth": True,
                        **{f"{split}_count": len(manifests[split]) for split in SPLITS},
                        "sources": {split: sorted({r["source_id"] for r in records if r["split"] == split}) for split in SPLITS},
                        "clips": {split: sorted({r["clip_id"] for r in records if r["split"] == split}) for split in SPLITS}}
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    test_manifest = destination / "test.jsonl"
    command = (".venv/bin/python scripts/evaluate/test_classification.py --manifest " + shlex.quote(str(test_manifest))
               + " --out " + shlex.quote(f"runs/classification_{name}.json"))
    return {**metadata, "development_manifest": str(destination / "development.jsonl"),
            "test_manifest": str(test_manifest), "metadata_path": str(destination / "metadata.json"), "command": command}
