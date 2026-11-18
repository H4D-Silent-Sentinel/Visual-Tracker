import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from classification_data import export_collection, load_annotations, remove_clip_annotations, save_annotation
from scripts.evaluate.test_classification import load_manifest


def jpeg(width=200, height=100, format="JPEG"):
    output = io.BytesIO()
    Image.new("RGB", (width, height), (40, 70, 100)).save(output, format=format)
    return output.getvalue()


class ClassificationDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "our_data"
        self.root.mkdir()
        self.raw = self.root / "recording.mov"
        self.raw.write_bytes(b"raw recording")
        self.legacy = self.root / "labels/legacy/0.txt"
        self.legacy.parent.mkdir(parents=True)
        self.legacy.write_text("legacy drone training label")

    def tearDown(self):
        self.directory.cleanup()

    def save(self, key="clip-1", frame=0, label="drone", split="test", source="test-source", **kwargs):
        return save_annotation(self.root, key, frame, [[0.25, 0.2, 0.75, 0.8]], label, split, source,
                               image_bytes=jpeg(), **kwargs)

    def test_manual_annotation_clamps_boxes_and_keeps_legacy_data_separate(self):
        annotation = save_annotation(self.root, "clip-1", 0, [[-0.1, 0.2, 1.1, 0.8]], "bird", "test",
                                     "session-1", image_bytes=jpeg())
        self.assertEqual(annotation["boxes"], [[0.0, 0.2, 1.0, 0.8]])
        self.assertEqual((annotation["width"], annotation["height"]), (200, 100))
        self.assertTrue(annotation["manual"])
        self.assertFalse(annotation["expected_is_drone"])
        self.assertEqual(load_annotations(self.root, "clip-1"), {"0": annotation})
        self.assertEqual(self.legacy.read_text(), "legacy drone training label")
        self.assertEqual(self.raw.read_bytes(), b"raw recording")

    def test_invalid_labels_metadata_boxes_and_images_write_no_annotation(self):
        arguments = dict(root=self.root, key="clip-1", frame=0, boxes=[[0.1, 0.1, 0.8, 0.8]],
                         label="drone", split="test", source_id="session-1", image_bytes=jpeg())
        invalid = [dict(label="unlisted"), dict(frame=True), dict(key="../outside"), dict(source_id=""),
                   dict(source_id="x"*81), dict(split="train"), dict(boxes=[[0.8, 0.1, 0.2, 0.8]]),
                   dict(boxes=[[0, 0, float("nan"), 1]]), dict(boxes=[[False, 0, 1, 1]]),
                   dict(image_bytes=b"invalid JPEG"), dict(image_bytes=jpeg(format="PNG"))]
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises((ValueError, OSError)):
                save_annotation(**{**arguments, **changed})
        self.assertFalse((self.root / "classification/labels/clip-1/0.json").exists())
        self.assertFalse((self.root / "classification/images/clip-1/0.jpg").exists())

    def test_no_drone_is_binary_truth_and_requires_explicit_empty_frame(self):
        annotation = save_annotation(self.root, "background", 3, [], None, "test", "background-source",
                                     no_drone=True, image_bytes=jpeg())
        self.assertIsNone(annotation["label"])
        self.assertFalse(annotation["expected_is_drone"])
        exported = export_collection(self.root, "background-set")
        row = load_manifest(exported["test_manifest"])[0]
        self.assertIsNone(row["expected_label"])
        self.assertIsNone(row["candidate_bbox"])
        self.assertIs(row["expected_is_drone"], False)
        for boxes, label, no_drone in (([], None, False), ([], "other", True), ([[0, 0, 1, 1]], "bird", True)):
            with self.subTest(boxes=boxes, label=label), self.assertRaises(ValueError):
                save_annotation(self.root, "invalid", 0, boxes, label, "test", "new-source",
                                no_drone=no_drone, image_bytes=jpeg())

    def test_annotation_can_update_without_reencoding_its_existing_image(self):
        original = self.save()
        image = self.root / original["image"]
        before = image.read_bytes()
        updated = save_annotation(self.root, "clip-1", 0, [[0.1, 0.1, 0.5, 0.5]], "kite", "test", "test-source")
        self.assertEqual(updated["label"], "kite")
        self.assertEqual(image.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "JPEG is required"):
            save_annotation(self.root, "new", 0, [[0, 0, 1, 1]], "bird", "test", "new-source")

    def test_clip_and_source_split_leakage_is_rejected_before_writes(self):
        original = self.save()
        for key, frame, source in (("clip-1", 1, "another-source"), ("clip-2", 0, "test-source")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "both development and test"):
                self.save(key=key, frame=frame, source=source, split="development")
            self.assertFalse((self.root / "classification/labels" / key / f"{frame}.json").exists())
        self.assertEqual(load_annotations(self.root, "clip-1"), {"0": original})

    def test_export_is_self_contained_and_converts_normalized_boxes_to_pixels(self):
        self.save(key="development-clip", split="development", source="development-source")
        self.save(key="test-clip", label="bird", source="test-source")
        exported = export_collection(self.root, "session-collection")
        rows = load_manifest(exported["test_manifest"])
        self.assertEqual(rows[0]["candidate_bbox"], [50.0, 20.0, 150.0, 80.0])
        self.assertEqual(rows[0]["expected_label"], "bird")
        self.assertIn("collections/session-collection/images/test-clip/0.jpg", rows[0]["image"])
        self.assertTrue(Path(rows[0]["image"]).is_file())
        self.assertEqual(exported["frames"], 2)
        self.assertEqual(exported["test_count"], 1)
        self.assertEqual(exported["sources"], {"development": ["development-source"], "test": ["test-source"]})
        frozen = Path(rows[0]["image"]).read_bytes()
        remove_clip_annotations(self.root, "test-clip")
        self.assertEqual(Path(rows[0]["image"]).read_bytes(), frozen)
        self.assertEqual(load_manifest(exported["test_manifest"])[0]["expected_label"], "bird")
        self.assertIn("scripts/evaluate/test_classification.py --manifest", exported["command"])

    def test_frozen_export_never_overwrites_existing_collection(self):
        self.save()
        exported = export_collection(self.root, "frozen")
        metadata = Path(exported["metadata_path"]).read_bytes()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            export_collection(self.root, "frozen")
        self.assertEqual(Path(exported["metadata_path"]).read_bytes(), metadata)
        with self.assertRaises(ValueError):
            export_collection(self.root, "../outside")

    def test_export_rechecks_split_leakage_and_manual_provenance(self):
        self.save(key="clip-1", source="source-1")
        second = self.save(key="clip-2", split="development", source="source-2")
        path = self.root / "classification/labels/clip-2/0.json"
        path.write_text(json.dumps({**second, "source_id": "source-1"}))
        with self.assertRaisesRegex(ValueError, "both development and test"):
            export_collection(self.root, "leaking")
        self.assertFalse((self.root / "collections/leaking").exists())
        path.write_text(json.dumps({**second, "manual": False}))
        with self.assertRaisesRegex(ValueError, "manual"):
            export_collection(self.root, "model-guesses")

    def test_export_rejects_image_dimensions_that_no_longer_match_truth(self):
        annotation = self.save()
        (self.root / annotation["image"]).write_bytes(jpeg(width=100, height=200))
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            export_collection(self.root, "changed-image")
        self.assertFalse((self.root / "collections/changed-image").exists())

    def test_explicit_annotation_delete_keeps_recordings_and_frozen_data(self):
        self.save()
        exported = export_collection(self.root, "saved")
        deleted = save_annotation(self.root, "clip-1", 0, [], None, "test", "test-source", delete=True)
        self.assertTrue(deleted["deleted"])
        self.assertEqual(load_annotations(self.root, "clip-1"), {})
        self.assertTrue(Path(exported["test_manifest"]).is_file())
        self.assertTrue(self.raw.is_file())
        self.assertTrue(self.legacy.is_file())


if __name__ == "__main__":
    unittest.main()
