import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from candidate_labels import DRONE_LABEL, LABELS, OTHER_LABEL
from scripts.evaluate.test_classification import evaluate_record, load_manifest, main, summarize, validate_box, validate_truth


def record(label=DRONE_LABEL):
    return {"image": "fixture.jpg", "expected_label": label, "candidate_bbox": [10, 10, 110, 110],
            "source_id": "source-1", "clip_id": "clip-1"}


class ClassificationTests(unittest.TestCase):
    def test_taxonomy_is_unique_and_supports_single_drone_class(self):
        self.assertEqual(len(LABELS), len(set(LABELS)))
        self.assertLessEqual(len(LABELS), 52)
        self.assertIn(DRONE_LABEL, LABELS)
        self.assertIn(OTHER_LABEL, LABELS)
        self.assertNotEqual(DRONE_LABEL, OTHER_LABEL)

    def test_background_box_cannot_replace_the_matching_candidate(self):
        proposals = [[180, 180, 280, 280], [20, 20, 100, 100], [10, 10, 110, 110]]
        scorer = Mock(return_value={"label": "drone", "is_drone": True})
        result = evaluate_record(record(), lambda image: proposals, scorer)
        self.assertEqual(result["status"], "classified")
        self.assertEqual(result["proposal"], proposals[2])
        scorer.assert_called_once_with("fixture.jpg", proposals[2])

    def test_missing_proposal_is_not_a_vlm_negative_or_a_vlm_call(self):
        scorer = Mock()
        result = evaluate_record(record(), lambda image: [[200, 200, 220, 220]], scorer)
        self.assertEqual(result["status"], "no_proposal")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["is_drone"])
        scorer.assert_not_called()
        metrics = summarize([result])
        self.assertEqual(metrics["no_proposal"], 1)
        self.assertEqual(metrics["no_proposal_by_label"], {"drone": 1})
        self.assertEqual(metrics["missed_drone_by_vlm"], 0)
        self.assertEqual(metrics["scored_proposals"], 0)
        self.assertIsNone(metrics["label_accuracy_on_scored_proposals"])

    def test_each_review_retains_model_and_prompt_identity_for_comparison(self):
        identity = {"model": "local-model", "model_revision": "revision-1", "prompt_hash": "prompt-1"}
        scorer = Mock(return_value={"label": "bird", "is_drone": False, **identity})
        result = evaluate_record(record("bird"), lambda image: [[10, 10, 110, 110]], scorer)
        self.assertEqual({key: result[key] for key in identity}, identity)

    def test_background_scores_every_proposal_without_inventing_class_truth(self):
        background = {**record(), "expected_label": None, "candidate_bbox": None, "expected_is_drone": False}
        proposals = [[10, 10, 110, 110], [200, 200, 250, 250]]
        scorer = Mock(side_effect=[{"label": "bird", "is_drone": False}, {"label": "drone", "is_drone": True}])
        result = evaluate_record(background, lambda image: proposals, scorer)
        self.assertEqual(scorer.call_count, 2)
        self.assertFalse(result["background_correct"])
        metrics = summarize([result])
        self.assertEqual(metrics["background_false_positive_frames"], 1)
        self.assertEqual(metrics["background_false_positive_proposals"], 1)
        self.assertEqual(metrics["false_confirmed_drone"], 1)
        self.assertEqual(metrics["scored_proposals"], 2)
        self.assertEqual(metrics["label_confusion"], {})
        self.assertIsNone(metrics["label_accuracy_on_scored_proposals"])

    def test_background_without_proposals_is_correct_without_a_vlm_verdict(self):
        background = {**record(), "expected_label": None, "candidate_bbox": None, "expected_is_drone": False}
        scorer = Mock()
        result = evaluate_record(background, lambda image: [], scorer)
        scorer.assert_not_called()
        self.assertTrue(result["background_correct"])
        self.assertIsNone(result["is_drone"])
        metrics = summarize([result])
        self.assertEqual(metrics["background_no_proposal"], 1)
        self.assertEqual(metrics["background_correct"], 1)
        self.assertEqual(metrics["no_proposal"], 0)
        self.assertEqual(metrics["missed_drone_by_vlm"], 0)

    def test_background_partial_service_failure_is_not_a_correct_negative(self):
        background = {**record(), "expected_label": None, "candidate_bbox": None, "expected_is_drone": False}
        scorer = Mock(side_effect=[{"label": "leaf", "is_drone": False}, TimeoutError("service failed")])
        result = evaluate_record(background, lambda image: [[10, 10, 20, 20], [30, 30, 40, 40]], scorer)
        self.assertIsNone(result["background_correct"])
        self.assertEqual(summarize([result])["errors"], 1)
        self.assertEqual(summarize([result])["background_correct"], 0)

    def test_yolo_false_positive_baseline_and_suppression_use_known_truth(self):
        detector = lambda image: [[10, 10, 110, 110]]
        background = {**record(), "expected_label": None, "candidate_bbox": None, "expected_is_drone": False}
        rejected = evaluate_record(record("bird"), detector, lambda image, box: {"label": "bird", "is_drone": False})
        confirmed = evaluate_record(record("kite"), detector, lambda image, box: {"label": "drone", "is_drone": True})
        clear_background = evaluate_record(background, detector, lambda image, box: {"label": "leaf", "is_drone": False})
        metrics = summarize([rejected, confirmed, clear_background])
        self.assertEqual(metrics["yolo_only_false_positive_proposals"], 3)
        self.assertEqual(metrics["reviewed_false_positive_proposals"], 3)
        self.assertEqual(metrics["false_confirmed_drone"], 1)
        self.assertEqual(metrics["vlm_suppressed_false_positive_proposals"], 2)
        self.assertAlmostEqual(metrics["false_positive_reduction"], 2 / 3)
        self.assertTrue(metrics["false_positive_comparison_complete"])

    def test_failed_reviews_never_look_like_false_positive_suppression(self):
        detector = lambda image: [[10, 10, 110, 110]]
        failed = evaluate_record(record("bird"), detector, Mock(side_effect=TimeoutError("no review")))
        metrics = summarize([failed])
        self.assertEqual(metrics["yolo_only_false_positive_proposals"], 1)
        self.assertEqual(metrics["reviewed_false_positive_proposals"], 0)
        self.assertFalse(metrics["false_positive_comparison_complete"])
        self.assertIsNone(metrics["vlm_suppressed_false_positive_proposals"])
        self.assertIsNone(metrics["false_positive_reduction"])
        no_proposal = evaluate_record(record("bird"), lambda image: [], Mock())
        self.assertIsNone(summarize([no_proposal])["false_positive_reduction"])

    def test_binary_background_truth_must_be_explicit(self):
        background = {**record(), "expected_label": None, "candidate_bbox": None, "expected_is_drone": False}
        self.assertTrue(validate_truth(background))
        for changed in ({"expected_label": "other"}, {"expected_is_drone": 0}, {"candidate_bbox": [0, 0, 1, 1]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_truth({**background, **changed})

    def test_false_confirmation_and_vlm_miss_have_separate_counts(self):
        proposals = lambda image: [[10, 10, 110, 110]]
        false_confirmation = evaluate_record(record("bird"), proposals,
                                              lambda image, box: {"label": "drone", "is_drone": True})
        missed = evaluate_record(record("drone"), proposals,
                                 lambda image, box: {"label": "kite", "is_drone": False})
        metrics = summarize([false_confirmation, missed])
        self.assertEqual(metrics["false_confirmed_drone"], 1)
        self.assertEqual(metrics["missed_drone_by_vlm"], 1)
        self.assertEqual(metrics["label_confusion"], {"bird": {"drone": 1}, "drone": {"kite": 1}})
        self.assertEqual(metrics["source_count"], 1)
        self.assertEqual(metrics["clip_count"], 1)

    def test_other_and_service_errors_remain_distinct(self):
        detector = lambda image: [[10, 10, 110, 110]]
        unknown = evaluate_record(record(), detector, lambda image, box: {"label": "other", "is_drone": False})
        error = evaluate_record(record(), detector, Mock(side_effect=TimeoutError("worker timed out")))
        metrics = summarize([unknown, error])
        self.assertEqual(metrics["unknown"], 1)
        self.assertEqual(metrics["errors"], 1)
        self.assertEqual(metrics["scored_proposals"], 1)
        self.assertEqual(metrics["missed_drone_by_vlm"], 1)
        self.assertIsNone(error["is_drone"])
        self.assertIn("timed out", error["error"])

    def test_invalid_vlm_contract_is_an_error(self):
        detector = lambda image: [[10, 10, 110, 110]]
        for response in ({"label": "drone", "is_drone": False}, {"label": "bird", "is_drone": 0},
                         {"label": "unlisted", "is_drone": False}, None):
            with self.subTest(response=response):
                result = evaluate_record(record(), detector, lambda image, box: response)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["label"])

    def test_invalid_boxes_and_manifest_labels_are_rejected(self):
        for box in (None, [1, 2, 3], [10, 10, 0, 0], [-1, 0, 1, 1], [0, 0, float("nan"), 1], [False, 0, 1, 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "fixture.jpg"
            image.touch()
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(record()) + "\n")
            rows = load_manifest(path)
            self.assertEqual(rows[0]["image"], str(image.resolve()))
            path.write_text(json.dumps(record("unlisted")) + "\n")
            with self.assertRaisesRegex(ValueError, ":1:"):
                load_manifest(path)

    def test_cli_help_and_argument_checks_need_no_model_load(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserve me")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--manifest", "absent.jsonl", "--out", str(output)])
            self.assertEqual(output.read_text(), "preserve me")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--manifest", "absent.jsonl", "--out", str(Path(directory) / "new.json"),
                      "--url", "http://example.com/verify"])


if __name__ == "__main__":
    unittest.main()
