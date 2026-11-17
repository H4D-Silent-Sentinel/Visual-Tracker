import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate.calibrate_vlm import evaluate_records, fit_calibration, load_records, main, validate_splits
from vlm_confidence import load_calibration, score_margin, validate_calibration


IDENTITY = {"model": "test-model", "prompt_hash": "test-prompt"}


def record(margin, label, source="calibration-source", clip="calibration-clip"):
    return {**IDENTITY, "raw_margin": margin, "label": label, "source_id": source, "clip_id": clip}


class ConfidenceTests(unittest.TestCase):
    def test_two_label_normalization_and_chosen_label_probability(self):
        yes = score_margin(math.log(9))
        no = score_margin(-math.log(9))
        self.assertTrue(yes["is_drone"])
        self.assertFalse(no["is_drone"])
        self.assertAlmostEqual(yes["p_drone"], 0.9)
        self.assertAlmostEqual(no["p_drone"], 0.1)
        self.assertAlmostEqual(no["confidence"], 0.9)
        self.assertFalse(yes["calibrated"])
        self.assertEqual(yes["raw_margin"], math.log(9))
        self.assertEqual(score_margin(0)["confidence"], 0.5)

    def test_extreme_margins_do_not_overflow(self):
        self.assertEqual(score_margin(1e308)["p_drone"], 1)
        self.assertEqual(score_margin(-1e308)["p_drone"], 0)
        calibration = {**IDENTITY, "temperature": 1e-308, "bias": 0}
        self.assertEqual(score_margin(1e308, calibration)["p_drone"], 1)
        self.assertEqual(score_margin(-1e308, calibration)["p_drone"], 0)
        self.assertFalse(score_margin(-1e-300)["is_drone"])

    def test_temperature_and_bias_can_change_selected_label(self):
        calibration = {**IDENTITY, "temperature": 2, "bias": -2}
        score = score_margin(2, calibration)
        self.assertAlmostEqual(score["p_drone"], 1 / (1 + math.e))
        self.assertFalse(score["is_drone"])
        self.assertAlmostEqual(score["confidence"], 1 - score["p_drone"])
        self.assertTrue(score["calibrated"])

    def test_invalid_raw_numbers_rejected(self):
        for margin in (True, "2", None, float("nan"), float("inf"), -float("inf"), 10**1000):
            with self.subTest(margin=repr(margin)[:30]), self.assertRaises(ValueError):
                score_margin(margin)

    def test_invalid_calibration_rejected(self):
        valid = {**IDENTITY, "temperature": 1, "bias": 0}
        invalid = [[], {}, {**valid, "temperature": 0}, {**valid, "temperature": -1},
                   {**valid, "temperature": float("inf")}, {**valid, "bias": float("nan")},
                   {**valid, "bias": True}, {**valid, "model": ""}, {**valid, "prompt_hash": None}]
        for calibration in invalid:
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                score_margin(1, calibration)

    def test_calibration_loader_checks_scorer_identity(self):
        calibration = {**IDENTITY, "temperature": 2, "bias": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(calibration))
            self.assertEqual(load_calibration(path, **IDENTITY), calibration)
            for identity in ({**IDENTITY, "model": "other"}, {**IDENTITY, "prompt_hash": "other"}):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_calibration(path, **identity)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_calibration(path, model=None, prompt_hash=IDENTITY["prompt_hash"])
        self.assertEqual(validate_calibration(calibration), calibration)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration_records = ([record(-12, True)] * 2 + [record(-12, False)] * 8
                                    + [record(12, True)] * 6 + [record(12, False)] * 4)
        self.evaluation_records = [record(-12, False, "test-source", "test-clip"),
                                   record(12, True, "test-source", "test-clip")]

    def test_fit_reduces_nll_and_recovers_temperature_and_bias(self):
        fitted = {**IDENTITY, **fit_calibration(self.calibration_records)}
        self.assertGreater(fitted["temperature"], 1)
        self.assertLess(fitted["bias"], 0)
        self.assertAlmostEqual(score_margin(-12, fitted)["p_drone"], 0.2, places=3)
        self.assertAlmostEqual(score_margin(12, fitted)["p_drone"], 0.6, places=3)
        raw = evaluate_records(self.calibration_records)
        adjusted = evaluate_records(self.calibration_records, fitted)
        self.assertLess(adjusted["nll"], raw["nll"])
        self.assertLess(adjusted["brier"], raw["brier"])

    def test_separable_and_reversed_scores_have_finite_fit(self):
        for samples in ([record(-2, False), record(2, True)],
                        [record(-2, True), record(2, False)],
                        [record(0, True), record(0, False)]):
            with self.subTest(samples=samples):
                fitted = fit_calibration(samples)
                self.assertTrue(math.isfinite(fitted["temperature"]))
                self.assertGreater(fitted["temperature"], 0)
                self.assertTrue(math.isfinite(fitted["bias"]))

    def test_overlap_by_clip_or_source_rejected(self):
        for source, clip in (("calibration-source", "other-clip"), ("other-source", "calibration-clip")):
            with self.subTest(source=source, clip=clip), self.assertRaisesRegex(ValueError, "overlap"):
                validate_splits(self.calibration_records, [record(1, True, source, clip)])
        self.assertEqual(validate_splits(self.calibration_records, self.evaluation_records), IDENTITY)

    def test_mixed_identities_and_one_class_fit_rejected(self):
        for field in ("model", "prompt_hash"):
            evaluation = [{**self.evaluation_records[0], field: "other"}]
            with self.assertRaisesRegex(ValueError, "same model"):
                validate_splits(self.calibration_records, evaluation)
        with self.assertRaisesRegex(ValueError, "both"):
            validate_splits([record(1, True)], self.evaluation_records)
        with self.assertRaisesRegex(ValueError, "both"):
            fit_calibration([record(1, True)])

    def test_metrics_and_abstention_use_selected_label_confidence(self):
        samples = [record(math.log(4), True), record(-math.log(4), False)]
        metrics = evaluate_records(samples)
        self.assertAlmostEqual(metrics["brier"], 0.04)
        self.assertAlmostEqual(metrics["nll"], -math.log(0.8))
        self.assertAlmostEqual(metrics["ece"], 0.2)
        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 0, "tn": 1, "fn": 0})
        self.assertEqual(metrics["coverage"], 0)
        self.assertIsNone(metrics["accepted_accuracy"])
        accepted = evaluate_records(samples, threshold=0.79)
        self.assertEqual(accepted["accepted"], 2)
        self.assertEqual(accepted["coverage"], 1)
        self.assertEqual(accepted["accepted_accuracy"], 1)

    def test_nll_remains_informative_at_saturated_probabilities(self):
        metrics = evaluate_records([record(-1000, True), record(1000, False)])
        self.assertEqual(metrics["nll"], 1000)
        self.assertEqual(metrics["brier"], 1)
        self.assertEqual(metrics["ece"], 1)
        self.assertEqual(metrics["accepted_accuracy"], 0)

    def test_manifest_validation_reports_bad_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps(record(1, True)) + "\n")
            self.assertEqual(load_records(path), [record(1, True)])
            for bad in ({**record(1, True), "label": 1}, {**record(1, True), "source_id": ""},
                        {**record(1, True), "raw_margin": float("nan")}, [1]):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps(bad) + "\n")
                    with self.assertRaisesRegex(ValueError, ":1:"):
                        load_records(path)

    def test_cli_writes_loadable_report_without_overwriting_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.jsonl"
            evaluation_path = Path(directory) / "evaluation.jsonl"
            output_path = Path(directory) / "fitted.json"
            for path, records in ((calibration_path, self.calibration_records),
                                  (evaluation_path, self.evaluation_records)):
                path.write_text("\n".join(json.dumps(r) for r in records))
            argv = ["--calibration", str(calibration_path), "--evaluation", str(evaluation_path),
                    "--output", str(output_path)]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            fitted = load_calibration(output_path, **IDENTITY)
            self.assertEqual(fitted["validation"]["evaluation_calibrated"]["count"], 2)
            self.assertEqual(fitted["validation"]["calibration_sources"], ["calibration-source"])
            original = output_path.read_text()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(output_path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
