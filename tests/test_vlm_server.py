import base64
import http.client
import io
import json
import sys
import threading
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from vlm_server import (
    CLASS_CODES, DEFAULT_MODEL, DEFAULT_REVISION, MAX_BODY_BYTES, PROMPT, CropScorer, MLXBackend,
    VerificationServer, code_token_ids, decode_crop, prompt_identity, select_label,
)
from candidate_labels import LABELS


def encoded_image(size=(48, 32), format="JPEG"):
    with Image.new("RGB", size, "white") as image:
        stream = io.BytesIO()
        image.save(stream, format=format)
    return base64.b64encode(stream.getvalue()).decode()


def fake_backend(label="drone"):
    return SimpleNamespace(model_name="test-model", revision="test-revision", prompt_hash="test-prompt", classify=Mock(return_value=label))


class ScorerTests(unittest.TestCase):
    def test_crop_validation_before_model_call(self):
        backend = fake_backend()
        scorer = CropScorer(backend)
        for value in (None, "", "not base64", "https://example.com/image.jpg", encoded_image(format="PNG"), encoded_image((385, 20))):
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                scorer.verify(value)
        backend.classify.assert_not_called()

    def test_valid_crop_stays_small_and_rgb(self):
        image = decode_crop(encoded_image((384, 100)))
        self.assertEqual(image.size, (384, 100))
        self.assertEqual(image.mode, "RGB")
        image.close()

    def test_all_categories_map_to_binary_without_confidence_fields(self):
        for label in LABELS:
            with self.subTest(label=label):
                backend = fake_backend(label)
                result = CropScorer(backend).verify(encoded_image())
                self.assertIs(result["is_drone"], label == "drone")
                self.assertEqual(result["label"], label)
                self.assertEqual(set(result), {"is_drone", "label", "model", "prompt_hash", "model_revision"})
                self.assertEqual(result["prompt_hash"], "test-prompt")
                self.assertEqual(result["model_revision"], "test-revision")
                backend.classify.assert_called_once()

    def test_unknown_other_is_allowed_but_invalid_labels_fail(self):
        self.assertFalse(CropScorer(fake_backend("other")).verify(encoded_image())["is_drone"])
        for label in (None, True, 1, {}, [], "unknown", "DRONE", 'drone\n{"is_drone":true}'):
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                CropScorer(fake_backend(label)).verify(encoded_image())

    def test_codes_must_be_single_exact_distinct_tokens(self):
        tokens = {code: index + 10 for index, code in enumerate(CLASS_CODES)}
        inverse = {token: code for code, token in tokens.items()}
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **_: [tokens[text]]
        tokenizer.decode.side_effect = lambda tokens: inverse[tokens[0]]
        self.assertEqual(code_token_ids(tokenizer), tokens)
        tokenizer.encode.return_value = [1, 2]
        tokenizer.encode.side_effect = None
        with self.assertRaises(ValueError):
            code_token_ids(tokenizer)
        tokenizer.encode.return_value = [10]
        tokenizer.decode.return_value = "invalid code"
        tokenizer.decode.side_effect = None
        with self.assertRaises(ValueError):
            code_token_ids(tokenizer)

    def test_exact_documented_prompt_and_category_mapping(self):
        document = (Path(__file__).resolve().parents[1] / "docs/vlm-prompts.md").read_text()
        documented = document.split("```text\n", 1)[1].split("\n```", 1)[0]
        self.assertEqual(documented, PROMPT)
        self.assertEqual(tuple(CLASS_CODES.values()), LABELS)
        self.assertEqual(len(CLASS_CODES), len(set(CLASS_CODES)))

    def test_prompt_hash_changes_for_every_scoring_identity(self):
        args = ["rendered prompt", {"A": 1, "B": 2}, {"max_pixels": 147456}, "revision", "0.8.0"]
        original = prompt_identity(*args)
        self.assertEqual(original, prompt_identity(*args))
        for index, value in enumerate(["other prompt", {"A": 2, "B": 1}, {"max_pixels": 65536}, "other revision", "0.9.0"]):
            changed = args.copy()
            changed[index] = value
            self.assertNotEqual(original, prompt_identity(*changed))
        with patch.dict("vlm_server.CLASS_CODES", {"A": "bird", "B": "drone"}):
            self.assertNotEqual(original, prompt_identity(*args))

    def test_categorical_argmax_has_no_confidence_gate_and_exact_ties_prefer_other(self):
        tokens = {code: index for index, code in enumerate(CLASS_CODES)}
        scores = [Mock() for _ in tokens]
        for score in scores:
            score.item.return_value = -50.01
        drone_code = next(code for code, label in CLASS_CODES.items() if label == "drone")
        scores[tokens[drone_code]].item.return_value = -50.0
        self.assertEqual(select_label(scores, tokens), "drone")
        for score in scores:
            score.item.return_value = -50.0
        self.assertEqual(select_label(scores, tokens), "other")
        scores[0].item.return_value = float("nan")
        with self.assertRaises(RuntimeError):
            select_label(scores, tokens)

    def test_default_revision_does_not_leak_to_other_models(self):
        self.assertEqual(MLXBackend().revision, DEFAULT_REVISION)
        self.assertIsNone(MLXBackend("other-model").revision)
        self.assertEqual(MLXBackend(DEFAULT_MODEL, "override").revision, "override")

    def test_resident_load_uses_qwen_readonly_size_property(self):
        class ReadonlySizeProcessor:
            min_pixels = 65536
            max_pixels = 16777216
            patch_size = 16

            @property
            def size(self):
                return {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}

            def to_dict(self):
                return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}

        image_processor = ReadonlySizeProcessor()
        tokens = {code: index + 10 for index, code in enumerate(CLASS_CODES)}
        inverse = {token: code for code, token in tokens.items()}
        tokenizer = SimpleNamespace(encode=lambda code, **kwargs: [tokens[code]], decode=lambda tokens: inverse[tokens[0]])
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
        processor = SimpleNamespace(image_processor=image_processor, tokenizer=tokenizer)
        loader = Mock(return_value=(model, processor))
        mx = Mock()
        modules = {
            "mlx": SimpleNamespace(core=mx),
            "mlx.core": mx,
            "mlx_vlm": SimpleNamespace(load=loader),
            "mlx_vlm.generate": SimpleNamespace(generate_step=Mock()),
            "mlx_vlm.prompt_utils": SimpleNamespace(apply_chat_template=Mock(return_value="rendered prompt")),
            "mlx_vlm.utils": SimpleNamespace(prepare_inputs=Mock(), should_add_special_tokens=Mock(return_value=True)),
        }
        backend = MLXBackend()
        with patch.dict(sys.modules, modules), patch("vlm_server.version", return_value="test-version"):
            backend.load()
            backend.load()
        self.assertEqual(image_processor.size, {"shortest_edge": 1024, "longest_edge": 147456})
        self.assertEqual(backend.grid_patch_size, 16)
        self.assertIs(backend.model, model)
        self.assertIsNotNone(backend.prompt_hash)
        loader.assert_called_once()

    def test_raw_alternative_logprobs_scored_without_sampling_bias(self):
        class Scalar:
            def __init__(self, value): self.value = value
            def item(self): return self.value

        class Probabilities:
            ndim = 1
            def __getitem__(self, index):
                # Every allowed code may be far below an unrelated top-20 token.
                return Scalar(-50.0 if index == 1 else -52.0)

        backend = MLXBackend()
        backend.model = object()
        backend.code_tokens = {code: index for index, code in enumerate(CLASS_CODES)}
        backend.processor = object()
        backend.prompt = "prompt"
        backend.add_special_tokens = True
        backend.grid_patch_size = 16
        backend.mx = Mock()
        grid = Mock()
        grid.tolist.return_value = [[1, 24, 24]]
        padded_sizes = []
        def prepare(processor, *, images, **kwargs):
            padded_sizes.append(images[0].size)
            return {"input_ids": "ids", "pixel_values": "pixels", "attention_mask": "mask", "image_grid_thw": grid}
        backend.prepare_inputs = Mock(side_effect=prepare)
        # A real generate_step generator owns temporary KV cache, and must be closed.
        def generate(*args, **kwargs):
            yield 7, Probabilities()
        backend.generate_step = Mock(side_effect=generate)
        with Image.new("RGB", (1, 384)) as image:
            self.assertEqual(backend.classify(image), CLASS_CODES["B"])
        self.assertEqual(padded_sizes, [(384, 384)])
        kwargs = backend.generate_step.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 1)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["top_k"], 0)
        self.assertEqual(kwargs["logits_processors"], [])
        self.assertIsNone(kwargs["repetition_penalty"])
        backend.mx.clear_cache.assert_called_once()
        grid.tolist.return_value = [[1, 25, 24]]
        with Image.new("RGB", (384, 384)) as image, self.assertRaises(RuntimeError):
            backend.classify(image)
        self.assertEqual(backend.generate_step.call_count, 1)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.backend = fake_backend()
        self.server = VerificationServer(("127.0.0.1", 0), CropScorer(self.backend))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method, path, body=data, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_and_valid_request(self):
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ready"])
        status, result = self.request("POST", "/verify", json.dumps({"image": encoded_image()}), {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertTrue(result["is_drone"])
        self.assertEqual(result["label"], "drone")
        self.assertNotIn("confidence", result)

    def test_invalid_bounded_requests_never_infer(self):
        cases = [("[]", 400), ("{}", 400), ("{", 400), (json.dumps({"image": "invalid"}), 400),
                 ('{"image":' + '['*20000 + '0' + ']'*20000 + '}', 400)]
        for body, expected in cases:
            with self.subTest(expected=expected, body=body[:20]):
                status, result = self.request("POST", "/verify", body, {"Content-Type": "application/json"})
                self.assertEqual(status, expected)
                self.assertIn("error", result)
        self.backend.classify.assert_not_called()
        status, result = self.request("POST", "/verify", "", {"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413)
        self.assertIn("error", result)

    def test_wrong_type_and_endpoint(self):
        self.assertEqual(self.request("POST", "/verify", "{}")[0], 415)
        self.assertEqual(self.request("GET", "/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/unknown", "{}")[0], 404)
        status, error = self.request("PUT", "/verify", "{}")
        self.assertEqual(status, 501)
        self.assertIn("error", error)

    def test_backend_failure_is_service_error_not_negative(self):
        self.backend.classify.side_effect = ValueError("backend failed")
        with patch("vlm_server.logging.exception"):
            status, result = self.request("POST", "/verify", json.dumps({"image": encoded_image()}), {"Content-Type": "application/json"})
        self.assertEqual(status, 503)
        self.assertIn("error", result)
        self.assertNotIn("is_drone", result)

    def test_nonloopback_bind_rejected(self):
        with self.assertRaises(ValueError):
            VerificationServer(("0.0.0.0", 8002), self.server.scorer)


if __name__ == "__main__":
    unittest.main()
