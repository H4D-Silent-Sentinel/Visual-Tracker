"""Resident, local-only crop scorer. Install requirements.txt to run MLX."""

import argparse
import base64
import binascii
import hashlib
import io
import json
import logging
import math
import socket
import string
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version

from candidate_labels import DRONE_LABEL, LABELS, OTHER_LABEL


DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"
DEFAULT_REVISION = "674aaa7240b91e8012fcad5d791b7dfe5ba90207"
MAX_SIDE = 384
MAX_BODY_BYTES = 750_000
MAX_JPEG_BYTES = 512_000
if not 0 < len(LABELS) <= 52:
    raise ValueError("Classification needs between 1 and 52 labels")
CLASS_CODES = dict(zip(string.ascii_uppercase + string.ascii_lowercase, LABELS))
PROMPT = (
    "Classify the central visible object in this cropped image using the menu below.\n"
    "If the image is blank, shows only background, or the central object cannot be identified, choose other. "
    "Otherwise choose the single category best supported by the object's visible structure.\n"
    "The drone category includes unmanned quadcopters, FPV drones, multirotors, and fixed-wing UAVs. "
    "Rotor arms and propellers can distinguish a drone; feathers or flapping wings indicate a bird; "
    "fabric or trailing strings can indicate a kite. Airplane and helicopter refer to ordinary crewed aircraft.\n"
    "Answer with exactly one case-sensitive menu code. Do not include words, punctuation, or reasoning.\n\n"
    + "\n".join(f"{code}: {label}" for code, label in CLASS_CODES.items())
)


def prompt_identity(formatted_prompt, code_tokens, processor_config, revision, package_versions):
    """Changing instructions, weights, tokenization or preprocessing changes scorer identity."""
    identity = {
        "scorer_version": 2,
        "prompt": PROMPT,
        "formatted_prompt": formatted_prompt,
        "class_codes": CLASS_CODES,
        "code_token_ids": code_tokens,
        "image_processor": processor_config,
        "model_revision": revision,
        "package_versions": package_versions,
        "image_input": "base64 JPEG RGB; side <=384; centered square gray127 padding without resizing",
        "scoring": "largest allowed first-token code logP; exact ties prefer other; no sampling penalties or thinking",
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def code_token_ids(tokenizer):
    ids = {}
    for code in CLASS_CODES:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != code:
            raise ValueError(f"Model must encode menu code {code!r} as one exact token")
        ids[code] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError("Model menu code token IDs must differ")
    return ids


def select_label(logprobs, code_tokens):
    scores = {code: float(logprobs[token_id].item()) for code, token_id in code_tokens.items()}
    if not all(math.isfinite(score) for score in scores.values()):
        raise RuntimeError("MLX returned non-finite category scores")
    code = max(scores, key=lambda code: (scores[code], CLASS_CODES[code] == OTHER_LABEL))
    return CLASS_CODES[code]


class MLXBackend:
    """Import/load on demand, then keep exactly one model resident."""

    def __init__(self, model_name=DEFAULT_MODEL, revision=None):
        self.model_name = model_name
        self.revision = revision or (DEFAULT_REVISION if model_name == DEFAULT_MODEL else None)
        self.model = None
        self.prompt_hash = None

    def load(self):
        if self.model is not None:
            return
        try:
            import mlx.core as mx
            from mlx_vlm import load
            from mlx_vlm.generate import generate_step
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt in an Apple Silicon environment") from exc

        model, processor = load(self.model_name, revision=self.revision, trust_remote_code=False)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.code_tokens = code_token_ids(tokenizer)
        # The checkpoint's default minimum area otherwise enlarges small crops substantially.
        image_processor = processor.image_processor
        if model.config.model_type.startswith("qwen"):
            image_processor.min_pixels = 32 * 32
            image_processor.max_pixels = MAX_SIDE * MAX_SIDE
        self.prompt = apply_chat_template(
            processor, model.config, PROMPT, num_images=1,
            add_generation_prompt=True, enable_thinking=False,
        )
        self.prompt_hash = prompt_identity(
            self.prompt, self.code_tokens, image_processor.to_dict(), self.revision,
            {name: version(name) for name in ("mlx-vlm", "mlx", "transformers", "Pillow")},
        )
        self.add_special_tokens = should_add_special_tokens(model.config.model_type, processor)
        self.grid_patch_size = image_processor.patch_size if model.config.model_type.startswith("qwen") else None
        self.mx = mx
        self.prepare_inputs = prepare_inputs
        self.generate_step = generate_step
        self.processor = processor
        self.model = model

    def classify(self, image):
        from PIL import Image

        self.load()
        side = max(image.size)
        with Image.new("RGB", (side, side), (127, 127, 127)) as padded:
            padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
            inputs = self.prepare_inputs(
                self.processor, images=[padded], prompts=self.prompt,
                add_special_tokens=self.add_special_tokens,
            )
        if self.grid_patch_size is not None:
            grid = inputs.get("image_grid_thw")
            if grid is None or any(max(row[1:]) * self.grid_patch_size > MAX_SIDE for row in grid.tolist()):
                raise RuntimeError("Model preprocessing exceeded the 384-pixel image bound")
        input_ids = inputs.pop("input_ids")
        pixel_values = inputs.pop("pixel_values", None)
        mask = inputs.pop("attention_mask", None)
        steps = self.generate_step(
            input_ids, self.model, pixel_values, mask, max_tokens=1,
            temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
            repetition_penalty=None, presence_penalty=None, frequency_penalty=None,
            logit_bias=None, logits_processors=[], **inputs,
        )
        try:
            _, logprobs = next(steps)
            if logprobs is None or logprobs.ndim != 1:
                raise RuntimeError("MLX did not return a full first-token probability vector")
            # Score every allowed code from the full vocabulary, independent of the sampled token.
            return select_label(logprobs, self.code_tokens)
        finally:
            steps.close()
            self.mx.clear_cache()


def decode_crop(encoded):
    from PIL import Image, UnidentifiedImageError

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image must be a nonempty base64 JPEG string")
    if len(encoded) > (MAX_JPEG_BYTES + 2) // 3 * 4:
        raise ValueError("JPEG exceeds the byte limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image must be valid base64") from exc
    if len(raw) > MAX_JPEG_BYTES:
        raise ValueError("JPEG exceeds the byte limit")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG":
                raise ValueError("image must be JPEG")
            if min(image.size) < 1 or max(image.size) > MAX_SIDE:
                raise ValueError(f"Crop sides must be between 1 and {MAX_SIDE} pixels")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("image is not a valid JPEG crop") from exc


class CropScorer:
    def __init__(self, backend):
        self.backend = backend

    def verify(self, encoded):
        image = decode_crop(encoded)
        try:
            label = self.backend.classify(image)
            if not isinstance(label, str) or label not in LABELS:
                raise ValueError("VLM returned an invalid category label")
        except Exception as exc:
            raise RuntimeError("VLM inference failed") from exc
        finally:
            image.close()
        return {
            "is_drone": label == DRONE_LABEL,
            "label": label,
            "model": self.backend.model_name,
            "prompt_hash": self.backend.prompt_hash,
            "model_revision": self.backend.revision,
        }


class VerificationServer(HTTPServer):
    request_queue_size = 1

    def __init__(self, address, scorer):
        if address[0] != "127.0.0.1":
            raise ValueError("The scorer must bind to 127.0.0.1")
        self.scorer = scorer
        super().__init__(address, VerificationHandler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


class VerificationHandler(BaseHTTPRequestHandler):
    def send_error(self, code, message=None, explain=None):
        self.reply(code, {"error": message or self.responses.get(code, ("HTTP error",))[0]})

    def reply(self, status, data):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        backend = self.server.scorer.backend
        self.reply(200, {
            "ready": backend.prompt_hash is not None,
            "model": backend.model_name,
            "model_revision": backend.revision,
            "prompt_hash": backend.prompt_hash,
            "labels": LABELS,
        })

    def do_POST(self):
        if self.path != "/verify":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Content-Type must be application/json"})
                return
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is unsupported; send Content-Length")
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                self.reply(413, {"error": "Request body exceeds the byte limit"})
                return
            if length <= 0:
                raise ValueError("Content-Length must be positive")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body")
            request = json.loads(body)
            if not isinstance(request, dict) or set(request) != {"image"}:
                raise ValueError("Request must be an object with only an image field")
        except (ValueError, UnicodeDecodeError, RecursionError, socket.timeout) as exc:
            self.reply(400, {"error": str(exc)})
            return
        try:
            self.reply(200, self.server.scorer.verify(request["image"]))
        except ValueError as exc:
            self.reply(400, {"error": str(exc)})
        except Exception:
            logging.exception("VLM inference failed")
            self.reply(503, {"error": "VLM inference failed; see scorer terminal"})

    def log_message(self, fmt, *args):
        logging.info(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Hugging Face commit; default model uses a pinned commit")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    backend = MLXBackend(args.model, args.revision)
    logging.info("Loading %s at %s locally", backend.model_name, backend.revision or "default revision")
    backend.load()
    with VerificationServer(("127.0.0.1", args.port), CropScorer(backend)) as server:
        logging.info("Scorer ready at http://127.0.0.1:%s; prompt_hash=%s", server.server_port, backend.prompt_hash)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
