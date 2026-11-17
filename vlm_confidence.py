"""Binary VLM scores; raw model preference is not measured correctness."""

import json
import math
from pathlib import Path


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def validate_calibration(calibration, *, model=None, prompt_hash=None):
    """Reject invalid parameters and, when supplied, a different scorer identity."""
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a JSON object")
    temperature = finite_number(calibration.get("temperature"), "temperature")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    finite_number(calibration.get("bias"), "bias")
    for field, expected in (("model", model), ("prompt_hash", prompt_hash)):
        actual = calibration.get(field)
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError(f"calibration {field} must be a nonempty string")
        if expected is not None and actual != expected:
            raise ValueError(f"calibration {field} does not match this scorer")
    return calibration


def load_calibration(path, *, model, prompt_hash):
    """Load only a calibration fitted for the current model and prompt pipeline."""
    if any(not isinstance(value, str) or not value.strip() for value in (model, prompt_hash)):
        raise ValueError("current model and prompt_hash must be nonempty strings")
    with Path(path).open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    return validate_calibration(calibration, model=model, prompt_hash=prompt_hash)


def score_margin(margin, calibration=None):
    """Score log P(yes) - log P(no), normalized over these two labels only."""
    margin = finite_number(margin, "raw_margin")
    logit = margin
    if calibration is not None:
        validate_calibration(calibration)
        logit = margin / calibration["temperature"] + calibration["bias"]
    p_drone = sigmoid(logit)
    is_drone = logit >= 0
    return {
        "is_drone": is_drone,
        "confidence": p_drone if is_drone else 1.0 - p_drone,
        "p_drone": p_drone,
        "raw_margin": margin,
        "calibrated": calibration is not None,
    }
