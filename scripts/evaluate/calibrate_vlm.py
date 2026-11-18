"""Fit two-parameter calibration on scored JSONL, then evaluate disjoint sources.

Run: python3 scripts/evaluate/calibrate_vlm.py --calibration scores.jsonl --evaluation held_out.jsonl --output calibration.json
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm_confidence import finite_number, score_margin, sigmoid, validate_calibration


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                record["raw_margin"] = finite_number(record.get("raw_margin"), "raw_margin")
                if not isinstance(record.get("label"), bool):
                    raise ValueError("label must be true (drone) or false")
                for field in ("source_id", "clip_id", "model", "prompt_hash"):
                    if not isinstance(record.get(field), str) or not record[field].strip():
                        raise ValueError(f"{field} must be a nonempty string")
                records.append(record)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no scored records")
    return records


def validate_splits(calibration_records, evaluation_records):
    """Different file paths alone do not prevent adjacent-frame/source leakage."""
    for field in ("source_id", "clip_id"):
        overlap = {r[field] for r in calibration_records} & {r[field] for r in evaluation_records}
        if overlap:
            raise ValueError(f"calibration/evaluation overlap in {field}: {sorted(overlap)}")
    identities = {(r["model"], r["prompt_hash"]) for r in calibration_records + evaluation_records}
    if len(identities) != 1:
        raise ValueError("all records must use the same model and prompt_hash")
    if {r["label"] for r in calibration_records} != {False, True}:
        raise ValueError("calibration split must contain both drone and non-drone labels")
    model, prompt_hash = identities.pop()
    return {"model": model, "prompt_hash": prompt_hash}


def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def fit_calibration(records):
    """Damped Newton fit of positive slope and bias; no image/model training."""
    if not records or {r["label"] for r in records} != {False, True}:
        raise ValueError("fitting requires both labels")
    scale = max(1.0, max(abs(r["raw_margin"]) for r in records))
    samples = [(r["raw_margin"] / scale, r["label"]) for r in records]
    count = len(samples)
    # The small penalty gives a finite fit even when the labeled split is separable.
    penalty = 1e-6
    minimum_slope = 1e-6

    def objective(slope, bias):
        loss = math.fsum(softplus(-(slope * x + bias) if y else slope * x + bias) / count
                         for x, y in samples)
        return loss + penalty * (slope * slope + bias * bias) / 2

    slope, bias = 1.0, 0.0
    converged = False
    for iteration in range(100):
        ga, gb = penalty * slope, penalty * bias
        haa, hab, hbb = penalty, 0.0, penalty
        for x, y in samples:
            p = sigmoid(slope * x + bias)
            residual, weight = (p - y) / count, p * (1 - p) / count
            ga += x * residual
            gb += residual
            haa += x * x * weight
            hab += x * weight
            hbb += weight
        projected_ga = 0.0 if slope == minimum_slope and ga > 0 else ga
        if max(abs(projected_ga), abs(gb)) < 1e-9:
            converged = True
            break
        determinant = haa * hbb - hab * hab
        delta_a = (hbb * ga - hab * gb) / determinant
        delta_b = (haa * gb - hab * ga) / determinant
        if slope == minimum_slope and delta_a > 0:
            delta_a, delta_b = 0.0, gb / hbb
        old_loss = objective(slope, bias)
        step = 1.0
        while step > 1e-10:
            next_a = max(minimum_slope, slope - step * delta_a)
            next_b = bias - step * delta_b
            if objective(next_a, next_b) < old_loss:
                slope, bias = next_a, next_b
                break
            step *= 0.5
        else:
            # Floating-point precision can stop progress near the optimum.
            converged = max(abs(projected_ga), abs(gb)) < 1e-7
            break
    if not converged:
        raise ValueError("calibration optimizer did not converge; inspect labeled scores")
    temperature = finite_number(scale / slope, "fitted temperature")
    return {
        "temperature": temperature,
        "bias": bias,
        "fit": {"method": "positive-slope logistic NLL with L2 penalty",
                "l2_penalty": penalty, "margin_scale": scale,
                "iterations": iteration + 1, "slope_at_lower_bound": slope == minimum_slope},
    }


def evaluate_records(records, calibration=None, threshold=0.85):
    threshold = finite_number(threshold, "threshold")
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.5 and 1")
    if not records:
        raise ValueError("evaluation split must not be empty")
    if calibration is not None:
        validate_calibration(calibration)
    confusion = dict(tp=0, fp=0, tn=0, fn=0)
    accepted_confusion = dict(tp=0, fp=0, tn=0, fn=0)
    bins = [[] for _ in range(10)]
    brier, nll = [], []
    accepted, accepted_correct = 0, 0
    for record in records:
        score = score_margin(record["raw_margin"], calibration)
        p, prediction, label = score["p_drone"], score["is_drone"], record["label"]
        key = ("tp" if label else "fp") if prediction else ("fn" if label else "tn")
        confusion[key] += 1
        confidence, correct = score["confidence"], prediction == label
        bins[min(9, int(confidence * 10))].append((confidence, correct))
        brier.append((p - label) ** 2)
        logit = record["raw_margin"]
        if calibration is not None:
            logit = logit / calibration["temperature"] + calibration["bias"]
        nll.append(softplus(-logit if label else logit))
        if confidence >= threshold:
            accepted += 1
            accepted_correct += correct
            accepted_confusion[key] += 1
    count = len(records)
    reliability = []
    for index, entries in enumerate(bins):
        if entries:
            reliability.append({"lower": index / 10, "upper": (index + 1) / 10,
                                "count": len(entries),
                                "mean_confidence": math.fsum(c for c, _ in entries) / len(entries),
                                "accuracy": sum(ok for _, ok in entries) / len(entries)})
    return {
        "count": count,
        "brier": math.fsum(value / count for value in brier),
        "nll": math.fsum(value / count for value in nll),
        "ece": math.fsum(abs(b["accuracy"] - b["mean_confidence"]) * b["count"] / count
                         for b in reliability),
        "reliability": reliability,
        "confusion": confusion,
        "threshold": threshold,
        "accepted": accepted,
        "coverage": accepted / count,
        "accepted_accuracy": accepted_correct / accepted if accepted else None,
        "accepted_confusion": accepted_confusion,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path, help="dedicated scored calibration JSONL")
    parser.add_argument("--evaluation", required=True, type=Path, help="separate held-out scored evaluation JSONL")
    parser.add_argument("--output", required=True, type=Path, help="new calibration JSON; existing files are preserved")
    parser.add_argument("--threshold", type=float, default=0.85, help="report confidence coverage; do not tune on test data")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        calibration_records = load_records(args.calibration)
        evaluation_records = load_records(args.evaluation)
        identity = validate_splits(calibration_records, evaluation_records)
        fitted = {**identity, **fit_calibration(calibration_records)}
        report = {
            "calibration_count": len(calibration_records),
            "calibration_sources": sorted({r["source_id"] for r in calibration_records}),
            "calibration_clips": sorted({r["clip_id"] for r in calibration_records}),
            "evaluation_sources": sorted({r["source_id"] for r in evaluation_records}),
            "evaluation_clips": sorted({r["clip_id"] for r in evaluation_records}),
            "evaluation_raw": evaluate_records(evaluation_records, threshold=args.threshold),
            "evaluation_calibrated": evaluate_records(evaluation_records, fitted, args.threshold),
        }
        fitted["validation"] = report
        encoded = json.dumps(fitted, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(encoded, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
