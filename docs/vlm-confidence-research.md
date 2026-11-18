# VLM confidence for drone verification

Researched on 2026-11-17. This note describes the original binary-confidence experiment. The active scorer now chooses a broad object label and returns a binary result without a confidence gate; see `vlm-prompts.md`. Its API no longer emits these binary scores or accepts this calibration artifact. This implementation uses a binary token-probability margin and optional post-hoc calibration. It does not ask a vision-language model to invent a confidence percentage. No drone calibration dataset has been fitted or drone correctness rate measured in this change; `0.85` is a provisional decision threshold.

## Algorithms and evidence

| Method | Evidence read | Cost and decision for this tracker |
| --- | --- | --- |
| Answer-token probability / logit margin | Kadavath et al., *Language Models (Mostly) Know What They Know* [1], abstract: formatted true/false probabilities can support self-evaluation, and calibration can deteriorate on new tasks. | Score `yes` and `no` before sampling. This provides a cheap model preference signal from one image, requiring no generated reasoning. The text-only result does not validate this small VLM on drones. |
| Temperature scaling | Guo et al., *On Calibration of Modern Neural Networks* [2], abstract and sections 4–5: learn a positive temperature using validation NLL. It often improved calibration on the studied vision/NLP classifiers. | One fitted scalar, negligible inference cost; it preserves the decision boundary. The reported result is not a guarantee for drone crops or an instruction-tuned VLM. |
| Platt-style logistic calibration | Lin, Lin, and Weng, *A Note on Platt's Probabilistic Outputs for Support Vector Machines* [3], sections 1–3: fit a sigmoid to scores; use Newton updates, backtracking, and stable objective calculations. Guo [2] describes fitting slope and bias on validation data. | This tool fits `sigmoid(margin / temperature + bias)` with positive temperature. Bias corrects a systematic preference for one class and can change the selected label. Two parameters require fewer data than a flexible calibration map. |
| Verbalized confidence and self-evaluation | Lin, Hilton, and Evans, *Teaching Models to Express Their Uncertainty in Words* [4], abstract: explicitly trained GPT-3 models can express calibrated verbal uncertainty. Kadavath [1] reports answer-conditioned `P(True)` and benefits from considering multiple samples. | A generic prompt asking for a percentage is not the trained setup in [4]. Separate self-evaluation adds inference and can repeat the original visual mistake. Neither is used by default. |
| Multi-sample consistency / semantic entropy | Kuhn, Gal, and Farquhar, *Semantic Uncertainty* [5], abstract: cluster answers by meaning to estimate uncertainty. Zhang, Zhang, and Zheng, *VL-Uncertainty* [6], abstract: perturb visual/textual prompts, sample responses, cluster meanings, and measure cluster entropy across 10 LVLMs/four benchmarks. | Multiple generations and semantic comparisons increase latency. With only `yes`/`no`, clusters are already the two labels; repeated samples estimate a binary distribution that the token logits already supply for the fixed image/prompt. Perturbations may reveal instability, but that signal still needs drone validation. |
| VLM visual grounding and decoupled confidence | Xiao, Xu, and Gan, *VL-Calibration* [7], abstract, sections 2–3, and appendix E: distinguish visual and reasoning confidence, measure visual grounding by output KL divergence under image perturbation, combine it with token entropy, and train using RL. The paper explicitly warns that language priors can conceal visual uncertainty. | Valuable evidence that a confident VLM can still misread an image. Its method requires extra image passes, generated rationales, and model training; its published setup is not a drop-in two-scalar calibration. Not implemented on an 8 GB laptop. |

Entropy of a single binary distribution is highest at `p_drone = 0.5`. It is useful for expressing uncertainty but is only a transformation of the same probability, not an independent confidence estimate. Agreement between repeated answers also cannot detect a systematic confident mistake. Neither a sharp crop nor a larger pixel count should be converted into a fabricated confidence bonus.

## Implemented score

The local scorer computes `margin = log P(yes | image, prompt) - log P(no | image, prompt)` before temperature sampling, top-k/top-p filtering, or text generation. The exact yes/no token representation and prompt pipeline belong to the scorer identity.

Without calibration:

```text
p_drone = sigmoid(margin)
```

This equals `P(yes) / (P(yes) + P(no))`: probability normalized over the two allowed answers. It can be high even when substantial model probability belongs to other answers. It is not a measured probability that the image actually contains a drone, an out-of-distribution detector, or an assurance of visual grounding.

With calibration:

```text
p_drone = sigmoid(margin / temperature + bias)
is_drone = (margin / temperature + bias >= 0)
confidence = p_drone if is_drone else 1 - p_drone
```

`score_margin(margin, calibration=None)` returns `is_drone`, `confidence`, `p_drone`, `raw_margin`, and `calibrated`. Without a calibration object, `calibrated` is false. True means a supplied calibration transformation was applied; its quality must still be judged from held-out metrics. `load_calibration(path, *, model, prompt_hash)` rejects missing/nonfinite parameters, nonpositive temperature, and model or prompt identity mismatches. Identity must change with the scorer prompt/template/tokenization or preprocessing that affects the scores.

## Fitting and evaluation

Supply two deliberately separated JSONL files. Each row is a scored verification event with its manually verified binary label and the identity returned by the scorer:

```json
{"raw_margin": 1.7, "label": true, "source_id": "original-capture-A", "clip_id": "capture-A-clip-01", "model": "mlx-community/Qwen3.5-2B-4bit", "prompt_hash": "copy-the-scorer-prompt-hash"}
```

`label: true` means the crop contains the target drone; `false` means a non-drone target. The identifiers must refer to the original source and clip, including alternate encodings, crops, or excerpts of that same source. Both identifiers are required because different excerpts of one recording should not silently become independent test material. Manifests must use the same model and prompt hash. The calibration split must contain both labels.

```sh
python3 scripts/evaluate/calibrate_vlm.py \
  --calibration scored-calibration.jsonl \
  --evaluation scored-heldout.jsonl \
  --output calibration.json \
  --threshold 0.85
```

The tool rejects any shared `source_id` or `clip_id` across the two files. It never creates a random frame split, extracts training examples from existing footage, changes ground truth, or overwrites an existing output. Existing held-out tracking clips and bird/kite negative clips remain evaluation data unless a distinct, deliberately selected calibration source set is created. A file with only negative bird/kite examples cannot fit both classes.

The dependency-free fit minimizes mean binary NLL plus a fixed `1e-6` L2 penalty on the slope and bias after scaling margins by their maximum absolute value (at least 1). The small penalty prevents an infinite fit on perfectly separable data; it is recorded in the artifact. A positive slope lower bound of `1e-6` preserves score ordering and finite temperature. If that bound is reached, the artifact records `slope_at_lower_bound: true`: the calibration labels provide no usable positive ranking evidence, and this must not be presented as successful calibration. The fitter raises an error if its damped Newton optimizer does not converge. This is a regularized Platt-style fit, not the exact target-smoothing implementation in [3].

The output contains parameters, identity, fitting details, source/clip manifests, and **raw and calibrated evaluation** results:

- Brier score: mean `(p_drone - label)^2`, lower is better.
- Binary NLL, computed from logits with stable softplus rather than clipping saturated probabilities, lower is better.
- Chosen-label ECE with 10 equal-width bins, plus each populated bin's mean confidence, accuracy, and count. ECE depends on binning and sample count; it is not a guarantee.
- `tp`, `fp`, `tn`, `fn` confusion counts for the binary decision, and the same counts for accepted decisions.
- Acceptance coverage at the supplied confidence threshold and accuracy among accepted examples (`null` when none are accepted).

Use one selected crop per verification event, matching deployment. Correlated frames within a source can still inflate the apparent sample size, even with disjoint evaluation sources; the report gives source/clip IDs and counts to make this inspectable. Include small/distant drones, blur, occlusion, birds, kites, background clutter, and negative detector proposals. Preserve realistic detector proposal/class prevalence: a balanced dataset can shift fitted bias and fail to represent live conditions. Choose thresholds on calibration/development data and then freeze them; do not optimize the threshold after inspecting the held-out report.

Changing model weights, quantization, prompt formatting, preprocessing, camera/domain, or the detector proposal distribution can invalidate calibration. Source identity validation prevents declared overlap; it cannot discover renamed duplicate videos or false labels. Real drone accuracy, per-class errors, reliability, latency, GPU contention, and peak memory remain unmeasured here.

## References

Only the source sections described above were read; abstract-only entries do not claim a replication or full-method review.

1. Kadavath et al. (2022), [*Language Models (Mostly) Know What They Know*](https://arxiv.org/abs/2207.05221).
2. Guo, Pleiss, Sun, and Weinberger (2017), [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599), [paper HTML](https://arxiv.org/html/1706.04599v2).
3. Lin, Lin, and Weng, [*A Note on Platt's Probabilistic Outputs for Support Vector Machines*](https://www.csie.ntu.edu.tw/~htlin/paper/doc/plattprob.pdf), university-hosted paper.
4. Lin, Hilton, and Evans (2022), [*Teaching Models to Express Their Uncertainty in Words*](https://arxiv.org/abs/2205.14334).
5. Kuhn, Gal, and Farquhar (2023), [*Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*](https://arxiv.org/abs/2302.09664).
6. Zhang, Zhang, and Zheng (2024), [*VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation*](https://arxiv.org/abs/2411.11919).
7. Xiao, Xu, and Gan (ACL 2026), [*VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning*](https://aclanthology.org/2026.acl-long.2074/), DOI [10.18653/v1/2026.acl-long.2074](https://doi.org/10.18653/v1/2026.acl-long.2074).
