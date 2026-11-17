# Local vision model selection

Evidence checked **2026-11-17**. Target: an Apple Silicon Mac with **8 GB total RAM**, sharing memory with macOS, YOLO/ByteTrack, video buffers, and the dashboard. One resident local VLM worker evaluates one selected crop per track event; the crop budget is at most 384 pixels per edge. It scores the first answer token with thinking disabled.

Use **`mlx-community/Qwen3.5-2B-4bit`** as the conservative default. It is a native vision model under Apache 2.0, has a compatible released MLX implementation, and leaves substantially more memory headroom than the newer Gemma checkpoints. This is a resource-based recommendation, **not a measured claim of best drone accuracy**. Qwen3.5-4B is the first larger challenger to benchmark if this Mac can keep it resident alongside the detector without swap or unacceptable frame-rate loss.

## Actual checkpoint size

These are exact sums of `.safetensors` file sizes from Hugging Face's public model-tree API, including vision weights: `https://huggingface.co/api/models/{model_id}/tree/main?recursive=true&limit=1000`. They are **download/storage sizes, not peak inference RAM**. For the first four rows, HTTP range reads of only the safetensors JSON metadata headers also verified the tensor groups; no weight tensors were downloaded. The selected default revision is `674aaa7240b91e8012fcad5d791b7dfe5ba90207`.

| MLX checkpoint | Full weight bytes | GiB | Included vision tensor bytes | Assessment for 8 GB total |
|---|---:|---:|---:|---|
| [Qwen3.5-2B-4bit](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit) | 1,722,271,785 | 1.60 | 662,833,152 | Default; still requires runtime measurement |
| [Qwen3.5-4B-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) | 3,034,300,695 | 2.83 | 667,028,480 | Higher-capacity challenger; tighter headroom |
| [gemma-4-e2b-it-4bit](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit) | 3,550,670,554 | 3.31 | 334,730,112 | More RAM than Qwen 4B; not the default |
| [gemma-4-e4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) | 5,146,800,534 | 4.79 | 334,730,112 | Too little comfortable shared-memory headroom |
| [Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 16,054,541,349 | 14.95 | Included in the three model shards | Exceeds all physical RAM before inference |

Gemma's standard MLX checkpoints also contain **609,650,176 bytes of audio-tower tensors** each. This image-only service does not need audio, but those tensors are part of the published checkpoint. Do not estimate its footprint from the effective language-model parameter count or assume the loader removes unused modalities.

The [official Qwen3.8 catalog](https://huggingface.co/api/models?author=Qwen&search=Qwen3.8&limit=100) currently has 27B, Flash-Next, and 2.4T-A95B families, plus FP8 variants. The smallest official checkpoint is **27B**, not a 2B or 4B successor. [Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a native vision MoE model with about **180B total stored parameters**; active parameters do not determine resident weight memory. Neither is suitable here.

The [Gemma 4 model card](https://huggingface.co/google/gemma-4-E2B-it) confirms that E2B and E4B both accept images. Their names mean **effective** parameters: approximately 2.3B and 4.5B effective, but **5.1B and 8B total with embedding tables**. Both also support audio. The [official memory guidance](https://gemma.ai/models/gemma-4/#gemma-4-inference-memory-requirements) gives approximate Q4_0 static-load figures of 2.9 GB and 4.5 GB; these are estimates for other quantization/runtime configurations, not measurements of these MLX artifacts or the complete tracker. KV cache, activations, software, and competing applications require further memory.

## What generic accuracy evidence establishes

The [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B) explicitly reports both thinking and non-thinking vision results. Selected **non-thinking** scores are MMMU 64.2, MMMU-Pro 47.7, RealWorldQA 71.2, MMStar 68.0, and OCRBench 85.4. Its comparison against Qwen3-VL-2B is mixed: higher MMMU, MMMU-Pro, RealWorldQA, and OCRBench, but slightly lower MMStar and MMBench. That supports trying the newer model; it does not justify universal superiority.

The [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B) reports stronger generic vision results, including MMMU 77.6, MMMU-Pro 66.3, RealWorldQA 79.5, and MMStar 78.3. The card defaults this model to thinking and reports generated-answer benchmarks without a matched first-token, non-thinking evaluation; those scores do not establish the same improvement for this crop classifier.

The Gemma 4 card reports MMMU-Pro **44.2 for E2B** and **52.6 for E4B**, plus other reasoning/document benchmarks. Cross-vendor settings differ, and document or visual-reasoning benchmarks are not distant drone-versus-bird/kite tests. Quantized versions can also differ from the published full-precision results. No source inspected supplied a directly applicable held-out drone/bird/kite comparison for these checkpoints.

## Released runtime and binary scoring

Pin **`mlx-vlm==0.8.0`**, [released 2026-11-16](https://github.com/Neywa-Labs/mlx-vlm/releases/tag/v0.8.0), rather than relying on its moving `main` branch. Its [package metadata](https://github.com/Neywa-Labs/mlx-vlm/blob/v0.8.0/pyproject.toml) requires Python >=3.10; its [dependencies](https://github.com/Neywa-Labs/mlx-vlm/blob/v0.8.0/requirements.txt) require MLX >=0.32.4 and Transformers >=5.14.0. Released source contains native `qwen3_5`, [Gemma 4](https://github.com/Neywa-Labs/mlx-vlm/blob/v0.8.0/mlx_vlm/models/gemma4/README.md), and [Qwen3.8-Flash-Next](https://github.com/Neywa-Labs/mlx-vlm/blob/v0.8.0/mlx_vlm/models/qwen4_exp/README.md) implementations. Native architecture support plus the inspected affine quantization/config metadata establishes a plausible loading path for the standard checkpoints; an actual weight-loading smoke test is still required.

Keep the model and processor loaded once. Format one image with the checkpoint's chat template and **`enable_thinking=False`**, run one forward pass, and read the final input position's logits. Python constructs the response JSON. Generated prose, chain of thought, and model-written confidence numbers are unnecessary.

The default Qwen [chat template](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit/blob/674aaa7240b91e8012fcad5d791b7dfe5ba90207/chat_template.jinja) supplies an empty, already-closed thinking block when thinking is disabled. The following single-token vocabulary entries were inspected in the actual published tokenizer JSON files; verify them using the loaded tokenizer before scoring, rather than sharing IDs between models.

| Model tokenizer | `Yes`, `yes`, ` Yes`, ` yes` | `No`, `no`, ` No`, ` no` |
|---|---|---|
| [Qwen3.5-2B](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit/resolve/674aaa7240b91e8012fcad5d791b7dfe5ba90207/tokenizer.json) | 9175, 9405, 7179, 9542 | 2665, 2083, 2233, 874 |
| [Gemma 4 E2B](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit/resolve/238767527555cb75a05732a84dff5d6ba0dd6809/tokenizer.json) | 10784, 4443, 8438, 11262 | 3771, 1904, 2301, 951 |

The implementation scores the exact lowercase `yes` and `no` tokens under an explicit one-word prompt, then normalizes over those two labels. The alternate variants above are not summed; changing the label mapping requires fresh calibration. This gives a **conditional label score**, not an established probability of correctness. See [vlm-confidence-research.md](vlm-confidence-research.md) for calibration and abstention. Include the exact checkpoint revision, rendered prompt, label mapping, preprocessing, and runtime version in calibration identity.

Qwen's [published image processor](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit/blob/674aaa7240b91e8012fcad5d791b7dfe5ba90207/preprocessor_config.json) defaults to 65,536 minimum pixels and 16,777,216 maximum pixels. **Resizing the supplied crop alone does not guarantee a 384-pixel processor budget**: minimum-area resizing can enlarge narrow crops. Bound actual processed dimensions/image-token count, including any patch rounding or padding, and do not rely on a maximum-area constraint to bound the longest edge. Interpolation adds no missing object detail.

## Licenses and newer mobile variants

The official Qwen3.5 and Qwen3.8-27B model cards specify **Apache 2.0**. Flash-Next uses **Qwen Community 1.0**, not Apache 2.0. The current official Gemma 4 cards and [license text](https://gemma.ai/resources/apache-2/) specify **Apache 2.0**. The inspected July community Gemma MLX conversion cards still say `license: gemma`; this is a metadata mismatch against the current official source, so retain the official license/attribution information and verify the conversion's stated source revision before redistribution.

[Gemma E2B QAT mobile](https://huggingface.co/mlx-community/gemma-4-E2B-it-qat-mobile) and [E4B QAT mobile](https://huggingface.co/mlx-community/gemma-4-E4B-it-qat-mobile) have smaller safetensors totals, 2,398,140,572 and 3,455,955,670 bytes, and image configs. However, their configs retain `quant_method: gemma` and mixed 2/4/8-bit module recipes, rather than MLX affine quantization. The released [loader](https://github.com/Neywa-Labs/mlx-vlm/blob/v0.8.0/mlx_vlm/utils.py) and [checkpoint transformer](https://github.com/Neywa-Labs/mlx-vlm/blob/v0.8.0/mlx_vlm/quantization/checkpoint.py) contain no handler for that quantization method. Their small files therefore do not establish usable MLX support. Google's still smaller mobile RAM estimates refer to **LiteRT-LM**, a different runtime. Standard E2B QAT-4bit is also not automatically smaller: its inspected MLX file is 4,329,240,325 bytes because some layers use higher precision. These variants are not selected for this implementation.

## Required local decision test

Use separated calibration and held-out clip/source splits with real distant drones, birds, kites, and ambiguous tiny targets. Evaluate the deployed crop selection, resolution, no-thinking prompt, quantization, and first-token scorer together. Compare Qwen 2B against Qwen 4B first; add Gemma E2B only if enough RAM remains. Report drone recall, false confirmations on birds/kites, precision, abstention coverage, calibration error, and results by target pixel size. Measure cold load, warm per-event p50/p95 latency, peak process/system memory, swap, and detector FPS with the VLM resident. A candidate wins only if the actual task metrics improve within the shared hardware budget.

## Native smoke check (2026-11-17)

The pinned default model was downloaded and ran locally with the final prompt. On a **16 GiB Mac**, model loading took 6.308 seconds; two synthetic white JPEG reviews in the same resident process took 1.084 and 0.415 seconds. A 384-by-100 crop and a 1-by-384 crop both produced the bounded `image_grid_thw = [[1, 24, 24]]` (384-by-384 pixels after aspect-preserving square padding). Both selected `no`, with raw confidence 0.679 and 0.706; both remain unverified at the provisional 0.85 threshold.

Recorded peaks: MLX allocation 2,286,810,394 bytes, process RSS 1,074,315,264 bytes, and macOS process memory footprint 3,584,855,232 bytes. These metrics overlap and must **not** be added together. The full local artifact is `runs/vlm_smoke.json` (ignored by Git). The real tracker CLI also completed a 12-frame synthetic video, and the dashboard HTTP endpoints passed a separate smoke check.

These checks establish runtime compatibility, not drone recognition quality, sustained detector FPS, concurrent shared-GPU performance, or 8 GB system headroom. No real drone calibration or held-out flight benchmark has been performed; the recommendation remains provisional for those task metrics.
