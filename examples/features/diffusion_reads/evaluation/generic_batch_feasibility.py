# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time batched generic question scoring with one shared image encoding."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from generic_feasibility import options, prompt


def repeat_features(features, count):
    result = copy.copy(features)
    for key, value in features.items():
        if key == "pooler_output":
            result[key] = value * count
        elif isinstance(value, list):
            result[key] = [v.repeat((count,) + (1,) * (v.ndim - 1)) for v in value]
        elif hasattr(value, "ndim"):
            result[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
        else:
            raise ValueError(f"Unsupported visual feature field: {key}")
    return result


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--preprocess-once", action="store_true")
    parser.add_argument("--diagnose-drift", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    root = Path(__file__).resolve().parent
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    schema = json.loads((root / "animal_schema.json").read_text())
    questions = list(schema["questions"].items())
    image = Image.open(root / "images/dog.jpg").convert("RGB")
    original = model.model.get_image_features
    result = {
        "cpu_threads": torch.get_num_threads(),
        "preprocess_once": args.preprocess_once,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": (
            "Decoded image to option probabilities; includes preprocessing, "
            "one visual encoding and batched language scoring. No HTTP/disk time."
        ),
        "runs": [],
    }

    def prepare(count, optimized=False):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": prompt(q, options(q), schema.get("state", {})),
                        },
                    ],
                }
            ]
            for _, q in questions[:count]
        ]
        if optimized:
            visual = processor.image_processor(images=[image], return_tensors="pt")
            image_tokens = int(visual["image_grid_thw"].prod().item()) // 4
            texts = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts = [
                text.replace(
                    processor.image_token, processor.image_token * image_tokens
                )
                for text in texts
            ]
            batch = processor.tokenizer(texts, padding=True, return_tensors="pt")
            batch["mm_token_type_ids"] = (
                batch["input_ids"] == model.config.image_token_id
            ).long()
            batch["pixel_values"] = visual["pixel_values"]
            batch["image_grid_thw"] = visual["image_grid_thw"].repeat(count, 1)
            batch = batch.to("cuda")
        else:
            batch = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
        ids = []
        for _, q in questions[:count]:
            labels = [chr(65 + i) for i in range(len(options(q)))]
            encoded = [
                processor.tokenizer.encode(s, add_special_tokens=False) for s in labels
            ]
            assert all(len(t) == 1 for t in encoded)
            ids.append([t[0] for t in encoded])
        return batch, ids

    def score(batch, ids):
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        return [
            logits[i, token_ids].softmax(-1).cpu().tolist()
            for i, token_ids in enumerate(ids)
        ]

    with torch.no_grad():
        for count in (1, 8, 24):
            batch, ids = prepare(count)
            ordinary = score(batch, ids)
            if args.preprocess_once:
                optimized, optimized_ids = prepare(count, optimized=True)
                assert ids == optimized_ids
                for key in batch:
                    if key == "pixel_values":
                        patches = optimized[key].shape[0]
                        assert torch.equal(batch[key][:patches], optimized[key])
                    else:
                        assert torch.equal(batch[key], optimized[key]), key
            durations = []
            for repeat in range(4):
                torch.accelerator.synchronize()
                torch.accelerator.reset_peak_memory_stats()
                start = time.perf_counter()
                batch, ids = prepare(count, optimized=args.preprocess_once)
                # Grid t*h*w counts raw patches before spatial merging.
                first_grid = batch["image_grid_thw"][:1]
                patch_count = int(first_grid.prod().item())
                features = original(batch["pixel_values"][:patch_count], first_grid)
                expanded = repeat_features(features, count)
                model.model.get_image_features = (
                    lambda *a, cached=expanded, **kw: cached
                )
                try:
                    cached = score(batch, ids)
                    torch.accelerator.synchronize()
                    durations.append(time.perf_counter() - start)
                finally:
                    model.model.get_image_features = original
            delta = max(
                abs(a - b) for ap, bp in zip(ordinary, cached) for a, b in zip(ap, bp)
            )
            seconds = sum(durations[1:]) / 3
            result["runs"].append(
                {
                    "questions": count,
                    "seconds": durations[1:],
                    "warmup_seconds": durations[0],
                    "mean_seconds": seconds,
                    "usd_per_million_requests": seconds * args.hourly_usd / 3600 * 1e6,
                    "peak_allocated_gib": torch.accelerator.max_memory_allocated()
                    / 2**30,
                    "reuse_max_probability_delta": delta,
                    "reuse_passed": delta <= 1e-5,
                    "ordinary_probabilities": ordinary,
                    "cached_probabilities": cached,
                    "feature_fields": list(features.keys()),
                }
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("Batch", count, "seconds", seconds, "reuse delta", delta, flush=True)
            if delta > 1e-5 and not args.diagnose_drift:
                raise RuntimeError("Batched visual reuse changed probabilities")
    print("BATCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
