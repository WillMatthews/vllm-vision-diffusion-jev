# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score a new image and question schema with the shared generic pilot model.

No labels or training manifests are required. Probabilities are normalized over
supplied options and are not calibrated confidence estimates. This prototype
supports independent choice/noul/score questions, not conditional dependencies.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from generic_feasibility import options
from train_generic_pilot import Scorer, write_json


def schema_rows(schema, image):
    """Validate supported schema semantics before loading any model weights."""
    if not isinstance(schema, dict):
        raise ValueError("Schema must be an object")
    unknown = set(schema) - {"questions", "state", "samples", "think"}
    if unknown:
        raise ValueError(f"Unsupported schema fields/dependencies: {sorted(unknown)}")
    if schema.get("samples", 1) != 1 or schema.get("think", 0) != 0:
        raise ValueError("Only samples=1 and think=0 are supported")
    questions = schema.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    state = schema.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    rows = []
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(question, dict):
            raise ValueError("Question IDs must be nonempty strings with object values")
        unknown = set(question) - {"type", "instructions", "criteria"}
        if unknown:
            raise ValueError(
                f"{qid}: unsupported fields/ask-if/dependencies: {sorted(unknown)}"
            )
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"{qid}: instructions must be a nonempty string")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in criteria.items()
            ):
                raise ValueError(f"{qid}: choice criteria must map names to text")
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or set(criteria) - {"true", "false"}
                or any(not isinstance(v, str) or not v for v in criteria.values())
            ):
                raise ValueError(f"{qid}: noul criteria only supports true/false text")
        elif kind == "score":
            if not isinstance(criteria, list) or any(
                not isinstance(v, str) or not v for v in criteria
            ):
                raise ValueError(f"{qid}: score criteria must be ordered text labels")
        else:
            raise ValueError(f"{qid}: unsupported type {kind!r}")
        if not 2 <= len(options(question)) <= 26:
            raise ValueError(f"{qid}: expected 2–26 options")
        rows.append(
            {"id": qid, "question": question, "state": state, "image": image.name}
        )
    return rows


def answer(question, probabilities):
    """Return JEV-like distributions and zero-based ordinal expected values."""
    if set(probabilities) != {key for key, _ in options(question)} or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()
    ):
        raise ValueError("Invalid option probabilities")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6):
        raise ValueError("Option probabilities must sum to one")
    result = {"type": question["type"], "probabilities": probabilities}
    if question["type"] == "noul":
        result["noul"] = probabilities["yes"]
    elif question["type"] == "score":
        result["legend"] = dict(options(question))
        result["expected_value"] = sum(
            float(key) * value for key, value in probabilities.items()
        )
    return result


def predict(scorer, rows, verify_reuse=False):
    """Decode afresh and encode one image once, then score each question."""
    torch = scorer.torch
    scorer.image_cache = None
    scorer.feature_owner.get_image_features = scorer.original_features
    answers, diagnostics = {}, {}
    reuse_delta = None
    try:
        with torch.no_grad():
            for index, row in enumerate(rows):
                choices = options(row["question"])
                inputs, ids = scorer.prepare(row, choices)
                if index == 0:
                    ordinary = scorer.logits(inputs, ids)[0] if verify_reuse else None
                    features = scorer.original_features(
                        inputs["pixel_values"], inputs["image_grid_thw"]
                    )
                    scorer.feature_owner.get_image_features = (
                        lambda *a, cached=features, **kw: cached
                    )
                logits, mass = scorer.logits(inputs, ids)
                probabilities = dict(
                    zip((key for key, _ in choices), logits.softmax(-1).cpu().tolist())
                )
                if index == 0 and verify_reuse:
                    reuse_delta = (
                        (ordinary.softmax(-1) - logits.softmax(-1)).abs().max().item()
                    )
                    if reuse_delta > 1e-5:
                        raise RuntimeError(
                            "Sequential visual reuse changed probabilities"
                        )
                answers[row["id"]] = answer(row["question"], probabilities)
                diagnostics[row["id"]] = {
                    "label_mass": mass.exp().item(),
                    "input_tokens": inputs["input_ids"].shape[1],
                }
    finally:
        scorer.feature_owner.get_image_features = scorer.original_features
        scorer.image_cache = None
    return {"answers": answers, "diagnostics": diagnostics}, reuse_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=512 * 512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-reuse", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.benchmark_repeats <= 100:
        parser.error("--benchmark-repeats must be between 0 and 100")
    if min(args.pixels, args.max_input_tokens, args.cpu_threads) < 1:
        parser.error("Pixels, token limit, and CPU threads must be positive")
    args.image = args.image.resolve(strict=True)
    args.image_root = args.image.parent
    rows = schema_rows(json.loads(args.schema.read_text()), args.image)
    scorer = Scorer(args)
    if args.adapter:
        from peft import PeftModel

        scorer.model = PeftModel.from_pretrained(scorer.model, args.adapter)
    scorer.model.eval()
    result, delta = predict(scorer, rows, args.verify_reuse)
    timings = []
    for _ in range(args.benchmark_repeats):
        scorer.torch.accelerator.synchronize()
        begin = time.perf_counter()
        result, _ = predict(scorer, rows)
        scorer.torch.accelerator.synchronize()
        timings.append(time.perf_counter() - begin)
    result["metadata"] = {
        "model": args.model,
        "adapter": args.adapter,
        "image": str(args.image),
        "schema": str(args.schema),
        "pixels": args.pixels,
        "questions": len(rows),
        "reuse_max_delta": delta,
        "probabilities": "Uncalibrated, normalized over supplied options",
        "ordinal_scale": "Zero-based criteria index",
    }
    result["benchmark"] = {
        "seconds": timings,
        "median_seconds": statistics.median(timings) if timings else None,
        "scope": (
            "Complete warm-model requests: image open/decode, preprocessing and "
            "answer-token validation, one fresh vision encoding, sequential "
            "question scoring and CPU probability vectors. Excludes model load, "
            "schema validation, output-file writing, and the optional ordinary "
            "versus cached reuse check. Filesystem cache is not flushed. "
            "One untimed complete request precedes repetitions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "questions": len(rows)}))


if __name__ == "__main__":
    main()
