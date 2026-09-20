# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify generic inputs and benchmark 100 fixed validation images per model.

Only questions, state and image paths are copied into requests. Targets are
neither passed to inference nor used for example or model selection.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from predict_generic_pilot import predict, schema_rows
from train_generic_pilot import Scorer, write_json


def requests(manifest, image_root, count=3):
    """Take the first image groups in manifest order, without label access."""
    groups = {}
    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = row["image"]
        if image not in groups and len(groups) == count:
            continue
        groups.setdefault(image, []).append(
            {
                "id": row["id"],
                "question": row["question"],
                "state": row.get("state", {}),
            }
        )
    if len(groups) != count:
        raise ValueError(f"Expected at least {count} validation images")
    root = image_root.resolve()
    result = []
    for image, rows in groups.items():
        path = (root / image).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes image root")
        if len(rows) != 4 or len({row["id"] for row in rows}) != 4:
            raise ValueError("Each selected image must have four unique questions")
        if any(row["state"] != rows[0]["state"] for row in rows):
            raise ValueError("Per-question state differs within an image")
        schema = {
            "state": rows[0]["state"],
            "questions": {row["id"]: row["question"] for row in rows},
        }
        schema_rows(schema, path)
        result.append((path, schema))
    return result


def benchmark_loaded_model(args):
    """Time decoded-image requests using one model load and no reuse checks."""
    cases = requests(args.manifest, args.image_root, count=100)
    identifiers = {}
    for line in args.manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            identifiers[(args.image_root / row["image"]).resolve()] = row["image_id"]
    prepared = [(image, schema_rows(schema, image)) for image, schema in cases]
    args.pixels = 512 * 512
    args.max_input_tokens = 2048
    args.cpu_threads = 4
    args.seed = 42
    scorer = Scorer(args)
    if args.benchmark_model == "trained":
        from peft import PeftModel

        scorer.model = PeftModel.from_pretrained(scorer.model, str(args.adapter))
    scorer.model.eval()
    for image, rows in prepared[:3]:
        scorer.image_root = image.parent
        predict(scorer, rows)
    result = {
        "complete": False,
        "model": args.benchmark_model,
        "images": 100,
        "questions_per_image": 4,
        "repeats": 2,
        "untimed_warmups": 3,
        "records": [],
        "scope": (
            "One loaded model; in-process complete requests including image "
            "open/decode, preprocessing, one fresh complete vision encoding, "
            "four sequential question scores, CPU probability vectors and CUDA "
            "synchronization. Excludes model load, schema validation, result "
            "file writes and ordinary/cached equivalence checks. OS filesystem "
            "cache is not flushed. Teacher HTTP measurements additionally "
            "include transport/server overhead; these boundaries differ."
        ),
    }
    output = args.output / f"representative-{args.benchmark_model}.json"
    write_json(output, result)
    try:
        for repeat in range(2):
            for image, rows in prepared:
                scorer.image_root = image.parent
                scorer.torch.accelerator.synchronize()
                started = time.perf_counter()
                prediction, _ = predict(scorer, rows)
                scorer.torch.accelerator.synchronize()
                elapsed = time.perf_counter() - started
                if len(prediction["answers"]) != 4:
                    raise RuntimeError("Expected four generic answers")
                result["records"].append(
                    {
                        "image_id": identifiers[image],
                        "image": str(image.relative_to(args.image_root.resolve())),
                        "question_ids": [row["id"] for row in rows],
                        "repeat": repeat,
                        "seconds": elapsed,
                    }
                )
                write_json(output, result)
        seconds = [record["seconds"] for record in result["records"]]
        result.update(
            complete=True,
            mean_seconds=statistics.mean(seconds),
            median_seconds=statistics.median(seconds),
            requests_per_second=len(seconds) / sum(seconds),
        )
    finally:
        write_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-model", choices=("base", "trained"))
    args = parser.parse_args()
    if args.benchmark_model:
        benchmark_loaded_model(args)
        return
    cases = requests(args.manifest, args.image_root)
    if not (args.adapter / "adapter_config.json").is_file():
        raise ValueError("The fixed final adapter checkpoint is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = {"complete": False, "records": [], "selection": "First three images"}
    summary = args.output / "summary.json"
    write_json(summary, progress)
    try:
        for index, (image, schema) in enumerate(cases, start=1):
            request = args.output / f"image-{index}-schema.json"
            write_json(request, schema)
            for name, adapter in (("base", None), ("trained", str(args.adapter))):
                output = args.output / f"image-{index}-{name}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("predict_generic_pilot.py")),
                    "--model",
                    args.model,
                    "--image",
                    str(image),
                    "--schema",
                    str(request),
                    "--output",
                    str(output),
                    "--verify-reuse",
                    "--benchmark-repeats",
                    "3",
                ]
                if adapter is not None:
                    command.extend(["--adapter", adapter])
                subprocess.run(command, check=True, timeout=120)
                result = json.loads(output.read_text())
                if result["metadata"]["questions"] != 4:
                    raise RuntimeError(
                        "Generic inference returned wrong question count"
                    )
                progress["records"].append(
                    {
                        "image": str(image),
                        "model": name,
                        "output": output.name,
                        "reuse_max_delta": result["metadata"]["reuse_max_delta"],
                        "seconds": result["benchmark"]["seconds"],
                        "median_seconds": result["benchmark"]["median_seconds"],
                    }
                )
                write_json(summary, progress)
        for name in ("base", "trained"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model",
                    args.model,
                    "--adapter",
                    str(args.adapter),
                    "--manifest",
                    str(args.manifest),
                    "--image-root",
                    str(args.image_root),
                    "--output",
                    str(args.output),
                    "--benchmark-model",
                    name,
                ],
                check=True,
                timeout=240,
            )
            progress.setdefault("representative_outputs", []).append(
                f"representative-{name}.json"
            )
            write_json(summary, progress)
        progress["complete"] = True
        print("GENERIC_INPUT_BENCHMARK_COMPLETE", flush=True)
    except Exception as exc:
        progress["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(summary, progress)


if __name__ == "__main__":
    main()
