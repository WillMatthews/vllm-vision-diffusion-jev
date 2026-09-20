# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score the frozen generic pilot manifests through local DiffusionGemma HTTP.

Uses each architecture's native answer representation with identical images,
question meanings, candidate descriptions and human targets. The teacher uses
JEV typed labels, not the student's letter-only generation prompt. Raw records
are flushed per image; evaluation.json is complete only after every image.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.train_generic_pilot import (  # noqa: E402
    aggregate,
    digest,
    load_rows,
    options,
    row_metrics,
    write_json,
)
from visual_client import read_image  # noqa: E402


def teacher_question(row):
    """Retain schema meaning while leaving answer formatting to the JEV server."""
    partition = row.get("template_partition", "familiar")
    question = row["question"]
    if partition == "familiar":
        prefix = (
            "Inspect the image and answer the question using visible evidence.\n"
            "Question: "
        )
    elif partition == "unseen":
        prefix = "Use the photograph as evidence for this task.\nTask instructions: "
    else:
        raise ValueError(f"Unknown template partition: {partition}")
    return {
        "type": question["type"],
        "instructions": prefix + question["instructions"],
        "criteria": question["criteria"],
    }


def canonical_probabilities(row, received):
    if received is None:
        raise ValueError(f"Unexpected skipped independent question: {row['id']}")
    choices = options(row["question"])
    if row["question"]["type"] == "score":
        # visual_client resolves server score indices to level descriptions.
        names = [description for _, description in choices]
        if len(set(names)) != len(names):
            raise ValueError("Ordinal descriptions must be unique for teacher mapping")
        if set(received) != set(names):
            raise ValueError("Teacher ordinal legend differs from supplied criteria")
        result = {key: received[description] for key, description in choices}
    else:
        result = received
    if set(result) != {key for key, _ in choices}:
        raise ValueError(f"Teacher returned incorrect option set: {row['id']}")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in result.values()):
        raise ValueError("Teacher returned nonfinite or out-of-range probabilities")
    if not math.isclose(sum(result.values()), 1, abs_tol=1e-6):
        raise ValueError("Teacher probabilities do not sum to one")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--questions-per-image", type=int, default=4)
    args = parser.parse_args()
    address = urlparse(args.endpoint)
    if (
        address.scheme != "http"
        or address.hostname not in {"127.0.0.1", "localhost", "::1"}
        or address.username
        or address.password
        or address.query
        or address.fragment
    ):
        parser.error("Teacher evaluation requires an HTTP loopback endpoint")
    if args.questions_per_image < 1 or args.timeout <= 0:
        parser.error("Question count and timeout must be positive")
    rows = load_rows(args.manifest)
    image_ids = {row["image_id"] for row in rows}
    for manifest in args.exclude_manifest:
        if image_ids & {row["image_id"] for row in load_rows(manifest)}:
            raise ValueError("Image leakage across teacher evaluation manifests")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)
    root = args.image_root.resolve()
    for image_path, group in grouped.items():
        path = (root / image_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid/missing local image: {image_path}")
        if len(group) != args.questions_per_image:
            raise ValueError(
                f"Unexpected question count for {image_path}: {len(group)}"
            )
        if len({row["image_id"] for row in group}) != 1:
            raise ValueError("One image path has multiple image identities")
        if any(row.get("state", {}) != group[0].get("state", {}) for row in group):
            raise ValueError("Grouped questions must share the same supplied context")
        for row in group:
            teacher_question(row)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use an empty teacher evaluation output directory")
    metadata = {
        "manifest_sha256": digest(args.manifest),
        "source_sha256": digest(__file__),
        "teacher_revision": args.model_revision,
        "endpoint": args.endpoint,
        "images": len(grouped),
        "questions": len(rows),
        "questions_per_request": args.questions_per_image,
        "samples": 1,
        "think": 0,
        "note": (
            "Same typed tasks and human targets; architecture-native answer prompts. "
            "HTTP timing includes encoding/transport/server work and is not directly "
            "comparable with learner kernel timings. No calibration fitted."
        ),
    }
    write_json(args.output / "metadata.json", metadata)
    records = []
    requests = []
    started = time.perf_counter()
    try:
        with (args.output / "records.jsonl").open("w") as stream:
            for image_path, group in sorted(grouped.items()):
                group = sorted(group, key=lambda row: str(row["id"]))
                questions = {
                    f"q{i}": teacher_question(row) for i, row in enumerate(group)
                }
                begin = time.perf_counter()
                response = read_image(
                    root / image_path,
                    {"questions": questions, "samples": 1, "think": 0},
                    args.endpoint,
                    state=group[0].get("state", {}),
                    timeout=args.timeout,
                )
                seconds = time.perf_counter() - begin
                if set(response["probabilities"]) != set(questions):
                    raise ValueError("Teacher returned missing or extra question IDs")
                pending = []
                for i, row in enumerate(group):
                    probs = canonical_probabilities(
                        row, response["probabilities"][f"q{i}"]
                    )
                    pending.append(
                        {
                            "id": row["id"],
                            "image": image_path,
                            "image_id": row["image_id"],
                            "family": row["family"],
                            "primitive": row["question"]["type"],
                            "template_partition": row.get(
                                "template_partition", "familiar"
                            ),
                            "target": row["target"],
                            "probabilities": probs,
                            "metrics": row_metrics(row, probs),
                            "request_index": len(requests),
                        }
                    )
                requests.append(
                    {
                        "image": image_path,
                        "questions": len(group),
                        "complete_request_seconds": seconds,
                        "http_latency_ms": response["latency_ms"],
                        "diagnostics": response["diagnostics"],
                        "usage": response["usage"],
                    }
                )
                records.extend(pending)
                for record in pending:
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
                if len(requests) % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "images": len(requests),
                                "questions": len(records),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
    finally:
        write_json(
            args.output / "evaluation.json",
            {
                "metadata": metadata,
                "records": records,
                "requests": requests,
                "complete": False,
            },
        )
    write_json(
        args.output / "evaluation.json",
        {
            "metadata": metadata,
            "records": records,
            "requests": requests,
            "summary": aggregate(records),
            "seconds": time.perf_counter() - started,
            "complete": True,
        },
    )
    print("TEACHER_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
