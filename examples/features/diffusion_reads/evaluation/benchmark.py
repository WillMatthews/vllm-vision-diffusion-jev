# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure sequential, warm question-count latency on a fixed photo suite."""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_client import read_image  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.hourly_usd <= 0:
        parser.error("repeats and hourly-usd must be positive")
    root = Path(__file__).resolve().parent
    schema = json.loads((root / "animal_schema.json").read_text())
    counts = [1, 2, 4, 8, 16, 24]
    photos = ["dog", "cat"]
    schemas = {
        n: {**schema, "questions": dict(list(schema["questions"].items())[:n])}
        for n in counts
    }
    records = []
    jobs = [(n, photo) for n in counts for photo in photos]
    for n, photo in jobs:
        read_image(root / f"images/{photo}.jpg", schemas[n], args.endpoint)
        print(f"Warmup: {n} questions, {photo}", flush=True)
    jobs *= args.repeats
    random.Random(42).shuffle(jobs)
    for n, photo in jobs:
        result = read_image(root / f"images/{photo}.jpg", schemas[n], args.endpoint)
        records.append({"questions": n, "image": photo, **result})
        args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
        print(
            f"{len(records)}/{len(jobs)}: {n} questions, "
            f"{photo}, {result['latency_ms']:.1f} ms",
            flush=True,
        )
    summary = []
    for n in counts:
        times = sorted(r["latency_ms"] for r in records if r["questions"] == n)
        mean = statistics.mean(times)
        selected = [r for r in records if r["questions"] == n]
        server_mean = statistics.mean(
            r["diagnostics"]["timing"]["total_ms"] for r in selected
        )
        summary.append(
            {
                "questions": n,
                "requests": len(times),
                "mean_ms": mean,
                "server_mean_ms": server_mean,
                "reads": sorted(
                    {r["diagnostics"]["timing"]["reads"] for r in selected}
                ),
                "server_usd_per_1000_images": server_mean * args.hourly_usd / 3600,
                "server_usd_per_1000_answers": server_mean * args.hourly_usd / 3600 / n,
                "p50_ms": statistics.median(times),
                "p95_ms": times[math.ceil(0.95 * len(times)) - 1],
                "usd_per_1000_images": mean * args.hourly_usd / 3600,
                "usd_per_1000_answers": mean * args.hourly_usd / 3600 / n,
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "settings": {
                    "hourly_usd": args.hourly_usd,
                    "repeats_per_photo": args.repeats,
                    "seed": 42,
                    "warmup_requests": len(counts) * len(photos),
                    "concurrency": 1,
                    "cache_state": "warm repeated images and schemas",
                },
                "summary": summary,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
