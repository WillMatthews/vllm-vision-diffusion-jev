# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare empty cache, another-image prompt reuse, and same-image reuse.

Run only against a dedicated private server: this clears its caches repeatedly.
Start vLLM with --enable-prefix-caching --enable-prompt-tokens-details.
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_client import read_image  # noqa: E402


def reset_caches(upstream):
    for endpoint in ("reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache"):
        for attempt in range(20):
            request = urllib.request.Request(
                upstream.rstrip("/") + "/" + endpoint, data=b"", method="POST"
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
            if not body or json.loads(body).get("success", True):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"{endpoint} could not be reset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 24])
    parser.add_argument(
        "--photos",
        nargs="+",
        default=["dog", "cat", "fox", "deer", "horse", "rabbit", "bird"],
    )
    parser.add_argument("--chunk-prompt", choices=["own", "shared"], default="own")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    root = Path(__file__).resolve().parent
    full = json.loads((root / "animal_schema.json").read_text())
    photos = args.photos
    if len(set(photos)) < 2 or any(
        not (root / f"images/{p}.jpg").is_file() for p in photos
    ):
        parser.error("provide at least two distinct existing photos")
    full["seed"] = 42
    counts = args.counts
    if any(n < 1 or n > len(full["questions"]) for n in counts):
        parser.error("counts must be between 1 and the number of schema questions")
    full["chunk_prompt"] = args.chunk_prompt
    modes = ["empty", "different_image", "same_image"]
    schemas = {
        n: {**full, "questions": dict(list(full["questions"].items())[:n])}
        for n in counts
    }
    for photo in photos:
        for n in counts:
            read_image(root / f"images/{photo}.jpg", schemas[n], args.endpoint)
        print(f"Kernel warmup: {photo}", flush=True)
    jobs = [
        (repeat, n, photo)
        for repeat in range(args.repeats)
        for n in counts
        for photo in photos
    ]
    rng = random.Random(42)
    rng.shuffle(jobs)
    records = []
    settings = {
        "repeats": args.repeats,
        "chunk_prompt": args.chunk_prompt,
        "counts": counts,
        "seed": 42,
        "request_seed": 42,
        "photos": photos,
        "concurrency": 1,
        "transport": "server localhost",
        "reset_before_each_arm": ["prefix", "multimodal", "encoder"],
        "warmup_requests": len(photos) * len(counts),
    }
    for repeat, n, photo in jobs:
        arms = list(modes)
        rng.shuffle(arms)
        for mode in arms:
            reset_caches(args.upstream)
            primer = None
            if mode != "empty":
                primer = (
                    photo
                    if mode == "same_image"
                    else photos[(photos.index(photo) + 1) % len(photos)]
                )
                read_image(root / f"images/{primer}.jpg", schemas[n], args.endpoint)
            result = read_image(root / f"images/{photo}.jpg", schemas[n], args.endpoint)
            usage = result["diagnostics"]["upstream_usage"]
            if not usage or any(u.get("prompt_tokens_details") is None for u in usage):
                raise RuntimeError(
                    "Missing per-read usage: enable prompt token details"
                )
            cached = [u["prompt_tokens_details"]["cached_tokens"] for u in usage]
            if any(value is None for value in cached):
                raise RuntimeError("Server did not report cached token counts")
            record = {
                "repeat": repeat,
                "questions": n,
                "image": photo,
                "mode": mode,
                "primer_image": primer,
                "sum_prompt_tokens": sum(u["prompt_tokens"] for u in usage),
                "sum_cached_tokens": sum(cached),
                **result,
            }
            records.append(record)
            args.output.write_text(
                json.dumps({"settings": settings, "records": records}, indent=2) + "\n"
            )
            print(
                f"{len(records)}/{len(jobs) * len(modes)}: {n}q {photo} {mode}: "
                f"{result['latency_ms']:.1f}ms, {sum(cached)} cached",
                flush=True,
            )
    summary = []
    for n in counts:
        for mode in modes:
            selected = [r for r in records if r["questions"] == n and r["mode"] == mode]
            summary.append(
                {
                    "questions": n,
                    "mode": mode,
                    "requests": len(selected),
                    "mean_http_ms": statistics.mean(r["latency_ms"] for r in selected),
                    "mean_server_ms": statistics.mean(
                        r["diagnostics"]["timing"]["total_ms"] for r in selected
                    ),
                    "mean_prompt_tokens": statistics.mean(
                        r["sum_prompt_tokens"] for r in selected
                    ),
                    "mean_cached_tokens": statistics.mean(
                        r["sum_cached_tokens"] for r in selected
                    ),
                }
            )
    args.output.write_text(
        json.dumps(
            {"settings": settings, "summary": summary, "records": records}, indent=2
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
