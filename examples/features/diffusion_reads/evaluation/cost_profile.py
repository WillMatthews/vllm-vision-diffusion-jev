# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile one cold-image request on a dedicated private server.

Start vLLM with --profiler-config '{"profiler":"torch",
"torch_profiler_dir":"/workspace/profiles","torch_profiler_with_stack":false}'.
Warm the server separately. Profiling latency is not a throughput measurement.
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.prefix_benchmark import reset_caches  # noqa: E402
from visual_client import read_image  # noqa: E402


def profile_endpoint(upstream, action):
    request = urllib.request.Request(
        upstream.rstrip("/") + "/" + action, data=b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()


def summarize_trace(path):
    """Aggregate GPU kernel durations; overlapping kernels are counted separately."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    totals = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        entry = totals[event["name"]]
        entry["count"] += 1
        entry["duration_us"] += event["dur"]
    return {
        "trace": str(path),
        "summed_kernel_ms": sum(v["duration_us"] for v in totals.values()) / 1000,
        "note": "Summed durations include overlap; not elapsed GPU wall time.",
        "top_kernels": [
            {"name": name, **values}
            for name, values in sorted(
                totals.items(), key=lambda item: item[1]["duration_us"], reverse=True
            )[:30]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="dog-animal-schema")
    parser.add_argument("--trace", type=Path, help="summarize existing trace only")
    args = parser.parse_args()
    if args.trace:
        args.output.write_text(json.dumps(summarize_trace(args.trace), indent=2) + "\n")
        return

    root = Path(__file__).resolve().parent
    image = root / "images/dog.jpg"
    schema = json.loads((root / "animal_schema.json").read_text())
    schema["seed"] = 42
    record = {
        "tag": args.tag,
        "endpoint": args.endpoint,
        "upstream": args.upstream,
        "request": {
            "schema": schema,
            "image": "images/dog.jpg",
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
        "cache_reset": "prefix, multimodal and encoder before profiling",
        "note": "Profiler overhead included; do not use for throughput pricing.",
    }
    reset_caches(args.upstream)
    profile_endpoint(args.upstream, "start_profile")
    started = time.perf_counter()
    try:
        record["result"] = read_image(image, schema, args.endpoint, timeout=300)
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["request_seconds"] = time.perf_counter() - started
        try:
            profile_endpoint(args.upstream, "stop_profile")
        finally:
            args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
