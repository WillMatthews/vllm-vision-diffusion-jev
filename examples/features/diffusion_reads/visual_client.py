# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read image probabilities and evaluate labelled images via /v1/systemone."""

import argparse
import json
import math
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pybase64 as base64


def read_image(image, schema, endpoint, *, state=None, api_key=None, timeout=120):
    """Send a local image and preserve the server's probability diagnostics."""
    image = Path(image)
    mime = mimetypes.guess_type(image.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("image must be PNG, JPEG, WebP, or GIF")
    body = dict(schema)
    body["state"] = state if state is not None else schema.get("state", {})
    body["images"] = [
        f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")
    ]
    body.setdefault("samples", 1)
    body.setdefault("think", 0)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise ValueError(f"server returned HTTP {exc.code}: {detail}") from exc
    return {
        "probabilities": distributions(result["answers"]),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "diagnostics": result.get("diagnostics", {}),
        "usage": result.get("usage", {}),
    }


def distributions(answers):
    """Convert Jev answer types to named distributions without recalibration."""
    result = {}
    for qid, answer in answers.items():
        if answer is None:
            result[qid] = None
            continue
        if answer["type"] == "noul":
            p = answer["noul"]
            probs = {"yes": p, "no": 1 - p}
        elif answer["type"] == "choice":
            probs = answer["probabilities"]
        elif answer["type"] == "score":
            probs = {
                answer["legend"][key]: p for key, p in answer["probabilities"].items()
            }
            if len(probs) != len(answer["probabilities"]):
                raise ValueError(f"{qid}: score legend contains duplicate names")
        else:
            raise ValueError(f"{qid}: unknown answer type {answer['type']!r}")
        if (
            not probs
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values())
            or not math.isclose(sum(probs.values()), 1, abs_tol=1e-6)
        ):
            raise ValueError(f"{qid}: invalid probability distribution")
        result[qid] = probs
    return result


def metrics(records, threshold=0.9):
    """Score labelled decisions; skipped answers reduce coverage, not accuracy."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    total, scored, accepted = 0, [], []
    for record in records:
        for qid, expected in record["expected"].items():
            total += 1
            if qid not in record["probabilities"]:
                raise ValueError(f"missing answer for {qid!r}")
            probs = record["probabilities"][qid]
            if probs is None:
                continue
            if expected not in probs:
                raise ValueError(f"{qid}: unknown expected label {expected!r}")
            prediction = max(probs, key=probs.get)
            confidence = probs[prediction]
            correct = int(prediction == expected)
            brier = sum((p - int(label == expected)) ** 2 for label, p in probs.items())
            scored.append(
                (confidence, correct, brier, -math.log(max(probs[expected], 1e-15)))
            )
            if confidence >= threshold:
                accepted.append(correct)
    if not total:
        raise ValueError("evaluation needs at least one expected label")
    n = len(scored)
    ece = 0.0
    for index in range(10):
        bucket = [r for r in scored if min(int(r[0] * 10), 9) == index]
        if bucket:
            ece += abs(sum(r[0] - r[1] for r in bucket)) / n
    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "images": len(records),
        "labelled_decisions": total,
        "scored_decisions": n,
        "skipped_decisions": total - n,
        "accuracy": sum(r[1] for r in scored) / n if n else None,
        "brier": sum(r[2] for r in scored) / n if n else None,
        "nll": sum(r[3] for r in scored) / n if n else None,
        "ece_10_bins": ece if n else None,
        "threshold": threshold,
        "coverage": len(accepted) / total,
        "selective_accuracy": sum(accepted) / len(accepted) if accepted else None,
        "latency_p50_ms": latencies[math.ceil(len(latencies) * 0.50) - 1],
        "latency_p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--timeout", type=float, default=120)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path)
    mode.add_argument(
        "--manifest", type=Path, help="JSONL: image, expected, optional state"
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    try:
        if not 0 <= args.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        schema = json.loads(args.schema.read_text())
        kwargs = {
            "schema": schema,
            "endpoint": args.endpoint,
            "api_key": os.environ.get("API_KEY"),
            "timeout": args.timeout,
        }
        if args.image:
            output = read_image(args.image, **kwargs)
        else:
            entries = [
                json.loads(line)
                for line in args.manifest.read_text().splitlines()
                if line.strip()
            ]
            if not entries:
                raise ValueError("manifest is empty")
            for entry in entries:
                expected = entry.get("expected")
                if not isinstance(expected, dict) or not expected:
                    raise ValueError(
                        "each manifest entry needs a non-empty expected map"
                    )
                if not set(expected) <= set(schema["questions"]):
                    raise ValueError("expected labels reference unknown questions")
                if not (args.manifest.parent / entry["image"]).is_file():
                    raise ValueError(f"image does not exist: {entry['image']}")
            records = []
            for entry in entries:
                result = read_image(
                    args.manifest.parent / entry["image"],
                    state=entry.get("state"),
                    **kwargs,
                )
                records.append(
                    dict(result, image=entry["image"], expected=entry["expected"])
                )
            output = {"metrics": metrics(records, args.threshold), "records": records}
        print(json.dumps(output, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, urllib.error.URLError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
