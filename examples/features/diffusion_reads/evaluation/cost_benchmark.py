# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure diffusion-read quality and new-image sweep throughput on a private server.

Caches are cleared between sweeps, outside timing. Every source image occurs only
once per sweep. Repeated sweeps are not additional independent quality examples.
"""

import argparse
import copy
import hashlib
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.prefix_benchmark import reset_caches  # noqa: E402
from visual_client import metrics, read_image  # noqa: E402


def remap_schema(schema, layout):
    """Change only question identifiers, retaining the exact question semantics."""
    schema = copy.deepcopy(schema)
    names = list(schema["questions"])
    if layout == "named":
        return schema, {name: name for name in names}
    mapping = {
        name: str(i) if layout == "numeric" else f"q{i}" for i, name in enumerate(names)
    }
    questions = {}
    for name, question in schema["questions"].items():
        if question.get("depends_on") is not None:
            question["depends_on"] = [mapping[key] for key in question["depends_on"]]
        for field in ("ask_if",):
            if field in question:
                question[field] = {
                    mapping[key]: value for key, value in question[field].items()
                }
        questions[mapping[name]] = question
    schema["questions"] = questions
    if schema.get("ask") is not None:
        schema["ask"] = [mapping[name] for name in schema["ask"]]
    return schema, {value: key for key, value in mapping.items()}


def load_jobs(root, layout, seed, suites):
    jobs = []
    for suite in suites:
        schema_name = (
            "generic_schema.json" if suite == "generic" else "animal_schema.json"
        )
        schema, names = remap_schema(
            json.loads((root / schema_name).read_text()), layout
        )
        schema["seed"] = seed
        for line in (root / f"{suite}.jsonl").read_text().splitlines():
            item = json.loads(line)
            jobs.append((suite, item, schema, names))
    return jobs


def summarize(records, sweeps, hourly_usd):
    total_seconds = sum(s["seconds"] for s in sweeps)
    summary = {
        "quality": metrics(records),
        "images_per_second": len(records) / total_seconds,
        "usd_per_million_images": hourly_usd
        * total_seconds
        / len(records)
        / 3600
        * 1e6,
        "mean_server_ms": statistics.mean(
            r["diagnostics"]["timing"]["total_ms"] for r in records
        ),
        "suites": {},
        "errors": [],
    }
    for suite in sorted({r["suite"] for r in records}):
        summary["suites"][suite] = metrics([r for r in records if r["suite"] == suite])
    for r in records:
        for qid, expected in r["expected"].items():
            probs = r["probabilities"][qid]
            if probs is None:
                continue
            predicted = max(probs, key=probs.get)
            if predicted != expected:
                summary["errors"].append(
                    {
                        "repeat": r["repeat"],
                        "image": r["image"],
                        "question": qid,
                        "expected": expected,
                        "predicted": predicted,
                        "probability": probs[predicted],
                    }
                )
    changes, comparisons, max_delta, skip_changes = 0, 0, 0.0, 0
    first = {}
    for r in records:
        key = (r["suite"], r["image"])
        if key not in first:
            first[key] = r
            continue
        for qid, probs in r["probabilities"].items():
            base = first[key]["probabilities"][qid]
            if base is None or probs is None:
                skip_changes += (base is None) != (probs is None)
                continue
            comparisons += 1
            changes += max(base, key=base.get) != max(probs, key=probs.get)
            max_delta = max(max_delta, max(abs(probs[k] - base[k]) for k in probs))
    summary["repeat_stability"] = {
        "top_answer_changes": changes,
        "question_comparisons": comparisons,
        "max_probability_delta": max_delta,
        "skip_status_changes": skip_changes,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=["named", "numeric", "short"], default="named"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-sweeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--server-config", type=Path)
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["animals", "generic", "holdout"],
        default=["animals", "generic"],
    )
    args = parser.parse_args()
    if (
        min(args.concurrency, args.repeats, args.warmup_sweeps) < 1
        or args.hourly_usd <= 0
    ):
        parser.error(
            "concurrency, repeats, warmup sweeps and hourly cost must be positive"
        )
    root = Path(__file__).resolve().parent
    jobs = load_jobs(root, args.layout, args.seed, args.suites)
    records, sweeps = [], []
    settings = {
        **vars(args),
        "output": str(args.output),
        "transport": "server localhost",
        "cache_reset": "all three caches before each sweep",
        "unique_images_per_sweep": len(jobs),
        "warmup_concurrency": args.concurrency,
    }
    settings["server_config"] = (
        json.loads(args.server_config.read_text()) if args.server_config else None
    )
    settings["source_sha256"] = {
        str(path.relative_to(root.parents[3])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (
            root.parent / "structured_server.py",
            root.parents[3] / "vllm/model_executor/models/diffusion_gemma.py",
            root.parents[3] / "vllm/v1/worker/gpu/model_runner.py",
            Path(__file__).resolve(),
        )
    }
    inputs = {root / item["image"] for _, item, _, _ in jobs}
    inputs.update(root / f"{suite}.jsonl" for suite in args.suites)
    inputs.update(
        root / ("generic_schema.json" if suite == "generic" else "animal_schema.json")
        for suite in args.suites
    )
    settings["input_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs)
    }
    if args.server_config:
        adapter_config = args.server_config.with_name("adapter-config.json")
        if adapter_config.exists():
            settings["adapter_config"] = json.loads(adapter_config.read_text())

    def run(job):
        suite, item, schema, names = job
        result = read_image(
            root / item["image"],
            schema,
            args.endpoint,
            state=item.get("state"),
            timeout=300,
        )
        result["probabilities"] = {
            names[k]: v for k, v in result["probabilities"].items()
        }
        return {"suite": suite, **item, **result}

    rng = random.Random(42)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for warmup in range(args.warmup_sweeps):
            order = list(jobs)
            random.Random(1000 + warmup).shuffle(order)
            reset_caches(args.upstream)
            list(pool.map(run, order))
            print(f"Cold-cache warmup {warmup}: {len(jobs)} requests", flush=True)
        for repeat in range(args.repeats):
            order = list(jobs)
            rng.shuffle(order)
            reset_caches(args.upstream)
            start = time.perf_counter()
            results = list(pool.map(run, order))
            elapsed = time.perf_counter() - start
            records.extend({"repeat": repeat, **r} for r in results)
            sweeps.append(
                {"repeat": repeat, "seconds": elapsed, "images": len(results)}
            )
            payload = {
                "settings": settings,
                "sweeps": sweeps,
                "summary": summarize(records, sweeps, args.hourly_usd),
                "records": records,
            }
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            q = payload["summary"]["quality"]
            print(
                f"{args.tag} sweep {repeat}: {elapsed:.3f}s, "
                f"accuracy={q['accuracy']:.4f}, Brier={q['brier']:.4f}",
                flush=True,
            )
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
