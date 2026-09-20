# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare cost_benchmark JSON files without treating repeats as new images."""

import argparse
import glob
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def decisions(records):
    result = {}
    for record in records:
        for question, expected in record["expected"].items():
            key = (record["suite"], record["image"], question, record["repeat"])
            probs = record["probabilities"][question]
            if key in result:
                raise ValueError(f"duplicate decision: {key}")
            result[key] = {"expected": expected, "probabilities": probs}
    return result


def score(decision):
    expected, probs = decision["expected"], decision["probabilities"]
    if probs is None:
        return None
    return {
        "accuracy": int(max(probs, key=probs.get) == expected),
        "brier": sum((p - int(k == expected)) ** 2 for k, p in probs.items()),
        "nll": -math.log(max(probs[expected], 1e-15)),
    }


def metrics(items):
    scored = [s for d in items if (s := score(d)) is not None]
    return {
        "decisions": len(items),
        "scored_decisions": len(scored),
        "coverage": len(scored) / len(items) if items else None,
        **{
            name: statistics.mean(s[name] for s in scored) if scored else None
            for name in ("accuracy", "brier", "nll")
        },
    }


def summarize(data):
    records, sweeps = data["records"], data["sweeps"]
    items = decisions(records)
    suites, questions, repeated = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    first = {}
    for key, decision in sorted(items.items()):
        suite, image, question, repeat = key
        suites[suite].append(decision)
        questions[f"{suite}/{question}"].append(decision)
        repeated[(suite, image, question)].append(decision)
        first.setdefault((suite, image, question), decision)
    changes, comparisons, max_delta, skip_changes = 0, 0, 0.0, 0
    for group in repeated.values():
        base = group[0]["probabilities"]
        for other in group[1:]:
            probs = other["probabilities"]
            if base is None or probs is None:
                skip_changes += (base is None) != (probs is None)
                continue
            comparisons += 1
            changes += max(base, key=base.get) != max(probs, key=probs.get)
            max_delta = max(max_delta, max(abs(base[k] - probs[k]) for k in base))
    seconds = sum(s["seconds"] for s in sweeps)
    image_count = sum(s["images"] for s in sweeps)
    if image_count != len(records):
        raise ValueError("sweep image count does not match records")
    latencies = sorted(r["latency_ms"] for r in records if "latency_ms" in r)
    return {
        "tag": data["settings"]["tag"],
        "settings": data["settings"],
        "unique_images": len({r["image"] for r in records}),
        "completed_sweeps": len(sweeps),
        "requested_sweeps": data["settings"]["repeats"],
        "complete": len(sweeps) == data["settings"]["repeats"],
        "images_per_second": image_count / seconds,
        "usd_per_million_images": (
            data["settings"]["hourly_usd"] * seconds / image_count / 3600 * 1e6
        ),
        "sweep_seconds": [s["seconds"] for s in sweeps],
        "latency_p95_ms": (
            latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None
        ),
        "quality_all_repeats": metrics(list(items.values())),
        "quality_first_observation": metrics(list(first.values())),
        "suites": {k: metrics(v) for k, v in sorted(suites.items())},
        "questions": {k: metrics(v) for k, v in sorted(questions.items())},
        "repeat_stability": {
            "top_answer_changes": changes,
            "question_comparisons": comparisons,
            "max_probability_delta": max_delta,
            "skip_status_changes": skip_changes,
        },
    }


def paired_comparison(baseline, candidate):
    base, other = decisions(baseline["records"]), decisions(candidate["records"])
    shared = sorted(base.keys() & other.keys())
    regressions, improvements, deltas = [], [], []
    skips, max_delta = [], 0.0
    for key in shared:
        a, b = base[key], other[key]
        if a["expected"] != b["expected"]:
            raise ValueError(f"label mismatch: {key}")
        sa, sb = score(a), score(b)
        identity = dict(zip(("suite", "image", "question", "repeat"), key))
        if sa is None or sb is None:
            if (sa is None) != (sb is None):
                skips.append({**identity, "candidate_skipped": sb is None})
            continue
        pa, pb = a["probabilities"], b["probabilities"]
        if pa.keys() != pb.keys():
            raise ValueError(f"answer vocabulary mismatch: {key}")
        max_delta = max(max_delta, max(abs(pa[k] - pb[k]) for k in pa))
        deltas.append({name: sb[name] - sa[name] for name in sa})
        if sa["accuracy"] != sb["accuracy"]:
            change = {
                **identity,
                "expected": a["expected"],
                "baseline_answer": max(pa, key=pa.get),
                "candidate_answer": max(pb, key=pb.get),
            }
            (regressions if sa["accuracy"] else improvements).append(change)
    return {
        "matched_decisions": len(shared),
        "matched_scored_decisions": len(deltas),
        "baseline_only_decisions": len(base.keys() - other.keys()),
        "candidate_only_decisions": len(other.keys() - base.keys()),
        "mean_candidate_minus_baseline": {
            name: statistics.mean(d[name] for d in deltas) if deltas else None
            for name in ("accuracy", "brier", "nll")
        },
        "max_probability_delta": max_delta,
        "regressions": regressions,
        "improvements": improvements,
        "skip_changes": skips,
    }


def load_results(path):
    """Read individual results or a bundle, excluding its holdout section."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        data = json.load(stream)
    if {"settings", "records", "sweeps"}.issubset(data):
        return [(str(path), data)]
    members = data.get("results")
    if isinstance(members, dict):
        return [(f"{path}#{tag}", result) for tag, result in members.items()]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON paths, directories, or globs")
    baseline_args = parser.add_mutually_exclusive_group(required=True)
    baseline_args.add_argument("--baseline", type=Path)
    baseline_args.add_argument("--baseline-tag", help="Select a tag from input results")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = set()
    for source in args.inputs:
        path = Path(source)
        if path.is_dir():
            paths.update(path.glob("*.json"))
            paths.update(path.glob("*.json.gz"))
        else:
            paths.update(map(Path, glob.glob(source)))
    if args.baseline:
        paths.add(args.baseline)
    inputs = [
        result
        for path in sorted(paths)
        if path.resolve() != args.output.resolve()
        for result in load_results(path)
    ]
    matches = [
        (source, data)
        for source, data in inputs
        if (
            source == str(args.baseline)
            if args.baseline
            else data["settings"]["tag"] == args.baseline_tag
        )
    ]
    if len(matches) != 1:
        parser.error("baseline must match exactly one individual result")
    baseline_source, baseline = matches[0]
    results = []
    for source, data in inputs:
        if not data["records"] or not data["sweeps"]:
            continue
        row = summarize(data)
        row["source"] = source
        row["paired_to_baseline"] = paired_comparison(baseline, data)
        results.append(row)
    counts = ", ".join(str(n) for n in sorted({r["unique_images"] for r in results}))
    limits = [
        f"Unique image counts per result: {counts or 'none'}. Repeated sweeps and "
        "multiple questions do not increase the independent sample size. Distinct "
        "paths are counted as distinct images; independence is not verified.",
        "Quality metrics across repeats describe observed request behavior. First "
        "observation metrics avoid counting repeated requests as new examples.",
        "Paired differences match suite, image, question, and repeat; execution "
        "order and concurrency can still change numerical behavior.",
        "No confidence intervals or general quality-equivalence claims are justified "
        "by this small fixture. Inspect individual regressions even if accuracy rises.",
        "Cost is occupied inference time at the stated hourly price, excluding "
        "warmup, setup, idle time, cache resets, and network charges. Storage is "
        "included when it is part of the supplied hourly rate.",
    ]
    args.output.write_text(
        json.dumps(
            {"baseline": baseline_source, "limitations": limits, "results": results},
            indent=2,
        )
        + "\n"
    )
    print(
        "tag | done | img/s | $/M | p95 ms | accuracy | Brier | NLL | "
        "reg/imp | repeat flips"
    )
    for row in results:
        q = row["quality_all_repeats"]
        p = row["paired_to_baseline"]
        values = [
            row["tag"],
            f"{row['completed_sweeps']}/{row['requested_sweeps']}",
            f"{row['images_per_second']:.3f}",
            f"{row['usd_per_million_images']:.2f}",
            f"{row['latency_p95_ms']:.0f}" if row["latency_p95_ms"] else "n/a",
            *[
                f"{q[k]:.4f}" if q[k] is not None else "n/a"
                for k in ("accuracy", "brier", "nll")
            ],
            f"{len(p['regressions'])}/{len(p['improvements'])}",
            str(row["repeat_stability"]["top_answer_changes"]),
        ]
        print(" | ".join(values))
    print("\n" + "\n".join(limits))
    print(f"\nFull per-suite/per-question and paired details: {args.output}")


if __name__ == "__main__":
    main()
