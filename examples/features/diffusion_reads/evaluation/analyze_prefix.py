# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute accuracy and score-stability summaries for saved cache experiments."""

import json
import statistics
from pathlib import Path


def analyze(data, expected):
    records = data["records"]
    summary = []
    for row in data["summary"]:
        group = [
            r
            for r in records
            if r["questions"] == row["questions"] and r["mode"] == row["mode"]
        ]
        correctness = [
            max(r["probabilities"][q], key=r["probabilities"][q].get) == label
            for r in group
            for q, label in expected[r["image"]].items()
            if q in r["probabilities"]
        ]
        summary.append(
            {
                **row,
                "labelled_decisions": len(correctness),
                "correct": sum(correctness),
                "server_median_ms": statistics.median(
                    r["diagnostics"]["timing"]["total_ms"] for r in group
                ),
            }
        )
    bases = {
        (r["repeat"], r["questions"], r["image"]): r
        for r in records
        if r["mode"] == "empty"
    }
    changes = []
    maximum = 0.0
    comparisons = 0
    for r in records:
        if r["mode"] == "empty":
            continue
        base = bases[r["repeat"], r["questions"], r["image"]]
        for q, probs in r["probabilities"].items():
            previous = base["probabilities"][q]
            delta = max(abs(probs[k] - previous[k]) for k in probs)
            maximum = max(maximum, delta)
            comparisons += 1
            if max(probs, key=probs.get) != max(previous, key=previous.get):
                changes.append(
                    {
                        "repeat": r["repeat"],
                        "questions": r["questions"],
                        "image": r["image"],
                        "mode": r["mode"],
                        "question": q,
                        "empty_answer": max(previous, key=previous.get),
                        "cached_answer": max(probs, key=probs.get),
                        "max_probability_delta": delta,
                    }
                )
    within = {}
    for mode in ("empty", "different_image", "same_image"):
        deltas, flips, count = [], 0, 0
        for n, photo in sorted({(r["questions"], r["image"]) for r in records}):
            group = sorted(
                (
                    r
                    for r in records
                    if r["questions"] == n and r["image"] == photo and r["mode"] == mode
                ),
                key=lambda r: r["repeat"],
            )
            first = group[0]["probabilities"]
            for r in group[1:]:
                for q, probs in r["probabilities"].items():
                    deltas.extend(abs(probs[k] - first[q][k]) for k in probs)
                    count += 1
                    flips += max(probs, key=probs.get) != max(
                        first[q], key=first[q].get
                    )
        within[mode] = {
            "max_probability_delta": max(deltas) if deltas else None,
            "mean_absolute_probability_delta": statistics.mean(deltas)
            if deltas
            else None,
            "top_answer_changes": flips,
            "question_comparisons": count,
        }
    return {
        "summary": summary,
        "cache_top_answer_changes": changes,
        "max_probability_delta": maximum,
        "within_condition_repeats": within,
        "cross_condition_question_comparisons": comparisons,
        "species_correct": sum(
            max(r["probabilities"]["species"], key=r["probabilities"]["species"].get)
            == expected[r["image"]]["species"]
            for r in records
        ),
        "target_requests": len(records),
    }


def main():
    root = Path(__file__).resolve().parent
    expected = {
        Path(r["image"]).stem: r["expected"]
        for r in map(json.loads, (root / "animals.jsonl").read_text().splitlines())
    }
    output = {
        name: analyze(json.loads((root / name).read_text()), expected)
        for name in (
            "prefix-results.json",
            "prefix-shared-results.json",
            "prefix-bf16-results.json",
        )
    }
    (root / "prefix-analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    for name, result in output.items():
        print(
            name,
            "species:",
            result["species_correct"],
            "/",
            result["target_requests"],
            "cross-condition top-answer changes:",
            len(result["cache_top_answer_changes"]),
        )


if __name__ == "__main__":
    main()
