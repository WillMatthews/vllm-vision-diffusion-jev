# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize labelled diagnostics; these fixtures are not a held-out test set."""

import argparse
import json
import math
from collections import defaultdict


def summarize(result):
    suites = defaultdict(list)
    flips = 0
    for record in result["records"]:
        for qid, expected in record["expected"].items():
            probs = record["probabilities"][qid]
            expected = str(expected)
            assert expected in probs, (qid, expected)
            suites[record["suite"]].append(
                {
                    "correct": max(probs, key=probs.get) == expected,
                    "log_loss": -math.log(max(probs[expected], 1e-12)),
                    "brier": sum(
                        (value - (key == expected)) ** 2 for key, value in probs.items()
                    ),
                }
            )
        original = next(iter(record["probabilities"].values()))
        reverse = record["reversed_first_question"]
        flips += max(original, key=original.get) != max(reverse, key=reverse.get)
    return {
        "suites": {
            name: {
                "labelled_decisions": len(rows),
                "correct": sum(row["correct"] for row in rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "log_loss": sum(row["log_loss"] for row in rows) / len(rows),
                "brier_sum_over_options": sum(row["brier"] for row in rows) / len(rows),
            }
            for name, rows in suites.items()
        },
        "first_question_order_flips": flips,
        "order_checks": len(result["records"]),
        "max_reuse_probability_delta": max(
            row["max_probability_delta"] for row in result["reuse_checks"]
        ),
        "image_tokens": [row["image_tokens"] for row in result["records"]],
        "latency": result["latency"],
        "training": result.get("training"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    args = parser.parse_args()
    with open(args.result) as file:
        print(json.dumps(summarize(json.load(file)), indent=2))
