# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare completed pilot runs; fit temperatures using validation labels only."""

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

EPSILON = 1e-12
METRICS = ("accuracy", "brier", "log_loss")


def distribution(values):
    if not values or any(
        not math.isfinite(v) or v < 0 or v > 1 for v in values.values()
    ):
        raise ValueError("Invalid probability distribution")
    total = sum(values.values())
    if abs(total - 1) > 1e-3:
        raise ValueError("Probability mass does not sum to one")
    return {key: value / total for key, value in values.items()}


def load_run(path):
    payload = json.loads(path.read_text())
    if payload.get("complete") is not True:
        raise ValueError(f"Incomplete run: {path}")
    rows = {}
    for row in payload["records"]:
        if row["id"] in rows:
            raise ValueError(f"Duplicate question ID: {row['id']}")
        probs = distribution(row["probabilities"])
        target = distribution(row["target"])
        if set(target) - set(probs):
            raise ValueError("Target option absent from probabilities")
        if "reversed_probabilities" in row:
            reverse = distribution(row["reversed_probabilities"])
            if set(reverse) != set(probs):
                raise ValueError("Reverse-order option keys differ")
        rows[row["id"]] = {
            **row,
            "probabilities": probs,
            "target": target,
            "source_target": row["target"],
        }
    if not rows:
        raise ValueError(f"Empty run: {path}")
    return {
        "payload": payload,
        "rows": rows,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def validate_alignment(first, second):
    if first.keys() != second.keys():
        raise ValueError("Question ID sets differ")
    for key, row in first.items():
        other = second[key]
        if row.get("source_target", row["target"]) != other.get(
            "source_target", other["target"]
        ):
            raise ValueError(f"Mismatched original target for {key}")
        for field in ("image_id", "family", "primitive", "target"):
            if row[field] != other[field]:
                raise ValueError(f"Mismatched {field} for {key}")
        if row["probabilities"].keys() != other["probabilities"].keys():
            raise ValueError(f"Mismatched options for {key}")


def temperature_scale(probabilities, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = {
        key: math.log(max(value, EPSILON)) / temperature
        for key, value in probabilities.items()
    }
    maximum = max(logits.values())
    weights = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def fit_temperature(validation_rows):
    """Minimize soft-human-target validation NLL over T in [0.05, 20]."""
    prepared = []
    for row in validation_rows.values():
        keys = list(row["probabilities"])
        prepared.append(
            (
                [math.log(max(row["probabilities"][k], EPSILON)) for k in keys],
                [row["target"].get(k, 0) for k in keys],
            )
        )

    def objective(log_temperature):
        inverse = math.exp(-log_temperature)
        loss = 0.0
        for logits, target in prepared:
            scaled = [value * inverse for value in logits]
            maximum = max(scaled)
            normalizer = maximum + math.log(sum(math.exp(v - maximum) for v in scaled))
            loss += normalizer - sum(t * value for t, value in zip(target, scaled))
        return loss / len(prepared)

    lower, upper = math.log(0.05), math.log(20)
    ratio = (math.sqrt(5) - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    fleft, fright = objective(left), objective(right)
    for _ in range(72):
        if fleft < fright:
            upper, right, fright = right, left, fleft
            left = upper - ratio * (upper - lower)
            fleft = objective(left)
        else:
            lower, left, fleft = left, right, fright
            right = lower + ratio * (upper - lower)
            fright = objective(right)
    candidates = [math.log(0.05), math.log(20), 0.0, (lower + upper) / 2]
    best = min(candidates, key=objective)
    return {
        "temperature": math.exp(best),
        "validation_log_loss": objective(best),
        "validation_raw_log_loss": objective(0),
        "validation_questions": len(prepared),
        "bounds": [0.05, 20],
        "at_bound": abs(best - candidates[0]) < 1e-5
        or abs(best - candidates[1]) < 1e-5,
    }


def row_metrics(row, probabilities):
    predicted = max(sorted(probabilities), key=probabilities.get)
    target = row["target"]
    result = {
        "accuracy": float(target.get(predicted, 0) == max(target.values())),
        "vote_agreement": target.get(predicted, 0),
        "brier": sum(
            (value - target.get(key, 0)) ** 2 for key, value in probabilities.items()
        ),
        "log_loss": -sum(
            value * math.log(max(probabilities[key], EPSILON))
            for key, value in target.items()
        ),
        "confidence": probabilities[predicted],
    }
    if row["primitive"] == "score":
        result["ordinal_expected_value_absolute_error"] = abs(
            sum(float(k) * v for k, v in probabilities.items())
            - sum(float(k) * v for k, v in target.items())
        )
    return result


def summarize(rows, temperature=1):
    groups = collections.defaultdict(list)
    scored = {}
    for key, row in rows.items():
        probs = (
            row["probabilities"]
            if temperature == 1
            else temperature_scale(row["probabilities"], temperature)
        )
        metrics = row_metrics(row, probs)
        scored[key] = metrics
        for group in (
            "overall",
            "family/" + row["family"],
            "primitive/" + row["primitive"],
        ):
            groups[group].append((row, metrics))
    output = {}
    for group, values in sorted(groups.items()):
        metric_keys = set().union(*(metrics.keys() for _, metrics in values))
        result = {
            metric: sum(m[metric] for _, m in values if metric in m)
            / sum(metric in m for _, m in values)
            for metric in metric_keys
        }
        bins = collections.defaultdict(list)
        for _, metrics in values:
            bins[min(14, int(metrics["confidence"] * 15))].append(metrics)
        result["ece_15_equal_width_bins"] = sum(
            abs(sum(v["confidence"] - v["accuracy"] for v in members))
            for members in bins.values()
        ) / len(values)
        result["questions"] = len(values)
        result["images"] = len({r["image_id"] for r, _ in values})
        output[group] = result
    return output, scored


def percentile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def paired_bootstrap(rows, adapted, baseline, samples=1000, seed=20260920):
    """Resample whole images, preserving pairing and questions within clusters."""
    clusters = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for key, row in rows.items():
        cluster = clusters[row["image_id"]]
        for i, metric in enumerate(METRICS):
            cluster[i] += adapted[key][metric] - baseline[key][metric]
        cluster[3] += 1
    values = [clusters[key] for key in sorted(clusters, key=str)]
    rng = random.Random(seed)
    draws = [[] for _ in METRICS]
    for _ in range(samples):
        sums = [0.0, 0.0, 0.0, 0]
        for _ in values:
            picked = values[rng.randrange(len(values))]
            for i in range(4):
                sums[i] += picked[i]
        for i in range(3):
            draws[i].append(sums[i] / sums[3])
    output = {}
    for i, metric in enumerate(METRICS):
        interval = [percentile(draws[i], 0.025), percentile(draws[i], 0.975)]
        output[metric] = {
            "adapted_minus_reference": sum(v[i] for v in values)
            / sum(v[3] for v in values),
            "paired_image_percentile_ci95": interval,
        }
    output["descriptive_unapproved_margin_comparison"] = {
        "accuracy_loss_at_most_0.01": output["accuracy"][
            "paired_image_percentile_ci95"
        ][0]
        >= -0.01,
        "brier_increase_at_most_0.01": output["brier"]["paired_image_percentile_ci95"][
            1
        ]
        <= 0.01,
        "log_loss_increase_at_most_0.02": output["log_loss"][
            "paired_image_percentile_ci95"
        ][1]
        <= 0.02,
        "acceptance_decision": None,
    }
    return {
        "images": len(values),
        "questions": len(rows),
        "samples": samples,
        "seed": seed,
        "metrics": output,
    }


def diagnostics(run):
    rows = run["rows"]
    reversed_rows = [r for r in rows.values() if "reversed_probabilities" in r]
    deltas, flips = [], 0
    for row in reversed_rows:
        normal, reverse = (
            row["probabilities"],
            distribution(row["reversed_probabilities"]),
        )
        flips += max(sorted(normal), key=normal.get) != max(
            sorted(reverse), key=reverse.get
        )
        deltas.append(max(abs(normal[k] - reverse[k]) for k in normal))
    checks = run["payload"].get("reuse_checks")
    reuse = {"status": "not_measured", "checks": 0}
    if checks is not None:
        if any(not math.isfinite(c["max_delta"]) or c["max_delta"] < 0 for c in checks):
            raise ValueError("Invalid reuse-check delta")
        maximum = max((c["max_delta"] for c in checks), default=None)
        reuse = {
            "checks": len(checks),
            "max_probability_delta": maximum,
            "status": "exact"
            if maximum == 0
            else "within_1e-5"
            if maximum is not None and maximum <= 1e-5
            else "failed"
            if maximum is not None
            else "not_measured",
            "images": len({r["image_id"] for r in rows.values()}),
            "unique_checked_images": len({c["image"] for c in checks}),
        }
        reuse["one_check_per_image"] = (
            reuse["checks"] == reuse["unique_checked_images"] == reuse["images"]
        )
        if not reuse["one_check_per_image"] and reuse["status"] in {
            "exact",
            "within_1e-5",
        }:
            reuse["status"] = "partial_" + reuse["status"]
    return {
        "option_order": {
            "checks": len(reversed_rows),
            "top_flips": flips,
            "flip_fraction": flips / len(reversed_rows) if reversed_rows else None,
            "max_probability_delta": max(deltas, default=None),
        },
        "visual_feature_reuse": reuse,
    }


def analyze(student_dir, teacher_dir=None, samples=1000, seed=20260920):
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    runs = {
        model: {
            split: load_run(student_dir / f"{model}-{split}" / "evaluation.json")
            for split in ("validation", "test")
        }
        for model in ("base", "adapted")
    }
    if teacher_dir:
        runs["teacher"] = {
            split: load_run(teacher_dir / split / "evaluation.json")
            for split in ("validation", "test")
        }
    for model, splits in runs.items():
        for split, run in splits.items():
            validate_alignment(runs["base"][split]["rows"], run["rows"])
        validation_ids = {r["image_id"] for r in splits["validation"]["rows"].values()}
        test_ids = {r["image_id"] for r in splits["test"]["rows"].values()}
        if (
            validation_ids & test_ids
            or splits["validation"]["rows"].keys() & splits["test"]["rows"].keys()
        ):
            raise ValueError(f"Validation/test leakage in {model}")
    report = {
        "models": {},
        "paired_comparisons": {},
        "acceptance_decision": None,
        "method": {
            "temperature_fit": "Validation-only soft-target NLL, T in [0.05,20]",
            "probability_floor": EPSILON,
            "renormalization": "Unit-mass tolerance1e-3; normalize accepted inputs",
            "bootstrap": "Whole-image paired percentile; fitted temperatures fixed",
            "accuracy": "Top answer matches a tied plurality human answer",
            "tie_break": "Lexicographically first option key",
            "brier": "Sum over options, not option-count normalized",
            "ece": "15 equal-width bins; plurality accuracy vs top confidence",
            "quality_policy": "Proposed margins descriptive only; not approved",
            "timing": "Evaluation timings include checks; not matched serving cost",
        },
    }
    scored = {}
    for model, splits in runs.items():
        calibration = fit_temperature(splits["validation"]["rows"])
        output = {"calibration": calibration, "splits": {}}
        scored[model] = {}
        for split, run in splits.items():
            raw, raw_scores = summarize(run["rows"])
            calibrated, calibrated_scores = summarize(
                run["rows"], calibration["temperature"]
            )
            output["splits"][split] = {
                "raw": raw,
                "calibrated": calibrated,
                "diagnostics": diagnostics(run),
                "source": {k: run[k] for k in ("path", "sha256")},
            }
            if split == "test":
                scored[model] = {"raw": raw_scores, "calibrated": calibrated_scores}
        report["models"][model] = output
    for reference in ("base", "teacher"):
        if reference not in runs:
            continue
        report["paired_comparisons"]["adapted-minus-" + reference] = {
            mode: paired_bootstrap(
                runs["adapted"]["test"]["rows"],
                scored["adapted"][mode],
                scored[reference][mode],
                samples,
                seed,
            )
            for mode in ("raw", "calibrated")
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    report = analyze(
        args.student_dir, args.teacher_dir, args.bootstrap_samples, args.seed
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                model: value["splits"]["test"]["calibrated"]["overall"]
                for model, value in report["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
