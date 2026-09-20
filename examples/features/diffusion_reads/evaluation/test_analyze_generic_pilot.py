# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard held-out calibration isolation, alignment and image-level uncertainty."""

import copy
import json
import math

import analyze_generic_pilot as analysis
import pytest


def row(identifier="q", image=1, target=None):
    return {
        "id": identifier,
        "image_id": image,
        "family": "colour",
        "primitive": "choice",
        "target": target or {"a": 0.8, "b": 0.2},
        "probabilities": {"a": 0.99, "b": 0.01},
    }


def test_temperature_fits_soft_votes_without_hardening_targets():
    record = row()
    fitted = analysis.fit_temperature({"q": record})
    probabilities = analysis.temperature_scale(
        record["probabilities"], fitted["temperature"]
    )
    assert probabilities["a"] == pytest.approx(0.8, abs=1e-6)
    assert fitted["temperature"] == pytest.approx(math.log(99) / math.log(4), rel=1e-5)
    assert fitted["validation_log_loss"] < fitted["validation_raw_log_loss"]


@pytest.mark.parametrize(
    "field,value",
    [("image_id", 9), ("target", {"a": 0.7, "b": 0.3}), ("family", "other")],
)
def test_paired_comparison_rejects_misaligned_supervision(field, value):
    original = row()
    changed = {**original, field: value}
    with pytest.raises(ValueError, match="Mismatched"):
        analysis.validate_alignment({"q": original}, {"q": changed})


def test_cluster_bootstrap_keeps_correlated_questions_together():
    rows, adapted, baseline = {}, {}, {}
    for image in (1, 2):
        for question in range(4):
            key = f"{image}-{question}"
            rows[key] = row(key, image)
            adapted[key] = {name: float(image == 1) for name in analysis.METRICS}
            baseline[key] = dict.fromkeys(analysis.METRICS, 0.0)
    result = analysis.paired_bootstrap(rows, adapted, baseline, 1000, 7)
    metric = result["metrics"]["accuracy"]
    assert metric["adapted_minus_reference"] == 0.5
    assert metric["paired_image_percentile_ci95"] == [0.0, 1.0]


def write_runs(root, test_target):
    for model in ("base", "adapted"):
        for split in ("validation", "test"):
            path = root / f"{model}-{split}"
            path.mkdir(parents=True, exist_ok=True)
            record = row(
                split,
                1 if split == "validation" else 2,
                test_target if split == "test" else None,
            )
            (path / "evaluation.json").write_text(
                json.dumps({"complete": True, "records": [record]})
            )


def test_changing_test_labels_cannot_change_fitted_temperature(tmp_path):
    write_runs(tmp_path, {"a": 0.8, "b": 0.2})
    first = analysis.analyze(tmp_path, samples=100)
    write_runs(tmp_path, {"a": 0.1, "b": 0.9})
    second = analysis.analyze(tmp_path, samples=100)
    for model in ("base", "adapted"):
        assert (
            first["models"][model]["calibration"]
            == second["models"][model]["calibration"]
        )
    assert first["acceptance_decision"] is None


def test_incomplete_result_cannot_be_mistaken_for_full_evaluation(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"complete": False, "records": [row()]}))
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.load_run(path)


def test_validation_image_overlap_is_rejected_before_calibration(tmp_path):
    write_runs(tmp_path, {"a": 0.8, "b": 0.2})
    for model in ("base", "adapted"):
        path = tmp_path / f"{model}-test" / "evaluation.json"
        payload = json.loads(path.read_text())
        payload["records"][0]["image_id"] = 1
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="leakage"):
        analysis.analyze(tmp_path, samples=100)


def test_soft_probability_metrics_and_order_diagnostics():
    original = row()
    metrics = analysis.row_metrics(original, {"a": 0.8, "b": 0.2})
    assert metrics["brier"] == 0
    assert metrics["log_loss"] == pytest.approx(
        -0.8 * math.log(0.8) - 0.2 * math.log(0.2)
    )
    changed = copy.deepcopy(original)
    changed["reversed_probabilities"] = {"a": 0.01, "b": 0.99}
    result = analysis.diagnostics(
        {
            "rows": {"q": changed},
            "payload": {"reuse_checks": [{"image": "one.jpg", "max_delta": 0.0}]},
        }
    )
    assert result["option_order"]["top_flips"] == 1
    assert result["visual_feature_reuse"]["status"] == "exact"
