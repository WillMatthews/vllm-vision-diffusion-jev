# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for label-free inference schemas and probability serialization."""

from pathlib import Path

import predict_generic_pilot as predictor
import pytest


def schema():
    return {
        "state": {"context": "Inspect the photograph"},
        "questions": {
            "new_task": {
                "type": "choice",
                "instructions": "Which custom category is visible?",
                "criteria": {"first_custom_key": "A bicycle", "second_key": "A bus"},
            },
            "new_binary": {"type": "noul", "instructions": "Is it raining?"},
            "new_ordinal": {
                "type": "score",
                "instructions": "How many vehicles are visible?",
                "criteria": ["None", "One", "Two or more"],
            },
        },
    }


def test_unseen_schema_needs_no_training_targets_and_keeps_custom_option_keys():
    rows = predictor.schema_rows(schema(), Path("/images/new.jpg"))
    assert [row["id"] for row in rows] == list(schema()["questions"])
    assert all("target" not in row for row in rows)
    assert rows[0]["state"] == schema()["state"]
    probabilities = {"first_custom_key": 0.3, "second_key": 0.7}
    assert predictor.answer(rows[0]["question"], probabilities)["probabilities"] == (
        probabilities
    )


@pytest.mark.parametrize("field", ["askif", "ask_if", "dependencies", "depends_on"])
def test_question_dependency_cannot_be_silently_scored_as_independent(field):
    request = schema()
    request["questions"]["new_binary"][field] = "new_task"
    with pytest.raises(ValueError, match="unsupported fields/ask-if/dependencies"):
        predictor.schema_rows(request, Path("new.jpg"))


def test_ordinal_expected_value_uses_zero_based_indices_and_binary_keeps_yes_mass():
    questions = schema()["questions"]
    ordinal = predictor.answer(questions["new_ordinal"], {"0": 0.1, "1": 0.2, "2": 0.7})
    assert ordinal["expected_value"] == pytest.approx(1.6)
    assert ordinal["legend"] == {"0": "None", "1": "One", "2": "Two or more"}
    binary = predictor.answer(questions["new_binary"], {"no": 0.15, "yes": 0.85})
    assert binary["noul"] == pytest.approx(0.85)


@pytest.mark.parametrize("count", [1, 27])
def test_option_count_outside_answer_token_contract_is_rejected(count):
    request = schema()
    request["questions"]["new_task"]["criteria"] = {
        f"key-{i}": f"Option {i}" for i in range(count)
    }
    with pytest.raises(ValueError, match="2–26 options"):
        predictor.schema_rows(request, Path("new.jpg"))


@pytest.mark.parametrize(
    "probabilities", [{"no": 0.2, "yes": 0.2}, {"no": 0.5, "yes": float("nan")}]
)
def test_invalid_probabilities_are_not_written_as_confident_answers(probabilities):
    with pytest.raises(ValueError):
        predictor.answer(schema()["questions"]["new_binary"], probabilities)
