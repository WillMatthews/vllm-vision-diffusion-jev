# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for human vote supervision and held-out image isolation."""

import json
import math

import prepare_generic_pilot as data
import pytest
import train_generic_pilot as runner


def example():
    return {
        "id": "question-1",
        "image": "images/one.jpg",
        "image_id": 1,
        "family": "colour",
        "question": {
            "type": "choice",
            "instructions": "What colour is the car?",
            "criteria": {"red": "Red", "blue": "Blue", "green": "Green"},
        },
        "target": {"red": 8, "blue": 2},
        "state": {},
    }


def test_soft_votes_follow_option_keys_after_shuffle():
    """Shuffling presentation must not silently turn minority votes into labels."""
    row = example()
    choices = [("green", "Green"), ("red", "Red"), ("blue", "Blue")]
    assert runner.target_values(row, choices) == pytest.approx([0, 0.8, 0.2])
    assert runner.target_values(row, choices, 0.3) == pytest.approx([0.1, 0.66, 0.24])


@pytest.mark.parametrize(
    "target",
    [{"absent": 1}, {"red": -1}, {"red": 0}, {"red": math.nan}],
)
def test_invalid_supervision_rejected_before_model_load(tmp_path, target):
    row = {**example(), "target": target}
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        runner.load_rows(manifest)


def test_probability_metrics_use_all_human_votes():
    row = example()
    metrics = runner.row_metrics(row, {"red": 0.8, "blue": 0.2, "green": 0.0})
    assert metrics["brier"] == pytest.approx(0)
    assert metrics["log_loss"] == pytest.approx(
        -0.8 * math.log(0.8) - 0.2 * math.log(0.2)
    )
    assert metrics["accuracy"] == 1
    assert metrics["vote_agreement"] == pytest.approx(0.8)


def test_training_refuses_shared_images_before_constructing_model(
    tmp_path, monkeypatch
):
    """Distinct question IDs do not make the same image a valid holdout."""
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "test.jsonl"
    train.write_text(json.dumps(example()) + "\n")
    heldout.write_text(json.dumps({**example(), "id": "different-question"}) + "\n")

    def unexpected_model_load(*args):
        pytest.fail("GPU/model initialization occurred before image leakage check")

    monkeypatch.setattr(runner, "Scorer", unexpected_model_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_generic_pilot.py",
            "train",
            "--model",
            "unused",
            "--manifest",
            str(train),
            "--exclude-manifest",
            str(heldout),
            "--image-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="Image leakage"):
        runner.main()


def source_row(votes, answer_type="other", family="other"):
    return {
        "question_id": 123,
        "image_id": 1,
        "text": "What animal is pictured?",
        "votes": votes,
        "majority": max(votes, key=votes.get),
        "answer_type": answer_type,
        "family": family,
        "question_type": "what is",
    }


def test_dataset_keeps_minority_answers_in_probability_targets():
    row = source_row({"red": 8, "blue": 2}, family="colour")
    converted = data.make_example(row, {"__all__": []}, 42)
    assert converted["target"]["red"] == pytest.approx(0.8)
    assert converted["target"]["blue"] == pytest.approx(0.2)
    assert sum(converted["target"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "row,pool",
    [
        (source_row({"cat": 7, "dog": 3}), ["cat", "dog", "horse"]),
        (source_row({"cat": 9, "unseen synonym": 1}), ["cat", "dog", "horse"]),
        (source_row({"16": 10}, answer_type="number", family="count"), []),
    ],
)
def test_dataset_rejects_ambiguous_or_missing_answers_without_relabelling(row, pool):
    assert data.make_example(row, {"__all__": pool}, 42) is None


def test_semantic_options_do_not_depend_on_the_gold_answer():
    cat = source_row({"cat": 10})
    dog = source_row({"dog": 10})
    first = data.make_example(cat, {}, 42)
    second = data.make_example(dog, {}, 42)
    assert first["options"] == second["options"]
    assert {"cat", "dog", "horse"} <= set(first["options"])
    assert first["question"]["instructions"] == cat["text"]
    assert first["state"] == {}


def test_heldout_wrapper_changes_wording_preserving_generic_instructions():
    row = {**example(), "state": {"scene": "outdoors"}}
    choices = runner.options(row["question"])
    familiar = runner.row_prompt(row, choices)
    unseen = runner.row_prompt({**row, "template_partition": "unseen"}, choices)
    assert familiar != unseen
    assert row["question"]["instructions"] in unseen
    assert json.dumps(row["state"]) in unseen
    assert all(text in unseen for _, text in choices)
    # Supervision and answer metadata must not become model input.
    poisoned = {
        **row,
        "template_partition": "unseen",
        "target": {"green": 10},
        "majority": "never leak this",
        "votes": {"never leak this": 10},
    }
    assert runner.row_prompt(poisoned, choices) == unseen


def test_unknown_wrapper_is_not_silently_treated_as_familiar():
    row = {**example(), "template_partition": "misspelled"}
    with pytest.raises(ValueError, match="Unknown template"):
        runner.row_prompt(row, runner.options(row["question"]))


def test_teacher_maps_ordinal_descriptions_back_to_supplied_indices():
    from evaluate_generic_teacher import canonical_probabilities

    row = {
        **example(),
        "question": {"type": "score", "criteria": ["none", "some", "many"]},
    }
    assert canonical_probabilities(row, {"none": 0.2, "some": 0.5, "many": 0.3}) == {
        "0": 0.2,
        "1": 0.5,
        "2": 0.3,
    }
    with pytest.raises(ValueError, match="legend"):
        canonical_probabilities(row, {"0": 0.2, "1": 0.5, "2": 0.3})


def test_teacher_groups_four_tasks_and_keeps_targets_out_of_request(
    tmp_path, monkeypatch
):
    import evaluate_generic_teacher as teacher

    image = tmp_path / "images/one.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image bytes unused by mocked HTTP client")
    manifest = tmp_path / "test.jsonl"
    rows = [
        {**example(), "id": f"question-{i}", "majority": "secret label"}
        for i in range(4)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    requests = []

    def fake_read(path, schema, endpoint, **kwargs):
        requests.append(schema)
        assert path == image
        assert len(schema["questions"]) == 4
        assert "secret label" not in json.dumps(schema)
        assert "target" not in json.dumps(schema)
        return {
            "probabilities": {
                key: {"red": 0.8, "blue": 0.2, "green": 0.0}
                for key in schema["questions"]
            },
            "latency_ms": 123.0,
            "diagnostics": {},
            "usage": {},
        }

    monkeypatch.setattr(teacher, "read_image", fake_read)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_generic_teacher.py",
            "--manifest",
            str(manifest),
            "--image-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--model-revision",
            "mock-revision",
        ],
    )
    teacher.main()
    result = json.loads((tmp_path / "out/evaluation.json").read_text())
    assert result["complete"] is True
    assert len(requests) == 1
    assert len(result["records"]) == 4
    assert result["summary"]["overall"]["brier"] == pytest.approx(0)
