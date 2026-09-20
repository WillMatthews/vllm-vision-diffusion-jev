# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests: transport, answer conversion, and evaluation arithmetic.

The client maps local images to requests and typed answers to probabilities.
A local HTTP stub catches dropped image bytes and diagnostics without a GPU;
pure metric tests catch misleading accuracy or coverage calculations.
"""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pybase64 as base64
from evaluation import analyze_cost, cost_benchmark
from structured_server import answer_text, decide, jev_schema, parse_schema
from visual_client import distributions, metrics, read_image


class VisualClientTests(unittest.TestCase):
    def test_compact_schema_preserves_question_meanings_and_dependencies(self):
        schema = {
            "ask": ["animal", "collar"],
            "state": {"context": "inspect animal"},
            "questions": {
                "animal": {"type": "noul", "instructions": "Animal visible?"},
                "collar": {
                    "type": "noul",
                    "instructions": "Collar visible?",
                    "depends_on": ["animal"],
                    "ask_if": {"animal": ["yes"]},
                },
            },
        }
        original = json.dumps(schema)
        for layout, first, second in (("numeric", "0", "1"), ("short", "q0", "q1")):
            with self.subTest(layout=layout):
                compact, names = cost_benchmark.remap_schema(schema, layout)
                self.assertEqual(names, {first: "animal", second: "collar"})
                self.assertEqual(compact["ask"], [first, second])
                self.assertEqual(compact["state"], schema["state"])
                self.assertEqual(
                    compact["questions"][first], schema["questions"]["animal"]
                )
                self.assertEqual(
                    compact["questions"][second],
                    {
                        **schema["questions"]["collar"],
                        "depends_on": [first],
                        "ask_if": {first: ["yes"]},
                    },
                )
                jev_schema(compact)
        self.assertEqual(json.dumps(schema), original)

    def test_cost_uses_sweep_wall_time_even_when_request_latencies_overlap(self):
        records = [
            {
                "suite": "toy",
                "image": str(i),
                "repeat": 0,
                "expected": {"animal": "yes"},
                "probabilities": {"animal": {"yes": 0.8, "no": 0.2}},
                "latency_ms": 9000,
                "diagnostics": {"timing": {"total_ms": 8000}},
            }
            for i in range(2)
        ]
        sweeps = [{"seconds": 10, "images": 2}]
        offline = analyze_cost.summarize(
            {
                "records": records,
                "sweeps": sweeps,
                "settings": {"tag": "toy", "repeats": 1, "hourly_usd": 0.72},
            }
        )
        online = cost_benchmark.summarize(records, sweeps, 0.72)
        for result in (online, offline):
            self.assertAlmostEqual(result["images_per_second"], 0.2)
            self.assertAlmostEqual(result["usd_per_million_images"], 1000)

    def test_repeats_are_paired_by_identity_and_first_observation_counts_once(self):
        records = [
            {
                "suite": "toy",
                "image": "same-image",
                "repeat": repeat,
                "expected": {"animal": "yes"},
                "probabilities": {"animal": {"yes": p, "no": 1 - p}},
            }
            for repeat, p in enumerate((0.8, 0.2))
        ]
        baseline = {
            "records": records,
            "sweeps": [{"seconds": 1, "images": 1}] * 2,
            "settings": {"tag": "toy", "repeats": 2, "hourly_usd": 1},
        }
        summary = analyze_cost.summarize(baseline)
        self.assertEqual(summary["unique_images"], 1)
        self.assertEqual(summary["quality_first_observation"]["decisions"], 1)
        self.assertEqual(summary["quality_first_observation"]["accuracy"], 1)
        self.assertEqual(summary["quality_all_repeats"]["accuracy"], 0.5)
        comparison = analyze_cost.paired_comparison(
            baseline, {"records": list(reversed(records))}
        )
        self.assertEqual(comparison["matched_decisions"], 2)
        self.assertEqual(comparison["mean_candidate_minus_baseline"]["accuracy"], 0)
        self.assertFalse(comparison["regressions"])
        self.assertEqual(comparison["max_probability_delta"], 0)
        with self.assertRaisesRegex(ValueError, "duplicate decision"):
            analyze_cost.decisions(records * 2)

    def test_cost_summary_accepts_conditionally_skipped_answers(self):
        records = [
            {
                "suite": "toy",
                "image": "same-image",
                "repeat": repeat,
                "expected": {"collar": "yes"},
                "probabilities": {"collar": probs},
                "latency_ms": 1,
                "diagnostics": {"timing": {"total_ms": 1}},
            }
            for repeat, probs in enumerate((None, {"yes": 0.8, "no": 0.2}))
        ]
        summary = cost_benchmark.summarize(records, [{"seconds": 1, "images": 2}], 1)
        self.assertEqual(summary["quality"]["skipped_decisions"], 1)
        self.assertEqual(summary["quality"]["accuracy"], 1)
        self.assertFalse(summary["errors"])
        self.assertEqual(summary["repeat_stability"]["question_comparisons"], 0)
        self.assertEqual(summary["repeat_stability"]["skip_status_changes"], 1)

    def test_cache_usage_keeps_every_sample_from_every_chunk(self):
        schema = parse_schema(
            {
                "samples": 2,
                "questions": [{"id": str(i), "type": "noul"} for i in range(2)],
            }
        )
        usages = [
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": i * 16}}
            for i in range(4)
        ]
        slot = {
            "probs": [0.8, 0.2],
            "entropy": 0.5,
            "label_mass": 0.9,
            "argmax_is_label": True,
        }

        def fake_reads(sub, *args):
            index = int(sub["questions"][0]["id"]) * 2
            return [[slot], [slot]], usages[index : index + 2]

        with (
            patch(
                "structured_server.chunk_groups",
                return_value=[[q] for q in schema["questions"]],
            ),
            patch(
                "structured_server.template_for",
                return_value=([10], [{"pos": 0, "label_ids": [1, 2]}]),
            ),
            patch("structured_server.read_many", side_effect=fake_reads),
        ):
            result, _ = decide(schema, "test state", 0)
        self.assertEqual(result["diagnostics"]["upstream_usage"], usages)
        self.assertEqual(result["diagnostics"]["timing"]["reads"], 4)

    def test_many_named_questions_keep_a_delimiter_before_labels(self):
        questions = [
            {"id": f"has_collar_{i}", "type": "noul", "instructions": "Collar visible?"}
            for i in range(12)
        ]
        schema = parse_schema({"questions": questions})
        text = answer_text(schema["questions"], [1] * 12, schema["format"])
        self.assertEqual(text.splitlines()[0], "has_collar_0: no")
        self.assertEqual(text.splitlines()[-1], "has_collar_11: no")

    def test_image_bytes_and_context_reach_server_and_diagnostics_survive(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured["path"] = self.path
                captured["auth"] = self.headers.get("Authorization")
                captured["body"] = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "answers": {"error": {"type": "noul", "noul": 0.8}},
                            "diagnostics": {
                                "questions": {"error": {"label_mass": 0.02}}
                            },
                        }
                    ).encode()
                )

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "image.png"
                payload = base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
                    "/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
                )
                path.write_bytes(payload)
                result = read_image(
                    path,
                    {"questions": {"error": {"type": "noul"}}},
                    f"http://127.0.0.1:{server.server_port}",
                    state={"task": "inspect"},
                    api_key="test-key",
                )
            self.assertEqual(captured["path"], "/v1/systemone")
            self.assertEqual(captured["auth"], "Bearer test-key")
            body = captured["body"]
            self.assertEqual(body["state"], {"task": "inspect"})
            self.assertEqual(base64.b64decode(body["images"][0].split(",")[1]), payload)
            self.assertEqual(body["samples"], 1)
            self.assertAlmostEqual(result["probabilities"]["error"]["no"], 0.2)
            self.assertEqual(
                result["diagnostics"]["questions"]["error"]["label_mass"], 0.02
            )
            self.assertGreater(result["latency_ms"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_choice_score_and_skipped_answers_keep_label_meanings(self):
        result = distributions(
            {
                "screen": {
                    "type": "choice",
                    "probabilities": {"login": 0.3, "other": 0.7},
                },
                "severity": {
                    "type": "score",
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"0": 0.4, "1": 0.6},
                },
                "conditional": None,
            }
        )
        self.assertEqual(
            result,
            {
                "screen": {"login": 0.3, "other": 0.7},
                "severity": {"low": 0.4, "high": 0.6},
                "conditional": None,
            },
        )
        for probs in ({"a": float("nan")}, {"a": 0.4}, {"a": 1.1, "b": -0.1}):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                distributions({"x": {"type": "choice", "probabilities": probs}})

    def test_metrics_penalize_confident_errors_and_count_skips_in_coverage(self):
        records = [
            {
                "probabilities": {
                    "a": {"yes": 0.8, "no": 0.2},
                    "b": {"yes": 0.1, "no": 0.9},
                    "c": None,
                },
                "expected": {"a": "yes", "b": "yes", "c": "no"},
                "latency_ms": 100,
            }
        ]
        result = metrics(records, threshold=0.85)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.85)
        self.assertAlmostEqual(result["ece_10_bins"], 0.55)
        self.assertAlmostEqual(result["coverage"], 1 / 3)
        self.assertEqual(result["selective_accuracy"], 0)
        self.assertEqual(result["skipped_decisions"], 1)
        self.assertEqual(result["latency_p95_ms"], 100)
        records[0]["expected"]["a"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown expected label"):
            metrics(records)

    def test_all_skipped_is_not_reported_as_perfect_accuracy(self):
        result = metrics(
            [{"probabilities": {"a": None}, "expected": {"a": "yes"}, "latency_ms": 1}]
        )
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["ece_10_bins"])
        self.assertIsNone(result["selective_accuracy"])
        self.assertEqual(result["coverage"], 0)


if __name__ == "__main__":
    unittest.main()
