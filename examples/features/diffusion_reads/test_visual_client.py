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
from structured_server import answer_text, decide, parse_schema
from visual_client import distributions, metrics, read_image


class VisualClientTests(unittest.TestCase):
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
