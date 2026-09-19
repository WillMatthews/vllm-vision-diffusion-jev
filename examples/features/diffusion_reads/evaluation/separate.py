# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure asking the same 24 questions as separate image requests."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_client import read_image  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    schema = json.loads((root / "animal_schema.json").read_text())
    records = []
    for photo in ("dog", "cat"):
        for qid, question in schema["questions"].items():
            result = read_image(
                root / f"images/{photo}.jpg",
                {**schema, "questions": {qid: question}},
                args.endpoint,
            )
            records.append({"image": photo, "question": qid, **result})
            args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
            print(f"{photo}: {qid}", flush=True)
    summary = []
    for photo in ("dog", "cat"):
        selected = [r for r in records if r["image"] == photo]
        summary.append(
            {
                "image": photo,
                "requests": len(selected),
                "http_total_ms": sum(r["latency_ms"] for r in selected),
                "server_total_ms": sum(
                    r["diagnostics"]["timing"]["total_ms"] for r in selected
                ),
            }
        )
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
