# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare question grouping, sample count, and class order on fixed labels."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_client import metrics, read_image  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    full = json.loads((root / "animal_schema.json").read_text())
    one = {**full, "questions": {"species": full["questions"]["species"]}}
    reversed_order = copy.deepcopy(one)
    q = reversed_order["questions"]["species"]
    q["criteria"] = dict(reversed(list(q["criteria"].items())))
    schemas = {
        "species_only": one,
        "species_reversed_options": reversed_order,
        "all_questions_four_samples": {**full, "samples": 4},
    }
    entries = [
        json.loads(line) for line in (root / "animals.jsonl").read_text().splitlines()
    ]
    output = {}
    for name, schema in schemas.items():
        records = []
        for entry in entries:
            result = read_image(root / entry["image"], schema, args.endpoint)
            expected = {
                k: v for k, v in entry["expected"].items() if k in schema["questions"]
            }
            records.append({**result, "image": entry["image"], "expected": expected})
            print(f"{name}: {entry['image']}", flush=True)
            output[name] = {"records": records}
            args.output.write_text(json.dumps(output, indent=2) + "\n")
        output[name]["metrics"] = metrics(records)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
