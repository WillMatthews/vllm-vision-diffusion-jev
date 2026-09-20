# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate public, deterministic geometry smoke-test images and schemas."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / "feasibility-fixtures"
root.mkdir(exist_ok=True)
cases = []
colors = {"red": "#d20c18", "blue": "#1554ce", "green": "#168334", "yellow": "#f4d522"}
for i, (color, hexcolor) in enumerate(colors.items()):
    for n in (1, 2):
        image = Image.new("RGB", (512, 384), "white")
        d = ImageDraw.Draw(image)
        shape = "circle" if (i + n) % 2 else "square"
        centers = [(256, 192)] if n == 1 else [(140, 192), (372, 192)]
        for x, y in centers:
            box = (x - 55, y - 55, x + 55, y + 55)
            if shape == "circle":
                d.ellipse(box, fill=hexcolor)
            else:
                d.rectangle(box, fill=hexcolor)
        name = f"{color}-{n}-{shape}.png"
        image.save(root / name)
        qs = {
            "colour": {
                "type": "choice",
                "instructions": "What colour are the shapes?",
                "criteria": {k: k for k in colors},
            },
            "shape": {
                "type": "choice",
                "instructions": "Which geometric shape is shown?",
                "criteria": {
                    "circle": "circles",
                    "square": "squares",
                    "triangle": "triangles",
                },
            },
            "count": {
                "type": "score",
                "instructions": "How many coloured shapes are visible?",
                "criteria": ["zero shapes", "one shape", "two shapes", "three shapes"],
            },
            "multiple": {
                "type": "noul",
                "instructions": "Are there at least two coloured shapes?",
            },
        }
        cases.append(
            {
                "suite": "synthetic",
                "image": "feasibility-fixtures/" + name,
                "schema": {"questions": qs},
                "expected": {
                    "colour": color,
                    "shape": shape,
                    "count": str(n),
                    "multiple": "yes" if n == 2 else "no",
                },
            }
        )
(root / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
