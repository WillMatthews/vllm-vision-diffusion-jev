# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create deterministic coloured PNGs and labels for an image-path smoke test."""

import json
import struct
import sys
import zlib
from pathlib import Path


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    colours = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    rows = []
    for name, rgb in colours.items():
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress((b"\0" + bytes(rgb) * 256) * 256))
        png += chunk(b"IEND", b"")
        (directory / f"{name}.png").write_bytes(png)
        rows.append(
            {
                "image": f"{name}.png",
                "expected": {
                    "colour": name,
                    "is_red": "yes" if name == "red" else "no",
                },
            }
        )
    schema = {
        "state": {},
        "samples": 1,
        "think": 0,
        "questions": {
            "colour": {
                "type": "choice",
                "instructions": "What colour fills the image?",
                "criteria": {name: name for name in colours},
            },
            "is_red": {"type": "noul", "instructions": "Is the image red?"},
        },
    }
    (directory / "schema.json").write_text(json.dumps(schema, indent=2))
    (directory / "manifest.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
