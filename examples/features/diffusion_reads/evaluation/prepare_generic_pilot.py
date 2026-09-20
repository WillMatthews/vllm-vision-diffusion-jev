# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a reproducible VQA-v2 pilot manifest, without running any model."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import random
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
COLORS = [
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "brown",
    "orange",
    "pink",
    "purple",
    "gray",
]
SPATIAL = (
    "left of",
    "right of",
    "behind",
    "in front of",
    "underneath",
    "above",
    "below",
    "between",
    "next to",
    "near",
    "inside",
    "outside",
    "under",
    "on top of",
    "on the left",
    "on the right",
)
SEMANTIC_POOLS = [
    (("season",), ["spring", "summer", "fall", "winter"]),
    (
        ("material", "made of", "made out of"),
        [
            "wood",
            "metal",
            "plastic",
            "glass",
            "stone",
            "brick",
            "concrete",
            "fabric",
            "leather",
            "paper",
            "ceramic",
            "rubber",
        ],
    ),
    (
        ("room",),
        [
            "kitchen",
            "living room",
            "bedroom",
            "bathroom",
            "dining room",
            "office",
            "garage",
            "classroom",
        ],
    ),
    (
        ("sport",),
        [
            "soccer",
            "basketball",
            "baseball",
            "tennis",
            "skiing",
            "snowboarding",
            "surfing",
            "skateboarding",
            "volleyball",
            "frisbee",
            "ice hockey",
            "cricket",
            "rugby",
            "badminton",
            "swimming",
            "boxing",
        ],
    ),
    (
        ("animal",),
        [
            "cat",
            "dog",
            "horse",
            "cow",
            "sheep",
            "goat",
            "pig",
            "elephant",
            "giraffe",
            "zebra",
            "bear",
            "bird",
            "rabbit",
            "monkey",
            "deer",
            "fish",
        ],
    ),
    (
        ("vehicle", "transportation"),
        [
            "car",
            "bus",
            "truck",
            "train",
            "bicycle",
            "motorcycle",
            "airplane",
            "helicopter",
            "boat",
        ],
    ),
]
ALIASES = {
    "grey": "gray",
    "autumn": "fall",
    "family room": "living room",
    "living": "living room",
    "wooden": "wood",
    "cloth": "fabric",
    "cats": "cat",
    "dogs": "dog",
    "horses": "horse",
    "cows": "cow",
    "goats": "goat",
    "pigs": "pig",
    "elephants": "elephant",
    "giraffes": "giraffe",
    "zebras": "zebra",
    "bears": "bear",
    "birds": "bird",
    "rabbits": "rabbit",
    "monkeys": "monkey",
    "cars": "car",
    "buses": "bus",
    "trucks": "truck",
    "trains": "train",
    "bicycles": "bicycle",
    "motorcycles": "motorcycle",
    "planes": "airplane",
    "plane": "airplane",
    "boats": "boat",
    "bike": "bicycle",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, path, limit=256 * 1024**2):
    """Download atomically with a strict per-file byte ceiling."""
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with (
            urllib.request.urlopen(url, timeout=60) as source,
            temporary.open("wb") as destination,
        ):
            size = 0
            while chunk := source.read(1024**2):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"Download exceeds {limit} bytes: {url}")
                destination.write(chunk)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_archive(path, field):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"Expected one JSON member: {path}")
        with archive.open(names[0]) as stream:
            data = json.load(stream)
    return data[field], data.get("license")


def normalize(answer):
    answer = answer.strip().lower()
    return ALIASES.get(answer, answer)


def family(question, annotation):
    padded = " " + question.lower().strip("?.!") + " "
    if padded.startswith(" where ") or any(f" {s} " in padded for s in SPATIAL):
        return "spatial_reserved"
    if annotation["answer_type"] == "yes/no":
        return "boolean"
    if annotation["answer_type"] == "number":
        return "count"
    if "color" in question.lower() or "colour" in question.lower():
        return "colour"
    return "other"


def records(cache, split):
    questions, _ = load_archive(cache / f"questions-{split}.zip", "questions")
    annotations, license_info = load_archive(
        cache / f"annotations-{split}.zip", "annotations"
    )
    questions = {q["question_id"]: q for q in questions}
    for annotation in annotations:
        question = questions[annotation["question_id"]]
        votes = collections.Counter(
            normalize(a["answer"]) for a in annotation["answers"]
        )
        yield {
            "image_id": question["image_id"],
            "question_id": question["question_id"],
            "text": question["question"],
            "family": family(question["question"], annotation),
            "question_type": annotation["question_type"],
            "answer_type": annotation["answer_type"],
            "votes": dict(votes),
            "majority": votes.most_common(1)[0][0],
            "source_split": split,
            "annotation_license": license_info,
        }


def make_example(row, pools, seed):
    votes = row["votes"]
    if max(votes.values()) < 0.8 * sum(votes.values()):
        return None
    if row["answer_type"] == "yes/no":
        choices, primitive = ["no", "yes"], "noul"
    elif row["answer_type"] == "number":
        choices, primitive = [str(n) for n in range(16)], "score"
    elif row["family"] == "colour":
        choices, primitive = COLORS.copy(), "choice"
    else:
        choices = next(
            (
                values.copy()
                for keywords, values in SEMANTIC_POOLS
                if any(word in row["text"].lower() for word in keywords)
            ),
            None,
        )
        if choices is None:
            return None
        primitive = "choice"
    if not 2 <= len(choices) <= 16 or set(votes) - set(choices):
        return None
    # Keep every human vote; never relabel discarded answer mass as certainty.
    if primitive == "choice":
        random.Random(f"{seed}:order:{row['question_id']}").shuffle(choices)
    total = sum(votes.values())
    criteria = dict(zip(choices, choices))
    if primitive == "noul":
        criteria = {"false": "No", "true": "Yes"}
    elif primitive == "score":
        criteria = choices
    return {
        **row,
        "id": f"vqav2-{row['question_id']}",
        "primitive": primitive,
        "question": {
            "type": primitive,
            "instructions": row["text"],
            "criteria": criteria,
        },
        "state": {},
        "options": choices,
        "target": {answer: votes.get(answer, 0) / total for answer in choices},
        "ordinal_values": [int(a) for a in choices] if primitive == "score" else None,
        "audit_status": "fixed_semantic_vocabulary_not_human_audited",
    }


def choose_images(rows, count, seed, reserved_count=0, preferred_ids=None):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    eligible = [i for i, group in grouped.items() if len(group) >= 4]
    random.Random(seed).shuffle(eligible)
    if preferred_ids:
        preferred = [i for i in preferred_ids if i in grouped and len(grouped[i]) >= 4]
        preferred_set = set(preferred)
        eligible = preferred + [i for i in eligible if i not in preferred_set]
    reserved = [
        i
        for i in eligible
        if any(r["family"] == "spatial_reserved" for r in grouped[i])
    ]
    if len(eligible) < count or len(reserved) < reserved_count:
        raise ValueError(
            f"Insufficient eligible images: {len(eligible)}/{count}; "
            f"spatial {len(reserved)}/{reserved_count}. Do not shrink silently."
        )
    selected = reserved[:reserved_count]
    selected += [i for i in eligible if i not in set(selected)][: count - len(selected)]
    output = []
    for image_id in selected:
        group = grouped[image_id]
        random.Random(f"{seed}:{image_id}").shuffle(group)
        if image_id in reserved[:reserved_count]:
            spatial = next(r for r in group if r["family"] == "spatial_reserved")
            group = [spatial] + [r for r in group if r is not spatial]
        output.extend(group[:4])
    return output


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def audit_images(args):
    """Fetch selected public images and flag cross-split visual duplicates."""
    from PIL import Image

    rows = [
        json.loads(line)
        for line in (args.output / "images.jsonl").read_text().splitlines()
    ]
    metadata = {}
    with zipfile.ZipFile(args.coco_metadata) as archive:
        for split in ("train", "val"):
            with archive.open(f"annotations/captions_{split}2014.json") as stream:
                source = json.load(stream)
            licenses = {item["id"]: item for item in source["licenses"]}
            for item in source["images"]:
                metadata[item["id"]] = {
                    **item,
                    "license_record": licenses[item["license"]],
                }
    args.image_cache.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    total = sum(p.stat().st_size for p in args.image_cache.glob("images/*.jpg"))

    def process(row):
        nonlocal total
        path = args.image_cache / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        url = row["url"].replace(
            "https://images.cocodataset.org/",
            "https://s3.amazonaws.com/images.cocodataset.org/",
        )
        if not path.exists():
            with lock:
                # Reserve the whole per-file ceiling before concurrent transfer.
                if total + 4 * 1024**2 > 2 * 1024**3:
                    raise ValueError("Image download allowance would exceed 2GiB")
                total += 4 * 1024**2
            for attempt in range(3):
                try:
                    fetch(url, path, 4 * 1024**2)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            with lock:
                total += path.stat().st_size - 4 * 1024**2
        with Image.open(path) as image:
            image.load()
            size = image.size
            small = list(image.convert("L").resize((9, 8)).getdata())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | (small[y * 9 + x] > small[y * 9 + x + 1])
        return {
            **row,
            "download_url": url,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "size": size,
            "dhash64": bits,
            "source_metadata": metadata[row["image_id"]],
        }

    output = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for result in executor.map(process, rows):
            output.append(result)
            if len(output) % 250 == 0:
                print(f"Verified {len(output)}/{len(rows)} images", flush=True)
    pairs = []
    for i, first in enumerate(output):
        for second in output[i + 1 :]:
            if first["split"] == second["split"]:
                continue
            distance = (first["dhash64"] ^ second["dhash64"]).bit_count()
            exact = first["sha256"] == second["sha256"]
            if exact or distance <= 4:
                pairs.append(
                    {
                        "images": [first["image_id"], second["image_id"]],
                        "splits": [first["split"], second["split"]],
                        "distance": distance,
                        "exact": exact,
                    }
                )
    write_jsonl(args.output / "image-provenance.jsonl", output)
    report = {
        "images": len(output),
        "bytes": sum(r["bytes"] for r in output),
        "licenses": dict(
            collections.Counter(
                r["source_metadata"]["license_record"]["name"] for r in output
            )
        ),
        "cross_split_duplicate_candidates": pairs,
        "duplicate_rule": "Exact SHA256 or grayscale64bit dHash Hamming <=4",
        "limitation": "dHash misses some crops/near duplicates; review flagged pairs",
        "coco_metadata_sha256": digest(args.coco_metadata),
    }
    (args.output / "image-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--audit-images", action="store_true")
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--coco-metadata", type=Path)
    parser.add_argument("--exclude-image-ids", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_images:
        if not args.image_cache or not args.coco_metadata:
            parser.error("--audit-images requires --image-cache and --coco-metadata")
        audit_images(args)
        return
    if any(args.output.iterdir()):
        raise ValueError("Use an empty output directory; preserve frozen manifests.")
    sources = []
    for split in ("train", "val"):
        for kind in ("questions", "annotations"):
            url = BASE + f"v2_{kind.title()}_{split.title()}_mscoco.zip"
            path = args.cache / f"{kind}-{split}.zip"
            if args.download:
                fetch(url, path)
            sources.append(
                {"url": url, "sha256": digest(path), "bytes": path.stat().st_size}
            )
    train_source = list(records(args.cache, "train"))
    excluded = (
        set(json.loads(args.exclude_image_ids.read_text()))
        if args.exclude_image_ids
        else set()
    )
    # Partition image IDs before any sampling; option vocabularies are fixed.
    train_source = [r for r in train_source if r["family"] != "spatial_reserved"]
    source_ids = sorted({r["image_id"] for r in train_source})
    random.Random(args.seed).shuffle(source_ids)
    vocabulary_ids = set(source_ids[:60000])
    train_source = [r for r in train_source if r["image_id"] not in excluded]
    pools = {}
    preferred = {}
    if args.base_manifest:
        for split in ("train", "validation", "test"):
            lines = (args.base_manifest / f"{split}.jsonl").read_text().splitlines()
            preferred[split] = list(
                dict.fromkeys(json.loads(line)["image_id"] for line in lines)
            )
    eligible_train = [
        example
        for row in train_source
        if row["image_id"] in vocabulary_ids
        if (example := make_example(row, pools, args.seed))
    ]
    train = choose_images(
        eligible_train, 5000, args.seed, preferred_ids=preferred.get("train")
    )
    eligible_val = [
        example
        for row in train_source
        if row["image_id"] not in vocabulary_ids
        if (example := make_example(row, pools, args.seed))
    ]
    val = choose_images(
        eligible_val, 500, args.seed + 1, preferred_ids=preferred.get("validation")
    )
    eligible_test = [
        example
        for row in records(args.cache, "val")
        if row["image_id"] not in excluded
        if (example := make_example(row, pools, args.seed))
    ]
    test = choose_images(
        eligible_test,
        1000,
        args.seed + 2,
        reserved_count=250,
        preferred_ids=preferred.get("test"),
    )
    groups = {"train": train, "validation": val, "test": test}
    sets = [{r["image_id"] for r in rows} for rows in groups.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Image ID leakage between partitions")
    images = {}
    summaries = {}
    for split, rows in groups.items():
        for row in rows:
            image_id = row["image_id"]
            source_split = row["source_split"] + "2014"
            filename = f"COCO_{source_split}_{image_id:012d}.jpg"
            row["image"] = "images/" + filename
            # Alternate schema wrapper is reserved for test evaluation.
            row["template_partition"] = "unseen" if split == "test" else "familiar"
            images[image_id] = {
                "image_id": image_id,
                "split": split,
                "path": row["image"],
                "url": f"https://images.cocodataset.org/{source_split}/{filename}",
            }
        write_jsonl(args.output / f"{split}.jsonl", rows)
        summaries[split] = {
            "images": len({r["image_id"] for r in rows}),
            "questions": len(rows),
            "families": dict(collections.Counter(r["family"] for r in rows)),
            "primitives": dict(collections.Counter(r["primitive"] for r in rows)),
            "sha256": digest(args.output / f"{split}.jsonl"),
        }
    write_jsonl(args.output / "images.jsonl", images.values())
    manifest = {
        "seed": args.seed,
        "excluded_image_ids": sorted(excluded),
        "base_manifest_sha256": digest(args.base_manifest / "manifest.json")
        if args.base_manifest
        else None,
        "sources": sources,
        "splits": summaries,
        "training_candidate_source_images": len(vocabulary_ids),
        "option_vocabulary": "fixed semantic categories; no empirical answer pools",
        "status": "candidate_requires_image_license_dedup_and_distractor_audit",
        "annotation_attribution": "VQA Consortium; Goyal et al. CVPR 2017",
        "annotation_license": "https://creativecommons.org/licenses/by/4.0/",
        "reserved_task_family": "spatial_reserved",
        "limitations": [
            "Image license/source attribution and duplicate checks not yet complete",
            "Consensus and vocabulary filters bias the sample toward easier items",
            "Test labels read solely by deterministic construction, not model tuning",
            "Counts are an ordinal proxy, not subjective rubric supervision",
            "Template marker requires distinct wrappers in the model runner",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
