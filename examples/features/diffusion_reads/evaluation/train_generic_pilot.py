# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train or evaluate one generic visual option scorer on public human labels.

JSONL rows contain id, image, image_id, family, question, target and optional
state. Targets map option keys to human vote weights. Question schemas use the
same choice/noul/score primitives as generic_feasibility.py. Images are resolved
against --image-root. Checkpoints include trusted local optimizer state; never
resume an untrusted checkpoint. This experiment does not implement ask-if.
"""

import argparse
import hashlib
import json
import math
import random
import string
import time
from collections import defaultdict
from pathlib import Path

from generic_feasibility import options, prompt


def row_prompt(row, choices):
    partition = row.get("template_partition", "familiar")
    familiar = prompt(row["question"], choices, row.get("state", {}))
    if partition == "familiar":
        return familiar
    if partition != "unseen":
        raise ValueError(f"Unknown template partition: {partition}")
    candidates = "\n".join(
        f"[{string.ascii_uppercase[index]}] {text}"
        for index, (_, text) in enumerate(choices)
    )
    return (
        "Use the photograph as evidence for this task.\n"
        f"Task instructions: {row['question']['instructions']}\n"
        f"Additional context: {json.dumps(row.get('state', {}))}\n"
        f"Candidate responses:\n{candidates}\n"
        "Select the best candidate and output its letter alone."
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate question ID: {row['id']}")
        seen.add(row["id"])
        choices = options(row["question"])
        keys = [key for key, _ in choices]
        if not 2 <= len(keys) <= 26 or len(set(keys)) != len(keys):
            raise ValueError("Expected 2–26 unique option keys")
        target = row["target"]
        if set(target) - set(keys):
            raise ValueError(f"Target absent from options: {row['id']}")
        weights = list(target.values())
        if not weights or any(not math.isfinite(v) or v < 0 for v in weights):
            raise ValueError("Targets must be finite nonnegative vote weights")
        if sum(weights) <= 0:
            raise ValueError("Target weights sum to zero")
        for required in ("image", "image_id", "family"):
            if required not in row:
                raise ValueError(f"Missing {required}")
    return rows


def target_values(row, choices, smoothing=0.0):
    total = sum(row["target"].values())
    return [
        (1 - smoothing) * row["target"].get(key, 0) / total + smoothing / len(choices)
        for key, _ in choices
    ]


def row_metrics(row, probabilities):
    choices = options(row["question"])
    truth = target_values(row, choices)
    predicted = max(probabilities, key=probabilities.get)
    maximum = max(truth)
    correct = any(
        key == predicted and weight == maximum
        for (key, _), weight in zip(choices, truth)
    )
    values = {
        "accuracy": float(correct),
        "vote_agreement": row["target"].get(predicted, 0) / sum(row["target"].values()),
        "log_loss": -sum(
            weight * math.log(max(probabilities[key], 1e-12))
            for (key, _), weight in zip(choices, truth)
        ),
        "brier": sum(
            (probabilities[key] - weight) ** 2
            for (key, _), weight in zip(choices, truth)
        ),
    }
    if row["question"]["type"] == "score":
        expected = sum(float(key) * probabilities[key] for key, _ in choices)
        target = sum(float(key) * weight for (key, _), weight in zip(choices, truth))
        values["ordinal_expected_value_absolute_error"] = abs(expected - target)
    return values


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups["overall"].append(record)
        groups[f"family:{record['family']}"].append(record)
        groups[f"primitive:{record['primitive']}"].append(record)
    result = {}
    for group, members in groups.items():
        metric_names = set().union(*(r["metrics"] for r in members))
        result[group] = {
            "questions": len(members),
            "images": len({r["image_id"] for r in members}),
            **{
                metric: sum(
                    r["metrics"][metric] for r in members if metric in r["metrics"]
                )
                / sum(metric in r["metrics"] for r in members)
                for metric in sorted(metric_names)
            },
        }
    return result


class Scorer:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        self.processor = AutoProcessor.from_pretrained(
            args.model, max_pixels=args.pixels
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda")
        self.model.config.use_cache = False
        self.image_root = args.image_root.resolve()
        self.max_tokens = args.max_input_tokens
        self.image_cache = None
        self.feature_owner = self.model.model
        self.original_features = self.feature_owner.get_image_features

    def image(self, row):
        from PIL import Image

        path = (self.image_root / row["image"]).resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError("Image path escapes image root")
        if self.image_cache is None or self.image_cache[0] != path:
            with Image.open(path) as image:
                self.image_cache = (path, image.convert("RGB"))
        return self.image_cache[1]

    def prepare(self, row, choices):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image(row)},
                    {
                        "type": "text",
                        "text": row_prompt(row, choices),
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] > self.max_tokens:
            raise ValueError(f"Input exceeds token limit: {row['id']}")
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        tokenizer = self.processor.tokenizer
        prefix = tokenizer.encode(text, add_special_tokens=False)
        ids = []
        for letter in string.ascii_uppercase[: len(choices)]:
            token = tokenizer.encode(letter, add_special_tokens=False)
            combined = tokenizer.encode(text + letter, add_special_tokens=False)
            if len(token) != 1 or combined != prefix + token:
                raise ValueError(f"Non-single-token answer boundary: {letter}")
            ids.append(token[0])
        return inputs.to("cuda"), self.torch.tensor(ids, device="cuda")

    def logits(self, inputs, ids):
        logits = (
            self.model(**inputs, use_cache=False, logits_to_keep=1)
            .logits[0, -1]
            .float()
        )
        if not self.torch.isfinite(logits).all():
            raise RuntimeError("Model produced nonfinite logits")
        return logits[ids], logits[ids].logsumexp(0) - logits.logsumexp(0)


def evaluate(args, scorer, rows):
    torch = scorer.torch
    if args.adapter:
        from peft import PeftModel

        scorer.model = PeftModel.from_pretrained(scorer.model, args.adapter)
    scorer.model.eval()
    records = []
    checks = []
    cached_image = None
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest_sha256": digest(args.manifest),
        "source_sha256": digest(__file__),
        "model": args.model,
        "adapter": args.adapter,
        "pixels": args.pixels,
        "mode": "sequential_complete_feature_reuse",
        "note": (
            "Accuracy accepts tied plurality answers; "
            "vote agreement is not VQA score. Uncalibrated probabilities."
        ),
    }
    output = args.output / "evaluation.json"
    try:
        with torch.no_grad():
            for index, row in enumerate(
                sorted(rows, key=lambda r: (r["image"], str(r["id"])))
            ):
                choices = options(row["question"])
                torch.accelerator.synchronize()
                begin = time.perf_counter()
                inputs, ids = scorer.prepare(row, choices)
                if cached_image != row["image"]:
                    scorer.feature_owner.get_image_features = scorer.original_features
                    ordinary, _ = scorer.logits(inputs, ids)
                    features = scorer.original_features(
                        inputs["pixel_values"], inputs["image_grid_thw"]
                    )
                    scorer.feature_owner.get_image_features = (
                        lambda *a, cached=features, **kw: cached
                    )
                    reused, _ = scorer.logits(inputs, ids)
                    delta = (
                        (ordinary.softmax(-1) - reused.softmax(-1)).abs().max().item()
                    )
                    checks.append({"image": row["image"], "max_delta": delta})
                    if delta > 1e-5:
                        raise RuntimeError(
                            "Sequential visual reuse changed probabilities"
                        )
                    cached_image = row["image"]
                logits, mass = scorer.logits(inputs, ids)
                probabilities = dict(
                    zip((key for key, _ in choices), logits.softmax(-1).cpu().tolist())
                )
                torch.accelerator.synchronize()
                record = {
                    "id": row["id"],
                    "image_id": row["image_id"],
                    "family": row["family"],
                    "primitive": row["question"]["type"],
                    "probabilities": probabilities,
                    "target": row["target"],
                    "label_mass": mass.exp().item(),
                    "seconds_with_reuse_check": time.perf_counter() - begin,
                    "input_tokens": inputs["input_ids"].shape[1],
                    "metrics": row_metrics(row, probabilities),
                }
                if index < args.order_checks:
                    reverse = list(reversed(choices))
                    inp, rev_ids = scorer.prepare(row, reverse)
                    rev_logits, _ = scorer.logits(inp, rev_ids)
                    rev_probs = dict(
                        zip(
                            (key for key, _ in reverse),
                            rev_logits.softmax(-1).cpu().tolist(),
                        )
                    )
                    record["reversed_probabilities"] = rev_probs
                    record["order_top_changed"] = max(
                        probabilities, key=probabilities.get
                    ) != max(rev_probs, key=rev_probs.get)
                records.append(record)
                if (index + 1) % 100 == 0:
                    write_json(
                        output,
                        {
                            "metadata": metadata,
                            "records": records,
                            "reuse_checks": checks,
                            "complete": False,
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "evaluated": index + 1,
                                "seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
    finally:
        scorer.feature_owner.get_image_features = scorer.original_features
        write_json(
            output,
            {
                "metadata": metadata,
                "records": records,
                "reuse_checks": checks,
                "complete": False,
            },
        )
    write_json(
        output,
        {
            "metadata": metadata,
            "records": records,
            "reuse_checks": checks,
            "summary": aggregate(records),
            "seconds": time.perf_counter() - started,
            "complete": True,
        },
    )
    print("EVALUATION_COMPLETE", flush=True)


def train(args, scorer, rows):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        scorer.model = PeftModel.from_pretrained(
            scorer.model, args.resume / "adapter", is_trainable=True
        )
    else:
        scorer.model = get_peft_model(
            scorer.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
    model = scorer.model
    if any(p.requires_grad for name, p in model.named_parameters() if "visual" in name):
        raise RuntimeError("Vision tower must be frozen")
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial_parameters = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    contract = {
        "manifest_sha256": digest(args.manifest),
        "model": args.model,
        "pixels": args.pixels,
        "epochs": args.epochs,
        "seed": args.seed,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "max_presentations": args.max_presentations,
    }
    done = 0
    updates = 0
    if args.resume:
        state = torch.load(
            args.resume / "state.pt", map_location="cpu", weights_only=False
        )
        if state["contract"] != contract:
            raise ValueError("Resume configuration or dataset differs from checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        done, updates = state["presentations"], state["updates"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    total = min(len(rows) * args.epochs, args.max_presentations)
    if done == total:
        print("TRAINING_ALREADY_COMPLETE", flush=True)
        return
    if done > total:
        raise ValueError("Checkpoint exceeds requested training presentations")
    if total < 1:
        raise ValueError("No training presentations requested")
    log_path = args.output / "training.jsonl"
    if log_path.exists() and not args.resume:
        raise ValueError(
            "Training output already exists; use a fresh output or --resume"
        )
    write_json(
        args.output / "training-config.json",
        {
            **contract,
            "source_sha256": digest(__file__),
            "questions": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "total_presentations": total,
        },
    )

    def checkpoint():
        directory = args.output / f"checkpoint-{done:06d}"
        if directory.exists():
            raise ValueError(f"Checkpoint already exists: {directory}")
        temporary = args.output / f".checkpoint-{done:06d}.tmp"
        temporary.mkdir()
        model.save_pretrained(temporary / "adapter")
        torch.save(
            {
                "contract": contract,
                "presentations": done,
                "updates": updates,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            temporary / "state.pt",
        )
        temporary.rename(directory)
        write_json(
            args.output / "latest.json",
            {"checkpoint": directory.name, "presentations": done},
        )
        return directory

    model.train()
    # Frozen vision features should not vary with any backbone train-mode dropout.
    scorer.feature_owner.visual.eval()
    optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter()
    torch.accelerator.reset_peak_memory_stats()
    losses = []
    order = []
    active_epoch = None
    with log_path.open("a") as log:
        while done < total:
            epoch, position = divmod(done, len(rows))
            if active_epoch != epoch:
                order = list(range(len(rows)))
                random.Random(args.seed + epoch).shuffle(order)
                active_epoch = epoch
            row = rows[order[position]]
            choices = options(row["question"])
            random.Random(args.seed * 1000003 + done).shuffle(choices)
            inputs, ids = scorer.prepare(row, choices)
            logits, _ = scorer.logits(inputs, ids)
            target = torch.tensor(
                target_values(row, choices, args.label_smoothing), device="cuda"
            )
            loss = -(target * logits.log_softmax(-1)).sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            # Last accumulation group may contain fewer examples.
            group_start = done - done % args.gradient_accumulation
            divisor = min(args.gradient_accumulation, total - group_start)
            (loss / divisor).backward()
            losses.append(loss.item())
            done += 1
            if done % args.gradient_accumulation == 0 or done == total:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates % args.log_every == 0 or done == total:
                    torch.accelerator.synchronize()
                    event = {
                        "presentations": done,
                        "updates": updates,
                        "mean_loss": sum(losses) / len(losses),
                        "elapsed_seconds_this_process": time.perf_counter() - begin,
                        "peak_allocated_gib": torch.accelerator.max_memory_allocated()
                        / 2**30,
                    }
                    log.write(json.dumps(event) + "\n")
                    log.flush()
                    print(json.dumps(event), flush=True)
                    losses.clear()
                if updates % args.checkpoint_every == 0 and done < total:
                    checkpoint()
        final = checkpoint() if done > 0 else None
    parameter_change = sum(
        (p.detach() - initial).abs().sum().item()
        for p, initial in zip(parameters, initial_parameters)
    )
    if not math.isfinite(parameter_change) or parameter_change == 0:
        raise RuntimeError("No finite adapter parameter update was observed")
    write_json(
        args.output / "complete.json",
        {
            "presentations": done,
            "updates": updates,
            "checkpoint": str(final),
            "complete": True,
            "adapter_parameter_l1_change": parameter_change,
        },
    )
    print("TRAINING_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--pixels", type=int, default=512 * 512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-presentations", type=int, default=40000)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--order-checks", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.label_smoothing < 1:
        parser.error("label-smoothing must be in [0,1)")
    if (
        min(
            args.epochs,
            args.max_presentations,
            args.gradient_accumulation,
            args.checkpoint_every,
            args.log_every,
            args.cpu_threads,
        )
        < 1
    ):
        parser.error("Training counts and CPU threads must be positive")
    if args.mode == "train" and args.adapter:
        parser.error("Use --resume to continue training")
    if args.mode == "evaluate" and args.resume:
        parser.error("Use --adapter to evaluate a checkpoint")
    rows = load_rows(args.manifest)
    images = {row["image_id"] for row in rows}
    for excluded in args.exclude_manifest:
        overlap = images & {row["image_id"] for row in load_rows(excluded)}
        if overlap:
            raise ValueError(f"Image leakage across manifests: {len(overlap)} images")
    scorer = Scorer(args)
    if args.mode == "train":
        train(args, scorer, rows)
    else:
        evaluate(args, scorer, rows)


if __name__ == "__main__":
    main()
