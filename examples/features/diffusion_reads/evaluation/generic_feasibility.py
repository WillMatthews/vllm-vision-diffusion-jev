# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded Qwen visual option-scoring and LoRA feasibility experiment.

This measures unadapted scoring before a disposable optimizer smoke test.
The optimizer test is not a trained model evaluation or a generalization claim.
"""

import argparse
import copy
import hashlib
import json
import string
import time
from pathlib import Path


def options(question):
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"].items())
    if kind == "noul":
        criteria = question.get("criteria") or {}
        return [
            ("no", criteria.get("false", "No")),
            ("yes", criteria.get("true", "Yes")),
        ]
    if kind == "score":
        return [(str(i), str(v)) for i, v in enumerate(question["criteria"])]
    raise ValueError(f"Unsupported question type: {kind}")


def prompt(question, choices, state):
    if not 2 <= len(choices) <= 26:
        raise ValueError("This feasibility scorer supports 2–26 options")
    lines = [f"{string.ascii_uppercase[i]}. {v}" for i, (_, v) in enumerate(choices)]
    return (
        "Inspect the image and answer the question using visible evidence.\n"
        f"Context: {json.dumps(state)}\nQuestion: {question['instructions']}\n"
        + "\n".join(lines)
        + "\nReply with only the letter of the best answer."
    )


def load_cases(root):
    cases = []
    for suite in ("animals", "generic"):
        schema = json.loads(
            (
                root / f"{'animal' if suite == 'animals' else suite}_schema.json"
            ).read_text()
        )
        for line in (root / f"{suite}.jsonl").read_text().splitlines():
            item = json.loads(line)
            cases.append({**item, "suite": suite, "schema": schema})
    cases.extend(json.loads((root / "feasibility-fixtures/cases.json").read_text()))
    return cases


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=512 * 512)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    root = Path(__file__).resolve().parent
    torch.manual_seed(42)
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.pixels)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    model.config.use_cache = False
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "pixels_limit": args.pixels,
        "cpu_threads": torch.get_num_threads(),
        "hourly_usd": args.hourly_usd,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": (
            "Natural fixtures are reused diagnostics. Synthetic cases are "
            "controlled smoke tests, not representative held-out quality evidence."
        ),
    }
    result = {"metadata": metadata, "records": [], "reuse_checks": [], "latency": []}

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def sync():
        torch.accelerator.synchronize()

    def inputs_for(image, q, choices, state):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt(q, choices, state)},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        prefix = processor.tokenizer.encode(text, add_special_tokens=False)
        ids = []
        for i in range(len(choices)):
            label = string.ascii_uppercase[i]
            token = processor.tokenizer.encode(label, add_special_tokens=False)
            combined = processor.tokenizer.encode(
                text + label, add_special_tokens=False
            )
            if len(token) != 1 or combined != prefix + token:
                raise ValueError(
                    f"Answer label does not append as a single token: {label}"
                )
            ids.append(token[0])
        return inputs.to(device), torch.tensor(ids, device=device)

    def scores(inputs, ids):
        logits = (
            model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        )
        return logits[ids].softmax(-1), logits[ids].logsumexp(0) - logits.logsumexp(0)

    original_features = model.model.get_image_features
    first_training = None
    for case in load_cases(root):
        if args.training_only and case["suite"] != "synthetic":
            continue
        image = Image.open(root / case["image"]).convert("RGB")
        schema = case["schema"]
        questions = list(schema["questions"].items())
        qid, q = questions[0]
        first_inputs, first_ids = inputs_for(
            image, q, options(q), schema.get("state", {})
        )
        with torch.no_grad():
            features = original_features(
                first_inputs["pixel_values"], first_inputs["image_grid_thw"]
            )
            ordinary, _ = scores(first_inputs, first_ids)
            model.model.get_image_features = lambda *a, cached=features, **kw: cached
            reused, _ = scores(first_inputs, first_ids)
        delta = (ordinary - reused).abs().max().item()
        result["reuse_checks"].append(
            {"image": case["image"], "max_probability_delta": delta}
        )
        if delta > 1e-5:
            model.model.get_image_features = original_features
            save()
            raise RuntimeError("Visual feature reuse changed the score")
        try:
            probs = {}
            masses = {}
            sync()
            start = time.perf_counter()
            with torch.no_grad():
                for qid, q in questions:
                    choices = options(q)
                    inputs, ids = inputs_for(image, q, choices, schema.get("state", {}))
                    p, mass = scores(inputs, ids)
                    probs[qid] = {k: v for (k, _), v in zip(choices, p.cpu().tolist())}
                    masses[qid] = mass.exp().item()
            sync()
            result["records"].append(
                {
                    "image": case["image"],
                    "suite": case["suite"],
                    "expected": case["expected"],
                    "probabilities": probs,
                    "label_mass": masses,
                    "questions": len(questions),
                    "warm_reused_seconds": time.perf_counter() - start,
                    "image_tokens": int(
                        first_inputs["image_grid_thw"].prod().item() // 4
                    ),
                }
            )
            # Reverse option order for one question per image before any training.
            choices = list(reversed(options(questions[0][1])))
            inputs, ids = inputs_for(
                image, questions[0][1], choices, schema.get("state", {})
            )
            with torch.no_grad():
                p, _ = scores(inputs, ids)
            result["records"][-1]["reversed_first_question"] = {
                k: v for (k, _), v in zip(choices, p.cpu().tolist())
            }
        finally:
            model.model.get_image_features = original_features
        # Time complete requests: preprocessing + one image encoding + scoring.
        if case["image"].endswith("dog.jpg"):
            for count in (1, 8, 24):
                elapsed = []
                for repeat in range(4):
                    sync()
                    torch.accelerator.reset_peak_memory_stats()
                    start = time.perf_counter()
                    image_copy = image.copy()
                    inp, _ = inputs_for(
                        image_copy,
                        questions[0][1],
                        options(questions[0][1]),
                        schema.get("state", {}),
                    )
                    with torch.no_grad():
                        fresh_features = original_features(
                            inp["pixel_values"], inp["image_grid_thw"]
                        )
                    model.model.get_image_features = (
                        lambda *a, cached=fresh_features, **kw: cached
                    )
                    try:
                        with torch.no_grad():
                            for _, q in questions[:count]:
                                inp, ids = inputs_for(
                                    image_copy, q, options(q), schema.get("state", {})
                                )
                                scores(inp, ids)
                        sync()
                        elapsed.append(time.perf_counter() - start)
                    finally:
                        model.model.get_image_features = original_features
                seconds = sum(elapsed[1:]) / 3
                result["latency"].append(
                    {
                        "questions": count,
                        "warmup_seconds": elapsed[0],
                        "seconds": elapsed[1:],
                        "mean_seconds": seconds,
                        "usd_per_million_requests": seconds
                        * args.hourly_usd
                        / 3600
                        * 1e6,
                        "peak_allocated_gib": torch.accelerator.max_memory_allocated()
                        / 2**30,
                    }
                )
        if case["suite"] == "synthetic" and first_training is None:
            first_training = (image.copy(), copy.deepcopy(case))
        save()
        print("Scored", case["image"], len(questions), flush=True)
        if args.training_only:
            break

    if not args.skip_training:
        from peft import LoraConfig, get_peft_model

        image, case = first_training
        qid, q = next(iter(case["schema"]["questions"].items()))
        choices = options(q)
        inputs, ids = inputs_for(image, q, choices, {})
        target = torch.tensor(
            [next(i for i, (k, _) in enumerate(choices) if k == case["expected"][qid])],
            device=device,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
        assert not any(
            p.requires_grad for n, p in model.named_parameters() if "visual" in n
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        initial_parameters = [p.detach().clone() for p in trainable]
        model.train()
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        measurements = []
        # Multiple input lengths exercise real forward/backward memory, not a
        # training-quality claim. No trained output is used in the scoring table.
        for tokens in (0, 512):
            extended = copy.deepcopy(q)
            if tokens:
                extended["instructions"] += " Context detail." * 170
            inputs, ids = inputs_for(image, extended, choices, {})
            torch.accelerator.reset_peak_memory_stats()
            for step in range(8):
                optimizer.zero_grad(set_to_none=True)
                sync()
                start = time.perf_counter()
                logits = (
                    model(**inputs, use_cache=False, logits_to_keep=1)
                    .logits[:, -1, ids]
                    .float()
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, target, label_smoothing=0.1
                )
                loss.backward()
                if not torch.isfinite(loss) or not all(
                    torch.isfinite(p.grad).all()
                    for p in trainable
                    if p.grad is not None
                ):
                    raise RuntimeError("Nonfinite training loss or gradients")
                optimizer.step()
                sync()
                measurements.append(
                    {
                        "extra_text": bool(tokens),
                        "step": step,
                        "input_tokens": inputs["input_ids"].shape[1],
                        "seconds": time.perf_counter() - start,
                        "loss": loss.item(),
                        "peak_allocated_gib": torch.accelerator.max_memory_allocated()
                        / 2**30,
                    }
                )
        parameter_change = sum(
            (p.detach() - initial).abs().sum().item()
            for p, initial in zip(trainable, initial_parameters)
        )
        if parameter_change == 0:
            raise RuntimeError("Optimizer did not change any adapter parameters")
        result["training"] = {
            "trainable_parameters": sum(p.numel() for p in trainable),
            "label_smoothing": 0.1,
            "adapter_parameter_l1_change": parameter_change,
            "measurements": measurements,
            "note": (
                "Disposable optimizer/throughput smoke test on one synthetic "
                "example; not model training or generalization evaluation."
            ),
        }
        save()
    print("FEASIBILITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
