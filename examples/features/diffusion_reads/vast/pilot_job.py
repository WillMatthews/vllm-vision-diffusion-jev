# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local job driver for a screened generic-model rental; credentials stay local."""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

from select_machine import remote, ssh_args

REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def transfer(run, instance, source, target, download=False):
    ssh = ssh_args(run, instance)
    port = ssh.index("-p")
    options = ssh[2:port]  # Drop ssh and its SSH-only -x option.
    remote_path = "root@" + instance["ssh_host"] + ":"
    if download:
        source = remote_path + source
    else:
        target = remote_path + target
    subprocess.run(
        ["scp", *options, "-P", str(instance["ssh_port"]), str(source), str(target)],
        check=True,
        timeout=600,
    )


def script(run, instance, name, text, timeout):
    target = "/workspace/pilot/" + name
    remote(run, instance, "cat > " + target, stdin=text.encode())
    try:
        remote(
            run,
            instance,
            f"timeout {timeout}s bash {target} > {target}.log 2>&1",
            timeout=timeout + 30,
        )
    finally:
        try:
            (run / (name + ".log")).write_bytes(
                remote(run, instance, "cat " + target + ".log")
            )
        except (OSError, subprocess.SubprocessError):
            print(f"Could not retrieve {name} log", flush=True)


def deploy(run, instance, data, image_root):
    evaluation = Path(__file__).resolve().parent.parent / "evaluation"
    rows = [
        json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()
    ]
    images = set()
    for split in ("train", "validation", "test"):
        for line in (data / (split + ".jsonl")).read_text().splitlines():
            item = json.loads(line)
            path = (image_root / item["image"]).resolve()
            if not path.is_relative_to(image_root.resolve()) or not path.is_file():
                raise ValueError("Missing or unsafe image path in manifest")
            images.add(path)
    # Only explicit source files, split manifests and their referenced images.
    archive = run / "pilot-input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("generic_feasibility.py", "train_generic_pilot.py"):
            tar.add(evaluation / name, arcname="evaluation/" + name)
        for split in ("train", "validation", "test"):
            tar.add(data / (split + ".jsonl"), arcname="data/" + split + ".jsonl")
        for path in sorted(images):
            tar.add(path, arcname="data/" + str(path.relative_to(image_root.resolve())))
        for name in ("manifest.json", "image-provenance.jsonl", "image-audit.json"):
            tar.add(data / name, arcname="data/" + name)
        # Smoke examples cover varied families and instruction lengths, train only.
        selected = sorted(rows, key=lambda r: len(r["question"]["instructions"]))
        indices = [round(i * (len(selected) - 1) / 15) for i in range(16)]
        smoke = run / "smoke.jsonl"
        smoke.write_text("".join(json.dumps(selected[i]) + "\n" for i in indices))
        tar.add(smoke, arcname="data/smoke.jsonl")
    remote(run, instance, "mkdir -p /workspace/pilot")
    transfer(run, instance, archive, "/workspace/pilot-input.tar.gz")
    remote(run, instance, "tar -xzf /workspace/pilot-input.tar.gz -C /workspace/pilot")
    setup = f"""set -eu
cd /workspace/acid-test
export PATH="$HOME/.local/bin:$PATH"
uv pip install transformers==5.17.0 peft==0.21.0 pillow==12.3.0 \\
    huggingface-hub==1.32.0
uv pip freeze > /workspace/pilot/requirements.txt
.venv/bin/python - <<'INNER'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-2B-Instruct', revision='{REVISION}',
    local_dir='/workspace/model',
    allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
INNER
"""
    script(run, instance, "setup.sh", setup, 600)


def commands(mode, manifest, output, extra=""):
    return (
        "/workspace/acid-test/.venv/bin/python evaluation/train_generic_pilot.py "
        f"{mode} --model /workspace/model --manifest data/{manifest}.jsonl "
        f"--image-root data --output {output} {extra}\n"
    )


def smoke(run, instance):
    text = "set -eu\ncd /workspace/pilot\n"
    text += "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu\n"
    text += commands(
        "train",
        "smoke",
        "smoke-training",
        "--max-presentations 16 --epochs 1 --gradient-accumulation 1 "
        "--log-every 1 --label-smoothing 0.1",
    )
    text += commands(
        "evaluate",
        "smoke",
        "smoke-evaluation",
        "--adapter smoke-training/checkpoint-000016/adapter --order-checks 4",
    )
    script(run, instance, "smoke.sh", text, 300)
    records = [
        json.loads(line)
        for line in remote(
            run, instance, "cat /workspace/pilot/smoke-training/training.jsonl"
        )
        .decode()
        .splitlines()
    ]
    complete = json.loads(
        remote(run, instance, "cat /workspace/pilot/smoke-training/complete.json")
    )
    evaluation = json.loads(
        remote(run, instance, "cat /workspace/pilot/smoke-evaluation/evaluation.json")
    )
    deltas = [
        b["elapsed_seconds_this_process"] - a["elapsed_seconds_this_process"]
        for a, b in zip(records[3:-1], records[4:])
    ]
    peak = max(row["peak_allocated_gib"] for row in records)
    mean = statistics.mean(deltas)
    passed = (
        complete["complete"]
        and complete["presentations"] == 16
        and complete["adapter_parameter_l1_change"] > 0
        and evaluation["complete"]
        and len(evaluation["records"]) == 16
        and all(
            all(math.isfinite(p) and 0 <= p <= 1 for p in row["probabilities"].values())
            and abs(sum(row["probabilities"].values()) - 1) < 1e-5
            for row in evaluation["records"]
        )
        and all(math.isfinite(r["mean_loss"]) for r in records)
        and mean <= 0.4
        and peak <= 18
    )
    result = {
        "passed": passed,
        "warm_mean_presentation_seconds": mean,
        "peak_allocated_gib": peak,
        "records": records,
        "thresholds": {"max_seconds": 0.4, "max_peak_gib": 18},
    }
    (run / "workload-probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    if not passed:
        raise RuntimeError("Real training workload failed its machine gate")


def train_and_evaluate(run, instance):
    text = "set -eu\ncd /workspace/pilot\n"
    text += "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu\n"
    # Fix the adapter before opening any test predictions. Validation has no
    # checkpoint selection in this fixed two-epoch pilot.
    text += commands(
        "train",
        "train",
        "training",
        "--exclude-manifest data/validation.jsonl --exclude-manifest data/test.jsonl",
    )
    for split in ("validation", "test"):
        text += commands("evaluate", split, "base-" + split)
        text += commands(
            "evaluate",
            split,
            "adapted-" + split,
            "--adapter training/checkpoint-040000/adapter",
        )
    text += "echo PILOT_COMPLETE\n"
    try:
        script(run, instance, "train-evaluate.sh", text, 5 * 3600)
    finally:
        # Explicit result directories only; no model weights or credentials.
        remote(
            run,
            instance,
            "cd /workspace/pilot && tar -czf /workspace/pilot-results.tar.gz "
            "--ignore-failed-read training base-validation adapted-validation "
            "base-test adapted-test requirements.txt smoke-training smoke-evaluation",
            timeout=120,
        )
        transfer(
            run,
            instance,
            "/workspace/pilot-results.tar.gz",
            run / "pilot-results.tar.gz",
            download=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "train"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--teacher-driver", type=Path)
    args = parser.parse_args()
    run = Path(os.environ["MACHINE_RUN_DIR"]).absolute()
    instance = json.loads((run / "connection.json").read_text())
    if args.mode == "smoke":
        deploy(run, instance, args.data.absolute(), args.image_root.absolute())
        smoke(run, instance)
    else:
        train_and_evaluate(run, instance)
        if args.teacher_driver:
            subprocess.run([sys.executable, str(args.teacher_driver)], check=True)


if __name__ == "__main__":
    main()
