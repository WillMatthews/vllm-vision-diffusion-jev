# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the frozen DiffusionGemma comparison on an already accepted Vast host.

MACHINE_RUN_DIR supplies local SSH metadata. No Vast token is read or uploaded.
The outer select_machine.py owns billing, lifecycle, and instance destruction.
This driver blocks until its bounded remote stage finishes and retrieves results
also on failure. It never starts another rental or changes serving parameters.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

BASE_COMMIT = "0323d37edf54cceebdd2cf3203e3d50f10f3efe4"
MODEL_REVISION = "ffc65bb9103ef37ba010cdfe259c2cde5401c579"
PUBLIC_REPO = "https://github.com/WillMatthews/vllm-vision-diffusion-jev.git"
ROOT = Path(__file__).resolve().parents[4]
ALLOWLIST = (
    "vllm/v1/worker/gpu/model_runner.py",
    "examples/features/diffusion_reads/structured_server.py",
    "examples/features/diffusion_reads/visual_client.py",
    "examples/features/diffusion_reads/evaluation/generic_feasibility.py",
    "examples/features/diffusion_reads/evaluation/train_generic_pilot.py",
    "examples/features/diffusion_reads/evaluation/predict_generic_pilot.py",
    "examples/features/diffusion_reads/evaluation/benchmark_generic_inputs.py",
    "examples/features/diffusion_reads/evaluation/evaluate_generic_teacher.py",
)


def remote_script():
    return (
        r"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/teacher/results
exec > >(tee -a /workspace/teacher/setup.log) 2>&1
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME=/workspace/teacher/hf
export HF_HUB_DISABLE_PROGRESS_BARS=1
export MAX_JOBS=6
export FLASHINFER_NVCC_THREADS=1
mkdir -p /workspace/teacher/generic-code /workspace/teacher/results/generic-input
tar -xzf /workspace/teacher/overlay.tar.gz -C /workspace/teacher/generic-code \
    examples/features/diffusion_reads/evaluation/generic_feasibility.py \
    examples/features/diffusion_reads/evaluation/train_generic_pilot.py \
    examples/features/diffusion_reads/evaluation/predict_generic_pilot.py \
    examples/features/diffusion_reads/evaluation/benchmark_generic_inputs.py
generic_status=0
generic_code=/workspace/teacher/generic-code/examples/features/diffusion_reads
timeout --signal=TERM --kill-after=15s 10m \
    /workspace/acid-test/.venv/bin/python \
    "$generic_code/evaluation/benchmark_generic_inputs.py" \
    --model /workspace/model \
    --adapter /workspace/pilot/training/checkpoint-040000/adapter \
    --manifest /workspace/pilot/data/validation.jsonl \
    --image-root /workspace/pilot/data \
    --output /workspace/teacher/results/generic-input \
    > /workspace/teacher/results/generic-input/benchmark.log 2>&1 \
    || generic_status=$?
printf '{"exit_code": %s}\n' "$generic_status" \
    > /workspace/teacher/results/generic-input/status.json
if [ "$generic_status" -ne 0 ]; then
    echo "Generic input benchmark failed ($generic_status); retaining logs"
    tail -40 /workspace/teacher/results/generic-input/benchmark.log
fi
free_kib=$(df -Pk /workspace | awk 'NR==2 {print $4}')
if [ "$free_kib" -lt 47185920 ]; then
    echo 'Teacher stage requires at least 45 GiB free disk before installation'
    exit 1
fi
apt-get update -qq
apt-get install -y -qq git curl ca-certificates python3.12-dev \
    build-essential ninja-build cmake
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
git init /workspace/teacher/repo
cd /workspace/teacher/repo
git remote add origin PUBLIC_REPO
git fetch --depth 1 origin BASE_COMMIT
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = BASE_COMMIT
tar -xzf /workspace/teacher/overlay.tar.gz -C /workspace/teacher/repo
uv venv --python 3.12
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
uv pip install -e . --torch-backend=cu130
uv pip install --no-deps 'flashinfer-jit-cache==0.6.18.post1+cu130' \
    --index-url https://flashinfer.ai/whl/cu130
unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_COMMIT VLLM_PRECOMPILED_WHEEL_VARIANT
uv pip freeze > /workspace/teacher/results/requirements.txt
nvidia-smi > /workspace/teacher/results/gpu-info.txt
.venv/bin/python - <<'PYMODEL'
from huggingface_hub import snapshot_download
snapshot_download('nvidia/diffusiongemma-26B-A4B-it-NVFP4',
                  revision='MODEL_REVISION', local_dir='/workspace/teacher/model',
                  allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model'])
PYMODEL
model_pid=''
adapter_pid=''
cleanup() {
    if [ -n "$adapter_pid" ]; then kill "$adapter_pid" 2>/dev/null || true; fi
    if [ -n "$model_pid" ]; then kill -- -"$model_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT
setsid .venv/bin/vllm serve /workspace/teacher/model --served-model-name dgemma \
    --host 127.0.0.1 --port 8000 --max-model-len 4096 \
    --diffusion-config '{"canvas_length":64}' --max-logprobs 32 \
    --max-num-seqs 1 --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --enable-prompt-tokens-details --attention-backend TRITON_ATTN \
    --enforce-eager --moe-backend marlin > /workspace/teacher/model.log 2>&1 &
model_pid=$!
ready=0
for _ in $(seq 1 900); do
    kill -0 "$model_pid" || { tail -80 /workspace/teacher/model.log; exit 1; }
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
test "$ready" = 1
.venv/bin/python examples/features/diffusion_reads/structured_server.py \
    --upstream http://127.0.0.1:8000 --model dgemma \
    --tokenizer /workspace/teacher/model --canvas 64 --canvas-step 16 \
    --host 127.0.0.1 --port 8011 > /workspace/teacher/adapter.log 2>&1 &
adapter_pid=$!
ready=0
for _ in $(seq 1 120); do
    kill -0 "$adapter_pid" || { tail -80 /workspace/teacher/adapter.log; exit 1; }
    if curl -fsS http://127.0.0.1:8011/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
test "$ready" = 1
evaluator=examples/features/diffusion_reads/evaluation/evaluate_generic_teacher.py
for split in validation test; do
    .venv/bin/python "$evaluator" \
        --manifest "/workspace/pilot/data/$split.jsonl" \
        --image-root /workspace/pilot/data \
        --exclude-manifest /workspace/pilot/data/train.jsonl \
        --model-revision MODEL_REVISION \
        --output "/workspace/teacher/results/$split"
done
printf 'TEACHER_JOB_COMPLETE\n'
test "$generic_status" = 0
""".replace("BASE_COMMIT", BASE_COMMIT)
        .replace("MODEL_REVISION", MODEL_REVISION)
        .replace("PUBLIC_REPO", PUBLIC_REPO)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=os.environ.get("MACHINE_RUN_DIR"))
    parser.add_argument("--max-minutes", type=int, default=90)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.run is None or not 1 <= args.max_minutes <= 90:
        parser.error("Provide MACHINE_RUN_DIR/--run and a 1–90 minute ceiling")
    run = args.run.resolve()
    local = run / "teacher"
    local.mkdir(parents=True, exist_ok=True)
    archive = local / "overlay.tar.gz"
    manifest = {}
    with tarfile.open(archive, "w:gz") as tar:
        for relative in ALLOWLIST:
            path = ROOT / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or symlinked allowlisted source: {relative}")
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            tar.add(path, arcname=relative, recursive=False)
    script = local / "remote.sh"
    script.write_text(remote_script())
    (local / "deployment.json").write_text(
        json.dumps(
            {
                "public_base_repository": PUBLIC_REPO,
                "base_commit": BASE_COMMIT,
                "model_revision": MODEL_REVISION,
                "source_overlay_sha256": manifest,
                "overlay_archive_sha256": hashlib.sha256(
                    archive.read_bytes()
                ).hexdigest(),
                "remote_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "max_minutes": args.max_minutes,
                "configuration": (
                    "seqs1/canvas64/eager/memory0.9/TRITON_ATTN/prefixcache/marlin"
                ),
                "generic_input_benchmark": {
                    "selection": "First three validation images in manifest order",
                    "questions_per_image": 4,
                    "models": ["base", "training/checkpoint-040000/adapter"],
                    "timed_repeats": 3,
                    "reuse_equivalence_check": True,
                    "representative_images": 100,
                    "representative_repeats": 2,
                    "representative_warmups": 3,
                    "representative_model_loads": "One per base/adapter worker",
                    "timeout_minutes": 10,
                },
            },
            indent=2,
        )
        + "\n"
    )
    if args.prepare_only:
        print("Teacher deployment prepared; no network calls or rental changes")
        return
    connection = json.loads((run / "connection.json").read_text())
    host = "root@" + connection["ssh_host"]
    common = [
        "-i",
        str(run / "id_ed25519"),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "UserKnownHostsFile=" + str(run / "known_hosts"),
    ]
    ssh = ["ssh", "-x", *common, "-p", str(connection["ssh_port"]), host]
    scp = ["scp", *common, "-P", str(connection["ssh_port"])]
    subprocess.run(
        [*ssh, "mkdir -p /workspace/teacher/results"], check=True, timeout=30
    )
    for path in (archive, script):
        subprocess.run(
            [*scp, str(path), host + ":/workspace/teacher/" + path.name],
            check=True,
            timeout=180,
        )
    try:
        command = (
            f"timeout --signal=TERM --kill-after=30s {args.max_minutes}m "
            "bash /workspace/teacher/remote.sh"
        )
        subprocess.run([*ssh, command], check=True, timeout=args.max_minutes * 60 + 60)
    finally:
        # Explicit result/log paths only; the archive never includes models,
        # credentials, user images, environments or unrelated remote files.
        command = (
            "tar --ignore-failed-read -czf /workspace/teacher-results.tar.gz "
            "-C /workspace/teacher results setup.log model.log adapter.log"
        )
        subprocess.run([*ssh, command], check=True, timeout=120)
        subprocess.run(
            [
                *scp,
                host + ":/workspace/teacher-results.tar.gz",
                str(local / "results.tar.gz"),
            ],
            check=True,
            timeout=180,
        )
    print("Teacher comparison finished and results downloaded", flush=True)


if __name__ == "__main__":
    main()
