#!/usr/bin/env bash
set -euo pipefail
cd /workspace/visjev
exec > >(tee -a /workspace/bootstrap.log) 2>&1
trap 'echo "Bootstrap exited with status $? at $(date -Is)"' EXIT
apt-get update -qq
apt-get install -y -qq python3.12-dev build-essential ninja-build cmake
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12; fi
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
if [ ! -x .venv/bin/vllm ]; then uv pip install -e . --torch-backend=cu130; fi
unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_COMMIT VLLM_PRECOMPILED_WHEEL_VARIANT
.venv/bin/python -c 'import torch, vllm; print(torch.__version__, torch.cuda.get_device_name()); print(vllm.__file__)'
if [ -f /workspace/visjev-kernel-cache.tar.gz ]; then
    mkdir -p "$HOME/.cache"
    tar -xzf /workspace/visjev-kernel-cache.tar.gz -C "$HOME/.cache"
fi
export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('nvidia/diffusiongemma-26B-A4B-it-NVFP4',
                  revision='ffc65bb9103ef37ba010cdfe259c2cde5401c579', local_dir='/workspace/model',
                  allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model'])
PY
.venv/bin/vllm serve /workspace/model --served-model-name dgemma \
    --host 127.0.0.1 --port 8000 --max-model-len 4096 \
    --diffusion-config '{"canvas_length":64}' --max-logprobs 32 \
    --max-num-seqs 1 --gpu-memory-utilization 0.9 \
    --enable-prefix-caching --attention-backend TRITON_ATTN \
    --enforce-eager > /workspace/vllm.log 2>&1 &
model_pid=$!
trap 'kill "$model_pid" 2>/dev/null || true' EXIT
for _attempt in $(seq 1 180); do
    kill -0 "$model_pid" || { tail -80 /workspace/vllm.log; exit 1; }
    if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
        exec .venv/bin/python examples/features/diffusion_reads/structured_server.py \
            --upstream http://127.0.0.1:8000 --model dgemma \
            --tokenizer /workspace/model --canvas 64 --host 127.0.0.1 --port 8011
    fi
    sleep 5
done
exit 1
