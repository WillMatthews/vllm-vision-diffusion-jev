# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, disposable machine checks; invoke with a uv-managed Python.

Each stage runs in a subprocess with a wall-clock timeout. Exit 0 means all
requested checks passed; exit 2 means reject the machine. Preflight deliberately
omits GPU checks and must be followed by a full probe after installing torch.
Disk eviction uses POSIX advice, which the kernel may ignore: read throughput
is a cold-read attempt, never a guarantee of physical storage performance.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

MIB = 1024**2
GIB = 1024**3


def capacity(config):
    """Read available filesystem capacity and host memory without allocations."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            memory[key] = int(value.split()[0]) * 1024 / GIB
    available = memory.get("MemAvailable", 0)
    cgroup_limit = None
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            limit = int(Path(limit_path).read_text())
            used = int(Path(used_path).read_text())
        except (OSError, ValueError):
            continue
        cgroup_limit = limit / GIB
        available = min(available, max(0, limit - used) / GIB)
        break
    return {
        "disk_free_gib": shutil.disk_usage(config["work_dir"]).free / GIB,
        "host_memory_total_gib": memory.get("MemTotal", 0),
        "host_memory_available_gib": available,
        "cgroup_memory_limit_gib": cgroup_limit,
        "cpu_count": os.cpu_count(),
    }


def disk(config):
    """Measure fsynced writes and advised-cold sequential/random reads."""
    size = config["disk_mib"] * MIB
    block = os.urandom(MIB)
    # Parent owns this directory and removes it even if the worker is killed.
    path = Path(config["scratch_dir"]) / "io.bin"
    with path.open("w+b", buffering=0) as stream:
        start = time.perf_counter()
        for _ in range(config["disk_mib"]):
            view = memoryview(block)
            while view:
                written = stream.write(view)
                if not written:
                    raise OSError("Short disk write")
                view = view[written:]
        os.fsync(stream.fileno())
        write_seconds = time.perf_counter() - start
        os.posix_fadvise(stream.fileno(), 0, size, os.POSIX_FADV_DONTNEED)
        stream.seek(0)
        start = time.perf_counter()
        read_bytes = 0
        while chunk := stream.read(MIB):
            read_bytes += len(chunk)
            if chunk != block:
                raise RuntimeError("Disk readback differs from written bytes")
        read_seconds = time.perf_counter() - start
        os.posix_fadvise(stream.fileno(), 0, size, os.POSIX_FADV_DONTNEED)
        rng = random.Random(42)
        start = time.perf_counter()
        for _ in range(config["random_reads"]):
            offset = rng.randrange(size // 4096) * 4096
            data = os.pread(stream.fileno(), 4096, offset)
            expected = block[offset % MIB : offset % MIB + 4096]
            if data != expected:
                raise RuntimeError("Random disk readback differs")
        random_seconds = time.perf_counter() - start
    return {
        "bytes": size,
        "write_fsync_mib_s": size / MIB / write_seconds,
        "advised_cold_read_mib_s": read_bytes / MIB / read_seconds,
        "random_4k_iops": config["random_reads"] / random_seconds,
        "eviction": "POSIX_FADV_DONTNEED (advisory)",
        "readback_correct": True,
    }


def network(config):
    """Download at most the configured payload, including connection latency."""
    size = config["download_mib"] * MIB
    request = urllib.request.Request(
        config["download_url"],
        headers={"Range": f"bytes=0-{size - 1}", "Accept-Encoding": "identity"},
    )
    start = time.perf_counter()
    total = 0
    with urllib.request.urlopen(request, timeout=15) as response:
        status = response.status
        if status not in (200, 206):
            raise RuntimeError(f"Unexpected HTTP status {status}")
        while total < size:
            chunk = response.read(min(MIB, size - total))
            if not chunk:
                break
            total += len(chunk)
    elapsed = time.perf_counter() - start
    if total != size:
        raise RuntimeError(f"Downloaded only {total} of {size} requested bytes")
    return {
        "bytes": total,
        "seconds": elapsed,
        "download_mib_s": total / MIB / elapsed,
        "http_status": status,
        "source_host": urllib.parse.urlsplit(config["download_url"]).hostname,
    }


def gpu(config):
    """Check BF16 GEMM correctness against CPU FP32 and sustained throughput."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    small_a = torch.randn(128, 128, dtype=torch.bfloat16)
    small_b = torch.randn(128, 128, dtype=torch.bfloat16)
    expected = (small_a.float() @ small_b.float()).to(torch.bfloat16).float()
    actual = (small_a.to(device) @ small_b.to(device)).float().cpu()
    error = (actual - expected).abs().max().item()
    if not torch.isfinite(actual).all() or not torch.allclose(
        actual, expected, rtol=0.02, atol=0.125
    ):
        raise RuntimeError(f"BF16 GPU GEMM correctness failed (max error {error})")
    n = 4096
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    for _ in range(5):
        torch.mm(a, b, out=out)
    torch.accelerator.synchronize()
    start = time.perf_counter()
    iterations = 0
    windows = []
    while time.perf_counter() - start < config["gpu_seconds"]:
        window_start = time.perf_counter()
        for _ in range(20):
            torch.mm(a, b, out=out)
        torch.accelerator.synchronize()
        windows.append(20 * 2 * n**3 / (time.perf_counter() - window_start) / 1e12)
        iterations += 20
    elapsed = time.perf_counter() - start
    if not torch.isfinite(out).all().item():
        raise RuntimeError("Non-finite output after sustained GEMM")
    free, total = torch.accelerator.get_memory_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "total_gib": total / GIB,
        "free_after_probe_gib": free / GIB,
        "correctness_max_abs_error": error,
        "finite": True,
        "bf16_tflops": iterations * 2 * n**3 / elapsed / 1e12,
        "window_min_tflops": min(windows),
        "window_max_tflops": max(windows),
        "seconds": elapsed,
        "iterations": iterations,
    }


STAGES = {"capacity": capacity, "disk": disk, "network": network, "gpu": gpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--download-url")
    parser.add_argument("--phase", choices=("preflight", "full"), default="full")
    parser.add_argument("--disk-mib", type=int, default=1024)
    parser.add_argument("--download-mib", type=int, default=128)
    parser.add_argument("--random-reads", type=int, default=512)
    parser.add_argument("--gpu-seconds", type=float, default=30)
    parser.add_argument("--stage-timeout", type=float, default=90)
    parser.add_argument("--min-disk-free-gib", type=float, default=30)
    parser.add_argument("--min-host-memory-gib", type=float, default=24)
    parser.add_argument("--min-write-mib-s", type=float, default=100)
    parser.add_argument("--min-read-mib-s", type=float, default=150)
    parser.add_argument("--min-random-iops", type=float, default=500)
    parser.add_argument("--min-download-mib-s", type=float, default=25)
    parser.add_argument("--min-gpu-gib", type=float, default=22)
    parser.add_argument("--min-bf16-tflops", type=float, default=50)
    parser.add_argument("--worker", choices=tuple(STAGES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(STAGES[args.worker](json.load(sys.stdin))))
        return 0
    if args.output is None or not args.download_url:
        parser.error("--output and an explicit --download-url are required")
    parsed = urllib.parse.urlsplit(args.download_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        parser.error("--download-url must be a public HTTPS URL without credentials")
    for key, value in vars(args).items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
            parser.error(f"{key} must be finite and positive")
    if args.disk_mib > 4096 or args.download_mib > 1024:
        parser.error("Disk/download bounds are 4096/1024 MiB")
    if args.random_reads > 4096 or args.gpu_seconds > 60 or args.stage_timeout > 180:
        parser.error("Bounds: 4096 random reads, 60 GPU seconds, 180s per stage")
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["work_dir"] = str(args.work_dir.resolve())
    thresholds = {
        "capacity": {
            "disk_free_gib": args.min_disk_free_gib,
            "host_memory_available_gib": args.min_host_memory_gib,
        },
        "disk": {
            "write_fsync_mib_s": args.min_write_mib_s,
            "advised_cold_read_mib_s": args.min_read_mib_s,
            "random_4k_iops": args.min_random_iops,
        },
        "network": {"download_mib_s": args.min_download_mib_s},
        "gpu": {"total_gib": args.min_gpu_gib, "bf16_tflops": args.min_bf16_tflops},
    }
    report = {
        "phase": args.phase,
        "passed": False,
        "stages": {},
        "thresholds": thresholds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stages = ("capacity", "disk", "network", "gpu")
    if args.phase == "preflight":
        stages = stages[:-1]
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="machine-probe-", dir=args.work_dir) as tmp:
        config["scratch_dir"] = tmp
        for name in stages:
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", name],
                    input=json.dumps(config),
                    capture_output=True,
                    text=True,
                    timeout=args.stage_timeout,
                    check=True,
                )
                metrics = json.loads(result.stdout)
                failures = [
                    f"{key}={metrics[key]:.3f} below {minimum}"
                    for key, minimum in thresholds[name].items()
                    if not math.isfinite(metrics[key]) or metrics[key] < minimum
                ]
                report["stages"][name] = {"metrics": metrics, "failures": failures}
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                # Omit stderr/URL details, which can contain signed redirect URLs.
                report["stages"][name] = {"failures": [type(exc).__name__]}
            report["elapsed_seconds"] = time.time() - started
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: {report['stages'][name]}", flush=True)
            if report["stages"][name]["failures"]:
                return 2
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
