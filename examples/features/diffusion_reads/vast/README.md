# Disposable Vast.ai smoke test

## Generic training machine screening

The generic pilot has **$10 additional user approval**, including rejected-host
screening, training, evaluation and transfers. The earlier $2 feasibility stage
is separate and complete. The target is one verified RTX 4090 (24 GB), at least
8 effective CPU cores, 64,000 MB advertised host RAM, 100 GB disk and CUDA 13
driver support. The accepted host on 2026-09-20 cost $0.4585/hour including disk;
the controller defaults to a $0.75/hour ceiling. These are transient quotes.

[`select_machine.py`](select_machine.py) searches without renting by default:

```bash
source ~/.tokens
export VAST_API_KEY
.venv/bin/python examples/features/diffusion_reads/vast/select_machine.py
```

With explicit `--rent`, it creates a fresh run directory and scoped teardown
timer before creating any server. It screens at most **three different physical
machine IDs**, with a **$1.50 screening allowance** within the total budget.
A rejected instance is destroyed and absence verified before another create.
An ambiguous create response stops automatic retries; the cleanup timer remains
armed. Unconfirmed deletion, missing telemetry or exhausted budgets stop the
entire run. A successful host is also destroyed when its foreground job driver
returns. Without a job, a passing probe is downloaded and the rental destroyed.

[`machine_probe.py`](machine_probe.py) performs bounded measurements on each
candidate. Every stage has a timeout and writes structured pass/fail evidence:

| Check | Default admission threshold |
| --- | --- |
| Capacity visible inside container | ≥30 GiB disk free, ≥24 GiB RAM available |
| Sequential storage, 1 GiB temporary file | ≥100 MiB/s fsynced writes, ≥150 MiB/s reads |
| Random storage, 512 × 4 KiB reads | ≥500 IOPS, correct readback |
| Model-source HTTPS download, 128 MiB | ≥25 MiB/s measured end-to-end |
| GPU memory and arithmetic | ≥22 GiB, BF16 result agrees with CPU FP32 tolerance |
| Sustained 4096² BF16 matrix multiply, 30 seconds | ≥50 TFLOP/s average, finite output |
| Optional real training workload | ≤0.4 s/warm presentation, ≤18 GiB allocated |

The disk test requests page-cache eviction but the kernel may ignore it; read
results are not guaranteed physical-disk bandwidth. The HTTP probe uses an
explicit public model object and stops at its byte limit even if the server
ignores Range. Only the tested route is characterized. The GPU test is a short
health/performance screen, not proof against every thermal or hardware fault.
Per-window GPU throughput is retained to expose variability.

The optional [`pilot_job.py`](pilot_job.py) workload gate selects 16 examples
from **training data only**, runs disposable LoRA updates, then checks the
adapter's option scoring and image-feature reuse. It downloads the checkpoint
and deploys the prepared public dataset onto a passing host so that accepted
machines can go directly into training. `--workload-probe` is a local Python
driver path; it receives `MACHINE_RUN_DIR`. `--job` takes a foreground local
command with the same environment. Both must wait for their remote work and
retrieve artifacts before exiting. A full job failure is not silently rerun on
another paid host.

Example probe-only rental (destroys the passing host too):

```bash
.venv/bin/python examples/features/diffusion_reads/vast/select_machine.py \
    --rent --run .vast-run/probe-unique-name \
    --budget-usd 10 --screening-usd 1.50 --max-attempts 3 \
    --download-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/model.safetensors
```

The controller keeps the API key local. It uploads only probe code until a local
workload driver explicitly deploys its allowlisted public inputs. It incorporates
the observed SSH permission repair and installs compiler/Python headers required
by Triton. Each attempt retains its quoted offer, logs, probes and rejection
reason. Failed setup is useful evidence, not a silent benchmark omission.

Budget controls combine observed account credit change with a conservative
elapsed-time estimate at the maximum hourly rate. Defaults allow at most eight
hours from the first rental, reserve $1 for transfers plus $2 for cleanup/lag,
and monitor cumulative container network bytes with a 100 GiB cutoff. Quotes
must charge ≤$0.005/GB in each direction. Container counters exclude provider
image pulls; the transfer reserve covers that additional overhead. The detached
timer requires this local machine to remain online and systemd to function;
billing lag and provider outages mean this is not a provider-enforced hard cap.
Never interpret unused approval as a spending target.

The lifecycle tests use fake APIs and create no rentals:

```bash
.venv/bin/python -m unittest discover \
    -s examples/features/diffusion_reads/vast -p test_control.py
```

## Measured first admission

The [first host passed](machine-screening-result.json), with no retries:
95.5 MiB/s model-source download, 2,386.6 MiB/s fsynced writes, 2,866.1 MiB/s
advisory-cold reads, 15,413 random-read IOPS and 131.6 BF16 TFLOP/s. Its actual
16-presentation adapter workload averaged 0.1167 seconds per warm presentation
and peaked at 5.02 GiB allocated GPU memory. These establish machine suitability,
not model quality. The [completed training pilot](../evaluation/GENERIC_PILOT.md)
used $0.88 of the $10 budget; this host was destroyed and zero account instances
confirmed. Rejection and cleanup ordering are also covered by fake-API
tests; no deliberately bad paid host was rented just to exercise retries.

## Original diffusion smoke test

This setup uses one RTX 5090, 100 GB ephemeral disk, the pinned NVFP4
checkpoint, and prebuilt CUDA 13.0 kernels matching the PR's base commit.
`bootstrap.sh` runs inside `/workspace/visjev` after cloning the `visual-probabilities` branch of
this fork. Model and adapter ports bind to localhost;
access the adapter over an SSH tunnel. The Vast API key stays on the local host.

The bootstrap installs the matching official CUDA 13.0 `flashinfer-jit-cache`
wheel. It includes the SM120 fused-MoE shared library and avoids compiling that
large module locally. This is separate from the optional local cache archive
below. The cost-optimization run verified the exact library in the installed
wheel after unrestricted local compilation exhausted host memory. See the
[official wheel index](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/).

For the tested CUDA 13.0 development container on x86-64, the bootstrap puts
the host driver libraries first in `LD_LIBRARY_PATH`. The bundled compatibility
driver caused CUDA error 803 on this host; the host library passed the GPU smoke
test. Compiler jobs default to six to bound fallback build memory use.

Before renting:

1. Run the client and lifecycle tests; lint the bootstrap.
2. Verify the wheel, checkpoint revision, Docker image, and current offer price.
3. Prepare `.vast-run/plan.json`, source overlay, SSH key, and smoke fixtures.
4. Exercise the exact cleanup command against the empty run label.
5. Arm and verify a detached cleanup timer **before** creating the instance.

For the original local run, the generated helper (not committed) was:

```bash
.vast-run/destroy.sh
```

It destroys only instances bearing this run's unique saved label and confirms
they disappear from the account listing. Destroy releases the ephemeral disk;
stopping the model process or stopping the instance is not equivalent.
Run artefacts and logs belong in the git-ignored `.vast-run/` directory.
Copy results back before destruction; preserve a copy of `plan.json` until
cleanup is confirmed. Never repeat a create request after an ambiguous timeout:
inspect the label using `control.py status` first.

The original run armed a local `visjev-budget-cleanup` systemd timer to destroy
the instance after two hours. A new run must create its own timer and plan.
The cleanup service retries failures. This survives the agent turn ending,
but requires this machine to stay online and its user service manager running.
It is not a provider-enforced spending cap. Manual fallback is the Destroy
button at <https://console.vast.ai/instances/>. The original run had a $5 budget;
compute/disk pricing and transfer charges must be rechecked for each rental.

```bash
systemctl --user status visjev-budget-cleanup.timer
journalctl --user -u visjev-budget-cleanup.service --no-pager
```

The colour fixtures test that images affect probabilities. They are not an
accuracy or calibration benchmark for real screenshots. GPU results and any
setup failures should be retained alongside the plan and destroyed-instance
confirmation.

## First GPU run: 2026-09-19

The image pipeline ran on a 32 GB RTX 5090 using the NVFP4 checkpoint. All six
labelled decisions across three solid-colour images were correct. The first
request took 6.89 seconds; the following two took 0.467 and 0.500 seconds over
an SSH tunnel. This used eager execution, one sample, and no generated thought.
These are smoke-test observations, not a representative benchmark.

The colour-choice distributions put 99.94%, 99.86%, and 96.57% on the correct
red, green, and blue labels respectively. However, total raw probability on
those allowed label tokens was only 32.0%, 9.1%, and 1.9%; the highest-scoring
vocabulary token was outside the label set in each case. This demonstrates why
normalized choice scores must not be interpreted as calibrated confidence.
The yes/no question had much higher allowed-label mass (93.9–98.9%).

A fresh CUDA development image also needed `python3.12-dev`, `ninja-build`,
and compiler tools for Triton/FlashInfer runtime builds. The bootstrap now
installs these explicitly. Initial FlashInfer compilation took several minutes
even with the prebuilt vLLM wheel. The resulting cache was saved locally as
`.vast-run/visjev-kernel-cache.tar.gz` with a SHA-256 sidecar. For a matching
GPU/software environment, upload that archive to `/workspace/` before running
the bootstrap to restore it automatically. Cache restoration is optional and
has not yet been timed on a second fresh host. The exact installed versions
are recorded in `.vast-run/requirements-freeze.txt`.

The first host was destroyed after a provisioning timeout. The replacement
was destroyed after saving results and logs; API listing confirmed no remaining
instances. Local artefacts include `visual-results.json`, `vllm.log`,
`bootstrap.log`, and `teardown-confirmation.json` under `.vast-run/`.

The portable teardown helper reads a run's unique `label` from `plan.json`:

```bash
export VAST_API_KEY  # load the value from your own credential store first
.venv/bin/python examples/features/diffusion_reads/vast/control.py destroy \
    --run /path/to/run-directory
```

A fresh clone does not include the original `.vast-run/` state, credentials,
timer, logs, or compiled cache. The sanitized [smoke results](smoke-results.json)
are committed for review. Provisioning is not automatic: prepare a fresh plan
and cleanup timer before using the create helper.
