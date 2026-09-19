# Disposable Vast.ai smoke test

This setup uses one RTX 5090, 100 GB ephemeral disk, the pinned NVFP4
checkpoint, and prebuilt CUDA 13.0 kernels matching the PR's base commit.
`bootstrap.sh` runs inside `/workspace/visjev` after cloning the `visual-probabilities` branch of
this fork. Model and adapter ports bind to localhost;
access the adapter over an SSH tunnel. The Vast API key stays on the local host.

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
