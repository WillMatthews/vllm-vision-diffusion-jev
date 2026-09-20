# Generic image-to-JEV feasibility

Completed 2026-09-20 on `cost-optimization-diffusion`. **Shared-adapter training
is technically feasible; a quality-preserving cheaper replacement is not yet
established.** Observed additional Vast spend was **$0.2487 / $2 approved**.
Both rentals were destroyed; the account listed zero instances at
01:18:05 UTC. Billing can post with a delay.

This tests one model conditioned on the supplied question and answer options,
not separately trained animal classifiers. The original diffusion serving path
has not been replaced. No production adapter checkpoint was trained or exported.

## What ran

- Qwen3-VL-2B-Instruct, revision
  `89644892e4d85e24eaac8bacfd4f463576704203`, BF16, SDPA, RTX 4090.
- Torch 2.13.0/CUDA 13, Transformers 5.17.0, PEFT 0.21.0.
- Nine existing public photos plus eight generated geometry images: 216 question
  inputs and 91 labelled decisions per full run. Choice, boolean and ordinal
  options are supplied in the request; verified single-token letters map back
  to those options. The largest existing question has 24 options.
- Two image pixel limits, sequential scoring, batched scoring, a CPU thread
  control, preprocessing the image once, and two disposable optimizer tests.
- Complete visual features, including DeepStack features, are reused within
  each request. The language model still processes visual tokens per question.

The [pinned upstream implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
defines the visual feature interface used here. This is an experimental scorer,
not a finished JEV API: dependency/ask-if execution, abstention and calibrated
probabilities remain unimplemented. The ordinal experiment returns distributions
over supplied levels; it does not yet expose the final expected-value API.

## Inference cost and the important failure

The table measures a decoded dog image through all option probabilities. It
includes preprocessing, one fresh GPU image encoding and every language-model
question pass. Each cell averages three warm repetitions after one warmup.
These are single-request measurements, not concurrent serving throughput.

| Method | 1 question | 8 questions | 24 questions | $/million 24-question requests |
| --- | ---: | ---: | ---: | ---: |
| Sequential, default CPU threads | 632 ms | 2,768 ms | 7,933 ms | $1,060.16 |
| Sequential, 4 CPU threads | 86 ms | 381 ms | 1,090 ms | $145.67 |
| Batched, 4 CPU threads | 63 ms | 205 ms | 504 ms | $67.36 |
| Batched, preprocess image once, 4 CPU threads | 61 ms | 128 ms | 361 ms | $48.21 |

Rate: $0.481111/hour including allocated disk. Cost is seconds × hourly rate /
3,600 × one million. It excludes model startup, idle capacity, HTTP/disk I/O,
transfers and unimplemented dependency handling. Actual stage spending includes
setup and failed infrastructure attempts. At low utilization, billed cost per
request will be higher. Only one image/schema was timed repeatedly; p95 and broad
workload performance are not established.

**The CPU-only change is supported by an exact-output control:** all option
probabilities across 216 questions matched the initial 512-limit run exactly.
Sequential image-feature reuse also had zero probability difference on each
image's first question at both resolutions.

**The batched fast path failed the equivalence check.** Ordinary batched image
encoding versus one-image feature reuse changed probabilities by up to 0.06149
at eight questions and 0.03121 at 24 questions. At 24 questions, the dog's
unlabelled `sharpness` answer changed from `yes` to `no`. Against sequential
scoring, the reused 24-question batch differed by up to 0.06218 and the same one
top answer. This was reproduced with both CPU/preprocessing configurations.
The cause has not been isolated; do not assume it is harmless BF16 noise.

The image-preprocessing optimization verified exact equality of the text IDs,
attention masks, modality IDs, grids and first-image pixels against normal
processing before timing. That establishes input equivalence, not batched-model
output equivalence. Later runs used an explicit `--diagnose-drift` flag to retain
failed checks and finish collecting timings; the default still fails closed.

Peak allocated memory was about 4.04 GiB for sequential scoring and 4.67 GiB for
the optimized 24-question batch. Allocator peaks exclude some driver/runtime
memory and are not a guarantee of compatibility with an 8 GB local card.

For context, the earlier diffusion animal-suite baseline cost $94.62/million,
while its quality-rejected concurrent 4090 candidate cost $26.11/million.
Those used different hardware/settings, image sweeps and HTTP serving boundaries.
The $48.21 result is therefore neither a matched 2x-win demonstration nor a
replacement recommendation. See [the diffusion experiments](COST_OPTIMIZATION.md).

## Diagnostic quality

All accuracy below is measured **before** optimizer updates.

| Fixture | Correct / labelled decisions, 512 limit | 768 limit |
| --- | ---: | ---: |
| Seven animal photos | 41/43 | 41/43 |
| Coffee and traffic-sign photos | 16/16 | 16/16 |
| Eight synthetic geometry images | 32/32 | 32/32 |
| Total | 89/91 | 89/91 |

The two disagreements with fixture labels were the horse's `multiple_animals`
answer and the bird's `coat_colour`. These are reused, partly AI-labelled
diagnostics, not independently audited ground truth. The synthetic cases test
simple colours, shapes, counting and booleans, not arbitrary instruction
following. No representative held-out quality claim follows from 89/91.

Reversing options for the first question on each image caused zero top-answer
flips in 17 checks; the largest probability change at the 512 limit was 0.01295.
This does not establish absence of position bias across all questions.
Natural images used 234–252 visual tokens at the 512² pixel cap and 396–567 at
the 768² cap. Synthetic images remained at 192 tokens under both caps.
The higher cap did not improve labelled accuracy here; this does not justify
reducing resolution for OCR or other detailed workloads.

The bundle includes per-suite log loss and sum-over-options Brier scores. These
tiny convenience fixtures cannot establish calibration or quality preservation.

## Training feasibility

The successful smoke test adapts **3,211,264 shared parameters** using rank-16
LoRA on language `q_proj`/`v_proj`, with the visual tower frozen. It uses one
synthetic example, eight optimizer steps at each of two sequence lengths, and
AdamW at 1e-4. Label smoothing of 0.1 prevents an already-correct, highly confident
answer from producing a useless zero-loss test.

| Input length, including image tokens | Mean warmed optimizer step | Peak allocated memory | First → last loss |
| --- | ---: | ---: | ---: |
| 250 tokens | 142 ms | 4.67 GiB | 1.297 → 0.401 |
| 760 tokens | 143 ms | 5.99 GiB | 0.372 → 0.350 |

Losses and gradients were finite. Adapter parameters changed (total absolute
change 2,778.69); the visual tower had no trainable parameters. The second length
continues the same disposable adapter. Its lower starting loss is not an
independent learning comparison. The original hard-label smoke test saturated at
zero loss and is retained in the bundle, but is not the basis for this conclusion.

These timings include forward/backward, gradient checks, optimizer work and GPU
synchronization, but exclude preparing each new input. They reuse a synthetic
image with 192 visual tokens. Longer/detailed real inputs, loading, validation,
checkpointing and sustained thermal behavior need measurement before committing
to a training schedule.

At 0.143 seconds, 40,000 presentations imply about **1.6 hours of optimizer work**.
Reserve roughly 3 hours for that stage until representative data batches confirm
the estimate. This is substantially below the proposal's unmeasured 6–12 hours.
Cloud BF16 memory figures make an 8 GB experiment plausible, but the RTX 2060
SUPER's actual FP16/quantized runtime, kernels, memory and speed were not tested.

## Recommended next stage and estimate

Proceed with a **quality/evaluation pilot**, not a production switch. Prepare
the public-data manifest and image/task-family splits locally first. Evaluate the
unadapted model before fitting adapters, then train the same generic shared
scorer and compare against it and diffusion on untouched images/questions.
Resolve the batched output drift before selecting its cheaper serving path.
The proposed non-inferiority criterion still needs agreement; it is not silently
replacing the original no-quality-sacrifice requirement.

A revised **$8 additional ceiling** is reasonable for the human-label-only pilot:

| Allocation | Allowance |
| --- | ---: |
| Setup, training, baseline/held-out evaluation: up to 6 GPU hours at ≤$0.75/hour | $4.50 |
| Downloads and result/checkpoint transfers | $1.00 |
| Contingency and teardown/billing lag | $2.50 |

Retain the proposal's 5,000 training images/~20,000 questions, up to two epochs,
500 validation images and 1,000 test images, subject to a reviewed data manifest.
This estimate excludes teacher-target generation and larger-model experiments.
Measure representative batches before the full run; if they do not fit this
allowance, stop and revise the estimate instead of shrinking evaluation silently.
**The $8 pilot is not approved.** Only the completed $2 feasibility stage was
authorized. No further rental remains active or scheduled.

## Reproduction and audit

- [Raw results, exact source snapshots, environment and commands](generic-feasibility-results.json.gz).
- [Sanitized run, spend and teardown record](generic-feasibility-run.json).
- [Sequential scorer and optimizer test](generic_feasibility.py).
- [Batching and single-preprocessing experiment](generic_batch_feasibility.py).
- [Diagnostic summary utility](analyze_generic_feasibility.py).
- [Deterministic synthetic fixture generator](make_feasibility_fixtures.py).

The bundle retains original and revised source versions because the optimizer
and batching diagnostics evolved after initial failures. It pins the checkpoint
and full installed package list. Reproduction requires a uv-managed environment,
the listed public photos/schemas and generated fixtures, plus `build-essential`
and Python development headers for Triton in the chosen runtime container.
No account credentials, private SSH keys or account balances are included.

Infrastructure history: the first host's long container pull was cancelled and
destruction confirmed before replacement. The replacement needed a startup
repair for provider-injected SSH file permissions, then compiler/header packages.
The scoped teardown timer was armed before rental and stopped only after final
absence was verified. Full results were retrieved before destruction.

Repository pre-commit checks passed for the benchmark scripts and fixture
generator. Synthetic fixtures regenerated byte-for-byte. Runtime validation
includes full input comparisons, sequential feature-reuse equality, finite
training updates, the CPU-only output control and explicitly retained batching
equivalence failures. These checks establish the reported feasibility, not broad
model quality.
