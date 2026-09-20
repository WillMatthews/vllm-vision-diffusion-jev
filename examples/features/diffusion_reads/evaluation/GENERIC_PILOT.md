# Generic image-to-JEV training pilot

2026-09-20, branch `cost-optimization-diffusion`. **Completed for $0.88 of the
approved $10; the rental was destroyed and zero account instances confirmed.**

The shared student improves measured quality, but **does not meet the 2× inference
cost-reduction target**. Four-question requests take about 175 ms for the trained
student and 180 ms for DiffusionGemma. The apparent 2.7% saving uses unequal
HTTP overhead and is not convincing evidence of cheaper serving.

## Completed student experiment

One shared Qwen3-VL-2B-Instruct scorer accepts new question text and options. It
does not train a separate classifier for each question. The vision tower stays
frozen; rank-16 LoRA adapts the language model's query/value projections
(3,211,264 trainable parameters). Images are capped at 512² pixels. Inference
encodes the image once, then scores each question sequentially using its complete
cached visual features. Language computation still occurs per question.

The frozen VQA-v2/COCO subset contains 5,000 training images/20,000 questions,
500 validation images/2,000 questions and 1,000 test images/4,000 questions.
Training used human-vote targets, two epochs, shuffled option order, learning
rate 1e-4, gradient accumulation 8 and seed 42. No teacher targets or test-based
checkpoint selection were used. See [dataset construction and limits](PILOT_DATA.md).

Training completed 40,000 presentations in **4,655 seconds (77.6 minutes)**,
with **5.03 GiB peak allocated GPU memory**. The final adapter and all checkpoint
and evaluation artifacts have been downloaded locally.

| Test metric | DiffusionGemma | Untrained Qwen | Trained Qwen |
| --- | ---: | ---: | ---: |
| Plurality-answer accuracy | 83.18% | 86.63% | 90.10% |
| Raw Brier score, lower is better | 0.2228 | 0.1837 | 0.1275 |
| Raw soft-target log loss, lower is better | 0.6191 | 0.6439 | 0.3699 |
| Validation-calibrated Brier | 0.2097 | 0.1577 | 0.1204 |
| Validation-calibrated log loss | 0.5317 | 0.4249 | 0.3468 |
| Reserved spatial-family accuracy, 257 questions | 80.16% | 86.77% | 89.49% |
| Top-answer flips on 100 reversed-option checks | Not measured | 19 | 1 |

The paired image-cluster bootstrap 95% interval for the accuracy gain is
**+2.45 to +4.40 percentage points** (1,000 bootstrap samples). Every reported
family improves in point-estimate accuracy versus both references. Versus
DiffusionGemma, the gain is **+6.93 points**, with 95% interval **+5.65 to +8.30**.
Calibrated Brier and log-loss differences also favor the student: intervals
[-0.1040, -0.0754] and [-0.2101, -0.1626], respectively. These are overall
image-cluster intervals; family-specific acceptance criteria were not approved.

| Test family | Questions | DiffusionGemma | Base | Trained |
| --- | ---: | ---: | ---: | ---: |
| Boolean | 2,606 | 85.30% | 86.38% | 89.87% |
| Colour | 334 | 84.73% | 92.22% | 96.41% |
| Count | 639 | 71.21% | 82.00% | 85.92% |
| Other fixed categories | 164 | 97.56% | 96.95% | 98.17% |
| Reserved spatial | 257 | 80.16% | 86.77% | 89.49% |

The teacher used NVFP4 on the same 4090 with Marlin, eager execution, canvas 64,
max sequences 1, TRITON attention, prefix caching, samples 1 and think 0. This is
a newly measured reference, not the earlier optimized concurrent 5090/4090
benchmark. Tasks, options and human labels match, but each architecture uses its
own prompts and image processing. It is not a controlled same-backbone ablation.

One temperature per model was fitted using validation labels only: 2.431716
for base, 1.272026 for adapted and 1.610779 for DiffusionGemma. Calibration optimizes soft-target log loss,
not every metric: the adapted model's ordinal expected-count error increases
from 0.266 raw to 0.400 calibrated. Generic CLI output remains raw normalized
probabilities unless separately calibrated; neither normalization nor these
limited calibration results establish confidence on arbitrary new domains.

Visual reuse produced exactly matching probabilities on the first checked
question of each of 1,500 held-out images, separately for both models. This is
one check per image, not an exhaustive equivalence proof over all inputs.

## Inference timing

The student benchmark used 100 fixed validation images, four questions each,
two repetitions and three untimed warmups. Excluding the first three teacher
requests from both sides leaves **99 matched images** (198 student measurements
and 99 teacher measurements):

| Model | Mean complete request | Requests/s | Occupied cost per million |
| --- | ---: | ---: | ---: |
| Untrained Qwen | 150.6 ms | 6.64 | $19.18 |
| Trained Qwen, unmerged adapter | 175.4 ms | 5.70 | $22.35 |
| DiffusionGemma, localhost HTTP | 180.3 ms | 5.55 | $22.96 |

Costs use $0.458519/hour including allocated disk. Timing includes fresh image
open/decode, preprocessing, one vision encoding, four sequential question scores
and CPU probability vectors. It excludes model load, schema validation, result
file writes, HTTP transport, idle capacity and the separate reuse verification.
Filesystem caches were not flushed. These are occupied-machine estimates, not
an all-in serving bill. Teacher HTTP timing additionally includes transport
and server overhead. Its internal adapter timer averages 177.0 ms on these
images, reducing the apparent gap to under 1%. Native visual-token budgets
differ; this is a practical task comparison, not equal kernel work. Teacher
timings were not repeated, so host/run variability is not fully characterized.

Training compute/disk alone cost approximately $0.593. Dividing it by the tiny
apparent HTTP saving gives an illustrative 0.96 million-request break-even,
but unequal overhead and missing teacher repeats make that estimate unreliable.
There is **no established economic break-even** from this pilot.

The likely bottleneck is repeated language-model work for each question, with
unmerged LoRA also adding overhead. A next experiment should measure cached
preprocessing and merged-adapter inference with probability/quality checks,
then compare identical serving boundaries and larger schemas. More training
alone does not address the measured cost problem. These follow-ups were not run.

## Machine selection

The first candidate passed the [hardware and workload screen](../vast/README.md)
without a retry. See [measured screening evidence](../vast/machine-screening-result.json).
The controller limits screening to three physical hosts and $1.50 within the
$10 total, verifies destruction before retrying and has detached cleanup.
Observed account credit decreased by **$0.87932** for this stage, including
setup, screening, training, evaluation and transfers. A follow-up API check
confirmed zero scoped instances and zero total account instances; the cleanup
timer is inactive after deletion. Earlier experiments are separate budgets.

## Scope

This is a bounded research pilot. Boolean questions dominate the dataset;
answer vocabularies and consensus filtering make it narrower than arbitrary
user tasks. Counts provide ordinal supervision, not subjective rubrics. The
reserved spatial family is defined by question-wording heuristics, and the
changed test wrapper measures wording transfer. Foundation-model exposure to
VQA/COCO is unknown. Exact/dHash duplicate screening cannot exclude every crop
or semantic duplicate. Conditional `askif`, OCR, video and production-specific
rubrics are not validated. The image corpus is not commercially cleared.

No deployment defaults have been changed and no universal quality-preserving
replacement is claimed. The proposed non-inferiority margins remain descriptive,
not a user-approved acceptance policy.

## Artifacts and reproduction

- [Full metrics, calibration, paired intervals and costs](generic-pilot-summary.json)
- [Raw predictions, teacher request diagnostics and inference timings](generic-pilot-results.json.gz)
- [Pinned revisions, configuration, artifact hashes and teardown](generic-pilot-run.json)
- [Frozen dataset manifest and split checksums](generic-pilot-data-manifest.json)
- [Training/evaluation runner](train_generic_pilot.py), [generic inference CLI](predict_generic_pilot.py), [offline analyzer](analyze_generic_pilot.py)

Local retained artifacts (gitignored):

- Final adapter archive: `.vast-run/generic-pilot-run/attempt-1/final-adapter.tar.gz`
- Ready adapter directory: `.vast-run/generic-pilot-run/attempt-1/final-adapter/training/checkpoint-040000/adapter`
- Full student checkpoints: `.vast-run/generic-pilot-run/attempt-1/student-results/training`
- Frozen split JSONL and image provenance: `.vast-run/generic-pilot-prep/final`
- Prepared public images: `.vast-run/generic-pilot-prep/image-cache`

The generic CLI takes an image and a fresh schema with `questions` and optional
`state`, without training targets. It supports independent `choice`, `noul` and
`score` questions and rejects unsupported dependencies. Use the pinned GPU
software environment recorded by the job driver; BF16 inference was measured
on the 4090, not the local RTX 2060. For example:

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/predict_generic_pilot.py \
  --model /path/to/pinned/Qwen3-VL-2B-Instruct \
  --adapter .vast-run/generic-pilot-run/attempt-1/final-adapter/training/checkpoint-040000/adapter \
  --image /path/to/image.jpg --schema /path/to/questions.json \
  --output .vast-run/prediction.json --verify-reuse
```

Recompute the full statistical analysis from the downloaded results:

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/analyze_generic_pilot.py \
  --student-dir .vast-run/generic-pilot-run/attempt-1/student-results \
  --teacher-dir .vast-run/generic-pilot-run/attempt-1/teacher/extracted/results \
  --output .vast-run/recomputed-pilot-analysis.json
```

The compressed public results bundle also contains the six complete evaluation
objects under `base_validation`, `base_test`, `adapted_validation`,
`adapted_test`, `teacher_validation` and `teacher_test`; these can be restored
as the corresponding `evaluation.json` files without model execution.

Validation: 35 focused CPU tests, seven fake-API lifecycle tests, repository
pre-commit checks, the actual hardware/workload screen, complete GPU training
and six split/model evaluations, and generic-input GPU checks all passed.
No deployment defaults changed, and no extra rental was made to spend the
remaining budget.
