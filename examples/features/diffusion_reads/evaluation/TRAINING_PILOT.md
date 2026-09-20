# Generic image-to-decision training proposal

Revision 4 — 2026-09-20. Replaces the fixed animal-predictor proposal and its $5
estimate. The approved **$2 feasibility stage is complete**, with $0.25 observed
spending and all rentals destroyed. The [measured feasibility report](GENERIC_FEASIBILITY.md)
proposed an $8 additional human-label-only pilot. The user has now approved that
experiment with a**$10 additional cap**, including machine screening/retries.
It supersedes the original unmeasured $20 estimate below. The user has explicitly
requested new question schemas at inference time. The [completed pilot](GENERIC_PILOT.md) used $0.88; all rentals are destroyed.
Measured quality improved, but the 2× inference-cost target was not met.
The design below is retained as the pre-run proposal; actual coverage and results
are recorded in that report.

Machine selection uses the [bounded screening loop](../vast/README.md): one
RTX 4090, maximum $0.75/hour including disk, at most three candidate hosts and
$1.50 screening within the $10 total. Public data is prepared and audited locally
before rental; accepted hosts proceed directly into the paid workload.

## Required behavior

One trained model accepts an image, optional text/JSON context, and a schema of
new questions, instructions and answer options. It returns JEV-shaped choice,
boolean and ordinal-score distributions. New question IDs, wording and options
must not require a new head or retraining. The animal schema is one evaluation
case, not the architecture's set of permitted outputs.

The first pilot targets single images and English schemas. It will exercise
2–16 options per question and batches of 1, 8 and 24 questions. These are measured
pilot limits, not evidence of unrestricted language, document, video or question
coverage. Requests outside supported limits must receive an explicit error;
never silently truncate options or substitute an animal classifier.

## Proposed model and inference path

Start from an existing small instruction-tuned vision-language model, with
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) as the first
candidate. The publisher provides image-and-text inference and open weights.
Its performance on our typed probability task is unmeasured. Pin model and
library revisions before experiments; run in a separate uv-managed environment.

Reuse the pretrained model's visual and language understanding. Adapt its shared
language layers with low-rank adapters, initially keeping the vision tower
frozen. There is one shared scorer, not one classifier per question or domain.
This is an architecture change from the current diffusion model, expressly
motivated by the user's newly clarified generic-training requirement.

For each question, the input contains its full instructions and option text.
Map options to verified single-token answer labels and read the logits for those
labels at the answer position, rather than generating a prose answer. Normalize
within the supplied option set, map back to user keys, and train directly on that
distribution. Shuffle option order during training and test position bias. Verify
tokenization in the exact answer context; do not silently drop multi-token labels.
Boolean questions use two options; ordinal questions use ordered levels and return
both the distribution and its expected value. An abstention flag is separate from
the user's option probabilities. Normalization alone is not calibration.

Encode the image once and reuse the complete visual features required by the
backbone across that request's questions, where implementation permits. Validate
reused-feature outputs against ordinary inference before relying on this path.
The language model can still process image tokens and question text per question;
we are NOT promising one shared language forward pass for all questions. Benchmark
complete 24-question requests, including preprocessing and all question scoring.
Dependency/ask-if behavior must remain explicit in the adapter and be tested
separately; initial independent-question timings do not establish its cost.

A custom tiny cross-attention scorer over cached image/text features remains a
possible later architecture. Training that from scratch to follow arbitrary
instructions is higher risk; it is not what this pilot's budget promises.

## Data and training

Provisional training target: 5,000 images with approximately 20,000 varied typed
questions, up to two training epochs. Add 500 validation/calibration images and
1,000 untouched test images, with actual question counts and coverage fixed in a
manifest before training. These are feasibility-scale targets, not enough to
claim universal generalization.

If no representative user dataset is supplied, assemble a bounded public-data
pilot from human-annotated visual QA sources such as
[VQA v2](https://visualqa.org/download.html) and
[GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), after checking the
selected assets' reuse terms. Fetch only the required image subset within local
disk limits. These sources are candidates, not a dataset already assembled or
validated. Include objects, attributes, counts, spatial relationships and varied
scenes; do not use only animal photographs. Specialized OCR, subjective ordinal
rubrics and production-specific tasks need separately labelled coverage before
claiming quality for them.

Convert questions into typed schemas using human answers, meaningful distractors
and shuffled options. Audit ambiguous alternatives, synonyms and missing correct
answers so schema construction does not manufacture easy or incorrect labels.
Retain multiple human answers when available. No answer information may leak
into the supplied context or instructions.

Split by image and near-duplicate/source groups, then reserve question families
and schema templates absent from adapter training. Test both new images with
familiar task types and new task types/templates. A wording paraphrase alone is
not an unseen task. These splits establish held-out adaptation data; they cannot
prove exclusion from the pretrained model's original corpus.

Use supervised probability losses first: human-answer cross-entropy, optionally
combined with a teacher-distribution loss. We do not need to assume that RL is
required. If useful, generate DiffusionGemma targets for training images with two
canvas seeds, grouping compatible questions into requests. Teacher disagreement
is diagnostic; teacher confidence is not ground truth. Human labels and untouched
test results take precedence. Do not distill the teacher's mistakes uncritically.

Fit calibration only on the validation split. Keep test labels out of training,
model selection, distractor tuning and calibration. The old nine-photo tuning set
and opened pigeon holdout remain historical diagnostics, not the new acceptance
set.

## Local versus cloud work

Verified local card: RTX 2060 SUPER, 8 GB VRAM (about 7 GB free at inspection).
Use it first for adapter development, untrained-model scoring, memory checks and
small training batches. Quantized inference or adapter training may fit, but
8 GB is not a promise that the full pilot fits. Check Turing-compatible kernels
and FP16/quantization support; do not assume native BF16 support. Do not lower
image detail merely to fit memory without measuring its quality effect.

Before cloud training, measure a representative batch's peak memory, forward and
backward throughput, and preprocessing cost. Extrapolate total work with the
actual image-token and text-length distribution. The previous diffusion inference
rates do not predict this student's training speed.

Keep teacher and student stages sequential on a rented GPU rather than assuming
both large models fit simultaneously. Save teacher outputs and checkpoints before
switching stages. Use local preparation time without leaving a cloud GPU idle.

## Original pre-measurement budget (superseded)

Retained for comparison with the measured feasibility result. Use the narrower
$8 estimate in [the feasibility report](GENERIC_FEASIBILITY.md) as the basis for
the approved $10 cap; its scope excludes teacher-target generation.

Live read-only quote at 2026-09-20 00:27:10 UTC: suitable single RTX 4090 offers
were $0.4811–$0.4904/hour with 100 GB allocated storage. Offers are transient;
private quote metadata is in `.vast-run/generic-training-price-check.json`.
[Vast pricing](https://docs.vast.ai/guides/instances/pricing) includes separate
bandwidth charges. Plan conservatively at a maximum $0.75/hour including disk.

| Stage | Planned paid time | Compute/disk ceiling |
| --- | ---: | ---: |
| Feasibility benchmark, environment setup and initial scoring | 1–2 hours | $1.50 |
| Optional teacher targets for the training subset | 1–3 hours | $2.25 |
| Adapter training, approximately 40,000 question presentations | 6–12 hours | $9.00 |
| Calibration, held-out comparisons, throughput and exports | 1–3 hours | $2.25 |
| Total | 9–20 hours | $15.00 |

Training time assumes roughly 1–2 question examples/second, giving 5.6–11.1 hours
for 40,000 presentations, rounded upward in the table. Teacher time assumes
roughly 1–3 grouped image requests/second for 10,000 requests, inferred from earlier
runs but unverified on these new prompts. Neither assumption is a benchmark.
All paid setup/warmup time is included in the stage allowances. Public human
labels may allow skipping the teacher stage entirely.

Planning estimate: **approximately $8–$17** for the pilot at conservative rates
and transfer allowances. Proposed ceiling: **$20 additional spending**, separate
from the earlier $6 budget. Its allocation is at most $15 compute/storage,
$2 bandwidth and $3 contingency for teardown, billing lag and small overruns.
Confirm actual transfer sizes/rates and remaining budget before rental. No paid
annotation, commercial dataset fees, external model APIs, or unrelated work are
included or authorized by this proposal.

Spend no more than **$2 of that ceiling on initial cloud feasibility** before
checking that measured speed and memory support the remaining plan. If they do
not, stop and produce a revised estimate; do not silently reduce the declared
training/evaluation workload to claim completion. If the small model fails basic
unseen-question tests, stop rather than spending the remainder on longer training.
Alternatively, the user can approve only this $2 feasibility stage first.

For each rental, arm scoped teardown before creation, monitor cumulative spending,
reserve funds for transfers and deletion lag, retrieve artifacts, destroy the
instance immediately after use, and confirm absence. A timer is a safeguard, not
a provider-enforced cap. Unused approval is not a spending target. More training,
a larger student or a second architecture needs a separate estimate if it will
exceed this ceiling.

## Evaluation and decision gates

Compare three systems on identical held-out images, schemas and option sets:
current DiffusionGemma, the unadapted small VLM, and the adapted student. Report
per-family and per-primitive accuracy, Brier/log loss, calibration, ordinal error,
option-order sensitivity, abstention coverage, latency, throughput and cost.
Include no-change repeat controls and group statistical resampling by image.
Measure matched concurrency and full request work; do not compare one student
question with 24 teacher questions or exclude repeated language computation.

Proposed statistical gate for discussion: a paired 95% confidence bound excludes
more than a one-percentage-point overall accuracy loss, a 0.01 absolute Brier
increase, and a 0.02 log-loss increase. Report each task family separately so
aggregate gains cannot conceal important failures; determine critical-family
criteria from the intended workload before training. This is a proposed
non-inferiority policy, not a previously approved relaxation of the old gate.
If the available test set cannot establish it, report the result as inconclusive.
The user has not yet chosen between statistical quality preservation and identical
outputs; do not claim the overall quality-preserving objective achieved without
an agreed acceptance criterion.

For the cost objective, target at least 2x lower occupied inference cost per full
schema request at matched service conditions, also reporting local latency and
peak memory. This is a go/no-go target, not a forecast. Include training amortization:
training dollars divided by measured saving per request gives the break-even
request count. Report a negative result if the smaller model is less accurate or
not cheaper after all per-question work is included.

## Deliverables and approval boundary

Deliver a pinned student checkpoint plus adapters, generic typed-schema scoring
adapter, dataset/split manifests, reproducible training commands, unseen-question
quality/cost report, and spend/teardown records. This $10 pilot buys evidence of
feasibility, not guaranteed Jev-level generality or a production-ready universal
predictor.

Approval recorded: the earlier **$2 feasibility is completed**, and the user
subsequently approved the human-label-only pilot with **$10 additional spending**.
This includes bounded machine screening, training and held-out comparisons.
It excludes teacher-target generation, paid annotations and larger architectures.
