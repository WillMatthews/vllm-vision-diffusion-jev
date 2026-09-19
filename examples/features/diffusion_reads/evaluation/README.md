# Real-photo evaluation

This suite uses seven animal photos and two general visual-question examples.
The animal class names come from `wm_animals/animal_taxonomy.json`; a snapshot is
included as [animal_taxonomy.json](animal_taxonomy.json). That source defines
labels, not question wording. The natural-language prompts in
[animal_schema.json](animal_schema.json) were written for this experiment.

The animal schema contains 24 questions: species, coat colour, coat pattern,
posture, view, eleven context flags, five quality flags, and three scene questions.
Every species question offers all 24 taxonomy classes. The general schema asks
eight questions about a coffee photo and a hand-symbol stop sign.

## Run it

Install the client dependency with `uv pip install pybase64`. Start this fork's
model server and adapter using the [GPU runbook](../vast/README.md), then run:

```bash
.venv/bin/python examples/features/diffusion_reads/visual_client.py \
    --schema examples/features/diffusion_reads/evaluation/animal_schema.json \
    --manifest examples/features/diffusion_reads/evaluation/animals.jsonl \
    --endpoint http://127.0.0.1:8011 > animals-results.json

.venv/bin/python examples/features/diffusion_reads/visual_client.py \
    --schema examples/features/diffusion_reads/evaluation/generic_schema.json \
    --manifest examples/features/diffusion_reads/evaluation/generic.jsonl \
    --endpoint http://127.0.0.1:8011 > generic-results.json

.venv/bin/python examples/features/diffusion_reads/evaluation/benchmark.py \
    --endpoint http://127.0.0.1:8011 --hourly-usd 0.8277777778 \
    --repeats 5 --output benchmark-results.json
```

Use your own current rental rate for `--hourly-usd`. The benchmark performs one
warmup per photo and question count, then five measured requests for each pair.
It uses the dog and cat, question counts 1, 2, 4, 8, 16, and 24, concurrency one,
and a fixed shuffled order. The first N questions in the schema are used.
The raw probabilities, HTTP latency, adapter timing, read counts, and model
usage diagnostics are retained for every measured request.

## Interpretation

- Expected labels were assigned by visual inspection before inference. Only
  selected, reasonably clear animal attributes are scored; all 24 questions are
  still returned. Horse headgear and the partially hidden fox body are not given
  speculative ground-truth context labels.
- These are convenience examples, not a representative test set. Internet photos
  may have appeared in model training. The pooled accuracy mixes tasks with very
  different difficulty and does not estimate deployment accuracy or calibration.
- Species is single-choice even when an application may require multiple animal
  detections. There are no boxes, instance masks, or per-animal assignments here.
- Scores are normalized within the supplied options. Label mass reports how much
  original vocabulary probability those options collectively received. A score
  near one is not a guarantee that the answer is correct.
- The benchmark repeats images and prompts with the existing server cache enabled.
  It measures warm sequential requests, not unseen-image throughput. No cache
  optimization is introduced in this experiment. HTTP timing includes transport;
  adapter timing excludes the client's network trip but is not a CUDA kernel timer.
- Cost estimates multiply measured elapsed time by the hourly rental price. They
  exclude startup, model download, idle time, and bandwidth charges. The provider
  bills the whole rental lifetime, not individual questions or model tokens.

Photo credits and license links are in [ATTRIBUTION.md](ATTRIBUTION.md).
[sources.json](sources.json) includes download URLs and SHA-256 checksums.

## How answers are scored

Each question owns a separate answer slot and a separate normalization. A
24-question request returns 24 distributions, not one distribution over the
questions. Multiple yes/no attributes may all be true at once.

For a choice question, the adapter maps the supplied options to answer tokens
(such as A, B, C), reads their log probabilities at that question's slot, and
normalizes over those allowed options. A boolean question similarly normalizes
its yes/no options. The client returns named distributions and preserves the
underlying diagnostics. No free-form prose parsing or top-answer extraction is
needed to get these distributions; taking the maximum is only a display or
decision policy.

A species choice is therefore a single-label classification. To ask whether an
image contains both dogs and cats, define two boolean questions instead. This
still does not locate or count individual animals. Questions share an image and
prompt, so their scores should not be treated as statistically independent.

## Additional comparisons

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/compare.py \
    --endpoint http://127.0.0.1:8011 --output comparison-results.json
```

This tests species alone, species alone with reversed option order, and all 24
questions with four samples. Compare with the original one-sample animal run.
These are small diagnostic checks, not statistical evidence that a setting is
better. Changing option order also changes the letter assigned to each label.

For a direct grouped-versus-separate comparison:

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/separate.py \
    --endpoint http://127.0.0.1:8011 --output separate-results.json
```

This sends each of the same 24 questions separately for the dog and cat. It is
one sweep per photo, after the main benchmark, so cache histories differ and
there are no meaningful tail-latency statistics for this comparison. Each
separate request uploads the photograph again.

Named question IDs require a delimiter before their answer token. This fork
keeps `id: label` formatting for named questions even above ten questions;
otherwise strings such as `has_collarno` can violate the one-token answer-slot
contract. Compact indexed formatting is retained for large schemas whose IDs
are ASCII decimal numbers. Named questions can consume more canvas rows and
therefore require more chunks. The actual read count is in each response.
