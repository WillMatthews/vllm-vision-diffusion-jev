# Reusing questions across images

The photo evaluation was completed, pushed, and its GPU destroyed before this
follow-up investigation. The follow-up has its own $5 rental budget and teardown.

## What can be reused

The question instructions are sent as the system message, before the user image.
The pinned checkpoint's chat template retains that order. DiffusionGemma uses
causal prompt encoding, so the state of those earlier question tokens does not
depend on a later image. The answer canvas uses a separate bidirectional denoising
pass over the encoded context. See [Google's architecture explanation](https://ai.google.dev/gemma/docs/diffusiongemma/explained)
and this fork's [implementation](../../../../vllm/model_executor/models/diffusion_gemma.py).

With `--enable-prefix-caching`, the server can reuse matching prompt KV blocks
from an earlier request. It does not need to rerun the text prefix's transformer
computation for every image while those blocks remain resident. Full blocks are
reused; the boundary and image-dependent suffix still need processing. Image
content participates in cache keys, so a different image cannot reuse the previous
image's KV suffix. [vLLM's cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
describes these rules.

The existing bootstrap already enables prefix caching. The adapter also caches
answer-template token positions on the CPU; that is distinct from the model's
GPU KV cache.

## Work that remains for every new image

- Image decoding, preprocessing, and vision encoding.
- Prompt encoding of the image-dependent suffix, including any text after the
  image. Image tokens still attend to the cached question context.
- Answer-canvas scoring, which still reads cached question and image context.
- Request handling and tokenization. The current API resends the schema; it has
  no registered `schema_id` API that avoids repeating that CPU/network work.

Caching therefore saves repeated question prefill. It does not turn this model
into a vision-only classifier, or make attention to a long question set free.
Caches can be evicted and are not automatically preserved across server restarts.
Different replicas must each acquire the relevant cache state.

Keep question text, option order, question order, and the chat template stable.
Put constant instructions before the image; place per-image metadata afterwards.
Changing an early part of the prompt reduces prefix reuse. A future schema
registration API could store fixed prompts and templates server-side, but that
would be an API optimization in addition to GPU prefix caching, not a replacement
for it. No such registration API was added here.

## Controlled experiment

Run only on a dedicated server: the benchmark repeatedly resets its caches.
The model and adapter listen on loopback, with developer cache endpoints enabled
only for this experiment. Add `--enable-prompt-tokens-details` to the model command
and enable `VLLM_SERVER_DEV_MODE=1` in its environment. Do not use these resets on
a server that is handling other work.

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/prefix_benchmark.py \
    --endpoint http://127.0.0.1:8011 --upstream http://127.0.0.1:8000 \
    --output prefix-results.json

.venv/bin/python examples/features/diffusion_reads/evaluation/prefix_benchmark.py \
    --endpoint http://127.0.0.1:8011 --upstream http://127.0.0.1:8000 \
    --counts 24 --chunk-prompt shared --output prefix-shared-results.json
```

The main experiment uses 1, 8, and 24 questions, seven animal images, and three
repetitions. For each image/count/repetition, three arms run in shuffled order:

1. **Empty:** clear prefix, multimodal processor, and encoder caches; request the
   target image.
2. **Different image:** clear all three caches; prime with the same question schema
   and a different animal photo; request the target image.
3. **Same image:** clear all three caches; prime with the target image and same
   schema; request the target image again.

Kernel warmups are excluded. Priming and reset calls are outside measured target
latency. The benchmark runs on the server itself, avoiding SSH uploads. It records
per-read prompt and cached-token counts, probabilities, and timing. Multi-read
requests may create reuse between their own reads even in the empty arm.

The adapter's new `diagnostics.upstream_usage` preserves each answer read's
upstream usage separately, including samples and chunks. It excludes any separate
optional thought-generation call. The older `usage.input_tokens` field is a
maximum chunk prompt length, not a sum of all reads or a count of uncached work.

## Sharing context across answer chunks

The default `chunk_prompt: own` sends only each chunk's questions in that chunk's
prompt. Each chunk has a different text prefix. Those fixed prefixes can each be
reused across images, but image-dependent language-model states differ between
chunks.

The existing `chunk_prompt: shared` mode sends the full question list for every
chunk, permitting the image-conditioned prompt to be reused across chunks too.
It still scores every answer chunk. Longer prompts and changed question context
can affect timing and scores, so this is measured separately rather than assumed
to be better. These tests use `max_num_seqs=1`; concurrent requests can have a
different cache-sharing pattern.

## If the requirement is literally text encoding once

A dual-encoder model such as SigLIP2 offers separate text and image encoders.
Precompute a matrix of text embeddings for descriptions such as “a photo of a
dog” and “a photo of a cat”. For each new image, run its image encoder once and
compare that vector with the stored matrix. The text encoder need not run again
until the descriptions or model change. [SigLIP2's API](https://huggingface.co/docs/transformers/main/model_doc/siglip2)
exposes separate text/image feature methods.

That is a different model and scoring objective. It produces image/text matching
scores, not the same question-answer probabilities as this adapter. Fixed species
classification is a sensible task on which to compare it; collar detection,
negation, relationships, and text-reading questions need their own labelled
validation. A sigmoid or softmax alone does not establish calibration. We did not
benchmark or deploy a dual-encoder model in this experiment.

## Recorded findings

The results and timing tables are in the [main README](../../../../README.md#can-the-prompts-be-processed-once-for-many-images).
Across 441 measured target requests, the species answer was correct every time
on these seven repeated photos. Attribute scores were less stable, including
variation between fixed-seed repeats within one cache condition. Cache-condition
comparisons therefore do not isolate caching as the cause of every difference.

The initial server selected `fp8_e4m3` KV storage from the checkpoint. We repeated
the main experiment after restarting with `--kv-cache-dtype bfloat16`; startup
logs confirmed the change. The NVFP4 weights and `FLASHINFER_CUTLASS` MoE backend
were unchanged. BF16 KV storage did not eliminate the score variation and is not
presented as a fix. The existing bootstrap defaults have been retained.

To reproduce that follow-up, restart the model command with the BF16 cache flag,
then run the same benchmark with a distinct output filename:

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/prefix_benchmark.py \
    --output prefix-bf16-results.json
.venv/bin/python examples/features/diffusion_reads/evaluation/analyze_prefix.py
```

The analysis script reads the three recorded result files from this directory.
If rerunning, place the outputs here before invoking it. It pairs cache conditions
by image, question count, and repetition, and separately compares each condition's
later repetitions against repetition zero. Top-answer changes include unlabelled
attributes and are not equivalent to classification errors.

With `chunk_prompt: own`, the different-image primer reused 288, 672, and 1,120
reported tokens for 1, 8, and 24 questions respectively. The 24-question figure is
summed across three reads. Timing was essentially unchanged for new images.
With `chunk_prompt: shared`, the different-image primer added no reported cache
hits beyond the 2,624 tokens already reused between chunks in the empty arm.
The model has 1,024-token sliding-window layers, but this experiment did not
isolate the exact reason for that longer-prompt cache behavior.

The follow-up instance was destroyed, and the final account listing was empty.
[Run provenance and billing snapshot](prefix-run.json) records the separate
budget and teardown. No prompt registration service, persistent cache store,
dual-encoder deployment, or production calibration was implemented.
