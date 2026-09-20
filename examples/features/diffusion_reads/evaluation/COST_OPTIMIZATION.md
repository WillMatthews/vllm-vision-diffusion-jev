# Diffusion inference cost experiments

Status: screening and isolated prefill follow-up complete on
`cost-optimization-diffusion`. No
quality-preserving lower-cost serving preset established; serving defaults remain
unchanged. All three rented instances were destroyed and the account had zero instances
at the latest check. Cumulative observed credit decrease: **$2.54 / $6 budget**
(provider billing may post with a delay).

## Final result

Batching, CUDA graphs and a cheaper GPU reduced occupied inference cost by
**72.5%**, from **$83.86 to $23.08 per million images**, but the cheapest finalist
failed the quality guardrail. This is an experimental option, not a recommended
replacement for the existing default.

| Final configuration | Images/s | $/million images | p95 request ms | Tuning accuracy | Brier ↓ | Log loss ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5090, original eager/sequential baseline | 2.944 | 83.86 | 417 | 95.08% | 0.0688 | 0.1361 |
| RTX 5090, graphs, concurrency 9, prefill fix | 8.971 | 27.52 | 1009 | 94.07% | 0.0722 | 0.1355 |
| RTX 4090, graphs, concurrency 9, Marlin, prefill fix | 5.901 | 23.08 | 1579 | 95.08% | 0.0612 | 0.1116 |

These are two seeds × five sweeps × nine tuning images per configuration: 90
image observations and 590 scored decisions, but only **nine distinct photos**.
Each request retains every question and full distributions (24 animal questions,
eight general questions). The separate animal-only cost was $94.62, $30.18 and
$26.11/million respectively. Its accuracy was 93.49%, 91.16% and 91.16%.
Cost uses occupied sweep wall time, includes the quoted storage rate, and excludes
startup, compilation, idle time, reset overhead and network charges. Real billing
requires sufficient sustained concurrent demand; latency increases substantially.
These are observed rental offers, not permanent GPU prices or a global minimum.

The finalists were frozen before opening predictions on one separately annotated
holdout photo, then tested five times in mixed ten-image batches:

| Configuration | Correct holdout decisions / 95 | Brier ↓ | Log loss ↓ | Paired regressions |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 92 | 0.0500 | 0.0881 | — |
| 5090 finalist | 89 | 0.0904 | 0.1441 | 4 |
| 4090 finalist | 89 | 0.0610 | 0.0952 | 3 |

The 4090 answered the pigeon feather-colour question `black` instead of the
schema's `other` in all five repetitions, versus two baseline repetitions.
There were also seven paired tuning regressions on the 4090 and twelve on the
5090 (paired within seed). Matching aggregate accuracy therefore does not imply
preserved behaviour. Labels are AI annotations, not human-audited ground truth;
95 holdout decisions are 19 labels repeated on **one photo**, not 95 independent
examples. No quality-equivalence or statistical significance claim is justified.
Baseline itself varies across repetitions. The observed differences could include
batching/numerical variation; these experiments do not isolate every cause.
No further tuning was done after unblinding the holdout.

## Isolated prefill follow-up

[Raw follow-up results and analysis](cost-prefill-followup.json.gz) contain eight
additional benchmark runs: original/patched/patched/original blocks on a fresh
5090, with all serving settings fixed to eager execution and concurrency one.
Each block uses two seeds and five sweeps per seed; no holdout predictions were
used in this follow-up. This adds 360 image observations, still only nine photos.

| Code | Occupied $/million | Accuracy | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Original, two blocks | 83.41 | 94.07% | 0.0761 | 0.1336 |
| Prefill patch, two blocks | 83.18 | 94.58% | 0.0716 | 0.1260 |

The apparent **0.28% saving is within run variation**: individual blocks cost
$83.57, $84.29, $82.06 and $83.26/million respectively. This does not establish
a robust end-to-end cost improvement.

Three additional GPU tests confirm exact equality of all sampler-state tensors,
draft tokens, sampled outputs and GPU RNG state when replacing NaN-filled old
prefill logits with zero rows. They cover unfinished, mixed-completion and fully
completed prompt chunks. This verifies the prefill sampler contract; it is not a
proof of broad image quality equivalence.

Same-code controls also change outputs: original A1→A2 had 64 changed top answers
across 1,840 question distributions and eight labelled regressions; patched
B1→B2 had 58 and nine respectively. Original→patched comparisons had 64/61 answer
changes and seven/five labelled regressions. These are repeated measurements on
the same small fixture, not independent test examples. They establish that
individual regressions cannot all be attributed to code changes, but they do not
prove quality preservation or justify relaxing the predeclared gate.

The extra rental was destroyed at 2026-09-19 23:52:55 UTC; the account listing
confirmed zero instances. Cumulative observed spend is $2.52 (prior posting lag
raised the first phase's $2.22 reading to $2.30 before this rental). The timer was
stopped after destruction was confirmed. The primary bundle retains its original
phase metadata; the follow-up bundle and updated run metadata record this phase.

The requested [Laya assessment](LAYA_ASSESSMENT.md) is separate research: released
Laya checkpoints are text-only and supply no verified image-inference speedup.

## Upstream updates and retained changes

The updated [structured-read PR #57250](https://github.com/vllm-project/vllm/pull/57250)
contains no new throughput optimization relative to our stack: 11/14 commits are
patch-identical and the scheduler improvement is already present. Wholesale
replacement would lose our image initializer and named-question delimiter fixes.
The image initializer also relates to [#57589](https://github.com/vllm-project/vllm/pull/57589).

Backported Matt Mastracci's separate [prefill patch #57416](https://github.com/vllm-project/vllm/pull/57416)
from head `646ad6ed8ea1a8812bbf206acefbb89b26ab0ad1`. It skips unused vocabulary
projection during diffusion-only prefill while preserving decode rows. Three
focused model-contract tests passed in the GPU environment. An initial controlled
5090 comparison improved sequential throughput from 4.785 to 4.975 images/s,
but batching changed from 8.900 to 8.843 images/s; subsequent repetitions varied.
This supports removing wasted work, not a robust large end-to-end speedup claim.
The final candidate changes multiple settings; its quality differences cannot be
attributed specifically to this patch.

Retained the benchmark, analysis, holdout, tests, and two bootstrap fixes: prefer
the host CUDA driver and install the pinned official FlashInfer JIT cache wheel.
Reverted sampler and vision-encoder compilation experiments. Serving preset
values remain unchanged. Source attribution and reversible experimental patches
are preserved inside the results bundle.

## Reproduction and artifacts

Model: `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, revision
`ffc65bb9103ef37ba010cdfe259c2cde5401c579`.
Base code: `0323d37edf54cceebdd2cf3203e3d50f10f3efe4`.
Use the [Vast bootstrap instructions](../vast/README.md) for the pinned environment.
The results bundle records exact server and adapter command arrays, input hashes,
source hashes, GPU/driver and hourly rate for each final run.

All final runs use named IDs, original sampler, full 280 image tokens, server and
adapter canvas 64, adapter canvas step 16, and TRITON_ATTN. Final baseline uses
original runner, eager mode, max sequences 1 and memory fraction 0.9. Final 5090
uses the prefill patch, graphs, max sequences 8, memory 0.8, batched tokens 4096,
image-only admission and automatic FlashInfer MoE backend. Final 4090 uses memory
0.94, batched tokens 2048 and Marlin; other candidate settings match. The raw
`settings.server_config.cmd` and `settings.adapter_config.cmd` arrays are the
source of truth for launch flags; model path was `/workspace/model`.

With server and adapter running, reproduce a finalist sweep with:

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/cost_benchmark.py \
  --layout named --concurrency 9 --repeats 5 --warmup-sweeps 2 --seed 42 \
  --hourly-usd 0.4903703703703704 --tag confirm-4090-seed42 \
  --server-config /workspace/server-config.json --output /tmp/confirm-4090-seed42.json
```

Repeat with seed 43; use concurrency 1 and the 5090 hourly rate for baseline.
`--suites animals` selects the seven-photo animal workload;
`--suites animals generic holdout` selects the frozen mixed holdout evaluation.
The server-config file is optional metadata containing the launch command and GPU
information; it does not configure the server. Final warmup clears all three
caches before each of two shuffled sweeps at measured concurrency; each timed
sweep also clears all three caches. Cache reset/warmup time is excluded from cost.

- [Raw results bundle](cost-results.json.gz): all 66 benchmark runs (2,382 timed
  image observations), four validation files, exact early/final harness source,
  final analysis script, upstream review and reversible experiment patches.
- [Final matched validation](cost-validation.json): both seeds, animal-only result,
  holdout metrics and every paired regression used in the final tables.
- [All-run summary](cost-summary.json): per-question/per-suite scores and paired
  comparisons against `confirm-baseline-seed42`. This screening table includes
  different seeds, workload mixes and warmup versions; use matched validation
  above for final comparisons, not its global regression column.
- [Run metadata](cost-run.json): frozen candidates, source/model revision, observed
  spend, confirmed teardown and raw bundle SHA-256.
- [Holdout annotation notes](holdout.md): labels fixed before viewing predictions.

Regenerate the screening summary locally:

```bash
.venv/bin/python examples/features/diffusion_reads/evaluation/analyze_cost.py \
  examples/features/diffusion_reads/evaluation/cost-results.json.gz \
  --baseline-tag confirm-baseline-seed42 --output /tmp/cost-summary.json
```

Earlier screens used sequential warmup, then parallel warmup without a prior
cache reset. The known `vision-compiled-…-n9` screen includes a 97-second first
sweep compilation pause: its $555.16/million is **not steady-state cost**. Its raw
result remains intact. All final comparisons use the corrected common warmup.
Screening is exploratory and repeated selection can overfit these tiny fixtures.

The next useful step is a larger human-labelled, production-representative image
set and an explicit tolerance for distribution drift and latency. Keep this
holdout out of future model selection. More micro-optimization against these nine
photos would not establish the user's quality requirement.

## Validation

- Final pre-commit checks on all changed source/docs passed, including Ruff,
  mypy, Markdown and shell checks; `git diff --check` passed.
- Visual client and benchmark/analysis regression tests: **10 passed** using
  `.venv/bin/python -m unittest discover -s examples/features/diffusion_reads -p test_visual_client.py -v`.
- Existing Vast lifecycle tests: **3 passed**.
- Retained prefill contract tests on GPU environment: **3 passed, 8 deselected**,
  `.venv/bin/python -m pytest --confcutdir=tests/v1/worker tests/v1/worker/test_gpu_model_runner_v2.py -k logits_selection -q`.
- Discarded sampler experiment: 35 GPU tests passed; no useful end-to-end gain.
- Both scoped rentals destroyed; zero account instances confirmed at
  2026-09-19 23:25:52 UTC. 5090 rental upper estimate 132.27 minutes;
  4090 rental upper estimate 23.76 minutes. Cleanup timers stopped only after
  confirming absence. No credentials or account balances are in public artifacts.

## Objective and guardrails

Reduce the cost of image-to-typed-probability inference with the existing pinned
DiffusionGemma NVFP4 checkpoint. Preserve all questions, options, distributions,
and single-sample/no-thought behavior. No alternative model architecture.

Vast.ai budget: $6 total for this run. Initial RTX 5090, quoted $0.888889/hour
with 100 GB ephemeral storage; later RTX 4090 at $0.490370/hour including storage. A run-scoped destruction timer was armed before rental
with a three-hour deadline, leaving substantial budget for transfers and billing
lag. Private credentials and lifecycle files stay in `.vast-run/cost/`.

## Measurement plan

- Nine existing labelled photos: seven animals (24 questions each) and two general
  visual examples (eight questions each), 59 labelled decisions per sweep.
- These are convenience fixtures, not a representative production-quality set.
  Repeats measure variability; they do not increase the independent sample count.
- Compare accuracy, multiclass Brier score, log loss, per-image errors, and
  repeat variability. Do not select a faster setting based only on argmax accuracy.
- Clear prefix, processor, and encoder caches between sweeps; each photo appears
  once per sweep. Warm kernels separately. This avoids repeated-image cache hits
  being misrepresented as new-image throughput.
- Time complete sweeps at server localhost, excluding resets and warmups. Record
  wall time, completed images/second, per-request latency, read count, and raw
  probabilities. Price throughput, not summed overlapping request latencies.
- Screen compact IDs, canvas capacity, batching, then graph execution. Retest
  promising settings with additional repetitions against a fresh baseline.
- Any observed quality regression rejects a candidate as a recommended default;
  absence of regressions on these fixtures is only limited evidence.

## Progress

1. Created a clean branch from `0323d37edf54cceebdd2cf3203e3d50f10f3efe4`.
2. Existing lifecycle tests (3) and visual-client tests (6) pass; bootstrap shell
   syntax passes. No pre-existing instances were present.
3. Armed and verified scoped automatic teardown; rented instance 51641563.
4. Preparing reproducible screening benchmark while the GPU provisions.

5. Fixed container CUDA error 803 by preferring `/usr/lib/x86_64-linux-gnu`
   in `LD_LIBRARY_PATH`: bundled CUDA compatibility driver 580.95.05 was being
   loaded instead of host driver 595.84. GPU tensor smoke test passes.
6. Exact tokenizer counts including closing token: animal named 169, numeric 90,
   short 138; general named 52, numeric 36, short 44. Numeric IDs still need
   canvas 128 to hold all animal answers in one read.
7. Prepared a terminal read-only sampler fast path and GPU equivalence tests.
   It preserves schedule-scaled argmax and raw logprobs; skips noise generation,
   self-conditioning, and convergence work only for one-step read-only tiles.
8. Rejected proposed top-k tuning before testing: protocol normalization already
   disables ordinary top-k selection when explicit answer-token IDs are supplied.

9. Initial unrestricted FlashInfer compilation exhausted host memory (container
   recorded OOM kills; serving process disappeared). Recovered using existing
   artifacts and capped compiler jobs. Two jobs were stable but left 81 tasks;
   six jobs used about 34 GB against the 91 GB container limit, with no new OOM
   kills at the check. Inference configuration remains unchanged for baseline.
10. Added one independently annotated local photograph as a separate holdout;
    see `holdout.md`. Do not use its outputs to tune candidates.
11. Installed the exact official `flashinfer-jit-cache==0.6.18.post1+cu130`
    wheel instead of continuing the large local build. Verified its SM120 MoE
    library and restarted successfully. Installation took about 15 seconds;
    this does not include dependencies, model download, or server warmup.
12. Ran 35 GPU sampler tests successfully. The sampler shortcut did not improve
    sequential end-to-end performance: $83.84 versus $83.91 per million images.
    It remains experimental pending batched comparison.
13. Completed the first 18 configurations, each with three nine-image sweeps.
    Baseline throughput was 2.942 images/s. Named IDs with concurrency eight
    reached 7.903 images/s ($31.24/million), compared with $83.91/million for
    baseline. Aggregate accuracy/Brier improved, but two paired decisions
    regressed; this does not pass the quality gate.
14. Compact numeric IDs plus canvas 128 reached 8.680 images/s ($28.45/million),
    with five paired regressions. Kept question IDs unchanged in the default.
15. Profiled a separate request: 9,893 GPU kernels, 8,230 shorter than 10 µs.
    This supports testing graph execution. Profiled timing is excluded from
    all throughput/cost estimates.
16. Testing graph execution, followed by 140/70 image-token budgets. The
    checkpoint's image default is 280 tokens. Rental estimate at one hour:
    $0.91; scoped destruction timer remains armed.
17. Graphs with full image input, named IDs, canvas 64 and concurrency eight
    reached $28.05/million. One paired cat-colour decision regressed, so this
    still fails the default quality gate.
18. Rejected lower image budgets: 140-token named/batched accuracy was 92.1%
    versus 94.4% baseline. The cheapest 70-token numeric configuration reached
    $21.74/million but misclassified the bird as an opossum in one repetition.
19. Repeated eager and graph batching with the original sampler, five sweeps
    each. Original-sampler graphs reached $28.74/million with two paired
    regressions against the first baseline. Removed the sampler patch from
    active source; retained its diff with the experiment artifacts. Its GPU
    tests passed, but the small timing difference did not justify extra code.
20. Testing larger prefill batches and native CUTLASS/Marlin expert backends.
    These retain the checkpoint, questions and full image input. Defaults remain
    unchanged. Credit decrease at 75.6 minutes: $1.12.
21. Larger sequence capacity (16) and unrounded canvas widths did not help.
    Native CUTLASS and Marlin on the 5090 were also slower than the default
    FlashInfer backend. All screening results, including failures, are retained.
22. Tested opt-in compilation of the HF vision encoder. The image path calls
    `encoder` directly, so compiling the outer vision tower would be ineffective.
    Sequential gain was small. A batched run included a 97-second compilation
    pause: caches from its predecessor let warmup bypass vision encoding. Its
    aggregate cost is not a steady-state inference estimate.
23. Corrected the harness to use two shuffled, cold-cache warmup sweeps at the
    measured concurrency, then independently clear all three caches before every
    timed sweep. Retesting the final configurations with this common procedure.
24. Reviewed updated [PR #57250](https://github.com/vllm-project/vllm/pull/57250)
    at `ceb8eebf3eedddb964a50180f33838a9a6b13ee2`: 11/14 commits were patch-identical
    to our stack; its scheduler speedup was already present. Retained our image
    initializer and named-question delimiter fixes instead of replacing the tree.
25. Applied the separate [prefill optimization #57416](https://github.com/vllm-project/vllm/pull/57416)
    by Matt Mastracci, head `646ad6ed8ea1a8812bbf206acefbb89b26ab0ad1`, for testing.
    It skips unused prefill vocabulary projection. Three focused model-contract
    tests pass in the GPU environment. First 5090 measurements suggest a small
    sequential benefit, with no batched benefit beyond run-to-run variation.
26. Added a hardware-price comparison: a separately scoped RTX 4090 rental at
    $0.490370/hour, using the same NVFP4 checkpoint and Marlin. Its 90-minute
    destruction timer was verified before creation. The first $0.458519 offer
    was unavailable and created no instance. Combined observed spend: $1.81.

27. Completed matched final baseline and both finalists at seeds 42/43, plus
    animal-only and frozen holdout validation. The 4090 costs $23.08/million
    versus $83.86 baseline, but both finalists fail the quality gate. No preset
    promoted; no additional tuning after viewing holdout outputs.
28. Retrieved results before destroying both servers. Final observed spend $2.22,
    zero remaining instances. Packaged 66 benchmark runs and final validation.

29. Follow-up: the larger serving changes still fail the quality gate. Started a
    separate controlled prefill-only experiment on a fresh RTX 5090 rental at
    $0.888889/hour, with a verified 90-minute scoped teardown timer. Delayed
    billing raised the prior observed cumulative spend from $2.22 to $2.30.
    The additional compute/storage ceiling is $1.34, within the original $6 cap.
    This follow-up does not reuse the opened holdout for tuning.
30. Source audit: diffusion-only prefill sampling reads only `logits.device`,
    then initializes canvas/draft state and returns no sampled tokens. Added a
    regression comparing empty logits against NaN-filled old projection rows,
    including all sampler-state tensors, draft tokens, outputs and GPU RNG.
    Covers unfinished, mixed-completion and complete prompt chunks.
31. Frozen follow-up protocol: eager, one sequence, named IDs, full image tokens,
    original sampler and fixed hardware/settings. Run original/patched/patched/
    original blocks, two seeds and five cold-cache sweeps per seed per block.
    Original/original and patched/patched comparisons provide repeat controls;
    compare the full distributions, not only aggregate accuracy.

32. The three added GPU prefill-state equivalence cases passed (3 passed,
    8 deselected). Reviewed the user-provided Laya project while the isolated
    benchmark runs; see [Laya assessment](LAYA_ASSESSMENT.md). Its released
    checkpoints are text-only; no Laya model was substituted into this workload.

33. Completed the eight isolated runs. The prefill patch's apparent 0.28% saving
    is smaller than block-to-block variation. Same-code controls also change
    distributions and labels. No robust new cost or quality-equivalence claim.
34. Retrieved follow-up results and destroyed instance 51657195; zero remaining
    account instances confirmed. Cumulative observed spend $2.52.

35. Analysed original-code repeats at fixed seed 42 without new GPU work.
    [Repeat diagnostics](cost-repeat-diagnostics.json) group comparisons by
    per-chunk cached-token counts. Even identical cache-count patterns show
    large animal-answer differences; this rules out cache-hit counts alone as
    the explanation, not cache contents or numerical execution as causes.
    General-image questions are much more stable. Pairs reuse the same ten
    observations per photo and are not independent examples.
36. Requested the user's quality criterion: statistical non-inferiority on a
    larger labelled image set, or identical probability outputs for identical
    requests. The existing strict per-decision gate rejects same-code controls;
    changing that gate without an agreed criterion would redefine success.
    No more rentals while this is unresolved. The goal remains unachieved.
37. Rechecked billing and infrastructure at 2026-09-19 23:58:27 UTC: zero account
    instances, cumulative observed spend $2.54 after delayed billing.
