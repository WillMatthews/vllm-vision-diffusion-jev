# Structured reads on DiffusionGemma

A discrete diffusion model denoises a whole canvas per forward pass. If the
canvas is seeded with the answer's fixed text and only the answer slots are
left as noise, one denoise step yields a distribution over each slot. Three `extra_args` fields (`vllm_xargs` on the OpenAI server) expose
that:

| field | type | meaning |
| --- | --- | --- |
| `diffusion_seed_canvas` | `list[int]`, exactly `canvas_length` ids | replaces the random initial canvas after prefill |
| `diffusion_max_steps` | `int` | denoise steps before the canvas is emitted |
| `diffusion_read_only` | `bool` | emit the argmax canvas as soon as the cap is reached, end the request there, and return temperature-1 logprobs at every position |

`structured_server.py` is the layer that turns a question schema into those
fields. It speaks `/v1/chat/completions`: the system message is the schema,
the user message is the state JSON, and the reply content is one
distribution per question with a standard error over a few noise draws.

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
python examples/features/diffusion_reads/structured_server.py \
    --upstream http://127.0.0.1:8000 --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

Question types: `noul` (yes/no), `choice` with `options`, `score` with
ordered `levels`. Each label must be a single token in the answer template,
which the server checks with the tokenizer before the first request.

The server also speaks the contract of the Jev decision API at
`POST /v1/systemone`: a body of `state`, `questions` (a map of id to
`type`, `instructions`, `criteria`) and `model`, with answers in that API's
shapes (`noul` probability; `choice` with `probabilities` and `confidence`;
`score` with a 0-indexed `legend`). The schema's options above go in the
same body as extensions. A question may declare `depends_on` (read in a later stage with those
answers in its prompt), `ask_if` (asked only when a named question's answer
is among the listed ones, else null) and `alone` (a read of its own). Images attach as `multipart/form-data` (the JSON
in a part named `request`, each image a file part) or as an `images` array
of data URLs. With `TEST_PAGE=1` in the environment, `GET /` serves
`playground.html`, a page for sending requests with an image file or
webcam frames.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

`"think": N` in the schema lets the model write up to N tokens in its
thought channel before the read. The thought is an ordinary generation with
the chat template's thinking marker on, and the read then runs with the
thought in its prompt, so the answer slots condition on it. One thought
serves every noise draw of a decision. `diagnostics.thought` returns the
text, its length in tokens, whether the model closed the channel itself and
the generation time.

## Visual probabilities prototype

`visual_client.py` accepts a local image and returns a named probability
map per question. It uses the existing `/v1/systemone` image path. The
client needs `pybase64` (`uv pip install pybase64`); model serving needs a separate
GPU environment with this fork installed. This branch includes the image-path
initialization fix from [vLLM PR #57589](https://github.com/vllm-project/vllm/pull/57589).
An 8 GB RTX 2060 is not sufficient for the referenced 26B serving setup.

On a suitable model host, install this fork using the vLLM development
installation instructions, then start the model and adapter in separate terminals:

```bash
uv run --no-project --python .venv/bin/python vllm serve google/diffusiongemma-26B-A4B-it \
    --served-model-name dgemma --diffusion-config '{"canvas_length": 64}' \
    --max-logprobs 32 --enable-prefix-caching
uv run --no-project --python .venv/bin/python --with transformers --with pybase64 \
    examples/features/diffusion_reads/structured_server.py \
    --upstream http://127.0.0.1:8000 --model dgemma \
    --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 --host 127.0.0.1
```

From this checkout, send a screenshot (change `--endpoint` for a remote adapter):

```bash
.venv/bin/python examples/features/diffusion_reads/visual_client.py \
    --image screenshot.png \
    --schema examples/features/diffusion_reads/visual_schema.json \
    --endpoint http://127.0.0.1:8011
```

Set `API_KEY` in the client environment if the adapter requires authentication.
The example schema asks whether an error is visible and classifies the screen.
Edit the questions and criteria to suit your task. `noul` becomes `yes`/`no`,
choices retain their names, scores use their legend names, and skipped questions
remain `null`. Output also preserves usage, diagnostics, and HTTP round-trip
latency. File loading/base64 encoding are outside that latency measurement.

The default is one sample with no generated thought. Explicit `samples` and
`think` values in the schema override it. Probabilities are normalized over the
allowed labels, **not empirically calibrated correctness probabilities**.
In particular, check `diagnostics.questions.<id>.label_mass`: a large normalized
probability can hide very little original probability on the allowed tokens.
That diagnostic describes the first read; repeated draws do not establish
calibration. Questions are marginal distributions, not a guaranteed consistent
joint decision.

### Evaluate labelled images

Create a JSONL manifest. Image paths are relative to the manifest; expected
labels use the output names (`yes`/`no` for booleans). Optional `state` overrides
schema context for that image. Ground truth is never sent to the model.

```json
{"image":"screens/login.png","expected":{"is_error_screen":"no","screen_type":"login"}}
{"image":"screens/error.png","expected":{"is_error_screen":"yes","screen_type":"error"}}
```

```bash
.venv/bin/python examples/features/diffusion_reads/visual_client.py \
    --manifest labelled-images.jsonl \
    --schema examples/features/diffusion_reads/visual_schema.json \
    --threshold 0.9 > visual-results.json
```

The report includes per-image predictions and diagnostics, accuracy, multiclass
Brier score (sum over classes), negative log likelihood (probabilities floored
at 1e-15), and top-label ECE with ten equal-width confidence bins. Metrics pool
all labelled questions. Skipped answers are excluded from accuracy and scoring
but count against threshold coverage. Selective accuracy is accuracy among
answers whose maximum probability reaches the threshold; it is `null` when
none qualify. Latency percentiles use nearest ranks and include cold requests.
Use representative held-out images; a tiny smoke test cannot establish calibration.
The runner fails on request errors rather than silently excluding failed images.

CPU contract checks (a local HTTP stub, no weights or vLLM installation needed):

```bash
.venv/bin/python -m unittest discover \
    -s examples/features/diffusion_reads -p 'test_visual_client.py' -v
```

These tests verify transport and scoring only. Real image accuracy, latency,
and the model initialization fix still require validation on the model host.
