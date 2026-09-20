# Laya and image-to-JEV inference

Reviewed 2026-09-20, repository revision
`6a5819129eb220570792e417e49723d697efd76f`.

Laya is useful architectural inspiration, but the released checkpoints cannot
replace this repository's image inference. Its [model card](https://huggingface.co/convaiinnovations/laya)
explicitly limits inputs to text. The English model uses ModernBERT and a trained
decision head, not a vision encoder or diffusion model.

The [implementation](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/agent.py)
builds a separate sequence for each question, then batches those sequences.
The [decision head](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/common.py)
scores option-marker positions. A single batched call does not mean the shared
state is encoded only once across questions.

## What the published results establish

The [project README](https://github.com/NandhaKishorM/laya#benchmarks) is more
qualified than the promotional headline:

- The 76.6% typed-decisions result requires task-specific fine-tuning. Base
  checkpoints score approximately 34–36%, versus 31.8% random.
- Jev scores come from separate published experiments; prompts and sample sizes
  differ. These are not a controlled comparison with our image workload.
- The shipped models are overconfident on evaluated distributions; domain
  temperature fitting improves calibration. Higher argmax accuracy does not
  guarantee better agreement with a teacher's full distributions.

The [website's](https://laya.convaiinnovations.com/) zero-dollar self-hosting
comparison excludes the compute bill. Model weights without a licence fee do
not imply free inference. Its latency figures concern text questions, not
end-to-end image processing.

Neither these materials nor the reviewed [TypeSafe overview](https://typesafe.ai/)
establish that Laya implements Jev's actual internal architecture. Similar typed
outputs and training terminology are insufficient evidence of identical internals.

## Implication for this project

The following is a proposed research direction, not a measured result:

For a stable question set, train a small vision model to predict every question's
option probabilities directly. Encode the image once, share its features across
question heads, and use DiffusionGemma distributions as training targets alongside
human labels. Fit calibration on a separate split and evaluate on untouched
images, including rare attributes and changed image sources. The existing nine
photos and opened one-photo holdout are insufficient for that work.

This could remove substantially more computation than tuning the present large
model's runtime. It also changes the trained model and needs a new quality
validation effort. Supporting arbitrary new questions would require a
question-conditioned visual model and broader training coverage, not merely a
fixed classifier head.

An image-caption-to-Laya pipeline would retain the cost of generating a caption
and could lose visual details before decision making. It is not supported by
these benchmarks as a quality-preserving shortcut. For the current diffusion
optimization loop, Laya supplies no directly applicable image checkpoint or
verified inference patch. No Laya weights were downloaded and no rental budget
was spent benchmarking its text-only models.
