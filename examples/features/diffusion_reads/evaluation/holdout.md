# Additional local holdout

Labels in `holdout.jsonl` were assigned by visual inspection before any model
predictions were available. They are AI annotations, not independently reviewed
human ground truth. Keep this image separate from the nine-image tuning fixture.

The photograph is the repository's existing
`tests/v1/ec_connector/integration/hato.jpg`; the manifest references it directly
without copying or modifying the asset. It shows a foreground pigeon, two more
pigeons, and two people on an outdoor pavement. It adds positive people and
multiple-animal examples missing from the existing animal fixture. The main
animal is fully within the frame and its body is unobstructed. Feathers map to
`other` for coat colour and pattern as explicitly required by the schema.

Posture, view direction, juvenile status, ear-tag visibility, and fine-detail
sharpness are deliberately unlabelled because they introduce avoidable ambiguity.
The existing schema still asks these questions; only the manifest labels are
scored. This is one additional independent photograph, not nineteen independent
examples. It cannot establish population-level quality equivalence.

The other inspected candidates, `tests/multimodal/assets/image1.png` and
`image2.png`, contain rendered text. `tests/multimodal/assets/rgba.png` is an
illustration of coloured dice. None is a photograph; they were excluded rather
than counted as additional photographic quality evidence.
