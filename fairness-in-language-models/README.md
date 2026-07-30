# Fairness Definitions in Language Models

Slides for a seminar talk given as part of the MSc neural-networks
course: a walk through how fairness is formally defined for language
models, following a survey by Yin, Wang, Palikhe and Zhang.

## The problem the talk addresses

There are dozens of fairness definitions and dozens of metrics, and very
little agreement on which one applies when. The talk presents a
taxonomy that organises them along two axes.

**Axis 1 — where the bias lives.**

- *Intrinsic* bias sits in the model's internal representations: the
  embedding space, the attention weights, the token probabilities. It is
  what the model has learned to encode, independent of any downstream
  task.
- *Extrinsic* bias shows up in task behaviour — the classifier's outputs,
  the generated text.

**Axis 2 — the architecture family**, because each one admits different
measurements:

| Family | Examples | How fairness is naturally measured |
| --- | --- | --- |
| Encoder-only | BERT, RoBERTa, ALBERT, DeBERTa | Masked-token probabilities, embedding-space similarity |
| Decoder-only | GPT-3, GPT-4 | Generated continuations, counterfactual prompt pairs |
| Encoder–decoder | T5, BART | Output of the full transduction |

Crossing the two axes gives a grid holding **fourteen definitions**. The
useful observation is that some definitions — counterfactual fairness,
stereotypical association — recur across architectures but get a
*different mathematical formulation* each time. That is exactly the
confusion the taxonomy is meant to clear up.

## What the talk covers

- Common notation: a sensitive topic `T`, demographic groups `G`, and
  each group's sensitive attribute set `A_i`, so definitions from
  different papers can be compared on the same terms.
- **Similarity-based disparity** — treating the embedding space as
  evidence of encoded belief; WEAT and its relatives.
- **Probability-based disparity** — whether a masked-token prediction
  favours stereotypical fillers (*"The doctor said that [MASK] went to
  the hospital"*); DisCo, LPBS, CBS.
- **Equal opportunity** on downstream classification, with the
  Bias-in-Bios occupation task as the worked example of unequal true
  positive rates between groups.
- **Attention-based metrics**, and why attributing bias to individual
  attention heads is harder than it looks.
- **Counterfactual fairness** — change only the sensitive attribute in
  the prompt and require the response to hold.
- **Allocative versus representational harm** — performance disparity
  (the model serves some users worse) against demographic representation
  (the model says less, or says stereotyped things, about some groups).
  The talk argues these are routinely conflated.
- **Algorithmic disparity** in encoder–decoder models, where the failure
  is homogenisation: linguistic variety collapsing toward whatever was
  most frequent in training, so forms disappear quietly rather than
  producing anything obviously offensive.

## Files

- `fairness-definitions-in-language-models.pptx` — 20 slides, with full
  speaker notes in Greek

The slides are mine. The surveyed papers are cited on the slides and are
not redistributed here.
