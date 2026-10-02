# Student Dropout Prediction — A Reproducible Benchmark

> **Full source: [Alexandros-Tsagkanos/edm-dropout-benchmark](https://github.com/Alexandros-Tsagkanos/edm-dropout-benchmark)**
>
> This directory is a summary. The code, results and technical report
> live in their own repository.

A multi-seed, significance-tested benchmark for **student-dropout
prediction**, supervised by Prof. Sotiris Kotsiantis at the University of
Patras, Department of Mathematics. It accompanies a technical report and
a paper in preparation.

## The study

The primary dataset is the UCI *Predict Students' Dropout and Academic
Success* data (Realinho): 4,424 students, binary dropout versus
graduate/enrolled, about 32% positive. A ten-phase pipeline runs over it:

- Three supervised baselines
- A VAE oversampler, and TabDDPM diffusion as a second generative arm
- Uncertainty-based active learning, and a DQN active learner
- A post-hoc fairness audit, plus an in-processing fairness penalty
- A multimodal extension pairing XGBoost with DistilBERT

On top of that sits a methodological-rigor layer: a classical SMOTE
baseline, bootstrap confidence intervals, a ten-seed robustness study
with paired significance tests, a self-labelled SSL baseline (Co-Forest),
a second dataset (OULAD, 32,593 enrolments, also ten seeds) for
cross-cohort generalisation, and a real-text validation on a genuine
educational corpus.

## What it found

The strongest configuration is multimodal XGBoost + DistilBERT at
F1 ≈ 0.85 / AUC ≈ 0.96, a ~0.06 F1 lift over the tabular baseline
(p = 0.002). That number is reported as **a ceiling rather than a
deployable result**, because the text in that phase is simulated and
injects 0.28 bits of label information by construction — roughly 31% of
the label entropy. Re-run on real forum text, the same protocol reaches
AUC ≈ 0.82 on urgency detection, confirming the encoding pipeline works
on genuine student writing.

The in-processing fairness penalty cuts the gender demographic-parity gap
from 0.21 to 0.08 — about 60%, p = 0.002 — at a real cost of roughly 1.6
accuracy points.

The recurring finding is the negative one: on moderately imbalanced
tabular data of this kind, **the elaborate levers do not beat a
well-regularised supervised baseline**. Generative oversampling, RL-driven
active learning and self-labelling all fail to improve on it. On the
second dataset, several of the oversampling and active-learning results
partly reverse — included deliberately as a cross-cohort honesty check.

The contribution is the integrated, honest, reproducible evaluation
rather than any single winning model.

## Reproducibility

Every claim is backed by a committed artefact, and every output under
`RESULTS/` carries the exact command that regenerates it. Provenance for
each dataset is tracked separately. A full single-seed run of all three
datasets takes about 20 minutes on CPU — no GPU needed — through a
one-command driver with per-platform launchers.
