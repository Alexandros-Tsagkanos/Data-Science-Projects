# Transfer Learning on Oxford Flowers 102

Fine-grained image classification: 102 flower species, many of which
differ only in petal shape or colour gradient, with relatively few
training images per class. A good setting for showing what transfer
learning buys you.

## The two experiments

**A CNN trained from scratch** — two conv/max-pool blocks (32 then 64
filters) into dense(64) and a 102-way softmax, trained 10 epochs with
Adam. It reached roughly 40% accuracy, which is the
honest result for a small network on 102 fine-grained classes with this
little data. The architecture is kept in the file, commented out, rather
than deleted, because the comparison is the point of the assignment.

**VGG16 transfer learning, in two stages** —

1. *Frozen base.* ImageNet-pretrained VGG16 with `include_top=False`,
   `base.trainable = False`, and a new head: flatten → dense(256, ReLU) →
   dropout(0.5) → dense(102, softmax). Trained 10 epochs with Adam at the
   default learning rate.
2. *Fine-tuning.* Unfreeze `block5` only, leave the earlier blocks
   frozen, recompile with Adam at `1e-5`, and train 10 more epochs.

The low learning rate on the second stage matters — fine-tuning at the
default rate destroys the pretrained filters before the new head has
stabilised.

The training curve plots both stages end to end with a vertical marker at
the fine-tuning boundary, so the effect of unfreezing is visible as a
step in validation accuracy rather than something you have to take on
faith.

## Details

- Images resized to 224×224 and scaled to `[0, 1]`; batch size 32.
- The official `train` / `validation` / `test` splits from
  `tensorflow_datasets` are used as-is.
- Evaluation is test-set accuracy plus a 102×102 confusion matrix,
  rendered as a heatmap — at this class count the useful reading is the
  diagonal's strength and where the off-diagonal mass clusters, not
  individual cells.

## Running it

```
python solution.py
```

Downloads Oxford Flowers 102 through `tensorflow_datasets` on first run
(about 330 MB). A GPU is strongly recommended; on CPU the 20 epochs of
VGG16 take hours. Needs `tensorflow`, `tensorflow-datasets`,
`scikit-learn`, `matplotlib`, `seaborn` and `numpy`.

## Files

- `solution.py` — both experiments, training curves and confusion matrix
- `report.pdf` — the submitted report
