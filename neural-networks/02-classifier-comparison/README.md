# MLP versus Classical Classifiers

A multi-layer perceptron benchmarked against four classical models on
three tabular datasets, to see whether the neural network is worth its
extra cost on this kind of data.

## Setup

**Datasets**, pulled from OpenML by name:

| Dataset | Character |
| --- | --- |
| `breast-cancer` | Small, mostly categorical |
| `credit-g` | Mixed types, moderately imbalanced |
| `adult` | Largest of the three, heavily imbalanced |

**Models**: KNN (k=5), decision tree (max depth 10), random forest (50
trees), MLP (one hidden layer of 50 units, 300 iterations), and SVM with
the default RBF kernel.

**Protocol**: one-hot encode categoricals, fill missing values with zero,
standardise, 80/20 train-test split, 3-fold cross-validated accuracy on
the training set, then weighted precision, recall and F1 on the held-out
test set. Weighted averaging is used because two of the three datasets
are imbalanced.

## What is deliberate, and what is a shortcut

The preprocessing is intentionally minimal, and the code says so — the
question is the relative ranking of model families under identical
treatment, not the best achievable score on any one dataset. Several
choices are shortcuts, and are marked as such in the comments:

- Missing values are filled with `0` rather than imputed, with a `TODO`
  noting that mean/median imputation was not tested.
- Hyperparameters are left near their defaults and are not tuned, so
  this compares families rather than tuned models.
- Cross-validation is 3-fold to keep runtime down.
- **The train/test split has no `random_state`.** This is on purpose —
  re-running shows how much of the gap between models is real and how
  much is split noise. It also means the numbers are not reproducible run
  to run, which is the trade being made.

## Running it

```
python solution.py
```

Fetches all three datasets from OpenML on first run and prints a summary
table per dataset. `adult` is the slow one, mostly because of the SVM.
Needs `scikit-learn`, `pandas` and `numpy`.

## Files

- `solution.py` — the full comparison loop
- `report.pdf` — the submitted report, with the results and discussion
