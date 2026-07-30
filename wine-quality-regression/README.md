# Wine Quality Regression

A regression study on the UCI red-wine quality dataset (1,599 samples,
11 physicochemical features, quality scored 0–10), run as a ten-phase
pipeline from exploratory analysis through to model explanation.

## Phases

1. **EDA** — descriptive statistics, target distribution, correlation
   heatmap, pairplot, IQR outlier counts per feature.
2. **Preprocessing** — duplicate check, outlier flagging, train/test
   split with a recorded split summary.
3. **Feature selection** — Pearson ranking, mutual information ranking,
   and a comparison of the best subset chosen by each method.
4. **Baselines** — six models fit under identical conditions.
5. **Cross-validation** — 10-fold shuffled CV with mean and standard
   deviation on every metric, so the ranking is read against its own
   noise.
6. **Tuning** — randomised search on the two strongest candidates.
7. **Diagnostics** — residuals vs predicted, Q-Q plot, residual
   distribution, actual vs predicted.
8. **Explanation** — SHAP summary, bar and dependence plots.
9. **Comparison** — permutation importance and a radar chart across
   models.
10. **Deliverables** — every table written to `results/` as CSV, every
    figure to `figures/`.

## Results

Test-set performance of the six baselines, ranked by RMSE:

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| **Random forest** | **0.549** | 0.422 | 0.539 |
| SVR | 0.593 | 0.454 | 0.462 |
| Gradient boosting | 0.602 | 0.485 | 0.446 |
| Linear regression | 0.625 | 0.504 | 0.403 |
| KNN | 0.659 | 0.506 | 0.335 |
| ElasticNet | 0.811 | 0.685 | −0.006 |

Random forest was tuned by randomised search
(`n_estimators=200`, `max_depth=30`, `max_features='log2'`,
`min_samples_leaf=1`, `min_samples_split=2`), improving the test RMSE
from 0.549 to **0.541** and R² from 0.539 to **0.553**.

Two things are worth reading off this table. **ElasticNet's negative R²**
means it does worse than predicting the mean — with the default
regularisation strength it shrinks the coefficients to near-nothing, and
it is left in as the honest floor of the comparison. And the tuning gain
is small: about 0.008 RMSE, well inside the cross-validation standard
deviation of 0.044, so the fair statement is that tuning did not
meaningfully improve on the default random forest.

Quality is an integer score treated here as continuous, which is why R²
around 0.55 is a reasonable ceiling for this dataset rather than a sign
of a broken model.

## Running it

```
python -X utf8 full_pipeline.py
```

The script downloads the dataset if it is missing, installs `tabulate`,
`shap` and `nbformat` if absent, sets `random_state=42` throughout, and
regenerates `figures/`, `results/` and the notebook. Also needs `numpy`,
`pandas`, `scikit-learn`, `matplotlib`, `seaborn` and `scipy`.

## Files

- `full_pipeline.py` — the whole pipeline, ~840 lines
- `wine_quality_regression.ipynb` — notebook version, phases as sections
- `report.pdf` — the submitted report, in Greek
- `data/winequality-red.csv` — the dataset
- `figures/` — 17 generated plots
- `results/` — 19 CSV tables (metrics, rankings, CV results, residuals)
