# Decision Analysis

Five decision-analysis case studies, worked through decision trees and
payoff tables, with expected-value reasoning, Bayesian revision of
probabilities, and the value of information.

| Case | What it asks |
| --- | --- |
| Chelsea Bush | A short warm-up: build the payoff table and the tree, solve by rollback, then vary the key probability to find where the decision flips |
| Steeley Associates | Decision tree over a consulting engagement with sequential choices |
| Smart Steering Support (BAAG) | Decision tree over a product-support strategy |
| The Carolina Cougars | Sequential decision under uncertainty, with a comparison of the candidate strategies |
| Brainy Business (Cerebrosoft) | The full treatment: payoff table, then Bayes' rule to revise the priors on market response given a survey result, then the value of the market research itself |

## Approach

Every tree in the report is **drawn by the code**, not by hand.
`solution.py` builds them from `matplotlib` patches, with the usual
convention — squares for decision nodes, circles for chance nodes:

```python
def vale_tetragono(ax, cx, cy, platos=0.15):   # tetragono = square, decision node
def vale_kyklo(ax, cx, cy, aktina=0.08):       # kyklos = circle, chance node
```

Doing it this way means the numbers on the branches come from the same
variables that compute the rollback, so the figure and the arithmetic
cannot drift apart.

The Chelsea Bush case also gets a sensitivity plot: expected value of
each alternative as a function of the state probability, with the
crossover point marked — that is the probability at which the
recommendation changes.

The Cerebrosoft case is the one that exercises the full method. It runs
in three parts: the payoff table under the prior, Bayesian revision of
the prior given the survey outcome, and then the comparison that prices
the survey.

## Running it

```
python -X utf8 solution.py
```

Writes eight PNGs into `figures/`, which the report includes. The script
sets the `Agg` backend, so it needs no display. Needs `numpy` and
`matplotlib`.

## Files

- `solution.py` — all five cases and every figure (about 1,300 lines)
- `report.tex` / `report.pdf` — the written report, in Greek
- `figures/` — the generated decision trees and sensitivity plots
