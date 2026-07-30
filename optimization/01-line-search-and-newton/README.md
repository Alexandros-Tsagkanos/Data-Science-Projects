# Line Search and Newton Methods

The theory half of the assignment asks for written answers on line-search
conditions, descent directions, quasi-Newton updates and conjugate
gradients. The applied half turns an optimization problem into a
nonlinear system and attacks it numerically.

## The problem

The test function is **Freudenstein–Roth**, a standard two-variable
nonlinear least-squares problem:

```
f(x1, x2) = (-13 + x1 + ((5 - x2) * x2 - 2) * x2)^2
          + (-29 + x1 + ((x2 + 1) * x2 - 14) * x2)^2
```

It is awkward on purpose. It has a global minimum at `(5, 4)` with
`f = 0` and a local minimum at `(11.41, -0.90)` with `f ≈ 48.98`, and the
local one has a wide basin — a gradient method started even at the origin
converges confidently to the wrong answer. That is the point of the
comparison.

## What the code does

`solution.py` produces every figure in the report and runs the numerical
comparison:

- `plot_armijo()` — the Armijo sufficient-decrease condition, shading the
  acceptable step-length region under the line `φ(0) + cαφ'(0)`.
- `plot_descent_direction()` — the geometry of the descent condition
  `∇f·p < 0`.
- `plot_freudenstein_3d()` / `plot_freudenstein_contour()` — surface and
  contour plots showing both basins.
- `gradient_freudenstein()` / `hessian_freudenstein()` — analytic
  derivatives, not finite differences.
- `one_iteration_methods()` — one hand-checkable iteration of each method
  from a common starting point, so the report's arithmetic can be
  verified against the code.
- The comparative study runs steepest descent and Fletcher–Reeves
  conjugate gradient, both with Armijo backtracking, and reports where
  each lands.

## Running it

```
python -X utf8 solution.py
```

The `-X utf8` matters — the plot labels are in Greek and Windows will
otherwise fail on the console encoding. Needs `numpy`, `matplotlib` and
`scipy`.

The script writes its four PNGs into the current working directory, so
run it from wherever you want them. The copies committed under
`figures/` are the ones used in the report.

## Files

- `solution.py` — all figures and the numerical comparison
- `report.tex` / `report.pdf` — the written report, in Greek
- `figures/` — the generated plots, as included in the report
- `presentation.pptx` — slides for the in-class presentation of exercise 4
