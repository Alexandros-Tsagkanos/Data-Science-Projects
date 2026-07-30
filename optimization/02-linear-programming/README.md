# Linear Programming Case Studies

Seven applied linear programs, each modelled from a word problem, solved
with PuLP, and then pushed on with sensitivity or scenario analysis.

| # | Case | Model |
| --- | --- | --- |
| 1 | Aircraft fuel tankering | Minimise fuel cost across four airports (ATH/LON/PAR/ROM) given per-airport price, minimum and maximum uplift, and tank continuity between legs |
| 2 | Wallasidis Juice Company | Minimise production and distribution cost, with a sensitivity analysis on the binding constraints |
| 3 | Diet problem | Maximise palatability subject to nutritional and calorie constraints; includes the graphical solution of the two-variable form |
| 4 | Helping Hand — humanitarian aid to Cuba | Minimise cost of an aid shipment under capacity and composition limits |
| 5 | Oakdale school busing | Minimise total busing distance while meeting racial-balance and capacity requirements |
| 6 | Patrol allocation | Minimise patrol cost subject to coverage requirements |
| 7 | Airport security staffing | Minimise staffing cost across shifts with overlapping coverage |

The report works each case through the same structure: problem
description, mathematical model (decision variables, objective,
constraints), then results and analysis. Exercise 3 additionally solves
the two-variable case graphically and then re-solves it under two
scenarios — a change in the palatability coefficient of meal A, and a
reduction in the calorie ceiling — to show which constraints bind.

## A note on the code file

`models-listing.py` is the collected PuLP code for all seven exercises,
**as transcribed into the report's appendix**. It is a listing, not a
runnable script: it was extracted back out of the typeset PDF, so it
carries typographic quotes (`’` instead of `'`) and the indentation of
some models has been flattened by the LaTeX `listings` environment.

It is included because it is the record of what was actually modelled and
submitted, and each model is readable as-is. To run one, copy the block
for that exercise, replace the curly quotes and restore the indentation.
Exercise 2 onward survived the round-trip largely intact; exercise 1 is
the worst affected.

## Files

- `models-listing.py` — the seven PuLP models, as listed in the report
- `report.tex` / `report.pdf` — the written report, in Greek

Needs `pulp`.
