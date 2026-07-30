# Database Design: ER Model to Working Schema

A university textbook-declaration domain, modelled as an ER diagram,
mapped to a relational schema, and then implemented and queried for real
rather than left on paper.

## The modelling problem

Seven relations: `ΠΑΝΕΠΙΣΤΗΜΙΟ` (university), `ΤΜΗΜΑ` (department),
`ΜΑΘΗΜΑ` (course), `ΦΟΙΤΗΤΗΣ` (student), `ΣΥΓΓΡΑΜΜΑ` (textbook), plus the
relationship tables `ΠΡΟΤΕΙΝΕΙ` (a course recommends a textbook) and
`ΔΗΛΩΝΕΙ` (a student declares a textbook).

The interesting part is that **`ΜΑΘΗΜΑ` is a weak entity**. The ER
diagram marks it with a double rectangle and a dashed underline under
`ΚωδικόςΜ`, meaning a course code is only unique within its department.
So the primary key is composite — `PK(ΜΑΘΗΜΑ) = (ΚωδικόςΤ, ΚωδικόςΜ)` —
and that composite key has to propagate into both `ΠΡΟΤΕΙΝΕΙ` and
`ΔΗΛΩΝΕΙ`.

The seed data exercises this deliberately: course code `1` is reused
across three different departments, which is legal precisely because of
the composite key. Getting this wrong is the standard failure mode on
this kind of assignment, and it only shows up when you actually insert
data.

## Why it is a notebook

Rather than submitting SQL as text, the schema is built and run
end-to-end in **SQLite** through Python's built-in `sqlite3`, with
`PRAGMA foreign_keys = ON` so the referential constraints are enforced
rather than decorative. Every query returns a real result set, rendered
through pandas.

The notebook runs against an in-memory database, so it leaves nothing
behind and can be re-run from a clean state.

## The queries

| Query | What it exercises |
| --- | --- |
| Q1 | `IS NULL` — textbooks with no listed price |
| Q2 | Three-way join across student → department → university |
| Q3 | `COUNT(DISTINCT ...)` over the declaration table |
| Q4 | Aggregation per publisher, ordered |
| Q5 | `MIN`/`MAX` per group with a `HAVING` filter |
| Q6 | `HAVING` on a join count — textbooks declared by more than two students |
| Q7 | Extremal row by year |
| Q8 | Grouping on the **composite** course key — the weak-entity payoff |
| Q9 | Set difference, done twice: once with `NOT IN`, once with `NOT EXISTS` |
| Q10 | Ranking students by declaration count |

Q9 being written both ways is the point of that exercise — the two
formulations differ in their handling of NULLs, and the notebook shows
both.

The notebook ends with a verification checklist (`Σημεία ελέγχου`) naming
what to confirm and the expected answer: `PRAGMA foreign_key_check`
returning empty proves the weak-entity keys propagated, both Q9 variants
must return the same publisher, and the one student with no declarations
is the reason Q3 returns 6 while Q10 excludes them.

## Running it

```
jupyter notebook schema_and_queries.ipynb
```

Needs only `pandas`; `sqlite3` ships with Python. No database server and
no setup — it creates, populates and queries an in-memory database on
each run.

## Files

- `schema_and_queries.ipynb` — schema, seed data, all queries, checks
- `report.pdf` — the written report, in Greek

Table, column and value names are in Greek throughout, matching the
assignment.
