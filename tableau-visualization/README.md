# Tableau Visualization

The final assignment from a Tableau course: a workbook over a retail
orders dataset, built to cover the main chart families and the design
decisions behind choosing between them.

## The workbook

`final-assignment.twbx` contains six worksheets, each answering a
different shape of question about the same `Orders` data source:

| Worksheet | Chart type | Suited to |
| --- | --- | --- |
| Bubble Chart | Packed bubbles | Relative magnitude across many categories at once |
| Column Chart | Vertical bars | Comparison across a discrete dimension |
| Continuous Line Chart | Line | Trend over a continuous date axis |
| Horizontal bar chart | Horizontal bars | Ranking with long category labels |
| Horizontal bar chart2 | Horizontal bars | A second cut of the same ranking |
| Pie Chart | Pie | Part-to-whole, at a deliberately small slice count |

The file is a **packaged** workbook (`.twbx`), so the extract travels
inside it — it opens and renders with no data-source reconnection. The
underlying spreadsheet is included separately as
`final-assignment-data.xlsx` for anyone who wants to rebuild the views
from scratch.

## Opening it

Needs Tableau Desktop, or Tableau Public, which is free. Open
`final-assignment.twbx` directly — there is nothing to configure.

## Files

- `final-assignment.twbx` — the packaged workbook, six worksheets
- `final-assignment-data.xlsx` — the source orders data

## What is not here

The course also shipped practice datasets for exercises on joins, data
blending and dashboards — the Brazilian e-commerce (olist) tables, the
Superstore sample, and a supermarket sales extract. Those are about
130 MB of third-party sample data with no workbooks of mine attached to
them, so they are not included. All are publicly available if you want
to follow the same exercises.
