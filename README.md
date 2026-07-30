# Data Science Projects

Data-science and operations-research work from my MSc, plus two larger
research projects that have grown past coursework: an agent-based retail
layout simulator written for a journal submission, and a reproducible
benchmark for student-dropout prediction.

The material spans nonlinear and linear optimization, decision analysis,
neural networks, regression modelling, relational database design,
interval-valued statistics in R, and dashboard work in Tableau.

## Projects

| Project | Area | Summary |
| --- | --- | --- |
| [optimization](optimization/) | Optimization / OR | Line-search and Newton methods, seven linear programs solved with PuLP, and five decision-analysis case studies with hand-drawn decision trees |
| [neural-networks](neural-networks/) | Deep learning | VGG16 transfer learning and fine-tuning on Oxford Flowers 102, and an MLP-versus-classical-classifier comparison across three OpenML datasets |
| [wine-quality-regression](wine-quality-regression/) | Machine learning | A ten-phase regression pipeline on the UCI red-wine dataset: EDA, feature selection, six models, tuning, diagnostics and SHAP |
| [database-design](database-design/) | Databases | An ER model with a weak entity mapped to a relational schema, implemented end-to-end in SQLite with the assignment's queries |
| [interval-analysis-returns](interval-analysis-returns/) | Statistics / R | Classical and interval-valued analysis of DIA ETF returns against 15 constituent stocks — regression, factor analysis, SEM |
| [tableau-visualization](tableau-visualization/) | Visualization | A six-worksheet Tableau workbook over a retail orders dataset |
| [fairness-in-language-models](fairness-in-language-models/) | Seminar | A talk on how fairness is formally defined in language models, and where the definitions conflict |
| [retail-layout-simulator](retail-layout-simulator/) | Simulation / research | Multi-floor agent-based retail simulator with dataset calibration and a genetic-algorithm layout optimizer — summary and link |
| [student-dropout-benchmark](student-dropout-benchmark/) | Research | Generative, semi-supervised and fair learning for dropout prediction — summary and link |

Each project directory has its own README describing the problem, the
approach and how to run it.

## Running things

There is no top-level build — each project stands alone.

- **Python projects** (optimization, neural networks, wine quality) are
  plain scripts and notebooks. Each README lists its dependencies; the
  usual stack is `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`,
  plus `pulp` for the linear programs and `tensorflow` for the neural
  networks.
- **Several scripts print Greek text.** On Windows run them as
  `python -X utf8 solution.py`, otherwise the console encoding mangles
  the output.
- **The R project** knits from RStudio or `rmarkdown::render()`. It
  installs its own missing packages on first run and reads its cached
  data from `data/`, so it does not need a network connection.
- **The Tableau workbook** needs Tableau Desktop or Tableau Public; the
  `.twbx` is packaged, so the data travels with it.

## A note on these projects

These are coursework and research projects, collected for presentation.
Two things are deliberately not here:

**Course material is excluded.** The source folders these came from also
held lecture slides, textbook scans, journal papers and classmates'
assignments. None of that is mine to publish, so only my own work is in
this repository. Where an assignment brief is needed to make sense of a
solution, the relevant README describes the problem instead of shipping
the professor's PDF.

**Two projects live in their own repositories.** The retail layout
simulator and the dropout benchmark are large enough to stand alone, and
duplicating them here would mean maintaining two copies that drift apart.
Their directories hold a written summary and a link.

Most reports, and many code comments, are in Greek — kept as originally
written. Variable and function names are mixed Greek-transliterated and
English, also as written.
