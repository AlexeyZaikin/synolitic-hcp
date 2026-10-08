# Supervised pairwise (synolitic) graphs for HCP task fMRI

Code accompanying the manuscript

> A. Zaikin, D. Vlasenko, T. Tyukina, O. Blyuss, D. Zakharov.
> *Supervised pairwise graphs recover cognitive-state information from
> individually uninformative brain regions.*

The repository contains the complete analysis: the cross-validated pipeline,
the preprocessing verification, the global-signal-regression sensitivity
analysis, the multiclass extension, the interpretability analysis, the figure
code, and the generator that writes every number and table of the manuscript
from the results files. The results files themselves are included, so all
numbers, tables and figures can be regenerated in seconds without the fMRI
data.

## What the method does

For every pair of brain regions (i, j) an elementary logistic-regression model
is trained on pairwise features of the two regional time series (means,
standard deviations and their correlation). Its output, the difference between
the posterior probabilities of the two brain states, is the weight of edge
(i, j). Each sample thus becomes a weighted graph over the regions (a
*synolitic*, or ensemble, graph; Vlasenko et al., 2025). Node summaries of
that graph are classified by a regularised logistic-regression meta-model.

The study applies the construction only to regions that are individually
uninformative (below 60 per cent accuracy, threshold sensitivity 55/65 per
cent), selected inside each outer training fold, and compares it with direct
classification of the same regions, a correlation graph, and a marginal-only
graph without the correlation feature.

## Repository layout

```
code/
  utils/data_loader.py          reads the parcellated HCP time series
  models/ensemble_graph.py      synolitic graph construction (edge models)
  models/run_pipeline.py        main experiment: region assessment, nested CV,
                                all methods, three thresholds
  analysis/preprocessing_check.py   z-scoring and global-signal checks (Fig. 7)
  analysis/gsr_sensitivity.py       whole experiment after global signal regression
  analysis/multiclass.py            14-class extension (7 paradigms x 2 conditions)
  analysis/interpretability.py      model weights by region and system (Figs. 8, 9)
  analysis/region_groups.py         HCP-MMP1.0 parcel -> anatomical system table
  analysis/make_paper_numbers.py    every number, table and traceability entry
  visualization/plot_results.py     Figures 1-6 and 10
results/
  metrics/                      results files (JSON) behind every reported value
  interpretability/             per-fold meta-model coefficients (NPZ)
```

Outputs are written to `results/`, `figures/` and `output/`.

## Installation

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.4/2.5, pandas 2.3, scikit-learn 1.8.0,
SciPy 1.17, matplotlib 3.10, seaborn 0.13, joblib 1.5 on macOS (Apple M4 Max,
16 cores).

## Reproducing the manuscript without the data

The included results files are sufficient for the numbers, tables and most
figures:

```bash
python code/analysis/interpretability.py        # Figs. 8-9, interpretability.json
python code/visualization/plot_results.py       # Figs. 1-6, 10
python code/analysis/make_paper_numbers.py      # output/generated/*.tex, traceability list
```

`make_paper_numbers.py` writes the LaTeX macros (`output/generated/numbers.tex`)
and tables used in the manuscript, plus `output/list_of_results/list_of_results.tex`,
which maps every reported value to the results file and script that produced
it. Without the data directory it skips only the subject counts (830 per
paradigm), which are read from the data file names.

## Reproducing the full analysis from the data

### Data

The data are the parcellated task-fMRI time series of the Human Connectome
Project 1200 Subject Release (WU-Minn HCP Consortium,
https://www.humanconnectome.org), parcellated with the HCP-MMP1.0 atlas
(360 cortical + 19 subcortical regions). Access requires accepting the HCP Open
Access Data Use Terms, and the data are **not** redistributed here.

The code expects one directory per paradigm, each holding one pickle per
subject, phase encoding and condition:

```
<data dir>/
  WM/          <subject>_LR_0bk.pickle   <subject>_LR_2bk.pickle   ...
  GAMBLING/    <subject>_LR_loss.pickle  <subject>_LR_win.pickle
  MOTOR/       ..._l / ..._r
  LANGUAGE/    ..._math / ..._story
  SOCIAL/      ..._mental / ..._rnd
  RELATIONAL/  ..._match / ..._relation
  EMOTION/     ..._fear / ..._neut
```

Each pickle is a list of `pandas.DataFrame`s, one per task block, with 379 rows
(regions, in the order of `MMP_REGIONS` in `code/utils/data_loader.py`) and one
column per time point. Only the LR phase encoding is used.

Place the data at `data/HCP_v4/split_data/split/` or point to it with

```bash
export HCP_DATA_DIR=/path/to/split
```

### Run

Run each script from the repository root:

```bash
python code/models/run_pipeline.py              # main experiment   (~5 h, 12 cores)
python code/analysis/preprocessing_check.py     # Fig. 7            (seconds)
python code/analysis/gsr_sensitivity.py         # GSR experiment    (~3 h)
python code/analysis/multiclass.py              # 14-class experiment
python code/analysis/interpretability.py
python code/visualization/plot_results.py
python code/analysis/make_paper_numbers.py
```

Runtimes are for a 16-core Apple M4 Max. Parallelism is set by `N_JOBS` at the
top of `run_pipeline.py` and `multiclass.py`. `gsr_sensitivity.py` accepts a
subset of paradigms, e.g. `python code/analysis/gsr_sensitivity.py WM MOTOR`.

The random seed is 42 for fold construction, internal folds and every
classifier. Cross-validation is 4-fold and grouped by subject, so the two
conditions of a subject never fall on opposite sides of a split; region
selection and graph construction for the training samples are nested inside
each outer training fold.

### Note on the package name

The source directory is called `code/`, matching the paths given in the
manuscript. This name shadows Python's standard-library module `code`, so
import the scripts by running them as files from the repository root (as
above), not with `python -m code....`.

## Citation

If you use this code, please cite the manuscript above and the original
description of the synolitic graph construction:

> D. Vlasenko, V. Ushakov, A. Zaikin, D. Zakharov. Ensemble-based graph
> representation of fMRI data for cognitive brain state classification.
> arXiv:2508.06118 (2025).

## Licence

MIT, see `LICENSE`.

## Acknowledgements

D.V., A.Z. and D.Z. acknowledge support from the Russian Science Foundation
(grant No. 24-68-00030). Data were provided by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657)
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research, and by the McDonnell Center for Systems Neuroscience at
Washington University.
