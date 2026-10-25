# Task26009

Implementation and evaluation of decision trees, manual bagging, and SAMME boosting, including masking robustness and bias–variance analysis using the Olivetti Faces dataset.

The project includes experiment outputs, saved predictions, trained models, reproducibility records, figures, and a PDF report. The current run is stored in `runs/m26mac009`. Before final submission, verify that `configs/default.json` contains the correct student name, roll number, repository URL, and final commit hash.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run Experiments](#run-experiments)
- [Validate Results](#validate-results)
- [Reproducibility](#reproducibility)
- [Report Compilation](#report-compilation)

## Project Structure

```text
.
├── configs/
│   ├── default.json
│   └── resolved.json
├── data/
│   ├── olivetti.npz
│   ├── resamples.npz
│   ├── split.npz
│   └── split_metadata.json
├── figures/
├── logs/
├── models/
├── report/
├── results/
├── runs/
│   └── m26mac009/
├── scripts/
├── task26009/
│   ├── __init__.py
│   ├── data.py
│   ├── experiments.py
│   ├── metrics.py
│   └── models.py
├── tests/
├── pytest.ini
├── requirements-lock.txt
└── README.md
```

## Setup

Python 3.14 is currently used by the existing Anaconda installation and project virtual environment. The locked dependencies should be installed in the virtual environment rather than the system Python environment.

From the project root:

```bash
source .venv/bin/activate

python --version
python -m pip install -r requirements-lock.txt
```

If `.venv` has not been created yet, create it using your available Python executable:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

If dependency installation fails, check the Python version required by the pinned packages before recreating the environment. Use Python 3.12 if the assignment dependencies or configuration explicitly require it.

## Run Experiments

Run the complete pipeline using the required roll number and the designated output directory:

```bash
python scripts/run_all.py \
  --root runs/m26mac009 \
  --roll-number M26MAC009 \
  --jobs 4
```

Replace `M26MAC009` with the complete roll number associated with your submission if it differs. The pipeline is responsible for executing the experiments, running tests, and generating the associated outputs according to its configuration.

### Run experiments separately

To execute only the experiment stage:

```bash
python scripts/run_experiments.py \
  --root runs/m26mac009 \
  --roll-number M26MAC009
```

Check `python scripts/run_experiments.py --help` if your version of the script uses different arguments.

## Validate Results

Run the unit tests:

```bash
python -m pytest -q
```

Validate the generated results:

```bash
python scripts/validate_results.py \
  --root runs/m26mac009
```

Run the reproducibility check:

```bash
python scripts/check_reproducibility.py
```

If a test fails with `ModuleNotFoundError: No module named 'task26009'`, verify that the virtual environment is active and that the project root is on Python's import path. The tests must import the renamed package consistently.

## Reproducibility

The experiment configuration and saved artifacts are retained to support reproducibility.

- `configs/default.json` contains the default experiment parameters.
- `configs/resolved.json` records the resolved configuration for the existing run.
- `data/split_metadata.json` stores metadata about the data split.
- `data/resamples.npz` contains the training resamples.
- `results/summary.json` contains consolidated experiment results.
- `results/predictions.npz` stores prediction outputs.
- `logs/` contains experiment and reproducibility records.

Use a separate output directory when comparing runs with different configurations. Confirm that the roll-number seed and resolved configuration match the requirements of your assignment before submitting the results.

## Report Compilation

The generated PDF report and its LaTeX source are stored under the run directory:

```text
runs/m26mac009/report/
├── report.pdf
├── report.tex
├── cvpr.sty
├── ieeenat_fullname.bst
└── figures/
```

If Tectonic is installed, compile the report from its source directory:

```bash
cd runs/m26mac009/report
tectonic --keep-logs report.tex
cd -
```

If the full pipeline supports a `--latex` option, consult the runner's help output before specifying a custom compiler path:

```bash
python scripts/run_all.py --help
```

## Main Components

- **`task26009/`** — data preparation, experiment logic, evaluation metrics, and model implementations.
- **`scripts/`** — experiment orchestration, validation, reproducibility checks, and report/figure generation.
- **`tests/`** — automated tests for data processing, metrics, and models.
- **`data/`** — dataset and preprocessing artifacts.
- **`models/`** — saved model artifacts.
- **`results/`** — prediction arrays and consolidated evaluation results.
- **`figures/`** — generated visualizations.
- **`report/`** — report source and generated PDF.
- **`logs/`** — experiment and validation records.
