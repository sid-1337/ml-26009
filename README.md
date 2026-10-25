# Task99

Decision trees, manual bagging and SAMME, masking robustness, and bias/variance on Olivetti faces.

The included run uses provisional seed `99`. Fill `student_name`, `roll_number`, `repository_url`, and `final_commit_hash` in `configs/default.json`, then rerun with the complete roll number. Code, analysis, figures, and report were generated with Codex.

## Setup

Python 3.12 was used. On this Mac:

```bash
cd /Users/hwalia1/Documents/Task99
/Users/hwalia1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

For another computer, use its Python 3.12 executable and install Tectonic. The exact Python and Tectonic paths shown here refer to this Mac.

In each new Terminal session:

```bash
cd /Users/hwalia1/Documents/Task99
source .venv/bin/activate
```

## Run

Reproduce the included seed-99 experiment, tests, figures, and official PDF:

```bash
python scripts/run_all.py --jobs 4 --latex /Users/hwalia1/Documents/Codex/2026-10-19/files-mentioned-by-the-user-064912a5/work/tectonic
```

Run with the required roll-number seed in a separate output directory; replace the placeholder:

```bash
python scripts/run_all.py --root runs/actual_roll --roll-number YOUR_COMPLETE_ROLL_NUMBER --jobs 4 --latex /Users/hwalia1/Documents/Codex/2026-10-19/files-mentioned-by-the-user-064912a5/work/tectonic
python scripts/validate_results.py --root runs/actual_roll
```

Rebuild figures and the report from saved results:

```bash
python scripts/run_all.py --skip-experiments --latex /Users/hwalia1/Documents/Codex/2026-10-19/files-mentioned-by-the-user-064912a5/work/tectonic
```

Run checks separately:

```bash
python -m pytest -q
python scripts/validate_results.py
python scripts/check_reproducibility.py
```

To use Tectonic installed on PATH, omit `--latex`. The pipeline requires a compiler to rebuild the official PDF; without one, it writes the TeX source. To compile that source manually:

```bash
cd report
tectonic --keep-logs report.tex
cd ..
```

## Files

- `report/`: official PDF, TeX source, CVPR style, and report figures.
- `task99/`, `scripts/`, `tests/`: complete implementation, figure/report scripts, and checks.
- `configs/`: default parameters and the configuration used for the included run.
- `data/`: dataset, saved split, seed metadata, and 30 training resamples.
- `results/`: consolidated scores, histories, case studies, and all prediction vectors.
- `models/`: selected fitted models used to reproduce case-study evidence.
- `logs/`: final experiment, test, and reproduction-check records.
- `requirements-lock.txt`: pinned Python environment.
