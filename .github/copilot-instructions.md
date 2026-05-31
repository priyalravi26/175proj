# Copilot Instructions for BE175 Final Project

This repository is a small data analysis project centered on drug response profiling and dimensionality reduction. There is no package structure or test suite; the main workflow is running the provided scripts against the CSV files in the repository root.

## Key Files
- `be175_finalproject.py` - primary analysis script. Reads `primary-screen-replicate-collapsed-logfold-change.csv` and `biomarkers.csv`, aligns drug metadata, performs PCA and PLS regression, and generates plots.
- `training.py` - exploratory PCA script. Reads `primary-screen-replicate-collapsed-treatment-info.csv`, `primary-screen-replicate-collapsed-logfold-change.csv`, and `primary-screen-cell-line-info.csv` for unsupervised analysis.
- `interactive.py` - currently empty and should be ignored unless the user adds intent for interactive workflow.

## How to run
- Use the repository root as working directory.
- Run the main analysis with:
  - `python3 be175_finalproject.py`
- Run the exploratory PCA script with:
  - `python3 training.py`

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Project-specific patterns and assumptions
- Scripts use relative CSV paths, so all data files must remain in the repo root.
- `be175_finalproject.py` depends on exact column matching between:
  - `primary-screen-replicate-collapsed-logfold-change.csv` columns and
  - `biomarkers.csv` `column_name` values.
- Data cleaning is explicit:
  - drop rows/columns with NaNs in the log-fold-change matrix
  - strip whitespace from drug names before matching
  - fill missing `target` values with `Unknown_Target`
- The main model split is by drug category:
  - targeted cancer drugs are used for training
  - noncancer drugs are used for testing
- `be175_finalproject.py` is exploratory and plot-heavy; it uses `plt.show()` rather than saving figures.

## Useful notes for AI agents
- Preserve the current alignment logic in `be175_finalproject.py`: intersection-based filtering of drug columns and metadata indexing by `column_name`.
- Do not assume a package or CLI entrypoint exists; this is a script-driven repository.
- Avoid refactoring into a package unless the user explicitly asks, because current scripts are written as standalone notebooks-to-Python exports.
- If debugging data issues, inspect column values in `logfold_raw.columns` and `drug_metadata['column_name']` for whitespace or mismatches.

## When to ask the user
- If a change touches data file naming, verify whether the user wants to preserve the current CSV layout.
- If modifications involve saving plot output, check if the user prefers generating files instead of interactive display.
- If adding testing or packaging, clarify that no existing test harness or setup configuration is present.