# Housing Price Predictor

## Project Goal
Build an end-to-end machine learning project that predicts California housing prices using tabular data.

This project is the first real portfolio project in the repository. It is designed to teach the full supervised learning workflow before moving into heavier GPU-based work.

## Why This Project Comes First
This project teaches the core machine learning loop without deep learning complexity:
- loading real data
- exploring features
- splitting train and test data
- building a preprocessing pipeline
- training baseline models
- evaluating results with RMSE
- documenting findings clearly

## Dataset
This project uses the California housing dataset from `scikit-learn`.

Target column:
- `MedHouseVal`

## Learning Objectives
By finishing this project, you should understand:
- the structure of a regression problem
- how to inspect and describe a tabular dataset
- how to avoid data leakage with train/test split
- how to use preprocessing pipelines
- how to compare baseline models
- how to report results like a portfolio project

## Project Structure
Recommended structure:

```text
projects/01-housing-price-predictor/
├── README.md
├── plan.md
├── run.md
├── run-next-stage.md
├── stage-3-guide.md
├── src/
│   ├── baseline_train.py
│   ├── crossval_compare.py
│   └── tune_random_forest.py
├── notes/
└── results/
```

## Project Milestones

### Milestone 1 — Data Understanding
- load dataset
- inspect `.head()`
- inspect `.describe()`
- plot histograms
- inspect correlations

### Milestone 2 — Baseline Pipeline
- create train/test split
- build preprocessing pipeline
- train Linear Regression baseline
- evaluate RMSE

### Milestone 3 — Stronger Baselines
- train Decision Tree Regressor
- train Random Forest Regressor
- compare RMSE values
- run cross-validation
- tune Random Forest

### Milestone 4 — Documentation
- summarize best model
- describe what worked and what failed
- list next improvements

## Evaluation Metric
Use RMSE (root mean squared error). Lower is better.

## Run It
One command:

```bash
python scripts/run_project1_housing.py
```

Or step by step:

```bash
python projects/01-housing-price-predictor/src/baseline_train.py
python projects/01-housing-price-predictor/src/crossval_compare.py
python projects/01-housing-price-predictor/src/tune_random_forest.py
```

## Notes
After running, update:
- `notes/first-results-template.md`
- `notes/stage-3-results-template.md`

## Next Step
After this project is stable, move to Project 2 for intensive classification and PyTorch work.
