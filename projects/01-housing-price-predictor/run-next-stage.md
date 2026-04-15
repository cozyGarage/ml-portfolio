# Next Stage Guide

## Goal
Move from a simple holdout baseline to a more reliable project workflow.

## Stage 2 Tasks
1. complete EDA using `notes/eda-checklist.md`
2. run `src/baseline_train.py`
3. run `src/crossval_compare.py`
4. compare holdout RMSE vs cross-validation RMSE
5. update project notes with best model and concerns

## Commands
```bash
python projects/01-housing-price-predictor/src/baseline_train.py
python projects/01-housing-price-predictor/src/crossval_compare.py
```

## Why This Matters
Cross-validation gives a more reliable estimate than one train/test split.

## After This Stage
The next upgrade should be:
- hyperparameter tuning for random forest
- better feature analysis
- stronger project documentation
