# Stage 3 Guide — Model Improvement

## Goal
Move from simple baselines to a tuned model with clearer comparison.

## Run Order
1. baseline models
2. cross-validation comparison
3. random forest tuning

## Commands
```bash
python projects/01-housing-price-predictor/src/baseline_train.py
python projects/01-housing-price-predictor/src/crossval_compare.py
python projects/01-housing-price-predictor/src/tune_random_forest.py
```

## What to Record
After running all three:
- baseline RMSE for each model
- cross-validation mean RMSE and std RMSE
- best random forest parameters
- tuned test RMSE
- whether tuning improved the result

## Decision Rule
If tuned random forest is clearly better and stable, use it as the current best model.

## Next Stage After This
- add feature importance analysis
- improve documentation
- connect this project to the CLI
