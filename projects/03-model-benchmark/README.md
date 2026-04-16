# Project 3 — Model Benchmark Tool

Project 3 is about systematic model comparison.

Instead of focusing on one model family, this project teaches you how to compare several models fairly across a dataset using shared metrics and cross-validation.

## Main Learning Goals
- compare models under the same evaluation setup
- understand trade-offs between accuracy and stability
- build reusable benchmarking habits
- document model selection decisions clearly

## Initial Version
The starter version benchmarks several classical classifiers on a selected dataset.

Default dataset:
- `wine`

Supported datasets in the starter script:
- `wine`
- `breast_cancer`
- `iris`

## Models Compared
- Logistic Regression
- KNN
- SVC (RBF)
- Random Forest
- Gradient Boosting

## Run It
```bash
python scripts/run_project3_benchmark.py --dataset wine
```

## Expected Outputs
Saved under:
```text
projects/03-model-benchmark/results/
```

Files:
- `benchmark_wine.json`
- similar JSON files for other datasets you run

## What to Look For
- best cross-validation accuracy
- best macro F1
- model stability across folds
- whether a more complex model really wins

## Why This Project Matters
Project 1 teaches regression workflow.
Project 2 teaches classification and neural training.
Project 3 teaches model selection discipline.
