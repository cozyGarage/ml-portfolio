# Project 2 — MNIST / Digits Classifier

This is the first intensive classification project in the repository.

It has two tracks:
- **Track A — Classical ML** using `load_digits()` for an offline-safe workflow
- **Track B — PyTorch MNIST** using `torchvision.datasets.MNIST` for Colab and GPU training

## What You Learn Here
- multiclass classification
- accuracy and macro F1
- confusion matrix analysis
- error analysis
- neural network training with PyTorch
- basic experiment tracking with saved artifacts

## Recommended Order
1. run the classical baseline comparison
2. inspect confusion analysis outputs
3. run the PyTorch training track in Colab
4. compare classical vs PyTorch results
5. write down lessons learned

## Quick Start
### Classical track
```bash
python scripts/run_project2_mnist.py --stage classical
```

### PyTorch track
```bash
python scripts/run_project2_mnist.py --stage pytorch --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

### Full Project 2 pipeline
```bash
python scripts/run_project2_mnist.py --stage all --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

## PyTorch Options
You can customize the neural run:

```bash
python scripts/run_project2_mnist.py \
  --stage pytorch \
  --model cnn \
  --epochs 8 \
  --batch-size 128 \
  --lr 0.001 \
  --patience 2 \
  --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

## Expected Classical Outputs
Saved under:
```text
projects/02-mnist-classifier/results/
```

Files:
- `classical_results.json`
- `confusion_matrix.csv`
- `classification_report.txt`
- `top_confusions.csv`
- `misclassified_examples.csv`

## Expected PyTorch Outputs
Saved under the artifact directory you pass in.

Files:
- `best_model.pt`
- `metrics.json`
- `history.csv`
- `test_confusion_matrix.csv`
- `test_classification_report.txt`
- `misclassified_examples.csv`
- `run_summary.md`

## Deliverables
A strong first version of Project 2 should include:
- baseline model comparison
- confusion analysis of the best classical model
- one PyTorch training run
- written comparison of classical vs neural results
- notes on where the model still fails

## Best Supporting Files
- `plan.md`
- `run.md`
- `colab-gpu-guide.md`
- `notes/first-results-template.md`
- `notes/error-analysis-template.md`
- `notes/pytorch-results-template.md`
