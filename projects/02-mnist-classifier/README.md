# Project 2 — MNIST / Digits Classifier

This project is a deeper classification project than Project 1.

It has two tracks:

## Track A — Classical ML (offline-safe)
Uses `sklearn.datasets.load_digits()` so it can run without downloading data.

Goals:
- understand multiclass classification
- compare multiple baseline models
- compute accuracy and macro F1
- inspect the confusion matrix
- perform error analysis

## Track B — PyTorch + Colab GPU
Uses `torchvision.datasets.MNIST` and is intended for Google Colab.

Goals:
- build a real neural classifier
- train on GPU when available
- save model and metrics artifacts
- compare classical vs neural approaches

## Recommended Order
1. run the classical baseline
2. run confusion/error analysis
3. run the PyTorch MNIST training in Colab
4. document results and lessons learned

## Project Structure
```text
projects/02-mnist-classifier/
├── README.md
├── plan.md
├── run.md
├── colab-gpu-guide.md
├── src/
│   ├── baseline_digits_sklearn.py
│   ├── confusion_analysis.py
│   └── pytorch_mnist_train.py
├── notes/
│   ├── first-results-template.md
│   └── error-analysis-template.md
└── results/
    └── README.md
```

## Quick Start
### Classical path
```bash
python scripts/run_project2_mnist.py --stage classical
```

### PyTorch path in Colab
```bash
python scripts/run_project2_mnist.py --stage pytorch --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

## Metrics to care about
- Accuracy
- Macro F1
- Confusion matrix
- Misclassified digits

## Deliverables
A strong first version of this project should include:
- baseline model comparison
- confusion matrix analysis
- one PyTorch training run
- written summary in notes
