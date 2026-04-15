# Run Guide

## Classical path
Run the baseline comparison and confusion analysis:

```bash
python scripts/run_project2_mnist.py --stage classical
```

## PyTorch path
Run the PyTorch training stage:

```bash
python scripts/run_project2_mnist.py --stage pytorch --artifact-dir ./artifacts/project2
```

## Everything
Run both classical and PyTorch stages:

```bash
python scripts/run_project2_mnist.py --stage all --artifact-dir ./artifacts/project2
```

## Colab recommendation
Use Colab for the PyTorch stage, especially if you want GPU.
