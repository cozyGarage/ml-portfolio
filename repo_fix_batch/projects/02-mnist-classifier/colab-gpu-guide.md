# Colab GPU Guide

Use this for the PyTorch MNIST stage.

## Colab setup
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf ml-portfolio
!git clone https://github.com/cozyGarage/ml-portfolio.git
%cd ml-portfolio
!pip install -r requirements-colab.txt
```

Enable GPU in Colab:
- Runtime
- Change runtime type
- GPU

## Run Project 2 PyTorch stage
```python
!python scripts/run_project2_mnist.py --stage pytorch --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

## Expected outputs
Inside the artifact directory:
- `model.pt`
- `metrics.json`
