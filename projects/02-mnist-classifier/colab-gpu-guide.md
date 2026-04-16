# Project 2 Colab GPU Guide

## Setup
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/cozyGarage/ml-portfolio.git
%cd ml-portfolio
!pip install -r requirements-colab.txt
```

Enable GPU in Colab:
- Runtime
- Change runtime type
- GPU

## Run Project 2 PyTorch
```python
!python scripts/run_project2_mnist.py --stage pytorch --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```
