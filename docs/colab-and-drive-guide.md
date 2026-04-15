# Colab and Drive Guide

## Start Colab

Open a new notebook.

## Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Clone Repo
```python
%cd /content
!git clone https://github.com/cozyGarage/ml-portfolio.git
%cd ml-portfolio
```

## Install Dependencies
```python
!pip install -r requirements-colab.txt
```

## Enable GPU
- Runtime → Change runtime type → GPU

## Run Training
```python
!python scripts/run_train.py
```

## Save Artifacts
```python
ARTIFACT_DIR = "/content/drive/MyDrive/ml-portfolio-artifacts"
```

Save models, metrics, and figures there.

## Principle
- Colab = compute
- Drive = storage
- GitHub = source
