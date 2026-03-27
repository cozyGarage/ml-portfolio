# Colab Workflow

This repository uses a Colab-first workflow for GPU-based learning and training.

## Option B: GitHub + Colab + Google Drive

### In Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content
!git clone https://github.com/cozyGarage/ml-portfolio.git
%cd ml-portfolio
!pip install -r requirements-colab.txt
```

## Save artifacts
Use a persistent folder in Google Drive, for example:

```python
ARTIFACT_DIR = "/content/drive/MyDrive/ml-portfolio-artifacts"
```

Recommended subfolders:
- models/
- metrics/
- figures/
- logs/

## Why this workflow
- GitHub stores source code
- Colab provides GPU access
- Google Drive stores persistent training outputs

## Notes
- No .env file is required for Google Drive in Colab
- Hugging Face or Kaggle can be added later if needed
