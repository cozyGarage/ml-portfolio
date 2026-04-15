# Machine Learning Portfolio — Usage Guide

This file is the practical usage guide for this repository.

## What this repo is for
- learn machine learning through projects
- keep code, notes, and progress in one place
- use Colab for compute and GitHub for source control

## Main workflow
- GitHub: code and docs
- Colab: training and experiments
- Google Drive: models, metrics, figures, logs
- local machine: editing and Git

## First files to read
1. `docs/START-HERE.md`
2. `docs/setup-guide.md`
3. `docs/colab-and-drive-guide.md`
4. `docs/study-workflow-guide.md`

## Quick setup
```bash
git clone https://github.com/cozyGarage/ml-portfolio.git
cd ml-portfolio
git pull origin main
```

Optional local environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Colab setup
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!git clone https://github.com/cozyGarage/ml-portfolio.git
%cd ml-portfolio
!pip install -r requirements-colab.txt
```

## Project 1
Run the whole Project 1 pipeline with one command:

```bash
python scripts/run_project1_housing.py
```

Or run the steps individually:

```bash
python projects/01-housing-price-predictor/src/baseline_train.py
python projects/01-housing-price-predictor/src/crossval_compare.py
python projects/01-housing-price-predictor/src/tune_random_forest.py
```

In Colab:

```python
!python scripts/run_project1_housing.py
```

## After running
Update:
- `projects/01-housing-price-predictor/notes/first-results-template.md`
- `projects/01-housing-price-predictor/notes/stage-3-results-template.md`

## Current active project
- `projects/01-housing-price-predictor/`

## Next planned project
- MNIST classifier with a clearer Colab-first workflow
