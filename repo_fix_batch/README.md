# Machine Learning Portfolio

This repository is an umbrella portfolio for learning machine learning through project-based practice using **Scikit-Learn**, **PyTorch**, and related tools.

It is structured as a long-term learning system rather than a single project. The workflow is designed so you can study locally, train in **Google Colab**, store large outputs in **Google Drive**, and keep code and documentation in **GitHub**.

---

## What This Repo Is For

Use this repository to:
- learn machine learning by building projects
- practice end-to-end workflows: data loading, preprocessing, training, evaluation, and documentation
- build a public portfolio over time
- keep a repeatable study and experiment system

---

## How This Repo Works

This repo is built around a simple split:

- **GitHub** → source code, docs, project history
- **Google Colab** → GPU training and heavier experiments
- **Google Drive** → persistent artifact storage for models, metrics, figures, and logs
- **Local machine** → editing, Git, reading, light debugging
- **GitHub Actions** → lightweight CI checks

---

## Start Here

Read these files in order:

1. `docs/START-HERE.md`
2. `docs/setup-guide.md`
3. `docs/colab-and-drive-guide.md`
4. `docs/study-workflow-guide.md`
5. `docs/project-readme-guide.md`

These are the repo-level usage guides.

---

## Current Repo Structure

```text
ml-portfolio/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-colab.txt
├── docs/
├── projects/
├── src/
├── scripts/
└── .github/
```

Important folders:
- `docs/` → study and usage guides
- `projects/` → individual portfolio projects
- `src/` → reusable code and CLI pieces
- `scripts/` → run scripts

---

## Quick Setup

### Clone the repo

```bash
git clone https://github.com/cozyGarage/ml-portfolio.git
cd ml-portfolio
```

### Pull latest changes

```bash
git pull origin main
```

### Optional local environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Colab Workflow

Use Colab when you want internet access, notebook convenience, or GPU.

### In Colab

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

Enable GPU in Colab:
- Runtime
- Change runtime type
- Hardware accelerator
- GPU

---

## Google Drive Artifact Folder

Create this folder in Drive:

```text
MyDrive/ml-portfolio-artifacts/
```

Recommended subfolders:

```text
models/
metrics/
figures/
logs/
```

Use Drive for large outputs instead of committing them to GitHub.

---

## How To Run the Repo

### Current root-level entry points

CLI starter:

```bash
python -m src.ml_portfolio.cli train
python -m src.ml_portfolio.cli evaluate
python -m src.ml_portfolio.cli init-project
```

Training starter:

```bash
python scripts/run_train.py
```

In Colab:

```python
!python scripts/run_train.py
```

---

## Project 1: Housing Price Predictor

This is the first active project in the portfolio.

Location:

```text
projects/01-housing-price-predictor/
```

### What is included
- baseline training script
- cross-validation comparison script
- random forest tuning script
- run guides
- notes templates

### Run Project 1 locally

```bash
python scripts/run_project1_housing.py
```

Or run the steps individually:

```bash
python projects/01-housing-price-predictor/src/baseline_train.py
python projects/01-housing-price-predictor/src/crossval_compare.py
python projects/01-housing-price-predictor/src/tune_random_forest.py
```

### Run Project 1 in Colab

```python
!python scripts/run_project1_housing.py
```

---

## Project 2: MNIST / Digits Classifier

This is the next intensive classification project.

Location:

```text
projects/02-mnist-classifier/
```

### What is included
- classical sklearn classification pipeline
- confusion matrix and error analysis step
- PyTorch MNIST training for Colab / GPU
- one-command project runner

### Run Project 2 locally

```bash
python scripts/run_project2_mnist.py --stage classical
```

### Run Project 2 in Colab

```python
!python scripts/run_project2_mnist.py --stage all --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

---

## What To Commit

Commit to GitHub:
- source code
- docs
- project READMEs
- configs
- lightweight notebooks
- result summaries

Do not commit:
- large model files
- large datasets
- temporary checkpoints
- huge logs

---

## Recommended Study Loop

For each project:
1. read the relevant concept
2. explore in a notebook
3. move stable code into `src/`
4. run training
5. save large outputs to Drive
6. update README and notes
7. commit progress

Example commit flow:

```bash
git add .
git commit -m "Update project progress"
git push origin main
```

---

## Best Entry Point Right Now

If you are new to this repo, start here:

1. `docs/START-HERE.md`
2. `projects/01-housing-price-predictor/plan.md`
3. `projects/02-mnist-classifier/README.md`

That gives you the fastest path to doing real work in the repo.
