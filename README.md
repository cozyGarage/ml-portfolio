# Machine Learning Portfolio

This repository is a project-based machine learning learning system built around:
- **GitHub** for code and documentation
- **Google Colab** for heavier training and GPU use
- **Google Drive** for saving large artifacts such as models, metrics, figures, and logs
- **Local development** for editing, Git, and smaller experiments

The goal is to make the repo easy to use even if someone opens it for the first time.

---

## Start Here

Read these files first:
1. `docs/START-HERE.md`
2. `docs/setup-guide.md`
3. `docs/colab-and-drive-guide.md`
4. `docs/study-workflow-guide.md`
5. `docs/project-readme-guide.md`

---

## Quick Setup

### Clone the repository
```bash
git clone https://github.com/cozyGarage/ml-portfolio.git
cd ml-portfolio
git pull origin main
```

### Optional local Python environment
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

## Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/cozyGarage/ml-portfolio.git
%cd ml-portfolio
!pip install -r requirements-colab.txt
```

Recommended Drive folder:

```text
MyDrive/ml-portfolio-artifacts/
├── project1/
├── project2/
└── project3/
```

---

## Current Active Projects

### Project 1 — Housing Price Predictor
Regression workflow with tabular data.

One-command run:
```bash
python scripts/run_project1_housing.py
```

Useful files:
- `projects/01-housing-price-predictor/README.md`
- `projects/01-housing-price-predictor/stage-3-guide.md`
- `projects/01-housing-price-predictor/stage-4-guide.md`

### Project 2 — MNIST / Digits Classifier
Intensive classification project with a classical track and a PyTorch track.

Classical track:
```bash
python scripts/run_project2_mnist.py --stage classical
```

PyTorch / Colab track:
```bash
python scripts/run_project2_mnist.py --stage pytorch --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

Everything:
```bash
python scripts/run_project2_mnist.py --stage all --artifact-dir /content/drive/MyDrive/ml-portfolio-artifacts/project2
```

### Project 3 — Model Benchmark Tool
Systematic model comparison across datasets and metrics.

Run:
```bash
python scripts/run_project3_benchmark.py --dataset wine
```

---

## Recommended Study Loop

For each project:
1. read the concept
2. run the project or stage
3. inspect outputs
4. write notes and interpretation
5. commit progress

Example:
```bash
git add .
git commit -m "Update project progress"
git push origin main
```

---

## What to Commit
Commit:
- code
- docs
- notes
- small results summaries
- lightweight notebooks

Do not commit:
- large model files
- large datasets
- huge logs
- temporary checkpoints

Store large outputs in Google Drive.

---

## Best Entry Point Right Now
If you are starting fresh:
1. run Project 1 once
2. run Project 2 classical path
3. run Project 2 PyTorch path in Colab
4. use Project 3 to compare models more systematically
