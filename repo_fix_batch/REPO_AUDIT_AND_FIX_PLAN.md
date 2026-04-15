# Repo Audit and Fix Plan

This batch is designed to fix the main repo usability gaps.

## Main fixes included

### 1. Replace root README
Why:
- the root README should explain how to use the repo, not just describe it
- it should show setup, Colab usage, and exact run commands

Included file:
- `README.md`

### 2. Replace Project 1 README
Why:
- the project README should explain the actual workflow, stages, files, and commands
- the earlier placeholder README is too thin for a user to run the project confidently

Included file:
- `projects/01-housing-price-predictor/README.md`

### 3. Add Project 2 to the repo
Why:
- Project 2 exists as a scaffold bundle but may not yet be present in GitHub
- this adds a serious classification project with both classical ML and PyTorch tracks

Included files:
- `projects/02-mnist-classifier/...`
- `scripts/run_project2_mnist.py`

### 4. Update local dependencies
Why:
- Project 2 PyTorch training uses `torchvision`
- root `requirements.txt` should include it

Included file:
- `requirements.txt`

## Recommended cleanup after applying
- keep `README-USE-THIS.md` temporarily if you want
- once the new `README.md` is committed, you can remove `README-USE-THIS.md` to avoid duplication
