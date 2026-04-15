# How to Apply This Fix Batch

## Option A — copy the files into the repo manually
1. Extract this folder.
2. Copy its contents into the root of `ml-portfolio`.
3. Allow overwrite when asked.

## Option B — use the helper script
Linux/macOS:
```bash
bash apply_fix_batch.sh /path/to/ml-portfolio
```

Windows PowerShell:
```powershell
./apply_fix_batch.ps1 -RepoPath C:\path\to\ml-portfolio
```

## Commit after applying
```bash
git add .
git commit -m "Apply repo usability fixes and add Project 2"
git push origin main
```
