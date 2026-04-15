# Setup Guide

## Local Machine

Clone repo:
```bash
git clone https://github.com/cozyGarage/ml-portfolio.git
cd ml-portfolio
```

Pull updates:
```bash
git pull origin main
```

Optional environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Google Drive Setup
Create:

```
MyDrive/ml-portfolio-artifacts/
```

Inside:
- models/
- metrics/
- figures/
- logs/

No .env file needed for Drive in Colab.
