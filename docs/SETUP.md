# Setup & Connections

Run these **on your machine** (git/DVC/Kaggle need real local access). Commands shown for
Windows PowerShell — swap `venv\Scripts\activate` for `source venv/bin/activate` on macOS/Linux.

## 0. Prerequisites
- Python 3.10+  (`python --version`; if < 3.10, install from python.org and re-check)
- Git  (`git --version`)
- A Kaggle account
- (Optional) Docker Desktop for Week 3 containerization

## 1. Python environment
```powershell
cd "C:\Ronak\Investment\Claude - AIML\ML Engineering\Mini Project\nyc-ride-eta-pipeline"
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Connect the Git remote & push the scaffold
The remote already has one commit (a README stub). We adopt its history with `git reset`
(no merge conflict), then commit our scaffold on top:
```powershell
# one-time identity — use the email on your GitHub account
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

git init
git remote add origin https://github.com/sharma-nidhi/nyc-ride-eta-pipeline.git
git fetch origin
git reset origin/main          # adopt remote history, keep our files as pending changes
git add .
git status                     # review what will be committed
git commit -m "Scaffold pipeline structure, docs, and config"
git branch -M main
git push -u origin main        # first push opens a browser to sign in to GitHub
```
**Access:** you must have write access to the repo — ask Nidhi to add you as a collaborator
(repo **Settings ▸ Collaborators**) and accept the email invite, or the push returns a 403.
Keep commits incremental and weekly (the rubric checks commit history — not a single
last-day upload). Full explanation in `docs/GIT_SETUP.md`.

## 3. Kaggle API token (for the dataset)
1. Kaggle → your avatar → **Settings** → **API** → **Create New API Token**. This downloads `kaggle.json`.
2. Put it where the CLI looks:
   ```powershell
   mkdir $env:USERPROFILE\.kaggle -Force
   move $HOME\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
   ```
3. **Accept the competition rules once** (required or the API returns 403):
   https://www.kaggle.com/competitions/nyc-taxi-trip-duration/rules
4. Download:
   ```powershell
   python data/download_data.py       # lands train.csv / test.csv in data/raw/
   ```
   *Fallback if you can't use Kaggle:* we generate a synthetic trip dataset with the same
   columns so the pipeline still runs end-to-end.

## 4. DVC — version the dataset (Week 1 deliverable)
```powershell
dvc init
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc data/.gitignore .dvc/config
git commit -m "Track raw dataset with DVC"
# choose a remote (simplest = a local folder; or Google Drive / S3):
dvc remote add -d localremote ../dvc-storage
dvc push
git tag data-v1
git push origin main --tags
```

## 5. MLflow (Week 2)
No server needed — runs log to `./mlruns` by default.
```powershell
python training/train.py
mlflow ui            # open http://127.0.0.1:5000, select runs, Compare
```

## 6. Serve (Week 3) & monitor (Week 4)
```powershell
uvicorn serving.api:app --reload --port 8000   # http://127.0.0.1:8000/docs
python monitoring/check_drift.py
```

## Troubleshooting
- **Kaggle 403 / 401:** token missing/expired, or competition rules not accepted.
- **`kaggle` not found:** activate the venv, then `pip install kaggle`.
- **DVC push to Google Drive:** `pip install "dvc[gdrive]"` and use a `gdrive://<folder-id>` remote.
