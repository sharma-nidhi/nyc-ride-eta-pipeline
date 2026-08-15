# DVC Setup — Beginner Walkthrough

Goal: version the dataset so any run is reproducible (a Week 1 / M2 deliverable). Each step
says **what**, **why**, and an **alternative**.

## What DVC is (and why we need it)
Git is great for code but terrible for big data files — it would bloat the repo and GitHub
rejects files over 100 MB (our `train.csv` is ~190 MB). **DVC (Data Version Control)** solves
this: it keeps a tiny text "pointer" file (`train.csv.dvc`) in Git, while the actual data lives
in a separate **DVC remote** (a storage location). Result: `git checkout` an old commit and
`dvc pull` gives you the exact dataset that went with it — versioned data that plays nicely with
versioned code.

Prereqs: `dvc` installed (it's in `requirements.txt`), and `data/raw/train.csv` in place.

---

## Step A — Initialize DVC in the repo
```powershell
dvc init
git status -s
```
**Why:** creates a `.dvc/` folder + config that turns on DVC for this project. `dvc init`
stages a few small files; commit them:
```powershell
git commit -m "W1: initialize DVC"
```

## Step B — Tell DVC to track the dataset
```powershell
dvc add data/raw/train.csv
```
**Why:** DVC moves the big file into its own cache and creates two things:
`data/raw/train.csv.dvc` (the tiny pointer, ~2 lines) and a `.gitignore` entry so Git never
tries to store the 190 MB file. You commit the *pointer*, not the data.
**Check:** `dir data\raw` now shows `train.csv` **and** `train.csv.dvc`.

## Step C — Commit the pointer to Git
```powershell
git add data/raw/train.csv.dvc .gitignore
git commit -m "W1: track raw dataset with DVC"
```
**Why:** now Git history records *which version* of the data this commit used (via the pointer's
hash), while the data itself stays out of Git.

## Step D — Configure a DVC remote (where the data is stored)
Simplest option — a local folder remote (good enough to demonstrate the full workflow):
```powershell
dvc remote add -d localremote ../dvc-storage
git add .dvc/config
git commit -m "W1: add DVC remote"
```
**Why:** `-d` makes it the default remote; `../dvc-storage` is a sibling folder DVC will copy
data into on `push`.
**Alternative (truly shared with your teammate):** Google Drive —
`pip install "dvc[gdrive]"` then `dvc remote add -d gdrive gdrive://<folder-id>`. More setup, but
Nidhi can `dvc pull` the same data. (With the local remote, a teammate just re-downloads from
Kaggle — fine for this project since the data is public.)

## Step E — Push the data to the remote
```powershell
dvc push
```
**Why:** uploads the cached data to your remote so it's backed up and (with a shared remote)
available to teammates. `git push` moves code + pointers; `dvc push` moves the data — two
separate stores, on purpose.

## Step F — Tag this dataset version
```powershell
git tag data-v1
git push origin main --tags
```
**Why:** `data-v1` is a named bookmark for "the dataset as of Week 1." If you later change or
expand the data, you tag `data-v2`, and can always return to v1. This is the "dataset version
tagged" milestone.

---

## Everyday use
- After cloning fresh (or on your teammate's machine): `git pull` then `dvc pull` to get both
  code and data.
- To see versions: `git tag` lists dataset tags; `git checkout data-v1` + `dvc checkout` restores
  that snapshot.

## Troubleshooting
- **`dvc: command not found`:** activate the venv, then `pip install dvc`.
- **`train.csv.dvc` got ignored by Git:** ensure `.gitignore` has `!data/raw/*.dvc` (already set).
- **Google Drive auth errors:** re-run the command; a browser opens to authorize DVC once.
