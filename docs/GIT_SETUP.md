# Git Setup — Beginner Walkthrough

Goal: connect this local folder to the GitHub repo and push your scaffold, keeping a clean
history. Each step says **what**, **why**, and an **alternative**.

Mental model: **Git** records snapshots (commits) of your code locally. **GitHub** is the
online copy your team shares. `push` uploads your commits; `pull` downloads teammates'.

---

## Prerequisites (check these first)
- You have a **GitHub account**.
- You have **write access** to `sharma-nidhi/nyc-ride-eta-pipeline`. Because Nidhi owns it,
  she must add you: repo **Settings ▸ Collaborators ▸ Add people ▸ your GitHub username**.
  You then accept the invite from your email. *Without this, the final `push` fails with 403.*

---

## Step A — Tell Git who you are (one-time, whole computer)
```powershell
git config --global user.name "Ronak Shah"
git config --global user.email "you@github-email.com"
```
**Why:** every commit is stamped with a name + email. Use the **email tied to your GitHub
account** so your commits are attributed to you on GitHub.
**Alternative:** drop `--global` to set identity for just this one repo.

## Step B — Start version control in the folder
```powershell
git init
```
**Why:** creates a hidden `.git` folder that turns this directory into a Git repository (starts
tracking history). Run it from the project root (the folder with `README.md`).

## Step C — Point Git at the GitHub repo
```powershell
git remote add origin https://github.com/sharma-nidhi/nyc-ride-eta-pipeline.git
```
**Why:** `origin` is a nickname for the GitHub URL, so later you just type `origin` instead of
the full link.
**Check:** `git remote -v` should print the URL twice (fetch + push).

## Step D — Adopt the repo's existing history
```powershell
git fetch origin
git reset origin/main
```
**Why:** the repo already has one commit (a README stub). `fetch` downloads it; `git reset
origin/main` makes your local branch start from that commit **without touching your files**.
This lets your better scaffold become the *next* commit on top — no messy merge conflict.
**Check:** `git status` now lists `README.md` as *modified* and everything else as *untracked* —
that's expected. **Paste this output to your guide before continuing.**
**Alternative (standard but conflict-prone):** commit first, then
`git pull origin main --allow-unrelated-histories` and manually resolve the README conflict.

## Step E — Stage and commit your scaffold
```powershell
git add .
git status
git commit -m "Scaffold pipeline structure, docs, and config"
```
**Why:** `add .` stages all changes; `status` lets you review exactly what's going in *before*
you commit; `commit` saves the snapshot locally with a message. `venv/`, data, and models are
skipped automatically because of `.gitignore`.
**Alternative:** use the **Source Control** panel in VS Code (`Ctrl+Shift+G`) — type a message,
click ✓ to commit. Same result, click-based.

## Step F — Name the branch and push
```powershell
git branch -M main
git push -u origin main
```
**Why:** `branch -M main` ensures your main line is called `main` (GitHub's default). `push -u`
uploads it and links local `main` to `origin/main`, so future pushes are just `git push`.
**Auth:** the first push opens a **browser window to sign in to GitHub** (Git Credential Manager,
installed with Git for Windows). Approve it once and it's remembered.
**Alternative:** SSH keys or a Personal Access Token — more setup; the browser sign-in is easiest.

---

## Everyday workflow after this (per the rubric: commit weekly, incrementally)
```powershell
git pull            # get teammate's latest first
# ... do your work ...
git add .
git commit -m "W1: data validation checks"
git push
```
Use short, meaningful messages tagged by week (e.g. `W1:`, `W2:`). Consider a branch per feature
(`git checkout -b feature/validation`) and Pull Requests for review — good practice, and it
shows collaboration in the history.

## If something goes wrong
- **`git reset origin/main` errors:** paste the message to your guide. Fallback is the
  commit-then-pull alternative in Step D.
- **push says 403 / permission denied:** you're not a collaborator yet (see Prerequisites).
- **push says "rejected / non-fast-forward":** run `git pull` first, then push.
