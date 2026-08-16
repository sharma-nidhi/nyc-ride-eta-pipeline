# VS Code Setup — Beginner Walkthrough (Windows)

Follow this top to bottom. Each step says **what** to do, **why** it matters, and an
**alternative** where one exists. Don't rush — finish one phase, confirm it worked, then move on.

The three tools we set up, and what each is for:
- **VS Code** — the editor where you'll write and run code.
- **Python** — the language the whole pipeline is written in.
- **Git** — version control; tracks your code history and pushes to GitHub (rubric checks this).

---

## Phase 1 — Install & verify the three tools

### 1.1 VS Code
**What:** Install from https://code.visualstudio.com (the "User Installer" for Windows).
During install, tick **"Add to PATH"** and **"Open with Code"** boxes.
**Why:** VS Code is a lightweight editor that combines a code editor, a terminal, Git, and a
debugger in one window — exactly what an engineering project needs.
**Alternative:** PyCharm Community (heavier, Python-focused). We use VS Code because it's free,
light, and matches the tools in your course.

### 1.2 Python
**What:** Install Python **3.10 or newer** from https://www.python.org/downloads/windows/.
On the **first installer screen, tick "Add python.exe to PATH"** before clicking Install.
**Why:** "Add to PATH" lets you type `python` in any terminal. Without it, the terminal won't
find Python and you'll get "python is not recognized".
**Verify** — open a fresh terminal (see Phase 2) and run:
```powershell
python --version
```
You want `Python 3.10.x` or higher. If it says 3.9 or lower, install a newer version.
**Note:** if `python` doesn't work, try `py --version` — Windows also ships a `py` launcher.
**Alternative:** Anaconda (bundles Python + many libraries). We're **not** using it because your
course tutorials use plain `venv` + `pip`, and mixing conda in adds confusing extra commands.

### 1.3 Git
**What:** Install from https://git-scm.com/download/win. Accept the defaults.
**Why:** Git records your work as a history of commits and pushes it to your teammate's GitHub
repo. The rubric explicitly rewards a weekly commit history (not one last-day upload).
**Verify:**
```powershell
git --version
```
**Alternative:** GitHub Desktop (a click-based Git app). Fine later, but learning the handful of
git commands in `SETUP.md` is worth it for this course.

---

## Phase 2 — Open the project in VS Code

### 2.1 Open the folder
**What:** VS Code → **File ▸ Open Folder…** → select:
`C:\Ronak\Investment\Claude - AIML\ML Engineering\Mini Project\nyc-ride-eta-pipeline`
**Why:** VS Code works on a "workspace" = one project folder. Opening the repo root means the
Explorer, terminal, and Git all point at the right place.
When prompted **"Do you trust the authors?"**, click **Yes** (it's your own folder).

### 2.2 Learn three UI spots you'll use constantly
- **Explorer** (top-left icon, or `Ctrl+Shift+E`): the file tree.
- **Integrated Terminal** (`Ctrl+` backtick, or menu **Terminal ▸ New Terminal**): where you run
  commands. It opens already inside the project folder.
- **Command Palette** (`Ctrl+Shift+P`): a search box for every VS Code action. We'll use it to
  pick the Python interpreter.

---

## Phase 3 — Install VS Code extensions

**What:** Click the Extensions icon (`Ctrl+Shift+X`), search and Install:
1. **Python** (by Microsoft) — this also pulls in **Pylance** (autocomplete/type hints).
2. **Jupyter** (by Microsoft) — run notebooks inside VS Code for EDA.

Optional but useful later:
3. **Docker** (Microsoft) — build/run the Week-3 container from the UI.
4. **GitLens** — nicer view of Git history.

**Why:** The Python extension is what makes VS Code understand Python — running files, debugging,
selecting the venv, linting. Without it, VS Code is just a text editor.
**Alternative:** You can run everything from the terminal without extensions, but you'd lose the
"Run" button, debugging, and interpreter selection — not worth it.

---

## Phase 4 — Create the virtual environment (venv)

### 4.1 Why a venv at all
A **virtual environment** is a private copy of Python + libraries that lives inside this project
(the `venv/` folder). Your course's first rule is *"never use system Python for ML work."*
Reasons: it keeps this project's exact library versions separate from other projects, makes the
setup reproducible for your teammate, and means a mistake here never breaks your whole computer.

### 4.2 Create it
Open the terminal (`Ctrl+` backtick) and run:
```powershell
python -m venv venv
```
**Why:** `-m venv venv` tells Python to build a virtual environment in a new folder named `venv`.
You'll see a `venv/` appear in the Explorer. (It's git-ignored — it never gets committed.)

### 4.3 Activate it
```powershell
.\venv\Scripts\Activate.ps1
```
When active, your terminal prompt shows `(venv)` at the start. **This is the signal that any
`pip install` or `python` command now uses the project's private Python.**

**Common Windows snag:** if you get *"running scripts is disabled on this system"*, PowerShell is
blocking the activation script. Fix it once with:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then run the `Activate.ps1` line again. (This only allows locally-created scripts to run — it's safe.)
**Alternatives:** click the dropdown next to the terminal `+` and choose **Command Prompt**, then
run `venv\Scripts\activate.bat`; or just do Phase 4.4 — after you select the interpreter, VS Code
auto-activates the venv in every new terminal for you.

### 4.4 Tell VS Code to use this venv
Press `Ctrl+Shift+P` → type **"Python: Select Interpreter"** → choose the one that shows
`.\venv\Scripts\python.exe` (usually labeled *Recommended*).
**Why:** this makes the Run button, the debugger, and new terminals all use your venv instead of
system Python.

**Beginner-friendly alternative to all of Phase 4:** `Ctrl+Shift+P` →
**"Python: Create Environment"** → **Venv** → pick your Python → tick **requirements.txt**.
VS Code then creates the venv, selects it, *and* installs dependencies (Phase 5) in one go.

---

## Phase 5 — Install the project's libraries

With `(venv)` showing in the terminal:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**Why:** `requirements.txt` lists every library the pipeline needs (pandas, scikit-learn,
FastAPI, MLflow, DVC, etc.) with versions. Installing from it means you and your teammate get the
*same* toolset — no "works on my machine" surprises. This takes a few minutes.
**Alternative:** install packages one by one (`pip install pandas` …). Don't — the file exists so
the environment is reproducible and reviewable.

---

## Phase 6 — Verify the environment works

```powershell
python --version
python -c "import pandas, sklearn, fastapi, mlflow, dvc; print('All core libraries import OK')"
```
If you see **"All core libraries import OK"**, your environment is ready.

---

## What's next (separate steps, in SETUP.md)
Environment done ✅. The remaining setup is:
1. **Git** — connect this folder to the GitHub repo and push (`docs/SETUP.md` §2).
2. **Kaggle** — API token + download the dataset (`docs/SETUP.md` §3).
3. **DVC** — version the dataset and tag `data-v1` (`docs/SETUP.md` §4).

We'll do those one at a time after the environment is confirmed working.
