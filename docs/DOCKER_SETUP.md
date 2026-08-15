# Docker Setup — Beginner Walkthrough (Windows)

Goal: package the ETA API into a **container** — a self-contained box holding the app + Python +
all its libraries — so it runs identically on any machine, no "works on my laptop" surprises.
That's what makes the model a deployable product instead of a script.

Key terms: a **Dockerfile** is the recipe; building it produces an **image** (a frozen snapshot);
running an image gives you a **container** (a live instance).

---

## Phase 1 — Install Docker Desktop
1. Download **Docker Desktop for Windows**: https://www.docker.com/products/docker-desktop/
2. Run the installer. Keep **"Use WSL 2 instead of Hyper-V"** checked (WSL2 = the lightweight
   Linux layer Docker runs on). Admin rights required.
3. **Restart** your PC if it asks.
4. If it says WSL2 isn't installed: open **PowerShell as Administrator** and run `wsl --install`,
   then restart and reopen Docker Desktop.
5. Launch **Docker Desktop** and wait until the whale icon in the system tray says
   **"Docker Desktop is running"** (bottom-left status is green).
6. Verify in a terminal:
   ```powershell
   docker --version
   docker run hello-world
   ```
   `hello-world` should download and print a "Hello from Docker!" message — that confirms the
   whole stack works.

*Alternative if you can't install Docker:* skip it — the running `uvicorn` API + `sample_requests.py`
already satisfy "working REST API + sample calls." The Dockerfile stands as your packaging design.

---

## Phase 2 — Build the image
From the repo root (where the `Dockerfile` is):
```powershell
docker build -t nyc-eta-api .
```
**Why:** `-t nyc-eta-api` names the image; the `.` is the build context (this folder). Docker reads
the `Dockerfile`, installs `requirements-serve.txt`, and copies in `serving/`, `features/`,
`models/`, `config/`. The `.dockerignore` keeps `venv/`, `data/`, and `mlruns/` out so the build
is fast and the image small. First build takes a few minutes (downloading the Python base image).

## Phase 3 — Run the container
```powershell
docker run --rm -p 8000:8000 nyc-eta-api
```
**Why:** `-p 8000:8000` maps your PC's port 8000 to the container's port 8000, so you can reach the
API at `http://127.0.0.1:8000`. `--rm` auto-removes the container when you stop it (Ctrl+C).
You should see the same uvicorn "Application startup complete" as before — but now it's running
*inside the container*.

## Phase 4 — Test it (same as the local API)
In another terminal:
```powershell
python serving/sample_requests.py
```
Or open **http://127.0.0.1:8000/docs**. Same responses as the local run = the container works.

---

## Basic latency/throughput note (for the report)
Time a request to show awareness (rubric mentions this):
```powershell
curl.exe -w "\nTime: %{time_total}s\n" -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"pickup_datetime\":\"2016-03-14T17:24:00\",\"pickup_longitude\":-73.9857,\"pickup_latitude\":40.7484,\"dropoff_longitude\":-73.9850,\"dropoff_latitude\":40.7580,\"passenger_count\":1}"
```

## Troubleshooting
- **`docker: command not found`:** Docker Desktop isn't running, or the terminal predates the
  install — restart the terminal / start Docker Desktop.
- **Build fails on `pip install`:** check `requirements-serve.txt` for typos; ensure internet access.
- **Port 8000 already in use:** stop your local `uvicorn` first, or map a different port
  (`-p 8001:8000` then use port 8001).
- **Model load error in container:** usually a scikit-learn version mismatch — pin `scikit-learn`
  in `requirements-serve.txt` to the version in your venv (`pip show scikit-learn`).
