"""
Download the NYC Taxi Trip Duration dataset from Kaggle into data/raw/.

Prerequisites (see docs/SETUP.md):
  1. pip install kaggle
  2. Create a Kaggle API token (Kaggle > Account > Create New API Token) and place
     kaggle.json at:  Windows  %USERPROFILE%\\.kaggle\\kaggle.json
                       macOS/Linux  ~/.kaggle/kaggle.json
  3. Accept the competition rules once at:
     https://www.kaggle.com/competitions/nyc-taxi-trip-duration/rules

Usage:
    python data/download_data.py
"""
import os
import sys
import zipfile
from pathlib import Path

COMPETITION = "nyc-taxi-trip-duration"
RAW_DIR = Path("data/raw")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle package not installed. Run: pip install kaggle")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Kaggle auth failed — check kaggle.json. Details: {e}")

    print(f"Downloading '{COMPETITION}' to {RAW_DIR} ...")
    api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)

    # Unzip everything the API dropped into data/raw/
    for zf in RAW_DIR.glob("*.zip"):
        print(f"Extracting {zf.name} ...")
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW_DIR)
        os.remove(zf)

    print("Done. Files in data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
