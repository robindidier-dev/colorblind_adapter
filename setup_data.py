"""Download and prepare all data required by the project."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
PHYSIOLOGY_DIR = ROOT / "data" / "physiology"

CIE_FILES = {
    "CIE_lms_cf_2deg.csv": (
        "https://files.cie.co.at/"
        "Publications-datasets/CIE_lms_cf_2deg.csv"
    ),
    "CIE_lms_cf_2deg.csv_metadata.json": (
        "https://files.cie.co.at/"
        "Publications-datasets/CIE_lms_cf_2deg.csv_metadata.json"
    ),
    "CIE_xyz_1931_2deg.csv": (
        "https://files.cie.co.at/"
        "Publications-datasets/CIE_xyz_1931_2deg.csv"
    ),
    "CIE_xyz_1931_2deg.csv_metadata.json": (
        "https://files.cie.co.at/"
        "Publications-datasets/CIE_xyz_1931_2deg.csv_metadata.json"
    ),
}


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination.relative_to(ROOT)}")
        return

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    print(f"Downloading: {destination.name}")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(destination)


def main() -> None:
    load_dotenv(ROOT / ".env")

    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is missing. Add it to the .env file at the repository root."
        )

    PHYSIOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in CIE_FILES.items():
        download_file(
            url,
            PHYSIOLOGY_DIR / filename,
        )

    print("Starting ICVL download and patch extraction...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "download_and_prepare_icvl.py"),
        ],
        cwd=ROOT,
        check=True,
    )

    print("Data setup complete.")


if __name__ == "__main__":
    main()
