"""Download one ICVL hyperspectral scene for a local access test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
REPO_ID = "ICVL-BGU/ICVL_HS_2016"
DEFAULT_FILE = "mat/rsh_0406-1441-1.mat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT_FILE, help="Path inside the HF dataset")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN absent du fichier .env")

    output_dir = ROOT / "data" / "ICVL" / "mat"
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=args.file,
        token=token,
        local_dir=output_dir,
    )
    print(f"Téléchargé: {Path(local_path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
