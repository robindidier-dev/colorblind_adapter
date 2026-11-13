"""Download a deterministic ICVL subset and extract 256x256 patches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from prepare_icvl_patches import extract_scene


ROOT = Path(__file__).resolve().parent
REPO_ID = "ICVL-BGU/ICVL_HS_2016"
DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "ICVL" / "scenes_60"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "ICVL" / "patches_256"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("count doit être positif")

    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN absent du fichier .env")

    api = HfApi(token=token)
    files = sorted(
        file_name
        for file_name in api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        if file_name.startswith("mat/") and file_name.endswith(".mat")
    )
    selected = files[: args.count]
    if len(selected) < args.count:
        raise RuntimeError(f"Seulement {len(selected)} scènes ICVL disponibles")

    print(f"Scènes sélectionnées: {len(selected)}")
    downloaded: list[Path] = []
    for index, remote_file in enumerate(selected, start=1):
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_file,
            token=token,
            local_dir=args.download_dir,
        )
        path = Path(local_path)
        downloaded.append(path)
        print(f"[{index:02d}/{len(selected)}] {path.name}")

    manifest_rows = []
    for path in downloaded:
        rows = extract_scene(path, args.output_dir, args.patch_size, args.stride)
        manifest_rows.extend(rows)
        print(f"  patches: {path.stem} -> {len(rows)}")

    manifest_path = args.output_dir / "manifest.csv"
    import csv

    fieldnames = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Total patches: {len(manifest_rows)}")
    print(f"Manifeste: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
