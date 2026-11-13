"""Extract fixed-size hyperspectral patches from ICVL MAT v7.3 scenes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "ICVL" / "mat"
DEFAULT_OUTPUT = ROOT / "data" / "ICVL" / "patches_256"


def starts(length: int, patch_size: int, stride: int) -> list[int]:
    """Return origins for complete patches only."""
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


def extract_scene(mat_path: Path, output_dir: Path, patch_size: int, stride: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scene_id = mat_path.stem

    with h5py.File(mat_path, "r") as source:
        if "rad" not in source or "bands" not in source:
            raise ValueError(f"{mat_path} ne contient pas les variables rad et bands")

        rad = source["rad"]
        bands = np.asarray(source["bands"]).reshape(-1).astype(np.float32)
        if rad.ndim != 3 or rad.shape[0] != len(bands):
            raise ValueError(f"Dimensions inattendues dans {mat_path}: rad={rad.shape}, bands={bands.shape}")

        _, height, width = rad.shape
        row_starts = starts(height, patch_size, stride)
        col_starts = starts(width, patch_size, stride)
        if not row_starts or not col_starts:
            raise ValueError(f"Scène trop petite pour {patch_size}x{patch_size}: {rad.shape}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for row in row_starts:
            for col in col_starts:
                patch_id = f"{scene_id}_r{row:04d}_c{col:04d}"
                output_path = output_dir / f"{patch_id}.npz"
                if not output_path.exists():
                    cube = np.asarray(
                        rad[:, row : row + patch_size, col : col + patch_size],
                        dtype=np.float32,
                    )
                    np.savez_compressed(
                        output_path,
                        hyperspectral=cube,
                        wavelengths_nm=bands,
                        scene_id=scene_id,
                        row=np.int32(row),
                        col=np.int32(col),
                    )

                rows.append(
                    {
                        "patch_id": patch_id,
                        "scene_id": scene_id,
                        "source_file": str(mat_path.relative_to(ROOT)),
                        "path": str(output_path.relative_to(ROOT)),
                        "row": row,
                        "col": col,
                        "height": patch_size,
                        "width": patch_size,
                        "bands": len(bands),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("patch-size et stride doivent être positifs")

    mat_files = sorted(args.input.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"Aucun fichier .mat trouvé dans {args.input}")

    manifest_rows: list[dict[str, object]] = []
    for mat_file in mat_files:
        rows = extract_scene(mat_file, args.output, args.patch_size, args.stride)
        manifest_rows.extend(rows)
        print(f"{mat_file.name}: {len(rows)} patches")

    manifest_path = args.output / "manifest.csv"
    fieldnames = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Total: {len(manifest_rows)} patches")
    print(f"Manifeste: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
