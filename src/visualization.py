"""Simple diagnostic visualizations for spectral and CVD responses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .color_spaces import linear_rgb_to_srgb
else:
    from color_spaces import linear_rgb_to_srgb


def _rgb_for_display(linear_rgb: np.ndarray) -> np.ndarray:
    """Normalize one linear RGB patch for a readable diagnostic display."""
    scale = float(np.max(linear_rgb))
    if scale <= 0.0:
        raise ValueError("linear_rgb must contain a positive value")

    encoded = linear_rgb_to_srgb(
        linear_rgb / scale,
    )

    return np.moveaxis(encoded, 0, -1)


def _show_scalar(ax, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def show_patch_comparison(
    normal_linear_rgb: np.ndarray,
    cvd_linear_rgb: np.ndarray,
    wavelengths_nm: np.ndarray,
    normal_sensitivities: np.ndarray,
    cvd_sensitivities: np.ndarray,
    cvd_label: str,
) -> None:
    """Display normal RGB, projected CVD RGB, and LMS fundamentals."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(_rgb_for_display(normal_linear_rgb))
    axes[0, 0].set_title("Normal RGB display")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_rgb_for_display(cvd_linear_rgb))
    axes[0, 1].set_title(f"{cvd_label} display projection")
    axes[0, 1].axis("off")

    channel_names = ("L", "M", "S")
    for index, name in enumerate(channel_names):
        axes[1, 0].plot(
            wavelengths_nm,
            normal_sensitivities[index],
            label=f"{name} normal",
        )
        axes[1, 1].plot(
            wavelengths_nm,
            cvd_sensitivities[index],
            label=f"{name} CVD",
        )

    axes[1, 0].set_title("Normal cone fundamentals")
    axes[1, 1].set_title(f"{cvd_label} cone fundamentals")

    for axis in axes[1]:
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Sensitivity")
        axis.grid(True, alpha=0.25)
        axis.legend()

    figure.suptitle(f"Spectral CVD diagnostic: {cvd_label}")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    if __package__:
        from .color_spaces import (
            cvd_lms_to_linear_rgb,
            fit_lms_to_xyz_matrix,
            xyz_to_linear_rgb,
        )
        from .data_types import CVDCondition, SpectralPatch
        from .physiology import (
            load_cie_lms,
            load_cie_lms_for_cvd,
            load_cie_xyz,
        )
        from .simulator import SpectralSimulator
    else:
        from color_spaces import (
            cvd_lms_to_linear_rgb,
            fit_lms_to_xyz_matrix,
            xyz_to_linear_rgb,
        )
        from data_types import CVDCondition, SpectralPatch
        from physiology import (
            load_cie_lms,
            load_cie_lms_for_cvd,
            load_cie_xyz,
        )
        from simulator import SpectralSimulator

    project_root = Path(__file__).resolve().parent.parent
    wavelengths_nm = np.arange(400, 701, 10, dtype=np.float32)

    lms_path = project_root / "data" / "physiology" / "CIE_lms_cf_2deg.csv"
    xyz_path = project_root / "data" / "physiology" / "CIE_xyz_1931_2deg.csv"
    patch_path = (
        project_root
        / "data"
        / "ICVL"
        / "patches_256"
        / "rsh_0406-1441-1_r0000_c0000.npz"
    )

    normal_lms_sensitivities = load_cie_lms(
        lms_path,
        wavelengths_nm,
    )
    xyz_sensitivities = load_cie_xyz(
        xyz_path,
        wavelengths_nm,
    )

    condition = CVDCondition(
        cvd_type="protan",
        severity=1.0,
    )
    cvd_lms_sensitivities = load_cie_lms_for_cvd(
        lms_path,
        wavelengths_nm,
        condition,
    )

    with np.load(patch_path) as data:
        patch = SpectralPatch(
            cube=data["hyperspectral"],
            wavelengths_nm=data["wavelengths_nm"],
            scene_id=str(data["scene_id"].item()),
            row=int(data["row"].item()),
            col=int(data["col"].item()),
        )

    simulator = SpectralSimulator(
        wavelengths_nm=wavelengths_nm,
        cone_sensitivities=normal_lms_sensitivities,
        xyz_sensitivities=xyz_sensitivities,
    )

    xyz_result = simulator.spectral_to_xyz(patch)
    normal_rgb = xyz_to_linear_rgb(xyz_result)
    normal_lms = simulator.spectral_to_lms(patch)
    cvd_lms = simulator.spectral_to_cvd_lms(
        patch,
        cvd_lms_sensitivities,
    )

    lms_to_xyz = fit_lms_to_xyz_matrix(
        normal_lms_sensitivities,
        xyz_sensitivities,
    )

    cvd_rgb = cvd_lms_to_linear_rgb(
        cvd_lms,
        lms_to_xyz,
    )

    show_patch_comparison(
        normal_linear_rgb=normal_rgb.linear_rgb,
        cvd_linear_rgb=cvd_rgb.linear_rgb,
        wavelengths_nm=wavelengths_nm,
        normal_sensitivities=normal_lms_sensitivities,
        cvd_sensitivities=cvd_lms_sensitivities,
        cvd_label=f"{condition.cvd_type}, severity={condition.severity}",
    )
