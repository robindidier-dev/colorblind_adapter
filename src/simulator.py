import numpy as np

if __package__:
    from .data_types import (
        ConeResponsePatch,
        Float32Array,
        SpectralPatch,
        XYZPatch,
    )
else:
    from data_types import (
        ConeResponsePatch,
        Float32Array,
        SpectralPatch,
        XYZPatch,
    )



class SpectralSimulator:
    def __init__(self, wavelengths_nm, cone_sensitivities, xyz_sensitivities=None):
        self.wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)
        self.cone_sensitivities = np.asarray(cone_sensitivities, dtype=np.float32)
        self.xyz_sensitivities = np.asarray(xyz_sensitivities, dtype=np.float32) if xyz_sensitivities is not None else None
        self.weights = np.gradient(self.wavelengths_nm).astype(np.float32)


    # Reusable method
    def _integrate_spectral(self, cube, sensitivities):
        channels = sensitivities.shape[0]
        bands, height, width = cube.shape

        responses = np.zeros(
            (channels, height, width),
            dtype=np.float32,
        )

        for channel in range(channels):
            for band in range(bands):
                responses[channel] += sensitivities[channel, band] * cube[band] * self.weights[band]

        return responses

    # For RGB images
    def spectral_to_xyz(self, patch: SpectralPatch) -> XYZPatch:
        if self.xyz_sensitivities is None:
            raise ValueError(
                "xyz_sensitivities are required for spectral_to_xyz"
            )

        responses = self._integrate_spectral(patch.cube, self.xyz_sensitivities)
        return XYZPatch(tristimulus=responses, scene_id=patch.scene_id, row=patch.row, col=patch.col)

    def spectral_to_lms(self, patch: SpectralPatch) -> ConeResponsePatch:
        return self.spectral_to_cvd_lms(
            patch,
            self.cone_sensitivities,
        )


    # For CB images
    def spectral_to_cvd_lms(self, patch: SpectralPatch, cvd_sensitivities: Float32Array) -> ConeResponsePatch:
        responses = self._integrate_spectral(
            patch.cube,
            cvd_sensitivities,
        )

        return ConeResponsePatch(
            responses=responses,
            scene_id=patch.scene_id,
            row=patch.row,
            col=patch.col,
        )

if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    if __package__:
        from .data_types import (
            CVDCondition,
            SpectralPatch,
        )
        from .physiology import (
            load_cie_lms,
            load_cie_lms_for_cvd,
            load_cie_xyz,
        )
        from .color_spaces import (
            linear_rgb_to_srgb,
            xyz_to_linear_rgb,
        )
    else:
        from data_types import (
            CVDCondition,
            SpectralPatch,
        )
        from physiology import (
            load_cie_lms,
            load_cie_lms_for_cvd,
            load_cie_xyz,
        )
        from color_spaces import (
            linear_rgb_to_srgb,
            xyz_to_linear_rgb,
        )

    project_root = Path(__file__).resolve().parent.parent

    wavelengths_nm = np.arange(
        400,
        701,
        10,
        dtype=np.float32,
    )

    lms_path = (
        project_root
        / "data"
        / "physiology"
        / "CIE_lms_cf_2deg.csv"
    )

    xyz_path = (
        project_root
        / "data"
        / "physiology"
        / "CIE_xyz_1931_2deg.csv"
    )

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
        severity=0.5,
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

    xyz_result = simulator.spectral_to_xyz(
        patch,
    )

    cvd_result = simulator.spectral_to_cvd_lms(
        patch,
        cvd_lms_sensitivities,
    )

    rgb_result = xyz_to_linear_rgb(
        xyz_result,
    )

    display_scale = rgb_result.linear_rgb.max()

    display_rgb = linear_rgb_to_srgb(
        rgb_result.linear_rgb / display_scale,
    )

    print("Patch cube:", patch.cube.shape)
    print(
        "Normal LMS sensitivities:",
        normal_lms_sensitivities.shape,
    )
    print(
        "CVD LMS sensitivities:",
        cvd_lms_sensitivities.shape,
    )
    print(
        "XYZ sensitivities:",
        xyz_sensitivities.shape,
    )

    print(
        "XYZ result:",
        xyz_result.tristimulus.shape,
    )
    print(
        "XYZ all finite:",
        np.isfinite(xyz_result.tristimulus).all(),
    )

    print(
        "CVD LMS result:",
        cvd_result.responses.shape,
    )
    print(
        "CVD LMS all finite:",
        np.isfinite(cvd_result.responses).all(),
    )

    print(
        "Linear RGB result:",
        rgb_result.linear_rgb.shape,
    )
    print(
        "Linear RGB all finite:",
        np.isfinite(rgb_result.linear_rgb).all(),
    )

    print(
        "Display RGB:",
        display_rgb.shape,
    )
    print(
        "Display RGB range:",
        display_rgb.min(),
        display_rgb.max(),
    )

    for name, channel in zip(
        ("X", "Y", "Z"),
        xyz_result.tristimulus,
    ):
        print(
            f"XYZ {name}:",
            "min=", channel.min(),
            "mean=", channel.mean(),
            "max=", channel.max(),
        )

    for name, channel in zip(
        ("L", "M", "S"),
        cvd_result.responses,
    ):
        print(
            f"CVD {name}:",
            "min=", channel.min(),
            "mean=", channel.mean(),
            "max=", channel.max(),
        )

    zero_condition = CVDCondition(
        cvd_type="protan",
        severity=0.0,
    )

    zero_sensitivities = load_cie_lms_for_cvd(
        lms_path,
        wavelengths_nm,
        zero_condition,
    )

    normal_result = simulator.spectral_to_cvd_lms(
        patch,
        normal_lms_sensitivities,
    )

    zero_result = simulator.spectral_to_cvd_lms(
        patch,
        zero_sensitivities,
    )

    print(
        "severity=0 equals normal:",
        np.allclose(
            normal_result.responses,
            zero_result.responses,
        ),
    )
