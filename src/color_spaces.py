"""Small, explicit color-space conversions used by the pipeline."""

import numpy as np

if __package__:
    from .data_types import Float32Array, RGBPatch, XYZPatch
else:
    from data_types import Float32Array, RGBPatch, XYZPatch


XYZ_TO_LINEAR_RGB = np.array(
    [
        [3.24096994, -1.53738318, -0.49861076],
        [-0.96924364, 1.87596750, 0.04155506],
        [0.05563008, -0.20397696, 1.05697151],
    ],
    dtype=np.float32,
)


def xyz_to_linear_rgb(patch: XYZPatch) -> RGBPatch:
    """Convert CIE XYZ to linear sRGB without clipping."""
    _, height, width = patch.tristimulus.shape
    linear_rgb = np.zeros(
        (3, height, width),
        dtype=np.float32,
    )

    for output_channel in range(3):
        for xyz_channel in range(3):
            linear_rgb[output_channel] += (
                XYZ_TO_LINEAR_RGB[output_channel, xyz_channel]
                * patch.tristimulus[xyz_channel]
            )

    return RGBPatch(
        linear_rgb=linear_rgb,
        scene_id=patch.scene_id,
        row=patch.row,
        col=patch.col,
    )


def fit_lms_to_xyz_matrix(
    lms_sensitivities: Float32Array,
    xyz_sensitivities: Float32Array,
) -> Float32Array:
    """Fit XYZ = matrix @ LMS on the shared spectral grid."""
    coefficients, _, _, _ = np.linalg.lstsq(
        lms_sensitivities.T,
        xyz_sensitivities.T,
        rcond=None,
    )
    return coefficients.T.astype(np.float32)


def cvd_lms_to_linear_rgb(
    patch,
    lms_to_xyz: Float32Array,
) -> RGBPatch:
    """Project CVD LMS responses to linear sRGB for display only."""
    _, height, width = patch.responses.shape
    xyz = np.zeros(
        (3, height, width),
        dtype=np.float32,
    )

    for xyz_channel in range(3):
        for lms_channel in range(3):
            xyz[xyz_channel] += (
                lms_to_xyz[xyz_channel, lms_channel]
                * patch.responses[lms_channel]
            )

    xyz_patch = XYZPatch(
        tristimulus=xyz,
        scene_id=patch.scene_id,
        row=patch.row,
        col=patch.col,
    )
    return xyz_to_linear_rgb(xyz_patch)


def linear_rgb_to_srgb(linear_rgb: Float32Array) -> Float32Array:
    """Encode linear RGB for display; out-of-gamut values are clipped."""
    clipped = np.clip(linear_rgb, 0.0, 1.0)
    encoded = np.zeros_like(clipped, dtype=np.float32)

    low = clipped <= 0.0031308
    encoded[low] = 12.92 * clipped[low]
    encoded[~low] = (
        1.055 * np.power(clipped[~low], 1.0 / 2.4)
        - 0.055
    )

    return encoded.astype(np.float32)
