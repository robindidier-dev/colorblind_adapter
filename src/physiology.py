if __package__:
    from .data_types import CVDCondition
else:
    from data_types import CVDCondition

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]

MAX_ANOMALY_SHIFT_NM = 20.0

def load_cie_lms(path, target_wavelengths_nm):
    table = np.loadtxt(path, delimiter=",")
    source_wavelengths = table[:, 0]
    source_lms = table[:, 1:].T

    interp = []

    for curve in source_lms:
        valid = np.isfinite(curve) #avoid NaN

        interp_curve = np.interp(
            target_wavelengths_nm,
            source_wavelengths[valid],
            curve[valid],
            left=0.0,
            right=0.0)

        interp.append(interp_curve)

    lms = np.stack(interp, axis=0).astype(np.float32)
    return lms


def load_cie_xyz(path, target_wavelengths_nm):
    table = np.loadtxt(path, delimiter=",")
    source_wavelengths = table[:, 0]
    source_xyz = table[:, 1:].T

    interp = []

    for curve in source_xyz:
        valid = np.isfinite(curve) #avoid NaN

        interp_curve = np.interp(
            target_wavelengths_nm,
            source_wavelengths[valid],
            curve[valid],
            left=0.0,
            right=0.0)

        interp.append(interp_curve)

    xyz = np.stack(interp, axis=0).astype(np.float32)
    return xyz



def load_cie_lms_for_cvd(path, target_wavelengths_nm, condition: CVDCondition,):
    table = np.loadtxt(path, delimiter=",",)

    source_wavelengths = table[:, 0]
    source_lms = table[:, 1:].T

    shift_nm = condition.severity * MAX_ANOMALY_SHIFT_NM

    interpolated_curves = []

    for channel, curve in enumerate(source_lms):
        valid = np.isfinite(curve)

        query_wavelengths = target_wavelengths_nm

        if condition.cvd_type == "protan" and channel == 0:
            # L cone shifted -> short wavelengths
            query_wavelengths = target_wavelengths_nm + shift_nm

        elif condition.cvd_type == "deutan" and channel == 1:
            # M cone shifted -> long wavelengths
            query_wavelengths = target_wavelengths_nm - shift_nm

        shifted_curve = np.interp(
            query_wavelengths,
            source_wavelengths[valid],
            curve[valid],
            left=0.0,
            right=0.0,
        )

        interpolated_curves.append(shifted_curve)

    return np.stack(
        interpolated_curves,
        axis=0,
    ).astype(np.float32)


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    lms_path = (
        project_root
        / "data"
        / "physiology"
        / "CIE_lms_cf_2deg.csv"
    )

    wavelengths = np.arange(
        400,
        701,
        10,
        dtype=np.float32,
    )

    normal = load_cie_lms(
        lms_path,
        wavelengths,
    )

    protan_zero = load_cie_lms_for_cvd(
        lms_path,
        wavelengths,
        CVDCondition(
            cvd_type="protan",
            severity=0.0,
        ),
    )

    print(
        "severity=0 equals normal:",
        np.allclose(normal, protan_zero),
    )

    protan = load_cie_lms_for_cvd(
        lms_path,
        wavelengths,
        CVDCondition(
            cvd_type="protan",
            severity=1.0,
        ),
    )

    deutan = load_cie_lms_for_cvd(
        lms_path,
        wavelengths,
        CVDCondition(
            cvd_type="deutan",
            severity=1.0,
        ),
    )

    print(
        "M unchanged for protan:",
        np.allclose(normal[1], protan[1]),
    )

    print(
        "S unchanged for protan:",
        np.allclose(normal[2], protan[2]),
    )

    print(
        "L unchanged for deutan:",
        np.allclose(normal[0], deutan[0]),
    )

    print(
        "S unchanged for deutan:",
        np.allclose(normal[2], deutan[2]),
    )

    print(
        "All protan values finite:",
        np.isfinite(protan).all(),
    )

    print(
        "All deutan values finite:",
        np.isfinite(deutan).all(),
    )