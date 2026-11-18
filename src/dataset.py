from pathlib import Path
import csv

import numpy as np

if __package__:
    from .data_types import CVDCondition, SpectralPatch
    from .physiology import load_cie_lms, load_cie_lms_for_cvd, load_cie_xyz
    from .simulator import SpectralSimulator
else:
    from data_types import CVDCondition, SpectralPatch
    from physiology import load_cie_lms, load_cie_lms_for_cvd, load_cie_xyz
    from simulator import SpectralSimulator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = PROJECT_ROOT/"data"/"ICVL"/"patches_256"/"manifest.csv"
LMS_PATH = PROJECT_ROOT/"data"/"physiology"/"CIE_lms_cf_2deg.csv"
XYZ_PATH = PROJECT_ROOT/"data"/"physiology"/"CIE_xyz_1931_2deg.csv"
WAVELENGTHS_NM = np.arange(400, 701, 10, dtype=np.float32)

CONDITIONS = [
    ("protan", 0.0),
    ("protan", 0.5),
    ("protan", 1.0),
    ("deutan", 0.5),
    ("deutan", 1.0),
]


class CVDDataset:
    def __init__(self) -> None:
        with MANIFEST_PATH.open(newline="", encoding="utf-8") as file:
            self.manifest = list(csv.DictReader(file))

        normal_lms = load_cie_lms(LMS_PATH, WAVELENGTHS_NM)
        xyz = load_cie_xyz(XYZ_PATH, WAVELENGTHS_NM)
        self.simulator = SpectralSimulator(WAVELENGTHS_NM, normal_lms, xyz)
        self.cvd_sensitivities = {}

        for cvd_type, severity in CONDITIONS:
            condition = CVDCondition(cvd_type, severity)
            self.cvd_sensitivities[(cvd_type, severity)] = load_cie_lms_for_cvd(LMS_PATH, WAVELENGTHS_NM, condition)

    def __len__(self) -> int:
        return len(self.manifest) * len(CONDITIONS)

    def __getitem__(self, index: int) -> dict:
        patch_index, condition_index = divmod(index, len(CONDITIONS))

        row = self.manifest[patch_index]
        cvd_type, severity = CONDITIONS[condition_index]

        patch_path = PROJECT_ROOT/row["path"].replace("\\", "/")

        with np.load(patch_path) as data:
            patch = SpectralPatch(
                cube=data["hyperspectral"],
                wavelengths_nm=data["wavelengths_nm"],
                scene_id=row["scene_id"],
                row=int(row["row"]),
                col=int(row["col"]),
            )

        condition_value = severity if cvd_type == "protan" else -severity
        

        xyz = self.simulator.spectral_to_xyz(patch)

        target_lms = self.simulator.spectral_to_cvd_lms(
            patch,
            self.cvd_sensitivities[(cvd_type, severity)],
        )

        return {
            "xyz": xyz.tristimulus.astype(np.float32),
            "condition": condition_value,
            "target_lms": target_lms.responses.astype(np.float32),
            "scene_id": row["scene_id"],
            "patch_id": row["patch_id"],
        }