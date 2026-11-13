from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]

CVDType = Literal["protan", "deutan"]


@dataclass
class SpectralPatch:
    cube: Float32Array
    wavelengths_nm: Float32Array
    scene_id: str
    row: int
    col: int

    def __post_init__(self):
        if self.cube.ndim != 3:
            raise ValueError("cube must be a 3D array")

        bands = self.cube.shape[0]

        if self.wavelengths_nm.ndim != (bands,):
            raise ValueError("wavelengths_nm must be a 1D array")

        if self.row < 0 or self.col < 0:
            raise ValueError(f"row, col must be positive")

@dataclass
class CVDCondition:
    cvd_type: CVDType
    severity: float

    def __post_init__(self):
        if not (0 <= self.severity <= 1):
            raise ValueError("severity must be between 0 and 1")