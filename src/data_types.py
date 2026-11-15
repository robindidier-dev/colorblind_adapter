from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]

CVDType = Literal["protan", "deutan"]

ConeResponseArray = NDArray[np.float32]

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

        if self.wavelengths_nm.shape != (bands,):
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


@dataclass
class ConeResponsePatch:
    responses: ConeResponseArray
    scene_id: str
    row: int
    col: int

    def __post_init__(self):
        if self.responses.ndim != 3:
            raise ValueError("responses must be a 3D array")

        channels = self.responses.shape[0]

        if channels != 3:
            raise ValueError("responses must have 3 channels")

        if self.row < 0 or self.col < 0:
            raise ValueError(f"row, col must be positive")

@dataclass
class XYZPatch:
    tristimulus: Float32Array
    scene_id: str
    row: int
    col: int

    def __post_init__(self):
        if self.tristimulus.ndim != 3:
            raise ValueError("tristimulus must be a 3D array")

        if self.tristimulus.shape[0] != 3:
            raise ValueError("tristimulus must have 3 channels")

        if self.row < 0 or self.col < 0:
            raise ValueError(f"row, col must be positive")


@dataclass
class RGBPatch:
    linear_rgb: Float32Array
    scene_id: str
    row: int
    col: int

    def __post_init__(self):
        if self.linear_rgb.ndim != 3:
            raise ValueError("linear_rgb must be a 3D array")

        if self.linear_rgb.shape[0] != 3:
            raise ValueError("linear_rgb must have 3 channels")

        if self.row < 0 or self.col < 0:
            raise ValueError(f"row, col must be positive")
