# Colorblind Adapter

A course project on physics-based color-vision-deficiency simulation from hyperspectral images.

## Current pipeline

```text
ICVL hyperspectral patch
        ├──> CIE XYZ -----------------> model input
        └──> shifted CIE LMS ---------> CVD target

XYZ / CVD LMS
        └──> display-only RGB projection
```

The scientific target is a three-channel CVD cone-response image `(L, M, S)`. RGB images are used only for input/display and are not used for the target loss.

## Data

ICVL scenes are stored under:

```text
data/ICVL/scenes_60/
data/ICVL/patches_256/
```

The patch manifest is:

```text
data/ICVL/patches_256/manifest.csv
```

Patches use the convention `(bands, height, width)` with 31 bands from 400 to 700 nm at 10 nm intervals. Scene-level train/validation/test splits will be added before model training.

## Data setup

### One-command setup

Create a `.env` file at the repository root:

```env
HF_TOKEN=hf_your_token_here
```

The token needs read access to the ICVL dataset. Never commit `.env`.

Then run this from the repository root:

```powershell
.\.cvp_venv\Scripts\python.exe setup_data.py
```

This command creates the required directories, downloads the CIE reference tables, downloads the selected ICVL scenes, and extracts the 256x256 patches.

### Manual reference

The CIE reference tables are stored in:

```text
data/physiology/
```

Download these files into it:

```text
CIE_lms_cf_2deg.csv
CIE_xyz_1931_2deg.csv
```

Official download links:

- [CIE LMS fundamentals](https://files.cie.co.at/Publications-datasets/CIE_lms_cf_2deg.csv)
- [CIE XYZ colour-matching functions](https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv)

The LMS metadata file may also be downloaded for reference:

```text
CIE_lms_cf_2deg.csv_metadata.json
```

The selected ICVL scenes and extracted patches are stored in:

```text
data/ICVL/scenes_60/
data/ICVL/patches_256/
```

The patch manifest is generated at:

```text
data/ICVL/patches_256/manifest.csv
```


## Source code

```text
src/data_types.py      Typed data containers
src/physiology.py      CIE LMS/XYZ loading and CVD cone shifts
src/simulator.py       Spectral integration into LMS and XYZ
src/color_spaces.py    XYZ/RGB conversions and display projection
src/visualization.py   Diagnostic Matplotlib visualizations
```

The current simplified CVD model shifts the L fundamental toward shorter wavelengths for protanomaly and the M fundamental toward longer wavelengths for deuteranomaly. Severity is normalized to `[0, 1]` and mapped to a maximum 20 nm shift.

## Running diagnostics

From the `src` directory:

```powershell
..\.cvp_venv\Scripts\python.exe simulator.py
..\.cvp_venv\Scripts\python.exe visualization.py
```

## Next step

Implement a dataset loader returning:

```text
{XYZ, CVD type, severity} -> {CVD LMS target}
```

Targets will be generated deterministically from the hyperspectral patches rather than stored as display RGB images.
