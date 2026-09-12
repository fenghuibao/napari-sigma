# SIGMA (Structurally-aware Intensity-ordered GMM-MRF Algorithm)

[![License: MIT](https://img.shields.io/pypi/l/napari-sigma.svg?color=green)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sigma.svg?color=green)](https://pypi.org/project/napari-sigma/)
[![Python](https://img.shields.io/pypi/pyversions/napari-sigma.svg?color=green)](https://pypi.org/project/napari-sigma/)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sigma)](https://napari-hub.org/plugins/napari-sigma)
[![npe2](https://img.shields.io/badge/napari-npe2-blue)](https://napari.org/stable/plugins/index.html)

SIGMA is an annotation-free, image-adaptive napari framework for fluorescence image analysis. It combines an image-specific Gaussian mixture model (GMM) with intensity-ordered class assignment and a Markov random field (MRF) prior guided by local structural constraints derived from Hessian analysis. By estimating model parameters from each image, SIGMA is designed to retain faint or thin fluorescent structures while limiting mergers between closely apposed objects.

Its four connected workflows quantify cellular morphology, dynamics, and spatial organization:

- **Segmentation**: image-specific GMM-MRF segmentation guided by tubular or sheet-like structural responses
- **Morphology Analysis**: object geometry, skeleton branches, endpoints, junctions, and network topology
- **Tracking Analysis**: adjacent-frame object matching, remodeling-event classification, visualization, and link refinement
- **Proximity Analysis**: image-resolution overlap, distance, Manders coefficients, and ROI-restricted spatial association

SIGMA supports 2D, 3D, and time-series data and can run on CPU, CUDA, or Apple MPS when available.

## Installation

### Desktop installers (no terminal required)

Download a prebuilt SIGMA 0.0.6 application from [GitHub Releases](https://github.com/fenghuibao/napari-sigma/releases/tag/v0.0.6). These packages include Python and all required analysis libraries, so no conda, pip, or terminal setup is needed.

| Platform | Download | Compute support |
| --- | --- | --- |
| Windows 10/11, Intel/AMD x64 | [Windows installer (.exe)](https://github.com/fenghuibao/napari-sigma/releases/download/v0.0.6/SIGMA-0.0.6-windows-x86_64-cu130.exe) | CUDA / CPU |
| macOS 14 or later, Apple Silicon (M-series) | [Apple Silicon installer (.dmg.zip)](https://github.com/fenghuibao/napari-sigma/releases/download/v0.0.6/SIGMA-0.0.6-macos-arm64.dmg.zip) | MPS / CPU |
| macOS 14 or later, Intel | [Intel Mac installer (.dmg.zip)](https://github.com/fenghuibao/napari-sigma/releases/download/v0.0.6/SIGMA-0.0.6-macos-x86_64.dmg.zip) | CPU |

### Terminal installation (napari plugin)

Use this method on Linux, for a custom Python environment, or to install SIGMA as a plugin in napari.

Create a clean environment and install the published napari plugin:

```bash
conda create -n sigma python=3.11
conda activate sigma
pip install --upgrade "napari-sigma[all]"
napari-sigma
```

The `all` extra installs a compatible napari 0.9 release with the PyQt6 backend and napari's optional runtime dependencies. The `napari-sigma` launcher selects PyQt6 before napari imports Qt and ignores inherited plugin paths from other Qt installations for that process. Then select **SIGMA** from napari's **Plugins** menu.

For an existing installation, upgrade the complete extra rather than upgrading only napari:

```bash
python -m pip install --upgrade "napari-sigma[all]"
napari-sigma
```

PyQt5 does not normally need to be uninstalled: the launcher selects PyQt6 and its matching plugins before napari starts. It configures Qt only inside its own process and does not modify global environment variables. A fresh environment remains the safest recovery path for an environment with unrelated binary-level Qt conflicts. The standard `napari` command remains available when another backend is preferred.

SIGMA uses `opencv-python-headless`: image resizing and video export remain available, while napari supplies the GUI. Do not install multiple OpenCV variants together because they share the `cv2` namespace. When upgrading a dedicated SIGMA environment from an older release, remove the old OpenCV variants before reinstalling:

```bash
python -m pip uninstall opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless
python -m pip install --upgrade "napari-sigma[all]"
```

If other applications in the same environment need OpenCV's own GUI, use a separate environment for SIGMA instead.

On minimal Ubuntu/Debian installations, Qt also needs system libraries that pip does not supply:

```bash
sudo apt-get install libegl1 libopengl0 libdbus-1-3 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-glx0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 \
  libxkbcommon-x11-0
```

### Development installation

From the repository root:

```bash
conda create -n sigma-dev python=3.11
conda activate sigma-dev
pip install -e ".[all]"
napari-sigma
```

## Input data

We recommend using TIFF files (`.tif`, `.tiff`) exported from Fiji/ImageJ.

SIGMA also supports:

- PNG: `.png`
- JPEG: `.jpg`, `.jpeg`

SIGMA preserves available physical-size and axis metadata. If physical size metadata is unavailable, pixel or voxel size defaults to `1.0`.

## Quick start

### Basic steps

1. **Desktop app users:** open **SIGMA**. **Plugin users:** start napari and open **Plugins > SIGMA**.
2. In the **Segmentation** tab, select **Open File** and load a fluorescence image. TIFF is recommended when axis metadata or calibrated pixel and voxel sizes are needed.
3. Confirm the interpreted channel, time, and slice ranges. Supported image organizations include `YX`, `ZYX`, `TYX`, and `TZYX`. Spatial 3D images open in napari's 3D display mode by default, while 2D time series remain in 2D.
4. Review the physical scale under **Image Info** and correct it before any analysis that reports calibrated measurements.
5. Choose a computational device from the **Device** dropdown (**CPU**, **CUDA**, or **MPS**, depending on availability).
6. Use **Segmentation** to extract local structural evidence and infer a binary foreground mask.
7. Use **Morphology Analysis** to measure connected structures and their skeleton topology.
8. For time series, use **Tracking Analysis** to associate objects across adjacent frames and classify linear, fission, fusion, or split-merge transitions.
9. For aligned fluorescence channels, use **Proximity Analysis** to quantify spatial association between independently segmented structures.

### Examples

- [Segmentation](./docs/segmentation.md)

  - [2D mitochondria](./docs/segmentation-2d.md)
  - [2D mitochondria (data from MoDL)](./docs/segmentation-2d-modl.md)
  - [3D mitochondria](./docs/segmentation-3d-mitochondria.md)
  - [3D ER](./docs/segmentation-3d-er.md)

- [Morphology Analysis](./docs/morphology-analysis.md)

  - [2D mitochondria](./docs/morphology-2d.md)
  - [3D mitochondria](./docs/morphology-3d-mitochondria.md)
  - [3D ER](./docs/morphology-3d-er.md)

- [Tracking Analysis](./docs/tracking.md)

  - [3D mitochondria time series](./docs/tracking.md)

- [Proximity Analysis](./docs/proximity-analysis.md)

  - [3D mitochondria-ER](./docs/proximity-analysis.md)

## Example data

The repository includes example fluorescence data for every documented workflow. Segmentation and morphology use the same raw images; tracking analysis and proximity analysis also include aligned segmentation inputs so those workflows can be run directly.

- **2D segmentation and morphology:** [`2d_example.tif`](./example/2d/2d_example.tif)
- **2D mitochondria (data from MoDL):** [`2d_example_MoDL.tif`](./example/2d/2d_example_MoDL.tif), originally `4.tif` from [MoDL](https://doi.org/10.1038/s41467-025-55825-x).
- **3D mitochondrial segmentation and morphology:** [`3d_example_Mitochondria.tif`](./example/3d/3d_example_Mitochondria.tif)
- **3D ER segmentation and morphology:** [`3d_example_ER.tif`](./example/3d/3d_example_ER.tif)
- **Tracking Analysis (3D time series):** [`tracking_example.tif`](./example/Tracking/tracking_example.tif) and [`tracking_example_segmentation.tif`](./example/Tracking/tracking_example_segmentation.tif)
- **Mitochondria-ER proximity assay:** [`mitochondria raw`](./example/Proximity_analysis/proximity_assay_example_mito.tif), [`mitochondria segmentation`](./example/Proximity_analysis/proximity_assay_example_mito_segmentation.tif), [`ER raw`](./example/Proximity_analysis/proximity_assay_example_ER.tif), and [`ER segmentation`](./example/Proximity_analysis/proximity_assay_example_ER_segmentation.tif)

## Segmentation

The Segmentation page contains five sections.

### Data / Axes / Device

- Open the source image.
- Select CPU, CUDA, or MPS.
- Select channel and frame ranges.
- Review the interpreted axis layout.

### Image Info

- Review image shape, axes, and physical scale.
- Apply corrected pixel or voxel sizes.

### Preprocessing

- Select the raw image layer used by preprocessing.
- Optionally apply median filtering, Gaussian background subtraction, and XY upsampling before extraction.

### Structural Awareness Extraction

- Extract Hessian-derived vesselness (tubular), sheetness, or their voxelwise combined response.
- Configure kernel radius, sigma range, PSF ratio, and response rescaling.
- Run on the selected computational device.

### GMM-MRF Segmentation

- Select raw and structural-response layers.
- Configure neighborhood smoothness and structure-weighted regularization.
- Set the numbers of intensity-ordered foreground and background Gaussian components and the EM sampling budgets.
- Run segmentation and return upsampled results to the original spatial resolution.

## Morphology Analysis

Morphology Analysis labels connected objects and provides calibrated 2D or 3D measurements:

- object size and shape measurements
- skeleton and branch statistics
- endpoint and junction detection
- branch-length distributions and topology summaries
- object and branch table export

## Tracking Analysis

Tracking Analysis operates on segmented time series under an adjacent-frame, minimum-displacement assumption and provides:

- intensity-weighted point sampling and frame-to-frame matching
- candidate-link costs based on distance, bidirectional coverage, and matched-point support
- linear, fission, fusion, and split-merge event classification
- matched-frame, event, and unlinked-object visualization
- interactive link refinement with undo support
- event table import and export

The **Max distance** control is expressed in the pixel or voxel units shown by the interface. Spatial anisotropy from the layer scale is incorporated into matching.

## Proximity Analysis

Proximity Analysis compares source and target structures segmented independently in aligned fluorescence channels. It provides:

- overlap and union measurements
- Dice and Jaccard coefficients
- Manders M1 and M2 coefficients
- surface and nearest-distance measurements
- full-image and polygon-ROI analysis
- summary table export

These measurements describe spatial association at the image resolution. They do not estimate physical membrane separation or, by themselves, establish molecular contact.

## Runtime dependencies

Core runtime dependencies include napari, PyTorch, NumPy, SciPy, scikit-image, scikit-learn, OpenCV, tifffile, Pillow, matplotlib, dask, QtPy, and openpyxl. See [`pyproject.toml`](./pyproject.toml) for the authoritative version requirements.

## License

SIGMA is distributed under the [MIT License](./LICENSE).
