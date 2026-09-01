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
- **Tracking**: adjacent-frame object matching, remodeling-event classification, visualization, and link refinement
- **Proximity Analysis**: image-resolution overlap, distance, Manders coefficients, and ROI-restricted spatial association

SIGMA supports 2D, 3D, and time-series data and can run on CPU, CUDA, or Apple MPS when available.

## Installation

Create a clean environment and install the published napari plugin:

```bash
conda create -n sigma python=3.11
conda activate sigma
pip install --upgrade "napari-sigma[all]"
napari
```

The `all` extra installs a compatible napari 0.9 release with the PyQt6 backend and napari's optional runtime dependencies. Then select **SIGMA** from napari's **Plugins** menu.

Do not install an older `napari-sigma[all]` and then upgrade only the `napari` package: pip does not automatically remove the Qt5 backend selected by an older napari release. If an existing environment contains PyQt5, clean it once before reinstalling SIGMA:

```bash
python -m pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip
python -m pip install --upgrade "napari-sigma[all]"
```

A fresh environment remains the recommended installation path. Environment variables such as `QT_API`, `QT_PLUGIN_PATH`, or `QT_QPA_PLATFORM_PLUGIN_PATH` can override backend discovery and should not be globally fixed to a different Qt installation.

### Development installation

From the repository root:

```bash
conda create -n sigma-dev python=3.11
conda activate sigma-dev
pip install -e ".[all]"
napari
```

## Input data

The napari reader currently supports:

- TIFF: `.tif`, `.tiff`
- PNG: `.png`
- JPEG: `.jpg`, `.jpeg`

Microscope container formats such as CZI, ND2, LIF, and LSM should first be exported to one of the supported formats.

SIGMA preserves available physical-size and axis metadata. If physical size metadata is unavailable, pixel or voxel size defaults to `1.0`.

## Quick start

1. Start napari and open **Plugins > SIGMA**.
2. In the **Segmentation** tab, select **Open File** and load a fluorescence image. TIFF is recommended when axis metadata or calibrated pixel and voxel sizes are needed.
3. Confirm the interpreted channel, time, and slice ranges. Supported image organizations include `YX`, `ZYX`, `TYX`, and `TZYX`. Spatial 3D images open in napari's 3D display mode by default, while 2D time series remain in 2D.
4. Review the physical scale under **Image Info** and correct it before any analysis that reports calibrated measurements.
5. Select a computational device: **CPU** works on all supported systems, **CUDA** uses a compatible NVIDIA GPU, and **MPS** uses Apple Silicon GPU acceleration.
6. Use **Segmentation** to extract local structural evidence and infer a binary foreground mask.
7. Use **Morphology Analysis** to measure connected structures and their skeleton topology.
8. For time series, use **Tracking** to associate objects across adjacent frames and classify linear, fission, fusion, or split-merge transitions.
9. For aligned fluorescence channels, use **Proximity Analysis** to quantify spatial association between independently segmented structures.

Each workflow operates on napari layers. Check the selected layer before running a command, especially when raw, processed, structural-response, and segmentation layers are open together. Results are returned as regular napari layers regardless of the selected computational device.

## Documentation

- [Segmentation](./docs/segmentation.md)
- [Morphology Analysis](./docs/morphology-analysis.md)
- [Tracking](./docs/tracking.md)
- [Proximity Analysis](./docs/proximity-analysis.md)

## Example data

The repository includes example fluorescence data for every documented workflow. Segmentation and morphology use the same raw images; tracking and proximity analysis also include aligned segmentation inputs so those workflows can be run directly.

- **2D segmentation and morphology:** [`2d_example_raw.tif`](./example/2d/2d_example_raw.tif)
- **3D mitochondrial segmentation and morphology:** [`3d_example_Mitochondria.tif`](./example/3d/3d_example_Mitochondria.tif)
- **3D ER segmentation and morphology:** [`3d_example_ER.tif`](./example/3d/3d_example_ER.tif)
- **Tracking:** [`tracking_example.tif`](./example/Tracking/tracking_example.tif) and [`tracking_example_segmentation.tif`](./example/Tracking/tracking_example_segmentation.tif)
- **Mitochondria-ER proximity assay:** [`mitochondria raw`](./example/Proximity_analysis/proximity_assay_example_mito.tif), [`mitochondria segmentation`](./example/Proximity_analysis/proximity_assay_example_mito_segmentation.tif), [`ER raw`](./example/Proximity_analysis/proximity_assay_example_ER.tif), and [`ER segmentation`](./example/Proximity_analysis/proximity_assay_example_ER_segmentation.tif)

Use TIFF for workflows that depend on axis metadata or physical pixel and voxel sizes. Files used together must have matching spatial dimensions, time axes, physical scales, and, for proximity analysis, spatial alignment.

Follow the corresponding guides for the complete examples:

- [Segmentation](./docs/segmentation.md): 2D mitochondria, 3D mitochondria, and 3D ER
- [Morphology Analysis](./docs/morphology-analysis.md): object and skeleton analysis of the same three datasets
- [Tracking](./docs/tracking.md): adjacent-frame matching, event classification, visualization, and refinement
- [Proximity Analysis](./docs/proximity-analysis.md): full-volume and ROI-restricted mitochondria-ER analysis

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

- Extract Hessian-derived vesselness, sheetness, or their voxelwise combined response.
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

## Tracking

Tracking operates on segmented time series under an adjacent-frame, minimum-displacement assumption and provides:

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

## Build validation

Build and validate the distribution locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Release process

SIGMA is published as **`napari-sigma`**. The GitHub workflow uses PyPI Trusted Publishing; no long-lived PyPI API token is stored in the repository.

1. Update `src/napari_sigma/_version.py`.
2. Run the build validation commands above.
3. Commit the release changes.
4. Create and push a matching version tag, for example:

```bash
git tag v0.0.1
git push origin v0.0.1
```

The tag triggers distribution validation and publishes the built package to PyPI. The PyPI Trusted Publisher must be configured for this repository, the `test_and_deploy.yml` workflow, and the `pypi` environment.

## Runtime dependencies

Core runtime dependencies include napari, PyTorch, NumPy, SciPy, scikit-image, scikit-learn, OpenCV, tifffile, Pillow, matplotlib, dask, QtPy, and openpyxl. See [`pyproject.toml`](./pyproject.toml) for the authoritative version requirements.

## License

SIGMA is distributed under the [MIT License](./LICENSE).

## Issues

When reporting a problem, include:

- operating system and hardware
- Python, napari, Qt, and SIGMA versions
- computational device (`cpu`, `cuda`, or `mps`)
- input shape and axis interpretation
- traceback or warning output
