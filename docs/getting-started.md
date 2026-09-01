# Getting Started

This guide covers the shortest path from a fluorescence image to an annotation-free SIGMA analysis. SIGMA estimates segmentation parameters from the selected image and uses local structural evidence to support faint tubular or sheet-like signal.

## 1. Install and open SIGMA

Create a clean environment and install the plugin:

```bash
conda create -n sigma python=3.11
conda activate sigma
pip install --upgrade "napari-sigma[all]"
napari
```

In napari, open **Plugins > SIGMA**.

## 2. Open an image

SIGMA reads TIFF, PNG, and JPEG files. TIFF is recommended for microscopy data because it can preserve axes and physical pixel or voxel sizes.

Use **Open File** in the Segmentation tab. Spatial 3D images open in napari's 3D display mode by default; 2D time series remain in 2D. Confirm the interpreted channel, time, and slice ranges before processing. If the physical size is missing or incorrect, update it under **Image Info** before running an analysis that reports physical measurements.

Supported image organizations include:

- 2D images: `YX`
- 3D volumes: `ZYX`
- 2D time series: `TYX`
- 3D time series: `TZYX`

## 3. Select a device

- **CPU** works on every supported computer.
- **CUDA** uses a compatible NVIDIA GPU.
- **MPS** uses Apple Silicon GPU acceleration.

Device selection applies to supported denoising, structural extraction, and segmentation operations. Results are returned to napari as regular layers regardless of the selected device.

## 4. Follow a workflow

The usual order is:

1. [Segment the image](./segmentation.md).
2. [Measure morphology](./morphology-analysis.md).
3. For time series, [associate objects between adjacent frames and classify remodeling events](./tracking.md).
4. To compare structures segmented in aligned fluorescence channels, run [Proximity Analysis](./proximity-analysis.md).

Each page operates on napari layers. Check the selected layer before running a command, especially when several raw, processed, and segmentation layers are open.

## 5. Try the examples

The repository [indexes every example dataset on its front page](../README.md#example-data). Select a raw image or aligned input set there, then follow the linked workflow guide.
