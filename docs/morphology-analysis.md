# Morphology Analysis

Morphology Analysis converts a SIGMA foreground mask into connected objects and measures their geometry and skeleton topology in 2D or 3D. Calibrated pixel or voxel spacing is used for physical area, surface area, volume, and branch-length measurements.

## Examples

- [2D mitochondria](./morphology-2d.md)
- [3D mitochondria](./morphology-3d-mitochondria.md)
- [3D ER](./morphology-3d-er.md)

## Run an analysis

1. Select a binary or labels layer under **Layer**.
2. Set **Min pixel/voxel size**.
3. Press **Analyze Objects**.

For a time series, the analysis follows the frame currently shown in napari. The scope label reports the active frame. **Refresh** rebuilds the layer list after layers are added or removed.

The minimum-size setting filters reported objects, tables, and plots. It does not modify the source segmentation layer.

## Measurements

Measurements include object size, calibrated area or volume, perimeter or surface area, skeleton length, branch statistics, endpoints, junctions, and topology summaries. Values are reported in pixels or voxels when physical calibration is unavailable.

In the skeleton graph, endpoints have one neighbor and junctions have three or more neighbors. Branches are paths between endpoints and junctions, measured using the physical pixel or voxel spacing. Selecting an object row displays the corresponding skeleton, junction points, and endpoints in the image. Selecting a branch row highlights that branch together with its topology markers.

## Distribution plots

The distribution area summarizes object and branch measurements for the current analysis. The branch-length Lorenz curve and Gini coefficient describe how branch length is distributed across the analyzed network. Plots update when the minimum-size filter or active frame changes.

## Branch Length List

The branch table contains individual skeleton branches and their lengths. Endpoint and junction markers are displayed at a compact size so they do not obscure the segmented structure.

## Export

- **Export** writes the complete time series for a temporal segmentation, or the current result for a static segmentation.
- Supported table formats are CSV, tab-delimited TXT, and XLSX.

Exported values retain frame identifiers and physical units when those values are available from the selected layer.
