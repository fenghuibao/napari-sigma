# Segmentation: 3D ER

[All examples](../README.md#examples) · [Segmentation reference](./segmentation.md)

This example segments a 3D endoplasmic reticulum (ER) volume. The source stack contains 50 Z slices with a voxel size of approximately 0.20 x 0.112 x 0.112 um.

**Example data:** [`3d_example_ER.tif`](../example/3d/3d_example_ER.tif)

## 1. Load the ER volume

Drag the 3D TIFF into napari or select it with **Open File**. Confirm the Z, Y, and X dimensions and voxel sizes before processing.

![Raw 3D ER volume loaded in SIGMA](./images/segmentation-3d-er/01-raw-volume.png)

## 2. Remove isolated noise

Under **Preprocessing**, run a median filter with a filter size of 3. The filtered volume is used as the intensity image for both structural awareness extraction and segmentation.

![Median-filtered 3D ER volume](./images/segmentation-3d-er/02-median-filtered.png)

## 3. Extract a combined structural response

Under **Structural Awareness Extraction**, choose **combined** mode. This computes a tubular vesselness response and a plate-like sheetness response, then uses their voxelwise maximum so that ER tubules and sheets can both contribute local structural evidence. In this example, vessel sigma spans 0.01 to 0.80 and sheet sigma spans 0.01 to 0.40, with five scales for each response.

![Combined vesselness and sheetness response](./images/segmentation-3d-er/03-combined-response.png)

## 4. Strengthen the vesselness signal

Enable rescaling and apply a 1.80% rescale to the vesselness response while leaving sheetness unscaled. This increases the contribution of weak tubular ER signal before the combined structural response is passed to SIGMA.

![Combined structural response after vesselness rescaling](./images/segmentation-3d-er/04-vessel-rescaled.png)

## 5. Initialize foreground with Otsu

Select the median-filtered volume as **Raw data layer** and the rescaled combined result as **Structural response layer**. For this ER volume, use `beta1 = 1.00`, `beta2 = 2.00`, `nforeground = 16`, and `nbackground = 2`.

Set **init** to **otsu** so that strong intensity signal above the Otsu threshold seeds the initial foreground assignment. SIGMA then estimates the ordered foreground and background intensity mixtures and refines the labels using intensity likelihood, neighborhood coherence, and the structural response.

![3D ER segmentation initialized with Otsu](./images/segmentation-3d-er/05-otsu-segmentation.png)

## 6. Adjust the projection for presentation

For a clearer view of the 3D mask, display the segmentation as a volume using **average** rendering and **mean** projection, with a lower gamma to reveal structures that occupy fewer Z slices. This changes only the napari presentation; it does not modify the segmentation data.

![Final 3D ER segmentation shown with average rendering](./images/segmentation-3d-er/06-average-projection.png)
