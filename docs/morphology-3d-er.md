# Morphology Analysis: 3D ER

[All examples](../README.md#examples) · [Morphology Analysis reference](./morphology-analysis.md)

**Example data:** [`3d_example_ER.tif`](../example/3d/3d_example_ER.tif). Generate a vesselness-guided segmentation from this volume before running morphology analysis, as described below.

## 1. Prepare a vesselness-aware segmentation

For ER morphology analysis, first create a segmentation guided by the tubular vesselness response from **Structural Awareness Extraction**. The important point is that vesselness contributes the local structural evidence used for segmentation; the extraction mode itself can be chosen to suit the data. If needed, strengthen the vesselness contribution with **Vessel rescale**, then run SIGMA to obtain a mask suited to skeleton and branch analysis.

![Prepare a vesselness-aware ER segmentation for morphology analysis](./images/morphology-3d-er/01-vesselness-aware-segmentation.png)

## 2. Analyze objects and review measurements

Select the resulting ER segmentation in **Morphology Analysis** and click **Analyze Objects**. SIGMA separates the connected ER structures and reports their object measurements, distribution plots, and individual branch lengths in the right panel.

![Analyze the segmented 3D ER network](./images/morphology-3d-er/02-analyze-objects.png)

## 3. Inspect ER topology interactively

Double-click an ER object in the viewer or click its row in **Measurements** to display its skeleton, junction points, and endpoints. Selecting a branch row highlights the corresponding branch and topology markers.

![Inspect the skeleton and topology of a 3D ER object](./images/morphology-3d-er/03-topology-interaction.png)
