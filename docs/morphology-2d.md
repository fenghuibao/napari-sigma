# Morphology Analysis: 2D mitochondria

[All examples](../README.md#examples) · [Morphology Analysis reference](./morphology-analysis.md)

**Example data:** [`2d_example.tif`](../example/2d/2d_example.tif). First generate its mask by following the [2D segmentation example](./segmentation-2d.md), then select that segmentation layer for morphology analysis.

## 1. Analyze objects and review measurements

After obtaining the 2D segmentation, open **Morphology Analysis**, select the segmentation layer, set the minimum object size, and click **Analyze Objects**. SIGMA labels the connected foreground objects and reports their morphology statistics in the right panel.

The plots summarize object and branch distributions, including the branch-length Lorenz profile and its Gini coefficient. **Measurements** reports values such as pixel count, perimeter, area, branch number, junction number, and endpoint number, while **Branch Length List** reports the individual skeleton branches.

![Analyze objects and review 2D morphology measurements](./images/morphology-2d/01-analyze-objects.png)

## 2. Inspect an object interactively

Double-click an object in the viewer or click its row in **Measurements** to inspect it. SIGMA displays the selected object's skeleton, junction points, and endpoints. A branch selected from **Branch Length List** is highlighted together with its topology markers.

![Interactive 2D skeleton, junction, and endpoint display](./images/morphology-2d/02-topology-interaction.png)
