# Morphology Analysis: 3D mitochondria

[All examples](../README.md#examples) · [Morphology Analysis reference](./morphology-analysis.md)

**Example data:** [`3d_example_Mitochondria.tif`](../example/3d/3d_example_Mitochondria.tif). First generate its mask by following the [3D mitochondrial segmentation example](./segmentation-3d-mitochondria.md), including the 2x XY upsampling step used to reduce bridging between closely apposed objects.

## 1. Analyze objects and review measurements

After obtaining the 3D mitochondrial segmentation, open **Morphology Analysis** and select **denoised 2x XY bilinear segmentation** under **Layer**. Set **Min pixel/voxel size** to **20** and click **Analyze Objects**. For this 3D example, the minimum-size setting is a voxel-count threshold, not a physical volume. The scope is **static layer**.

SIGMA identifies connected foreground objects and displays them as a colored object-label layer. **Distribution Plots** include volume and branch-number distributions. **Measurements** lists object-level values such as voxel count, surface area, volume, branch number, junction number, and endpoint number; **Branch Length List** reports individual skeleton branches and their lengths. Surface area, volume, and branch length use the segmentation layer's voxel calibration and are reported in um^2, um^3, and um, respectively.

![Analyze objects and review 3D mitochondrial morphology measurements](./images/morphology-3d-mitochondria/01-analyze-objects.png)

## 2. Inspect an object interactively

Double-click a mitochondrion in the viewer or click its object row in **Measurements**. SIGMA links the image and table selection, then displays the selected object's skeleton, junction points, and endpoints. Selecting a row in **Branch Length List** additionally highlights that individual branch.

The screenshot below shows object **71** selected in **Measurements**. Lower the object-label layer's **opacity** if needed to see the skeleton and topology markers more clearly; the illustrated view uses approximately **0.22**. This changes only the display, not the segmentation or measurements.

![Object 71 selected in Measurements with its skeleton and topology markers visible through the dimmed object-label layer](./images/morphology-3d-mitochondria/02-topology-interaction.png)
