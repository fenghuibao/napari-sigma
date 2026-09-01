# SIGMA

**Structurally-aware Intensity-ordered GMM-MRF Algorithm**

SIGMA is an annotation-free, image-adaptive napari framework for fluorescence image analysis. It combines intensity-ordered GMM-MRF inference with Hessian-derived local structural constraints to retain faint or thin features while limiting mergers between closely apposed structures.

Its integrated interface provides:

- tubular and sheet-like structural response extraction followed by image-specific GMM-MRF segmentation
- calibrated object geometry, skeleton, branch, endpoint, junction, and topology measurements
- adjacent-frame matching, remodeling-event classification, visualization, and interactive link refinement
- image-resolution overlap, distance, Manders, and ROI-restricted spatial-association analysis

SIGMA supports 2D, 3D, and time-series TIFF data and can use CPU, CUDA, or Apple MPS when available. Results and metadata can be exported for reproducible downstream analysis. Proximity measurements describe spatial association between segmented fluorescence signals at image resolution; they do not estimate physical membrane separation.

Install the plugin with:

```bash
pip install "napari-sigma[all]"
```

Then select **SIGMA** from napari's **Plugins** menu.
