# Segmentation

The Segmentation tab implements SIGMA's annotation-free, image-adaptive segmentation workflow. SIGMA combines an image-specific Gaussian mixture model (GMM) with intensity-ordered class assignment and a Markov random field (MRF) prior guided by Hessian-derived local structural evidence. No annotated training masks are required. The interface organizes image setup, optional preprocessing, structural response extraction, and GMM-MRF inference across five sections.

## Recommended workflow

1. Open the image under **Data / Axes / Device**.
2. Confirm axes and physical size under **Image Info**.
3. Optionally preprocess the intensity layer under **Preprocessing**.
4. Run **Structural Awareness Extraction**.
5. Select the raw and structural-response layers under **Segmentation**.
6. Run SIGMA and inspect the inferred foreground mask in napari.

## Examples

- [2D mitochondria](./segmentation-2d.md)
- [2D mitochondria (data from MoDL)](./segmentation-2d-modl.md)
- [3D mitochondria](./segmentation-3d-mitochondria.md)
- [3D ER](./segmentation-3d-er.md)

## Data / Axes / Device

- **Open File** loads the source image.
- **Device** selects CPU, CUDA, or MPS.
- **Channel** and **Time** choose the displayed channel or frame.
- **Time range** and **Slice range** restrict the data used by later processing.

## Image Info

Image Info reports the interpreted shape, physical extent, and pixel or voxel size. Use **Edit size** only when the source metadata is absent or incorrect. Physical size affects scale-aware structural extraction and morphology or proximity measurements.

## Preprocessing

Choose the source under **Raw image layer** before running a preprocessing step.

### Median filter

The median filter suppresses isolated noise while preserving sharp boundaries. Larger filter sizes are more aggressive and can remove narrow structures.

### Gaussian background

Gaussian background subtraction removes slowly varying background intensity.

- **Sigma** controls the background spatial scale.
- **Alpha** controls how strongly the estimated background is subtracted.

### Upsample XY

XY upsampling increases lateral sampling before structural extraction and segmentation. It can improve the representation of narrow gaps between closely apposed objects and thereby reduce MRF bridging, but it increases memory use and computation time. It does not increase the optical resolution of the source image.

When an upsampled layer is segmented, SIGMA returns the final mask to the original XY resolution while retaining separated labels where the original grid permits it.

## Structural Awareness Extraction

SIGMA derives local structural evidence from multiscale, scale-normalized Hessian responses. Select the input layer and choose a response mode:

- **vesselness** uses a Frangi-style response to emphasize tubular structures.
- **sheetness** emphasizes plate-like structures in 3D.
- **combined** computes vesselness and sheetness and takes their voxelwise maximum.

**sheetness** and **combined** require a 3D volume with a selected Z depth greater than **Kernel radius**. These modes are disabled for 2D images, single Z slices, and `TYX` time series; use **vesselness (tubular)** instead.

Key settings:

- **PSF z/xy** is the axial-to-lateral point-spread-function ratio used to account for optical anisotropy in 3D derivative filtering.
- **Kernel radius** controls the spatial support of the derivative filters.
- **Sigma min/max** set the smallest and largest structural scales to evaluate.
- **Count** sets the number of scales between the selected limits.
- **Enable rescale** enables optional percentile rescaling after extraction.

Use **View Vessel**, **View Sheet**, or **View Max** to inspect available components. Rescaling is applied only after pressing **Apply Rescale**.

## GMM-MRF Segmentation

Select the intensity image under **Raw data layer** and its aligned extraction result under **Structural response layer**. SIGMA balances three terms during label inference: the image-specific GMM intensity likelihood, pairwise neighborhood coherence, and class-specific local structural support.

- **beta1 (smoothness)** weights pairwise spatial coherence between neighboring labels.
- **beta2 (structure)** weights structure-aware regularization from the selected response.
- **nforeground** sets the number of intensity-ordered foreground Gaussian components.
- **nbackground** sets the number of intensity-ordered background Gaussian components.
- **maxiter** sets the maximum number of GMM-MRF iterations.
- **init** selects random or Otsu-based initialization.
- **EM foreground points** sets the foreground sample budget; `All` uses every available point.
- **EM background points** sets the background sample budget; `All` uses every available point.

The preview displays the effective settings before execution. Progress is updated after each segmentation iteration.

## Output

Segmentation produces a binary foreground mask aligned with the source image. Use this layer directly in Morphology Analysis or Tracking Analysis, or pair it with an independently segmented fluorescence channel in Proximity Analysis. Save as TIFF when axes and physical scale need to remain embedded in the output file.
