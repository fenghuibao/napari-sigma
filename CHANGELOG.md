# Changelog

## 0.0.6 — Bug fixes

- Restore TIFF calibration and units during file loading and drag-and-drop, while preserving edited pixels and explicit reader choices.
- Fix repeated file opening and synchronize image information and processing inputs with the newly opened layer.
- Preserve segmentation and structural-response metadata when saving and reopening TIFF files.
- Correct analysis, tracking, proximity and segmentation edge cases; consolidate shared metadata and processing helpers.
- Improve device selection and interface state synchronization.
- Add regression coverage for calibration, repeated imports, cancelled/failed opens and preservation of existing data.

This is a napari plugin release. No desktop installers, example datasets, diagnostic scripts or audit reports are included.

## 0.0.5

- Fix the reader factory's parameter name to accept npe2's `path=` keyword, restoring SIGMA reads through napari's file-open/drop flow.
- Add manifest-dispatch regression tests for TIFF/PNG/JPEG, single-file stack calls, and calibrated uint64 time-series labels through both writer and reader commands.
- Track all outstanding analysis workers, including superseded requests; cancel and join them on panel disposal, and reject queued results from an earlier panel lifecycle after reopening.
- Restore full selected-time-range processing for unannotated 4D images in Median, Gaussian background subtraction, and shared Frangi/segmentation input extraction.
- Normalize TIFF axes by name, fixing RGB channel/Z swaps and supporting RGB movies/volumes, OME C/T/Z permutations, and unannotated I/Q page stacks. Expose unlabelled-page axis assumptions in metadata instead of treating the default Z interpretation as acquisition metadata.
- Respect native napari RGB state in both writer dispatch and the custom save menu. Preserve spatial calibration when RGB scale excludes the samples axis, while retaining legacy TYXC export compatibility.
- Preserve large integer instance IDs in Proximity by using compact internal distance-map IDs with an exact original-ID lookup. Keep existing binary-mask labelling and repeated-ID time aggregation behavior.
- Add pixel-exact TIFF round trips, large uint32/uint64 ID comparisons, real preprocessing-button/save-menu checks, and overlapping-worker close/reopen regression tests.

## 0.0.4

### Correctness and data protection

- Fix Frangi output writes for transposed/non-contiguous input arrays in 2D/3D.
- Keep the out-of-place foreground pairwise expression before its final copy, avoiding the reported MPS chained-view write issue.
- Accept uint16 segmentation inputs by normalizing to float32; reject empty, non-finite and constant images with explicit messages. Reset both classes when recovering an empty class.
- Correct unsigned-subtraction overcounting of 3D surface area. Existing measurements of concave/hollow objects should be recomputed.
- Compute fallback principal-axis lengths in physical coordinates; use calibrated contours for non-square-pixel perimeter estimates. Square-pixel perimeter behavior is preserved.
- Preserve labels, physical calibration and SIGMA metadata in non-ImageJ TIFFs, including uint32 labels; explicitly write resolution units and grayscale photometric interpretation.
- Correct TYX/TZYX Labels shapes and scales; reorder multichannel volumes for ImageJ export without scrambling channel/Z data.
- Do not reload and replace existing edited layers at plugin startup or upon insertion. Respect explicit napari reader choices.

### Startup, lifecycle and safety

- Load Matplotlib/charts on first analysis-tab use, openpyxl on export, and Torch when a computation requests a device. Respect the user's Matplotlib cache configuration.
- Use headless OpenCV for image/video operations and import it only when needed, avoiding its bundled Qt plugins conflicting with napari/PyQt6 on Linux. Existing environments should keep only one OpenCV package; see the upgrade instructions.
- Run Proximity in a cancellable background worker; avoid full-volume raw float64 copies and retaining per-object full-volume masks simultaneously.
- Release viewer callbacks, scale subscriptions and context-menu patches when the panel is removed/closed; make cleanup idempotent.
- Bound NIS metadata expansion, nesting, string sizes and item counts.

### Maintenance

- Share layer-axis/unit helpers, exposed-face geometry and reader/UI array conversion.
- Extract a lightweight reader; remove global file-open/drop interception and the obsolete internal TZCYX display branch.
- Share 2D/3D segmentation label-update code while retaining backend-specific EM behavior.
- Add scientific, metadata round-trip, safety, lazy-import and GUI lifecycle regression tests. Gate release builds on the test matrix and matching version tags.

This release does not change the existing full-image Proximity aggregation of repeated object IDs across time, or promise bitwise equivalence between different compute backends.
