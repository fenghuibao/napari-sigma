# Changelog

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
