# SIGMA artwork

- `sigma-logo-original.png`: original supplied design, retained without changes.
- `sigma-logo.png`: original sigma emblem only, no lower wordmark, true RGBA
  transparency. No image-generation result is used in this asset.
- `../prepare_logo.py`: user-authorized deterministic preparation. Finds the
  connected navy emblem, retains its enclosed cells/white separators, removes
  the exterior white matte and centers the original-resolution pixels in a
  1024-pixel square. Edge alpha is recovered against the original white matte.

Run preparation manually with Pillow, NumPy and SciPy available. Normal installer
builds use the checked-in prepared PNG and do not run the extraction script.
