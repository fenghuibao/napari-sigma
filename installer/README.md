# SIGMA desktop installers

Native, offline installers for the **unchanged published napari-sigma 0.0.5 wheel**.
The desktop launcher opens napari and the SIGMA panel automatically. No terminal,
Python installation, shell initialization, or administrator rights are needed for
a per-user installation. macOS also offers installation for all users.

| Target | Runtime | Compute default | Minimum OS |
| --- | --- | --- | --- |
| Apple Silicon | Python 3.13, Torch 2.13, NumPy 2.5 | auto (MPS/CPU) | macOS 14 |
| Intel Mac | Python 3.11, Torch 2.2.2, NumPy 1.26 | CPU | macOS 14 |
| Windows x64 | Python 3.13, Torch 2.13 CPU, NumPy 2.5 | CPU | Windows 10/11 |

Intel uses the last available official Intel macOS Torch wheels. Windows CUDA and
Windows ARM are not included. Separate backends are not promised bit-identical.
Intel TIFF I/O uses tifffile 2026.3.3, which still supports Python 3.11/NumPy 1.x
and includes the upstream high-resolution rational rounding fix. Calibration
tests retain the same precision thresholds as the modern platforms.
The CI uses macOS 15 and Windows Server 2022; this is not a substitute for testing
every supported consumer OS/hardware combination.
Windows rendering requires a working OpenGL driver. The GitHub Windows VM uses
a hash-pinned Mesa software driver installed **only on the disposable CI host**;
this driver is not bundled with SIGMA and no end-user system driver is changed.

## Build

Use a clean conda-forge build environment with Python 3.13, constructor 3.16.1,
conda-standalone, menuinst 2.5.2, Pillow, and PyYAML. Run on the **target architecture**:

```sh
python installer/build.py --work-dir /path/to/empty-build-dir --output-dir dist-desktop
python -m unittest discover -s installer/tests -v
```

The builder creates a separate minimal conda runtime. It downloads binary wheels,
generates a SHA-256 requirements lock and manifest, then constructs a PKG/NSIS
installer. Windows Torch is obtained only from PyTorch's official CPU index; the
remaining wheels use PyPI. Installation uses `--no-index --require-hashes --no-deps`
and runs `pip check` before creating shortcuts. Installation is fully offline.
Wheel archives are retained for third-party licenses and diagnosis (additional disk
space). Exact dependency resolutions are included beside and inside each artifact.
To intentionally update dependencies, build in a new work directory and re-test.

The build runtime must remain conda-only until constructor has finished: do not
install the wheelhouse into that environment manually. This keeps conda's package
inventory complete. `--prepare-only` can be used to inspect payload/configuration.
For launcher/configuration-only fixes, `--reuse-wheelhouse` reuses the previous
hash-verified wheel set and puts constructor in offline mode. It requires the
complete existing build/runtime/cache directories and refuses altered wheel files.
Full builds also require the standalone conda executable to run successfully;
restricted sandbox semaphore failures must be resolved on a normal build runner,
not treated as a successful build. SHA-256 files accompany the generated installers.

## Verification and CI

The dedicated `Desktop installers (unsigned)` workflow builds all three native
targets, installs the actual generated installer in a disposable runner, and runs:

- private import-origin and dependency checks;
- GUI startup, auto-opened SIGMA panel, and screenshot;
- calibrated uint64 TIFF round trip through napari's reader protocol;
- CPU Frangi and segmentation smoke tests;
- the existing core regression suite against the **installed wheel**, not `src/`;
- shortcut existence and Windows uninstallation checks.

The smoke-test TIFF directory is owned by the verifier and removed after the GUI
child exits; Windows cannot delete an actively displayed memory-mapped image.
Regression fixtures likewise remain until test locals/viewers have been released.
Cleanup errors are not ignored, and no pixel or numerical assertion is skipped.
On displays narrower than 1280 logical pixels, layer controls/list share tabs with
SIGMA to keep the image canvas visible; the smoke test also checks canvas size.

Artifacts are uploaded even on failure for diagnosis; **only artifacts from a
fully passing target should be distributed**. Logs/screenshots and the verification
record are included. The workflow does not upload to PyPI or create a public release.

## Signing and distribution

These are **unsigned test builds**. macOS Gatekeeper or Windows SmartScreen may
block a downloaded installer. Do not instruct users to disable OS security.
Production distribution needs the publisher's Apple Developer ID Installer and
Application certificates/notarization, and a Windows Authenticode signing identity.
Neither secrets nor signing workarounds are embedded in this repository.

Third-party components retain their own licenses in the installation and wheel
archives. Review applicable redistribution requirements before public distribution.
The included user guide describes supported platforms, logs, uninstalling, and
side-by-side upgrades. Core Python-package publishing is independent of this build.
