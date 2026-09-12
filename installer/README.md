# SIGMA desktop packaging

For the current 0.0.6 Apple Silicon app, use the [local-source build](#current-local-source-apple-silicon-build)
instructions below. They preserve the canonical checkout's current UI wording.
The older cross-platform instructions in the next section are historical.

The current Apple Silicon title-alignment update is in
`../../SIGMA-Desktop-0.0.6/centered-title/`. `mac_window.py` and the small
`mac_titlebar.m` bridge center a passive label in the existing native title bar,
without adding a toolbar or changing the core/plugin or interface wording.
Native close/minimize/fullscreen controls are retained. The geometry regression
measures the actual AppKit title at several window widths, and the native
launcher smoke test checks the packaged app's title center as well.

## Historical cross-platform packaging reference

Native, offline installers for the **unchanged published napari-sigma 0.0.5 wheel**.
The desktop launcher opens napari and the SIGMA panel automatically. No terminal,
Python installation, shell initialization, or administrator rights are needed for
a per-user installation. On macOS, drag the app from its DMG into Applications
(or your personal Applications folder); delete the app to uninstall its runtime.

Application, Dock and shortcut names are always **SIGMA**, without a version
suffix: macOS puts those in the menu bar and Dock, where a long name does not
fit. The napari window title bar and the SIGMA panel's own title bar both read
**SIGMA (Structurely-aware Intensity-ordered GMM-MRF Algorithm)**, using the
author's exact spelling. When a narrow display tabifies the panel beside the
layer controls, that same full name becomes its tab label.
Internal bundle metadata and installer filenames retain
the core version for compatibility checks and diagnosis. The user-supplied
`assets/sigma-logo-original.png` is retained. `prepare_logo.py` deterministically
extracts the emblem, removes the lower wordmark and white exterior, and preserves
its interior artwork; the result `assets/sigma-logo.png` has real RGBA transparency.
That prepared PNG is copied unchanged to the runtime and only resized when
encoding ICO/ICNS. The same artwork appears on the transparent startup screen
and Qt windows. App-icon encoding is tested for pixel and alpha preservation.

`mac_icon.py` also gives the disk image that artwork. The image is built
read/write so its volume root can be flagged `kHasCustomIcon` next to the
staged `.VolumeIcon.icns`, then compressed to UDZO exactly as before, so the
mounted volume shows the emblem. The finished `.dmg` file additionally gets an
`icns` resource written straight into its resource fork, without the deprecated
Rez/DeRez/SetFile tools. That fork is filesystem metadata: it survives a local
copy or `ditto`, but a zip, a CI artifact or a browser download strips it, so a
downloaded `.dmg` will still show Finder's generic disk-image icon. The volume
icon travels inside the image and is always preserved.

The desktop app uses Matplotlib 3.11.1's official `MPL_IGNORE_SYSTEM_FONTS=1`
setting and a pre-generated index containing only the fonts shipped in that wheel.
The builder generates the index in a separate disposable, hash-locked font-only
venv (never in the conda runtime that constructor packages). Relative font paths,
exact Matplotlib version, and index/font hashes are checked on the user's machine.
A missing or corrupt per-user cache is restored atomically from the bundled index;
it does not trigger operating-system font discovery. Damaged installed assets or
incompatible library versions produce an explicit error instead of a silent scan.
No system-wide font settings, Matplotlib source files, or other Python environments
are changed. Chart fonts are the bundled DejaVu/STIX and standard Matplotlib fonts;
custom system fonts and CJK chart labels require an explicitly supplied matching
font rather than automatic system discovery. Qt's normal UI fonts are unaffected.

The desktop adapter still prepares Matplotlib's non-GUI modules in a single
background worker. Opening Morphology Analysis while it is pending shows
a loading message without blocking the Qt event loop. Figures and Qt canvases
are created only on the GUI thread after preparation completes. Closing a panel
never waits for font enumeration. This desktop-layer improvement leaves the
published core wheel and scientific algorithms unchanged; a separately installed
PyPI plugin does not acquire this desktop adapter automatically.

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

The builder uses an isolated conda environment to download binary wheels and
generate a SHA-256 requirements lock and font index. Windows retains its NSIS/conda
installation workflow. Mac apps instead embed an official, relocatable
[python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901)
runtime (Python 3.13.15 ARM / 3.11.16 Intel), pinned by archive SHA-256. The native
launcher derives its runtime path from its own bundle and embeds Python using
isolated PyConfig initialization. There is no conda-unpack step, first-run repair,
external /Library runtime, PATH dependency, or write into the application bundle.
Matplotlib/Numba caches and preferences remain per-user outside the app.

Windows Torch is obtained only from PyTorch's official CPU index; other wheels use
PyPI. Both platforms install wheels using `--no-index --require-hashes --no-deps`
and run `pip check` (at build time for Mac, install time for Windows). End-user
installation is fully offline. Windows retains the wheel archives. Mac retains
installed package/license files plus the complete Python runtime license directory
and upstream PYTHON.json; duplicate wheel archives are not shipped in the app.
Exact dependency resolutions are included beside and inside each artifact.
To intentionally update dependencies, build in a new work directory and re-test.

The build runtime must remain conda-only until constructor has finished: do not
install the wheelhouse into that environment manually. This keeps conda's package
inventory complete. `--prepare-only` can be used to inspect payload/configuration.
For launcher/configuration-only fixes, `--reuse-wheelhouse` reuses the previous
hash-verified wheel set (and puts Windows constructor in offline mode). It requires
the existing build/runtime/cache directories and refuses altered wheel files.
Mac reuses a hash-checked Python archive cache but requires a fresh `mac-dmg`
staging directory and output DMG path; existing apps are never silently overwritten.
DMGs use APFS (supported by all target macOS versions). Disk-image creation emits
verbose diagnostics and has a 15-minute timeout; a stalled OS utility is a failed
build, never a distributable artifact. The CI retains failure diagnostics.
Windows full builds require the standalone conda executable to run successfully;
restricted sandbox semaphore failures must be resolved on a normal build runner,
not treated as a successful build. SHA-256 files accompany the generated installers.

## Verification and CI

The dedicated `Desktop installers (unsigned)` workflow builds all three native
targets, installs the actual generated artifact in a disposable runner, and runs:

- private import-origin and dependency checks;
- GUI startup, auto-opened SIGMA panel, and screenshot;
- calibrated uint64 TIFF round trip through napari's reader protocol;
- CPU Frangi and segmentation smoke tests;
- the existing core regression suite against the **installed wheel**, not `src/`;
- shortcut existence/names, installed artwork, native icon and Windows uninstallation checks.
- Mac native startup directly from the read-only DMG; copying and moving the app
  to a path containing spaces and Unicode, with the original build directory
  renamed/unavailable; full regression tests after ejecting the DMG; signature
  verification before/after tests; deletion of the test app, confirming no runtime
  was created in /Library or ~/Library. No real user's installation is deleted.
- desktop GUI regressions for pending/failed imports, responsive tab changes,
  main-thread canvas creation, shared preparation, and closing during loading.
- isolated cold-font tests for missing/corrupt caches, relocated installations,
  incompatible versions, damaged assets, and real PNG/SVG/PDF rendering. A call
  recorder asserts zero font-discovery calls during these tests and the actual
  installed app smoke launch; this is independent of timing thresholds.

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
Mac apps have an ad-hoc integrity signature for native execution, not an identified
publisher signature or notarization. Production distribution needs the publisher's
Apple Developer ID Application certificate/notarization and a Windows Authenticode
signing identity. A Developer ID Installer certificate is not needed for this DMG.
Neither secrets nor signing workarounds are embedded in this repository.

Third-party components retain their own licenses in the installation and wheel
archives. Review applicable redistribution requirements before public distribution.
The included user guide describes supported platforms, logs and uninstalling.
Quit Mac SIGMA before replacing the app. Old PKG runtimes are deliberately not
deleted or reused by this new layout; the guide describes their separate cleanup.
Uninstall the previous Windows build before replacement. The stable SIGMA app name
is shared across versions, so side-by-side shortcuts are not supported.
Core Python-package publishing is independent of this build.

## Current local-source Apple Silicon build

This maintained packaging directory was restored from the untouched
`../sync-0.0.6-20260911/installer` snapshot. The canonical core remains at
`../../napari-sigma`; do not maintain another core checkout here or in `/tmp`.
Claude's desktop wording, layout and transparent logo are preserved.

Use `build_current_macos.py` for local builds, not the older PyPI-based
`build.py` command above. It builds a fresh wheel from the canonical checkout,
compares every packaged Python/YAML file byte-for-byte with that checkout,
verifies cached dependencies before reuse, and records source hashes and the
Git commit in `source-provenance.json`. Existing installed apps are untouched.

```sh
/private/tmp/sigma-desktop-build/tools/bin/python -B build_current_macos.py \
  --source-dir /Users/huibao/Documents/Segmentation/napari-sigma \
  --build-python /Users/huibao/miniconda3/envs/sigma-qt6/bin/python \
  --dependency-cache /private/tmp/sigma-desktop-build/mac-arm64-v2 \
  --work-dir /private/tmp/sigma-desktop-build/mac-arm64-006-current \
  --output-dir /Users/huibao/Documents/Segmentation/SIGMA-Desktop-0.0.6
```

Use a fresh work directory for each build. `hdiutil` requires normal macOS
disk-image permissions and cannot run inside a restrictive process sandbox.
The output directory is persistent; temporary work directories contain only
build artifacts. Keep the hash-locked dependency cache for offline rebuilds.

Verify the resulting DMG using `ci_macos_app.py --dmg PATH --build-dir WORK
--output-dir CHECKS --source-dir CANONICAL_SOURCE`. This launches the native app
from a read-only image, relocates a test-owned copy, tests the installed core,
and removes only that temporary copy. Existing SIGMA installations and user
preferences are preserved. Verification uses disposable home directories.
