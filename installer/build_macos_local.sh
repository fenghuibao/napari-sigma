#!/bin/bash
# Build and verify a SIGMA disk image on this Mac, mirroring the CI job.
#
# Usage:  installer/build_macos_local.sh [OUTPUT_DIR]
#
# Produces the DMG for this Mac's own architecture, runs the same verification
# the CI runner does (read-only launch from the mounted image, volume icon,
# relocation, uninstall), and writes SHA256SUMS.txt next to the result.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$REPO/dist-desktop}"
WORK="${TMPDIR:-/tmp}/sigma-build-$(date +%Y%m%d-%H%M%S)"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mError: %s\033[0m\n' "$1" >&2; exit 1; }

step "Checking this machine"
[ "$(uname -s)" = "Darwin" ] || fail "This script builds the macOS app and must run on macOS."
echo "macOS $(sw_vers -productVersion) on $(uname -m)"
# The build compiles the native launcher and ad-hoc signs every Mach-O file.
xcode-select --print-path >/dev/null 2>&1 || fail "Xcode command line tools are missing. Run: xcode-select --install"
for tool in /usr/bin/clang /usr/bin/codesign /usr/bin/hdiutil /usr/bin/xattr; do
    [ -x "$tool" ] || fail "Missing required tool: $tool"
done

step "Checking build tools"
command -v conda >/dev/null 2>&1 || fail "conda is not on PATH. See the environment setup below."
for module in constructor menuinst PIL yaml pip setuptools; do
    python -c "import $module" >/dev/null 2>&1 || fail "Python module '$module' is missing in the active environment.
Create the builder environment the CI job uses:
  conda create -n sigma-builder -c conda-forge python=3.13 --yes
  conda activate sigma-builder
  conda install --override-channels -c conda-forge \\
      constructor=3.16.1 conda-standalone menuinst=2.5.2 pillow pyyaml pip setuptools --yes"
done
echo "Using $(python --version) from $(command -v python)"

# Verification preserves existing installations and deletes only its test copy.

step "Preparing output directory"
[ -e "$OUTPUT" ] && fail "Output directory already exists, refusing to overwrite: $OUTPUT"
mkdir -p "$OUTPUT"
echo "Output:   $OUTPUT"
echo "Scratch:  $WORK"

step "Packaging unit tests"
cd "$REPO"
python -m unittest discover -s installer/tests -v

step "Building the installer (this downloads the Python runtime and takes a while)"
python installer/build.py --source-dir "$REPO" --work-dir "$WORK" --output-dir "$OUTPUT"

step "Verifying the disk image"
shopt -s nullglob
IMAGES=("$OUTPUT"/*.dmg)
shopt -u nullglob
[ "${#IMAGES[@]}" -eq 1 ] || fail "Expected exactly one disk image in $OUTPUT, found ${#IMAGES[@]}"
DMG="${IMAGES[0]}"
python installer/ci_macos_app.py --dmg "$DMG" --build-dir "$WORK" --output-dir "$OUTPUT/verification" --source-dir "$REPO"

step "Recording checksums"
cd "$OUTPUT"
shasum -a 256 ./*.dmg > SHA256SUMS.txt
cat SHA256SUMS.txt

step "Done"
echo "Disk image:   $DMG"
echo "Verification: $OUTPUT/verification"
echo
echo "The icon on the .dmg file itself lives in its resource fork, so it"
echo "survives a Finder copy or 'ditto' but not a zip or a browser download."
echo "The volume icon travels inside the image and is always preserved."
echo
echo "Scratch directory left for inspection; remove it with:"
echo "  rm -rf $WORK"
