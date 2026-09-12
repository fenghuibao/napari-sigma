"""Build one offline Windows EXE from the current core and verified CUDA wheels.

Wheels are installed at build time, not shipped a second time or downloaded by
Setup. A portable CPython runtime is compressed with solid LZMA2 by Inno Setup.
The previous tested ZIP is a dependency cache only, never a source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

import build
from build_current_macos import digest, verify_source_wheel

HERE = Path(__file__).resolve().parent
CACHE_SHA256 = "8efd4490b45b1be4f9c3d36f9b4cd094d51f68fff4b831303ae9fd66184d0ca2"
PYTHON = {
    "version": "3.13.15", "provider": "astral-sh/python-build-standalone",
    "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.13.15%2B20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
    "sha256": "63d263ab0162f34a241a56dc5b283c22d6e131f5516117e6a921350c69ba7d4f",
}
INNO = {
    "version": "7.1.0",
    "url": "https://github.com/jrsoftware/issrc/releases/download/is-7_1_0/innosetup-7.1.0-x64.exe",
    "sha256": "0362a383ed217d4c4239b5933866dd96d3eb2102737da92f80f6057a4b40df2f",
}


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def download(record, path):
    build.run(["curl.exe", "--fail", "--location", "--retry", "3", "--proto", "=https",
               "--tlsv1.2", "--output", path, record["url"]])
    check_hash(path, record["sha256"])
    return path


def extract_wheels(package, destination):
    check_hash(package, CACHE_SHA256)
    destination.mkdir()
    with zipfile.ZipFile(package) as archive:
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            if len(name.parts) != 2 or name.parts[0] != "wheelhouse" or name.suffix != ".whl":
                continue
            if "\\" in name.name or name.name in {".", ".."}:
                raise ValueError("Invalid wheel path")
            target = destination / name.name
            with archive.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)


def pip_environment(work):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("PIP_", "PYTHON"))}
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str(work / "font-builder-cache"), MPL_IGNORE_SYSTEM_FONTS="1")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dependency-package", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if build.native_platform() != "win-64":
        parser.error("Build and verify on a native Windows x64 runner")
    source, package, work, output = [p.resolve() for p in (
        args.source_dir, args.dependency_package, args.work_dir, args.output_dir)]
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    wheels = work / "wheelhouse"
    extract_wheels(package, wheels)
    # Replace only the core wheel with a fresh build of the requested checkout.
    current = work / "current-core"
    current.mkdir()
    build.run([sys.executable, "-I", "-B", "-m", "pip", "--isolated", "wheel",
               "--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir",
               "--disable-pip-version-check", "--wheel-dir", current, source])
    wheel, = current.glob("*.whl")
    version, core_files = verify_source_wheel(source, wheel)
    if version != build.VERSION:
        raise ValueError("Packaging version does not match current source")
    for old in wheels.glob("napari_sigma-*.whl"):
        old.unlink()
    shutil.copy2(wheel, wheels)
    lock = work / "requirements.lock"
    records = build.lock_wheels(wheels, lock)
    torch = next(r for r in records if r["name"] == "torch")
    if torch["version"] != build.WINDOWS_TORCH:
        raise ValueError("The full CUDA-enabled Torch wheel is required")

    archive = download(PYTHON, work / "python.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    runtime = work / "python"
    python = runtime / "python.exe"
    if not python.is_file() or not (runtime / "pythonw.exe").is_file():
        raise ValueError("Portable Python layout is not supported")
    env = pip_environment(work)
    build.run([python, "-I", "-B", "-c",
               f"import sys; assert sys.version.split()[0] == {PYTHON['version']!r}"], env=env)
    build.run([python, "-I", "-B", "-m", "pip", "--isolated", "install", "--no-index",
               "--find-links", wheels, "--require-hashes", "--no-deps", "--no-compile",
               "--no-cache-dir", "--disable-pip-version-check", "-r", lock], env=env)
    build.run([python, "-I", "-B", "-m", "pip", "check"], env=env)
    desktop = runtime / "sigma-desktop"
    desktop.mkdir()
    for name in ("launch.py", "desktop_widget.py", "font_cache.py", "mac_window.py",
                 "record_windows_install.py", "QUICKSTART.txt", "NOTICE.txt"):
        shutil.copy2(HERE / name, desktop / name)
    shutil.copy2(source / "LICENSE", desktop / "SIGMA-LICENSE.txt")
    shutil.copy2(lock, desktop / "requirements.lock")
    build.make_icons(desktop)
    (desktop / "sigma.icns").unlink()  # Mac-only format; retain PNG and multi-size ICO.
    build.run(["pwsh", "-NoProfile", "-File", HERE / "prepare_windows_crt.ps1", "-Runtime", runtime])
    build.run([python, "-I", "-B", HERE / "generate_font_cache.py", "--output", desktop / "font-cache"], env=env)
    core_record = next(r for r in records if r["name"] == "napari-sigma")
    provenance = {
        "source_commit": subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip(),
        "core_files": core_files, "core_wheel": core_record,
        "desktop_files": {name: digest(HERE / name) for name in (
            "launch.py", "desktop_widget.py", "font_cache.py", "mac_window.py",
            "mac_titlebar.m", "mac_launcher.m", "assets/sigma-logo.png")},
        "packaging": "windows-inno-standalone", "python": PYTHON, "inno_setup": INNO,
        "dependency_cache_sha256": CACHE_SHA256,
    }
    bundle = {
        "schema": 1, "sigma_version": version, "platform": "win-64", "default_device": "cuda",
        "packaging": "windows-inno-standalone", "wheelhouse_location": "installed-at-build-time",
        "source_commit": provenance["source_commit"], "core_wheel_sha256": core_record["sha256"],
        "branding": {"name": "SIGMA", "logo_sha256": digest(desktop / "sigma.png")},
        "font_cache": {"mode": "bundled-only", "manifest_sha256": digest(desktop / "font-cache/manifest.json")},
        "signed": False, "wheels": records, "python": PYTHON,
    }
    (desktop / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    # Reuse verification records, but deliver no duplicate wheel archives/cache.
    for filename, data in (("source-provenance.json", provenance), ("bundle-win-64.json", bundle)):
        (output / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
    shutil.copy2(lock, output / "requirements-win-64.lock")
    for name in ("QUICKSTART.txt", "NOTICE.txt"):
        shutil.copy2(HERE / name, output / name)
    # Include an exact payload manifest for post-install/relocation integrity checks.
    files = {p.relative_to(runtime).as_posix(): digest(p) for p in sorted(runtime.rglob("*")) if p.is_file()}
    (output / "payload-sha256.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(f"Installed payload: {len(files)} files, {sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file()):,} bytes", flush=True)
    compiler_setup = download(INNO, work / "inno-setup.exe")
    compiler = work / "inno-compiler"
    build.run([compiler_setup, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER",
               f"/DIR={compiler}", f"/LOG={work / 'inno-compiler-install.log'}"], timeout=300)
    build.run([compiler / "ISCC.exe", f"/DRuntimeDir={runtime}", f"/DOutputDir={output}",
               f"/DAppVersion={version}", HERE / "windows_installer.iss"], timeout=5400)
    exe, = output.glob("*.exe")
    if list(output.glob("*.bin")):
        raise RuntimeError("Split payload files are not permitted")
    exe.with_suffix(".exe.sha256").write_text(f"{digest(exe)}  {exe.name}\n", encoding="utf-8")
    (output / "package-size.json").write_text(json.dumps({
        "filename": exe.name, "bytes": exe.stat().st_size,
        "fits_github_release_asset": exe.stat().st_size < 2 ** 31,
    }, indent=2), encoding="utf-8")
    print(f"Single offline EXE: {exe.name}, {exe.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
