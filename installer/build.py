"""Build offline SIGMA installers on each target OS/architecture.

The current source checkout is packaged unchanged. A small desktop launcher is shipped
alongside a private Python runtime and a hash-locked wheelhouse.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import yaml

HERE = Path(__file__).resolve().parent
VERSION = "0.0.6"
PLATFORMS = {"osx-arm64", "osx-64", "win-64"}
WINDOWS_TORCH = "2.13.0+cu130"
WINDOWS_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def native_platform() -> str:
    if sys.platform == "win32" and platform.machine().lower() in {"amd64", "x86_64"}:
        return "win-64"
    if sys.platform == "darwin":
        return "osx-arm64" if platform.machine() == "arm64" else "osx-64"
    raise RuntimeError("Build on a native Windows x64 or macOS runner.")


def run(command, **kwargs):
    command = [str(arg) for arg in command]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def requirements(target: str) -> list[str]:
    # Intel macOS stopped receiving official PyTorch wheels after 2.2.x.
    # That wheel must use NumPy 1.x; it is tested separately from modern Torch.
    torch = "2.2.2" if target == "osx-64" else "2.13.0"
    if target == "win-64":
        torch = WINDOWS_TORCH
    result = [
        f"napari-sigma[all]=={VERSION}", "napari==0.9.0",
        f"torch=={torch}", f"numpy=={'1.26.4' if target == 'osx-64' else '2.5.2'}",
        "PyQt6==6.11.0", "PyQt6-Qt6==6.11.2", "qtpy==2.4.3",
        "openpyxl==3.1.5", "scikit-image==0.26.0",
        "matplotlib==3.11.1",
    ]
    # 2026.3.3 still supports Python 3.11/NumPy 1.x and includes the upstream
    # high-resolution TIFF rational rounding fix (2026.2.20, issue #318).
    result.append("tifffile==2026.3.3" if target == "osx-64" else "tifffile==2026.8.23")
    return result


def lock_wheels(wheelhouse: Path, destination: Path) -> list[dict]:
    records = []
    seen = set()
    for wheel in sorted(wheelhouse.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            # setuptools also vendors other distributions with nested metadata.
            entries = [name for name in archive.namelist()
                       if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise ValueError(f"Expected one wheel metadata file: {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(entries[0]))
        name = re.sub(r"[-_.]+", "-", metadata["Name"]).lower()
        if name in seen:
            raise ValueError(f"Duplicate wheel for {name}; start with a clean wheelhouse.")
        seen.add(name)
        with wheel.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"name": name, "version": metadata["Version"],
                        "filename": wheel.name, "sha256": digest})
    if "napari-sigma" not in seen or "torch" not in seen:
        raise ValueError("Wheelhouse is missing SIGMA or Torch.")
    destination.write_text("\n".join(
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}"
        for item in records) + "\n", encoding="utf-8")
    return records


def validated_cached_wheels(payload: Path, target: str, *, version=None) -> list[dict]:
    """Reuse only the exact previously resolved wheel set, with no downloads."""
    bundle = json.loads((payload / "bundle.json").read_text(encoding="utf-8"))
    if bundle["platform"] != target or bundle["sigma_version"] != (version or VERSION):
        raise ValueError("Cached bundle is for a different platform or SIGMA version")
    with tempfile.TemporaryDirectory() as directory:
        lock = Path(directory) / "requirements.lock"
        records = lock_wheels(payload / "wheelhouse", lock)
        if records != bundle["wheels"] or lock.read_bytes() != (payload / "requirements.lock").read_bytes():
            raise ValueError("Cached wheels or hash lock changed; use a fresh build directory")
    return records


def build_font_index(python: Path, work: Path, payload: Path, records: list[dict]) -> dict:
    # Never pip-install into the conda runtime before constructor inventories it.
    # The font-only venv is disposable and excluded from the delivered runtime.
    names = {"matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "numpy",
             "packaging", "pillow", "pyparsing", "python-dateutil", "six"}
    selected = [record for record in records if record["name"] in names]
    if {record["name"] for record in selected} != names:
        raise RuntimeError("Locked wheelhouse is missing a font-index build dependency")
    lock = work / "font-requirements.lock"
    lock.write_text("\n".join(
        f"{record['name']}=={record['version']} --hash=sha256:{record['sha256']}"
        for record in selected) + "\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    env.update(PIP_CONFIG_FILE=os.devnull, MPLCONFIGDIR=str(work / "builder-font-cache"),
               MPL_IGNORE_SYSTEM_FONTS="1")
    with tempfile.TemporaryDirectory(prefix="font-index-", dir=work) as directory:
        environment = Path(directory)
        run([python, "-I", "-m", "venv", "--without-pip", environment])
        pip = [python, "-I", "-m", "pip", "--isolated", "--python", environment]
        run(pip + ["install", "--no-index", "--find-links", payload / "wheelhouse", "--require-hashes",
                   "--no-deps", "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "-r", lock], env=env)
        run(pip + ["check"], env=env)
        font_python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        run([font_python, "-I", HERE / "generate_font_cache.py", "--output", payload / "font-cache"], env=env)
    manifest = (payload / "font-cache/manifest.json").read_bytes()
    return {"mode": "bundled-only", "manifest_sha256": hashlib.sha256(manifest).hexdigest()}


def menu_metadata(version: str) -> dict:
    return {
        "$schema": "https://schemas.conda.org/menuinst-1-1-3.schema.json",
        "menu_name": "SIGMA",
        "menu_items": [{
            "name": "SIGMA", "description": "Microscopy segmentation and analysis",
            "command": ["{{ PYTHON }}", "-I", "{{ PREFIX }}/sigma-desktop/launch.py"],
            "icon": "{{ PREFIX }}/sigma-desktop/sigma.{{ ICON_EXT }}",
            "activate": False, "terminal": False,
            "platforms": {
                "osx": {"CFBundleVersion": version,
                        "CFBundleDisplayName": "SIGMA",
                        "CFBundleName": "SIGMA",
                        "CFBundleIdentifier": "org.fenghuibao.sigma.desktop",
                        "LSMinimumSystemVersion": "14.0"},
                "win": {"command": ["{{ PYTHONW }}", "-I", "{{ PREFIX }}/sigma-desktop/launch.py"],
                        "desktop": True, "app_user_model_id": "SIGMA.Desktop"},
            },
        }],
    }


def make_icons(destination: Path):
    """Package the supplied artwork unchanged; resize only for native formats."""
    from PIL import Image
    source = HERE / "assets/sigma-logo.png"
    with Image.open(source) as original:
        if original.width != original.height or original.width < 1024:
            raise ValueError("The app logo must be square and at least 1024 pixels wide")
        image = original.convert("RGBA").resize((1024, 1024), Image.Resampling.LANCZOS)
    shutil.copy2(source, destination / "sigma.png")
    image.save(destination / "sigma.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    image.save(destination / "sigma.icns")


def constructor_config(target: str, runtime: Path, payload: Path) -> dict:
    config = {
        "name": "SIGMA", "version": VERSION, "company": "SIGMA",
        "environment": str(runtime), "channels": ["conda-forge"],
        "license_file": str(HERE / "NOTICE.txt"),
        "initialize_conda": False, "register_envs": False, "menu_packages": [],
        "keep_pkgs": False,
        "build_outputs": [{"hash": {"algorithm": "sha256"}}, "info.json", "pkgs_list"],
        "extra_files": [{str(path): str(Path("sigma-desktop") / path.relative_to(payload))}
                        for path in sorted(payload.rglob("*")) if path.is_file()
                        and not (target == "win-64" and path.is_relative_to(payload / "wheelhouse"))],
        "post_install": str(HERE / ("post_install.bat" if target == "win-64" else "post_install.sh")),
    }
    if target == "win-64":
        config.update(
            installer_type="exe", installer_filename="SIGMA-Setup.exe",
            default_prefix=f"%LOCALAPPDATA%\\SIGMA\\{VERSION}",
            default_prefix_domain_user=f"%LOCALAPPDATA%\\SIGMA\\{VERSION}",
            default_prefix_all_users=f"%ALLUSERSPROFILE%\\SIGMA\\{VERSION}",
            register_python=False, check_path_spaces=False,
            uninstall_name="SIGMA", pre_uninstall=str(HERE / "pre_uninstall.bat"),
            welcome_image_text="SIGMA", header_image_text="SIGMA",
            icon_image=str(payload / "sigma.ico"),
            conclusion_text="SIGMA is installed.\nOpen SIGMA from the Start menu or desktop.",
        )
    else:
        arch = "AppleSilicon" if target == "osx-arm64" else "Intel"
        config.update(
            installer_type="pkg", installer_filename=f"SIGMA-{VERSION}-macOS-{arch}-unsigned.pkg",
            default_location_pkg="Library", pkg_name=f"sigma-{VERSION}",
            pkg_domains={"enable_anywhere": False, "enable_currentUserHome": True, "enable_localSystem": True},
            virtual_specs=["__osx>=14"], reverse_domain_identifier="org.fenghuibao.sigma",
            welcome_image="", welcome_text="SIGMA microscopy analysis. No Python setup is required.",
            readme_file=str(HERE / "QUICKSTART.txt"),
            conclusion_text="Open SIGMA from your Applications folder. This test installer is not notarized.",
        )
    return config


def package_windows(output: Path, wheelhouse: Path) -> Path:
    """Keep the CUDA payload outside NSIS's 2 GiB executable limit, in one ZIP."""
    setup = output / "SIGMA-Setup.exe"
    if not setup.is_file():
        raise ValueError("Windows setup executable is missing")
    destination = output / f"SIGMA-{VERSION}-Windows-x64-CUDA-CPU-unsigned.zip"
    # Wheels and the EXE are already compressed. ZIP64 works with Explorer's
    # Extract All; archive.write streams even multi-gigabyte CUDA wheels.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(setup, setup.name)
        archive.write(output / "QUICKSTART.txt", "QUICKSTART.txt")
        for wheel in sorted(wheelhouse.glob("*.whl")):
            archive.write(wheel, f"wheelhouse/{wheel.name}")
    with destination.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{sha256}  {destination.name}\n", encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Current core checkout, built and verified byte-for-byte")
    parser.add_argument("--conda", default=shutil.which("conda"))
    parser.add_argument("--constructor", default=shutil.which("constructor"))
    parser.add_argument("--standalone-conda", type=Path,
                        default=Path(sys.prefix) / "standalone_conda/conda.exe")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-wheelhouse", action="store_true",
                        help="Rebuild with the previously locked dependencies; require a complete local cache")
    args = parser.parse_args()
    target = native_platform()
    if not args.conda or not args.constructor:
        parser.error("conda and constructor are required in the build environment")
    if not args.prepare_only and target == "win-64":
        # Constructor can otherwise fall back after a failed executable probe
        # and still emit an installer with incorrectly detected capabilities.
        run([args.standalone_conda, "--version"])
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    runtime = work / "runtime"
    payload = work / "payload"
    wheels = payload / "wheelhouse"
    for path in (work, output, wheels):
        path.mkdir(parents=True, exist_ok=True)
    source = args.source_dir.resolve()
    local_wheels = work / "current-core"
    local_wheels.mkdir(exist_ok=True)
    run([sys.executable, "-I", "-B", "-m", "pip", "--isolated", "wheel",
         "--no-deps", "--no-build-isolation", "--no-cache-dir",
         "--disable-pip-version-check", "--wheel-dir", local_wheels, source])
    from build_current_macos import verify_source_wheel, digest
    local_core = list(local_wheels.glob("*.whl"))
    if len(local_core) != 1:
        raise ValueError("Expected one wheel built from the current source")
    version, core_files = verify_source_wheel(source, local_core[0])
    if version != VERSION:
        raise ValueError(f"Update the packaging version to match the source: {version}")
    if not (runtime / "conda-meta" / "history").is_file():
        if args.reuse_wheelhouse:
            parser.error("--reuse-wheelhouse requires an existing runtime and payload")
        python_version = "3.11" if target == "osx-64" else "3.13"
        run([args.conda, "create", "--prefix", runtime, "--override-channels", "-c", "conda-forge",
             "--no-default-packages", f"python={python_version}", "pip", "menuinst=2.5.2", "--yes", "--quiet"])
    python = runtime / ("python.exe" if target == "win-64" else "bin/python")
    expected_minor = 11 if target == "osx-64" else 13
    run([python, "-I", "-c", f"import sys; assert sys.version_info[:2] == (3, {expected_minor})"])
    download = [python, "-I", "-m", "pip", "download", "--only-binary=:all:",
                "--disable-pip-version-check", "--progress-bar", "off", "--dest", wheels]
    if target.startswith("osx-"):
        # Building on a newer Mac must not silently select macOS 15+ wheels.
        download += ["--platform", "macosx_14_0_" + ("arm64" if target == "osx-arm64" else "x86_64")]
    if args.reuse_wheelhouse:
        records = validated_cached_wheels(payload, target)
        cached_core = next(record for record in records if record["name"] == "napari-sigma")
        verify_source_wheel(source, wheels / cached_core["filename"])
    else:
        if target == "win-64":
            run(download + ["--no-deps", "--index-url", WINDOWS_TORCH_INDEX, f"torch=={WINDOWS_TORCH}"])
        # Passing the local wheel explicitly prevents same-version PyPI code
        # from replacing current interface edits during dependency resolution.
        requested = [f"{local_core[0]}[all]"] + requirements(target)[1:]
        run(download + ["--index-url", "https://pypi.org/simple", "--find-links", wheels] + requested)
        records = lock_wheels(wheels, payload / "requirements.lock")
    core_record = next(record for record in records if record["name"] == "napari-sigma")
    if target == "win-64" and next(record["version"] for record in records if record["name"] == "torch") != WINDOWS_TORCH:
        raise ValueError("Windows requires the CUDA-enabled Torch wheel; use a fresh build directory")
    verify_source_wheel(source, wheels / core_record["filename"])
    provenance = {
        "source_commit": subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip(),
        "core_files": core_files, "core_wheel": core_record,
        "desktop_files": {name: digest(HERE / name) for name in
                          ("launch.py", "desktop_widget.py", "font_cache.py", "mac_window.py",
                           "mac_titlebar.m", "mac_launcher.m", "assets/sigma-logo.png")},
    }
    (output / "source-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    font_cache = build_font_index(python, work, payload, records)
    for name in ("launch.py", "desktop_widget.py", "font_cache.py", "mac_window.py", "install.py", "QUICKSTART.txt", "NOTICE.txt"):
        shutil.copy2(HERE / name, payload / name)
    shutil.copy2(source / "LICENSE", payload / "SIGMA-LICENSE.txt")
    make_icons(payload)
    (payload / "menu.json").write_text(json.dumps(menu_metadata(VERSION), indent=2), encoding="utf-8")
    (payload / "bundle.json").write_text(json.dumps({
        "schema": 1, "sigma_version": VERSION, "platform": target,
        "source_commit": provenance["source_commit"],
        "core_wheel_sha256": core_record["sha256"],
        # The panel lists only devices torch reports as usable, best first, so
        # there is no "auto" to request. Naming the platform's likely
        # accelerator is enough; launch.py ignores a device that is absent.
        "default_device": {"osx-arm64": "mps", "osx-64": "cpu", "win-64": "cuda"}[target],
        "wheelhouse_location": "next-to-installer" if target == "win-64" else "bundled",
        "branding": {"name": "SIGMA", "logo_sha256": hashlib.sha256((payload / "sigma.png").read_bytes()).hexdigest()},
        "font_cache": font_cache,
        "signed": False, "wheels": records,
    }, indent=2), encoding="utf-8")
    (work / "construct.yaml").write_text(yaml.safe_dump(
        constructor_config(target, runtime, payload), sort_keys=False), encoding="utf-8")
    shutil.copy2(payload / "requirements.lock", output / f"requirements-{target}.lock")
    shutil.copy2(payload / "bundle.json", output / f"bundle-{target}.json")
    shutil.copy2(HERE / "QUICKSTART.txt", output / "QUICKSTART.txt")
    if not args.prepare_only:
        if target.startswith("osx-"):
            from mac_app import build_app
            build_app(target, VERSION, work, payload, output)
        else:
            env = os.environ.copy()
            if args.reuse_wheelhouse:
                env["CONDA_OFFLINE"] = "true"
            run([args.constructor, "--conda-exe", args.standalone_conda,
                 "--output-dir", output, "--cache-dir", work / "constructor-cache", work], env=env)
            package_windows(output, wheels)
            # The EXE alone is incomplete. Deliver only the complete offline ZIP.
            (output / "SIGMA-Setup.exe").unlink()
    print(f"Bundle prepared for {target}: {output}", flush=True)


if __name__ == "__main__":
    main()
