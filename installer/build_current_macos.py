"""Build the canonical local SIGMA source using verified offline dependencies.

No core source is copied into this packaging directory. The old packaging
snapshot is retained separately and is never modified by this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from email.parser import BytesParser

import build
import mac_app

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source_wheel(source, wheel):
    """Require byte-for-byte equality, including the current UI wording."""
    files = {}
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist()
                    if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Invalid local wheel metadata")
        info = BytesParser().parsebytes(archive.read(metadata[0]))
        if info["Name"] != "napari-sigma":
            raise ValueError("The local wheel must contain napari-sigma")
        for path in sorted((source / "src").rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml"}:
                continue
            relative = path.relative_to(source / "src").as_posix()
            if archive.read(relative) != path.read_bytes():
                raise ValueError(f"Wheel differs from current source: {relative}")
            files[relative] = digest(path)
        bundled = {name for name in archive.namelist()
                   if name.endswith((".py", ".yaml"))}
        if bundled != set(files):
            raise ValueError("Wheel contains missing or stale source files")
    return info["Version"], files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-python", type=Path, required=True)
    parser.add_argument("--dependency-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = build.native_platform()
    if target != "osx-arm64":
        parser.error("This entry point currently verifies Apple Silicon only")
    source, cache, work, output = [path.resolve() for path in
                                  (args.source_dir, args.dependency_cache, args.work_dir, args.output_dir)]
    if work.exists():
        raise FileExistsError(f"Use a fresh build directory: {work}")
    if not (source / "pyproject.toml").is_file() or not (source / "tests").is_dir():
        raise ValueError("A complete canonical source checkout is required")
    old_bundle = json.loads((cache / "payload/bundle.json").read_text())
    records = build.validated_cached_wheels(cache / "payload", target,
                                           version=old_bundle["sigma_version"])
    work.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)
    local_wheels = work / "current-core"
    local_wheels.mkdir()
    build.run([args.build_python, "-I", "-B", "-m", "pip", "--isolated", "wheel",
               "--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir",
               "--disable-pip-version-check", "--wheel-dir", local_wheels, source])
    wheels = list(local_wheels.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one freshly built SIGMA wheel")
    version, source_files = verify_source_wheel(source, wheels[0])
    if (output / f"SIGMA-{version}-macOS-AppleSilicon-unsigned.dmg").exists():
        raise FileExistsError("Refusing to overwrite an existing disk image")
    build.VERSION = version
    payload = work / "payload"
    wheelhouse = payload / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    for record in records:
        if record["name"] != "napari-sigma":
            shutil.copy2(cache / "payload/wheelhouse" / record["filename"], wheelhouse)
    shutil.copy2(wheels[0], wheelhouse)
    records = build.lock_wheels(wheelhouse, payload / "requirements.lock")
    font_cache = build.build_font_index(args.build_python, work, payload, records)
    for name in ("launch.py", "desktop_widget.py", "font_cache.py", "mac_window.py", "QUICKSTART.txt", "NOTICE.txt"):
        shutil.copy2(HERE / name, payload / name)
    shutil.copy2(source / "LICENSE", payload / "SIGMA-LICENSE.txt")
    build.make_icons(payload)
    desktop_files = {name: digest(HERE / name) for name in
                     ("launch.py", "desktop_widget.py", "font_cache.py", "mac_window.py",
                      "mac_titlebar.m", "mac_launcher.m", "assets/sigma-logo.png")}
    provenance = {
        "source_directory": str(source),
        "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True),
        "core_wheel": {"filename": wheels[0].name, "sha256": digest(wheels[0])},
        "core_files": source_files, "desktop_files": desktop_files,
    }
    bundle = {
        "schema": 1, "sigma_version": version, "platform": target, "default_device": "mps",
        "branding": {"name": "SIGMA", "logo_sha256": digest(payload / "sigma.png")},
        "font_cache": font_cache, "signed": False, "wheels": records,
        "source_commit": provenance["source_commit"],
        "core_wheel_sha256": provenance["core_wheel"]["sha256"],
    }
    (payload / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (output / "source-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for name in ("requirements.lock", "QUICKSTART.txt"):
        shutil.copy2(payload / name, output / name)
    runtime = mac_app.runtime_record(target)
    archive = mac_app.checked_archive(cache / "portable-python-cache", runtime)
    runtime_cache = work / "portable-python-cache"
    runtime_cache.mkdir()
    shutil.copy2(archive, runtime_cache / archive.name)
    mac_app.build_app(target, version, work, payload, output)
    print(f"Built current source {version}: {output}", flush=True)


if __name__ == "__main__":
    main()
