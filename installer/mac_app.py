"""Build a relocatable, offline SIGMA.app and drag-to-install disk image.

The conda environment is a build tool only. The delivered app contains the
official python-build-standalone runtime plus the same hash-locked core wheels.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RELEASE = "20260901"
RUNTIMES = {
    "osx-arm64": {
        "python": "3.13.15", "triple": "aarch64-apple-darwin",
        "sha256": "46685a8e6dbad3e94534e0d73f483d08149867707200aa39e99690b76002053f",
    },
    "osx-64": {
        "python": "3.11.16", "triple": "x86_64-apple-darwin",
        "sha256": "b81cc36535920c15c0f4e2dd055a27b02dc6222ca18966eb8cee9b408f8bc220",
    },
}
MACHO_MAGIC = {bytes.fromhex(value) for value in (
    "feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe", "bebafeca", "cafebabf", "bfbafeca")}


def run(command, **kwargs):
    command = [str(arg) for arg in command]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def runtime_record(target):
    record = dict(RUNTIMES[target])
    record["release"] = RELEASE
    record["provider"] = "astral-sh/python-build-standalone"
    record["filename"] = f"cpython-{record['python']}+{RELEASE}-{record['triple']}-pgo+lto-full.tar.zst"
    record["url"] = f"https://github.com/{record['provider']}/releases/download/{RELEASE}/{record['filename'].replace('+', '%2B')}"
    return record


def checked_archive(cache, record):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / record["filename"]
    if not archive.exists():
        # Download to a temporary file so an interrupted request is not reused.
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            download = Path(directory) / "download"
            run(["/usr/bin/curl", "--fail", "--location", "--retry", "3",
                 "--proto", "=https", "--tlsv1.2", "--output", download, record["url"]])
            check_digest(download, record["sha256"])
            download.replace(archive)
    check_digest(archive, record["sha256"])
    return archive


def check_digest(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Python archive SHA-256 mismatch: {path}")


def check_internal_links(root):
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink() and (os.path.isabs(os.readlink(path)) or
                                  not path.resolve().is_relative_to(root) or not path.exists()):
            raise ValueError(f"App contains an external or broken symlink: {path}")


def app_plist(version):
    return {
        "CFBundleName": "SIGMA", "CFBundleDisplayName": "SIGMA",
        "CFBundleExecutable": "SIGMA", "CFBundleIdentifier": "org.fenghuibao.sigma.desktop",
        "CFBundlePackageType": "APPL", "CFBundleVersion": version,
        "CFBundleShortVersionString": version, "CFBundleIconFile": "sigma.icns",
        "LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
    }


def sign_app(app):
    # Sign nested code inside-out. Ad-hoc signing permits local execution on
    # Apple Silicon; it is NOT Developer ID signing or notarization.
    binaries = []
    for path in app.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                if stream.read(4) in MACHO_MAGIC:
                    binaries.append(path)
    print(f"Ad-hoc signing {len(binaries)} native binaries…", flush=True)
    for path in sorted(binaries):
        result = subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", "--timestamp=none", str(path)],
                                capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Signing failed: {path}\n{result.stderr.decode(errors='replace')}")
    bundles = [path for path in app.rglob("*") if path.is_dir() and not path.is_symlink()
               and path.suffix in {".framework", ".app"}]
    for path in sorted(bundles, key=lambda path: len(path.parts), reverse=True) + [app]:
        run(["/usr/bin/codesign", "--force", "--sign", "-", "--timestamp=none", path])
    run(["/usr/bin/codesign", "--verify", "--deep", "--strict", app])


def build_app(target, version, work, payload, output):
    record = runtime_record(target)
    archive = checked_archive(work / "portable-python-cache", record)
    # A fresh staging directory prevents stale packages/scripts from surviving
    # rebuilds. Existing output applications are never silently overwritten.
    staging = work / "mac-dmg"
    if staging.exists():
        raise FileExistsError(f"Use a fresh staging directory (already exists): {staging}")
    app = staging / "SIGMA.app"
    resources = app / "Contents/Resources"
    native = app / "Contents/MacOS"
    resources.mkdir(parents=True)
    native.mkdir()
    with tempfile.TemporaryDirectory(prefix="portable-extract-", dir=work) as directory:
        # Only the hash-verified official archive is extracted. Build objects
        # are excluded; retain all runtime files, metadata and runtime licenses.
        run(["/usr/bin/tar", "-xf", archive, "-C", directory,
             "python/install", "python/licenses", "python/PYTHON.json"])
        extracted = Path(directory) / "python"
        shutil.move(extracted / "install", resources / "runtime")
        shutil.copytree(extracted / "licenses", resources / "python-licenses")
        shutil.copy2(extracted / "PYTHON.json", resources / "PYTHON.json")
    prefix = resources / "runtime"
    python = prefix / "bin/python3"
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE="1")
    run([python, "-I", "-B", "-c", f"import sys; assert sys.version.split()[0] == {record['python']!r}"])
    run([python, "-I", "-B", "-m", "pip", "--isolated", "install", "--no-index",
         "--find-links", payload / "wheelhouse", "--require-hashes", "--no-deps",
         "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "-r", payload / "requirements.lock"], env=env)
    run([python, "-I", "-B", "-m", "pip", "check"], env=env)
    # Installed dist-info/licenses are retained; no duplicate wheelhouse is
    # needed at runtime. Do not ship the old external-prefix install machinery.
    desktop = prefix / "sigma-desktop"
    desktop.mkdir()
    for name in ("launch.py", "desktop_widget.py", "font_cache.py", "QUICKSTART.txt",
                 "NOTICE.txt", "SIGMA-LICENSE.txt", "requirements.lock", "bundle.json",
                 "sigma.png", "sigma.icns", "sigma.ico"):
        shutil.copy2(payload / name, desktop / name)
    shutil.copytree(payload / "font-cache", desktop / "font-cache")
    bundle = json.loads((desktop / "bundle.json").read_text(encoding="utf-8"))
    bundle.update(packaging="self-contained-app", runtime=record, code_signature="ad-hoc")
    (desktop / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    shutil.copy2(desktop / "bundle.json", output / f"bundle-{target}.json")
    shutil.copy2(payload / "sigma.icns", resources / "sigma.icns")
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps(app_plist(version)))
    minor = ".".join(record["python"].split(".")[:2])
    run(["/usr/bin/clang", "-fobjc-arc", "-mmacosx-version-min=14.0", "-framework", "Cocoa",
         "-I", prefix / f"include/python{minor}", "-L", prefix / "lib", f"-lpython{minor}",
         "-Wl,-rpath,@executable_path/../Resources/runtime/lib", HERE / "mac_launcher.m",
         "-o", native / "SIGMA"])
    check_internal_links(app)
    sign_app(app)
    (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
    shutil.copy2(HERE / "QUICKSTART.txt", staging / "READ ME.txt")
    arch = "AppleSilicon" if target == "osx-arm64" else "Intel"
    dmg = output / f"SIGMA-{version}-macOS-{arch}-unsigned.dmg"
    run(["/usr/bin/hdiutil", "create", "-volname", "SIGMA", "-srcfolder", staging,
         "-format", "UDZO", "-fs", "HFS+", dmg])
    with dmg.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    dmg.with_suffix(".dmg.sha256").write_text(f"{digest}  {dmg.name}\n", encoding="utf-8")
    return app
