#!/usr/bin/env python3
"""Reproducible local build. Run with uv run --no-project --python 3.11.

Dependencies, build products and local exports stay in this project's
ignored .scratch directory. Python needs only its standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"
DEPS = SCRATCH / "deps"
DOWNLOADS = SCRATCH / "downloads"
BUILD = SCRATCH / "build"
GAME = ROOT / "godot"
LINUX = platform.system() == "Linux"
ARCH = platform.machine()
GODOT = DEPS / ("Godot_v4.5.2-stable_linux.x86_64" if LINUX else "Godot.app/Contents/MacOS/Godot")
EXPORT = SCRATCH / ("export/tennis-server.x86_64" if LINUX else "export/Tennis Sim Game.app")
GODOT_CPP_COMMIT = "e83fd0904c13356ed1d4c3d09f8bb9132bdc6b77"
GODOT_URL = "https://github.com/godotengine/godot-builds/releases/download/4.5.2-stable/"
DOWNLOAD_INFO = {
    "godot-linux.zip": (GODOT_URL + "Godot_v4.5.2-stable_linux.x86_64.zip", "87f6e6be292929e363d15ed9052f277b2ba4e95ed994e1e099048097be2dfd03"),
    "mujoco-linux.tar.gz": ("https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-linux-x86_64.tar.gz", "a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2"),
    "godot.zip": (GODOT_URL + "Godot_v4.5.2-stable_macos.universal.zip", "2a3f35cf5813b0d26e3f4c15dabc5e7c58407fceec7bae5291740772f72d141a"),
    "templates.tpz": (GODOT_URL + "Godot_v4.5.2-stable_export_templates.tpz", "ea50e5183f057d42b22120d0d0340205b98f51f8ff13c33c70fd4ac9077d2019"),
    "mujoco.dmg": ("https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-macos-universal2.dmg", "8410882d724c3637b935dc0482b0de90efed44b17bbcfcb4a165ee46285a4865"),
}


def run(*args: str | Path, cwd: Path = ROOT) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def check_godot_output(output: str, required: tuple[str, ...] = ()) -> None:
    """Godot can exit zero after a script load or assertion failure."""
    if re.search(r"(?m)^\s*(?:SCRIPT ERROR|ERROR):", output):
        raise RuntimeError("Godot reported an error; see the command output above")
    for marker in required:
        if marker not in output:
            raise RuntimeError(f"Godot did not complete the expected check: {marker}")


def run_godot(*args: str | Path, required: tuple[str, ...] = ()) -> None:
    print("+", GODOT, " ".join(map(str, args)), flush=True)
    result = subprocess.run([str(GODOT), *map(str, args)], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    check_godot_output(result.stdout, required)


def download(name: str) -> Path:
    url, expected = DOWNLOAD_INFO[name]
    target = DOWNLOADS / name
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        run("curl", "--fail", "--location", "--retry", "3", "--silent", "--show-error", url, "-o", temporary)
        temporary.rename(target)
    with target.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {target}; remove the invalid task download and retry")
    return target


def dependencies() -> None:
    if not ((platform.system() == "Darwin" and ARCH in ("arm64", "x86_64")) or (LINUX and ARCH == "x86_64")):
        raise RuntimeError("Build on macOS arm64/x86_64 or Linux x86_64; cross-compilation is not configured.")
    for path in (DEPS, DOWNLOADS, BUILD):
        path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free space is needed before dependency/build work")
    if LINUX:
        if not GODOT.exists():
            with zipfile.ZipFile(download("godot-linux.zip")) as archive:
                archive.extractall(DEPS)
            GODOT.chmod(0o755)
        for template in ("linux_debug.x86_64", "linux_release.x86_64"):
            if not (DEPS / "templates" / template).exists():
                with zipfile.ZipFile(download("templates.tpz")) as archive:
                    archive.extract("templates/" + template, DEPS)
                (DEPS / "templates" / template).chmod(0o755)
        if not (DEPS / "mujoco/lib/libmujoco.so.3.12.0").exists():
            with tarfile.open(download("mujoco-linux.tar.gz")) as archive:
                archive.extractall(DEPS, filter="data")
            (DEPS / "mujoco-3.12.0").rename(DEPS / "mujoco")
    else:
        if not GODOT.exists():
            run("ditto", "-xk", download("godot.zip"), DEPS)
        if not (DEPS / "templates/macos.zip").exists():
            with zipfile.ZipFile(download("templates.tpz")) as archive:
                archive.extract("templates/macos.zip", DEPS)
        library = DEPS / "mujoco/lib/libmujoco.3.12.0.dylib"
        if not library.exists() or ARCH not in subprocess.check_output(["lipo", "-archs", str(library)], text=True).split():
            volume = DEPS / "mujoco-volume"
            run("hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", volume, download("mujoco.dmg"))
            try:
                framework = volume / "mujoco.framework/Versions/A"
                library.parent.mkdir(parents=True, exist_ok=True)
                include = DEPS / "mujoco/include/mujoco"
                shutil.copytree(framework / "Headers", include, dirs_exist_ok=True)
                run("lipo", framework / "libmujoco.3.12.0.dylib", "-thin", ARCH, "-output", library)
                run("install_name_tool", "-id", "@rpath/libmujoco.3.12.0.dylib", library)
                run("codesign", "--force", "--sign", "-", library)
                for name in ("LICENSE", "THIRD_PARTY_NOTICES"):
                    shutil.copy2(volume / name, DEPS / "mujoco" / name)
            finally:
                run("hdiutil", "detach", volume)
    cpp = DEPS / "godot-cpp"
    if not cpp.exists():
        run("git", "clone", "--depth", "1", "--branch", "godot-4.5-stable", "https://github.com/godotengine/godot-cpp.git", cpp)
    commit = subprocess.check_output(["git", "-C", str(cpp), "rev-parse", "HEAD"], text=True).strip()
    if commit != GODOT_CPP_COMMIT:
        raise RuntimeError(f"godot-cpp revision differs from pin: {commit}")
    run(GODOT, "--headless", "--version")



def main():
    parser = argparse.ArgumentParser(description="Install pinned MuJoCo and Godot dependencies")
    parser.add_argument('command', choices=['deps'])
    parser.parse_args()
    dependencies()


if __name__ == '__main__':
    main()
