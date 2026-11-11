#!/usr/bin/env bash
# Build the Linux runtime, the frozen add-on and its archive, inside the
# AlmaLinux 8 container.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# The bundle carries the build interpreter's libpython and its OpenSSL, and
# NeMo-Speech.cpp is compiled here, so the add-on runs on any glibc at least as
# new as the build's. The image (tools/linux-build.Dockerfile) is AlmaLinux 8,
# glibc 2.28, as for the other Subtitld add-ons, with the distribution's
# Python 3.12. tools/check_glibc.py then refuses the archive if any file in it
# needs a newer glibc after all.
#
#     tools/build_linux.sh [--jobs N]
#
# writes, into the repository, owned by you:
#     parakeet-nemotron-<version>-linux-x86_64.zip   (+ .sha256)
#     build/assets/, dist/parakeet-nemotron-addon/   (as tools/build_native.py and PyInstaller do)
#
# The NeMo-Speech.cpp checkout and its build tree stay in build/nemo-speech-src/
# between runs. Needs Docker and the network.

set -euo pipefail

BUILD_IMAGE="${BUILD_IMAGE:-subtitld-parakeet-nemotron-build:almalinux8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/parakeet_nemotron_addon/__init__.py")"
ARCHIVE="parakeet-nemotron-${VERSION}-linux-x86_64.zip"
# pip's cache survives between builds.
PIP_CACHE="${PIP_CACHE:-$HOME/.cache/subtitld-parakeet-nemotron-build-pip}"
mkdir -p "$PIP_CACHE"

# No build context: the Dockerfile copies nothing in.
docker build -q -t "$BUILD_IMAGE" - < "$ROOT/tools/linux-build.Dockerfile" >/dev/null

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$ROOT:/src" \
    -v "$PIP_CACHE:/tmp/.cache/pip" \
    -w /src \
    "$BUILD_IMAGE" \
    bash -euo pipefail -c "
        command -v nm >/dev/null || { echo 'nm is required for the libexpat check' >&2; exit 1; }
        # The distribution's interpreter, which has a shared libpython.
        /usr/bin/python3.12 -m venv /tmp/build-venv
        . /tmp/build-venv/bin/activate
        pip install -q --upgrade pip
        pip install -q '.[build,dev]'

        # A failed build must not leave older outputs that look current.
        rm -rf build/pyinstaller dist release-staging '$ARCHIVE' '$ARCHIVE.sha256'

        # CMake and Ninja come from /opt/build-tools, still on PATH.
        python tools/build_native.py --platform linux $*
        python -m pytest -q -p no:cacheprovider tests/test_engine.py

        # /tmp/.cache belongs to root here, made by the pip mount.
        PYINSTALLER_CONFIG_DIR=/tmp/pyinstaller-config \\
            pyinstaller pyinstaller.spec --noconfirm --log-level WARN --workpath build/pyinstaller
        dist/parakeet-nemotron-addon/parakeet-nemotron-addon --self-test

        mkdir -p release-staging/bin
        cp -R dist/parakeet-nemotron-addon/. release-staging/bin/
        cp manifest.json LICENSE README.md release-staging/
        python tools/assemble_licences.py --bundle dist/parakeet-nemotron-addon \\
            --out release-staging/THIRD_PARTY_LICENSES
        python tools/make_archive.py release-staging '$ARCHIVE'
        python tools/check_glibc.py --max 2.28 '$ARCHIVE'
        sha256sum '$ARCHIVE' | cut -d' ' -f1 > '$ARCHIVE.sha256'
    "

echo "built $ARCHIVE, sha256 $(cat "$ROOT/$ARCHIVE.sha256")"
