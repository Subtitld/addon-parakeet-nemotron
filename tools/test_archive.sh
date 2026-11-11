#!/usr/bin/env bash
# Run tests/test_installed.py against the release archive, in a clean container.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# The archive is unpacked the way Subtitld's installer does it (zipfile's
# extractall, then the executable bit), into a distribution that has none of the
# build's packages. The container gets only what the tests drive the binary
# with: Python and pytest (tests/wav_ffmpeg.py stands in for ffmpeg). Without
# PARAKEET_NEMOTRON_TEST_MODELS_DIR the binary downloads the models itself
# (1.6 GB), which also checks that it finds the system's CA certificates.
#
#     tools/test_archive.sh almalinux:8
#     PARAKEET_NEMOTRON_TEST_MODELS_DIR=~/.cache/subtitld/models tools/test_archive.sh ubuntu:20.04
#
# Needs Docker and the network.

set -euo pipefail

IMAGE="${1:?usage: tools/test_archive.sh <image> [archive]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/parakeet_nemotron_addon/__init__.py")"
ARCHIVE="$(realpath "${2:-$ROOT/parakeet-nemotron-${VERSION}-linux-x86_64.zip}")"
[ -f "$ARCHIVE" ] || { echo "no archive at $ARCHIVE; run tools/build_linux.sh" >&2; exit 1; }

MOUNTS=(-v "$ARCHIVE:/artifact/addon.zip:ro" -v "$ROOT/tests:/repo/tests:ro"
        -v "$ROOT/parakeet_nemotron_addon:/repo/parakeet_nemotron_addon:ro")
MODELS=/tmp/models
if [ -n "${PARAKEET_NEMOTRON_TEST_MODELS_DIR:-}" ]; then
    MOUNTS+=(-v "$(realpath "$PARAKEET_NEMOTRON_TEST_MODELS_DIR"):/models:ro")
    MODELS=/models
fi

docker run --rm -e "SUBTITLD_MODELS_DIR=$MODELS" -e PYTHONDONTWRITEBYTECODE=1 "${MOUNTS[@]}" "$IMAGE" \
    bash -euo pipefail -c '
    PY=python3
    . /etc/os-release
    if command -v apt-get >/dev/null; then
        apt-get update -qq >/dev/null
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates python3 python3-pytest >/dev/null
    elif [ "${PLATFORM_ID:-}" = platform:el8 ]; then
        # EL8 python3 is 3.6, and its pytest 3.4: the tests need 3.8 or newer.
        dnf install -y -q python3.12 python3.12-pip >/dev/null
        python3.12 -m pip install -q --root-user-action=ignore pytest
        PY=python3.12
    elif command -v dnf >/dev/null; then
        dnf install -y -q python3 python3-pytest >/dev/null
    elif command -v pacman >/dev/null; then
        pacman -Sy --noconfirm archlinux-keyring >/dev/null
        pacman -Syu --noconfirm --needed python python-pytest >/dev/null
    else
        echo "no supported package manager in this image" >&2; exit 1
    fi
    echo "image: $PRETTY_NAME, $(ldd --version | head -1), test driver $($PY --version)"
    $PY - <<PY
import os, stat, zipfile
zipfile.ZipFile("/artifact/addon.zip").extractall("/opt/addon")
exe = "/opt/addon/bin/parakeet-nemotron-addon"
os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
    command -v ffmpeg >/dev/null && echo "ffmpeg is installed; the tests use it" || true
    cd /tmp
    PARAKEET_NEMOTRON_TEST_BINARY=/opt/addon/bin/parakeet-nemotron-addon \
        $PY -m pytest -q -rs -p no:cacheprovider /repo/tests/test_installed.py
'
