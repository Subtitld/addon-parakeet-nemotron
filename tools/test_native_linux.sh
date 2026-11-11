#!/usr/bin/env bash
# Load build/assets/ in clean containers of the distributions the add-on
# supports and transcribe a clip with it (tools/smoke_native.py).
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
#     tools/test_native_linux.sh MODEL.gguf CLIP.f32 [IMAGE ...]
#
# CLIP.f32 is raw 16 kHz mono float32, so the containers need Python but not
# ffmpeg:  ffmpeg -i clip.flac -ac 1 -ar 16000 -f f32le clip.f32
# Each container only gets Python from its own repositories; a library
# missing there fails the run. Needs Docker and the network.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$(realpath "$1")"
CLIP="$(realpath "$2")"
shift 2
IMAGES=("$@")
if [ ${#IMAGES[@]} -eq 0 ]; then
    IMAGES=(almalinux:8 ubuntu:20.04 ubuntu:22.04 ubuntu:24.04 debian:12 fedora:41 archlinux:latest)
fi

python_for() {
    case "$1" in
        almalinux:*|fedora:*) echo 'dnf install -y -q python3 >/dev/null' ;;
        ubuntu:*|debian:*) echo 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 >/dev/null' ;;
        archlinux:*) echo 'pacman -Sy --noconfirm --quiet python >/dev/null' ;;
        *) echo "unknown image $1" >&2; exit 2 ;;
    esac
}

failed=()
for image in "${IMAGES[@]}"; do
    echo "== $image"
    if docker run --rm \
        -v "$ROOT/build/assets:/assets:ro" \
        -v "$ROOT/tools:/tools:ro" \
        -v "$MODEL:/model.gguf:ro" \
        -v "$CLIP:/clip.f32:ro" \
        "$image" \
        sh -c "$(python_for "$image") && python3 /tools/smoke_native.py /model.gguf /clip.f32 --language en --assets /assets 2>/dev/null | grep -E '^(runtime|transcript)|real time'"; then
        :
    else
        failed+=("$image")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
echo "all ${#IMAGES[@]} images passed"
