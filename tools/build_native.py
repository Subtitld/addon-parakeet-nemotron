"""Put NeMo-Speech.cpp's ASR libraries into build/assets/.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The add-on calls the runtime's C ABI (include/nemo_speech/asr.h) through
ctypes. This script provides that library and everything it loads:

- Linux: built from source at a pinned commit, for glibc 2.28. Run it
  through tools/build_native_linux.sh, which starts the AlmaLinux 8 container.
- Windows and macOS: NVIDIA's release archives, pinned by SHA-256 and cut
  down to the libraries the ASR C ABI loads.

    python tools/build_native.py [--platform linux|windows|macos] [--jobs N]

Writes build/assets/ (the libraries and native.json, which records where
they came from and each file's SHA-256) and build/licences/nemo-speech/ (the
runtime's licence and third-party notices). Another platform's archive, such
as --platform windows on Linux, goes to build/assets-windows/ and
build/licences-windows/ instead, leaving this platform's in place.

x86-64 builds need AVX2 (x86-64-v3), as NVIDIA's archives do: the runtime
calls ggml's CPU backend directly, so ggml's per-CPU variants, which are
loaded as plugins, are not an option.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
ASSETS = BUILD / 'assets'
LICENCES = BUILD / 'licences' / 'nemo-speech'

REPO = 'https://github.com/NVIDIA/NeMo-Speech.cpp'
VERSION = '0.3.0'
COMMIT = '20999cc921d2b14aa58708f61756a8b2e787ba20'  # tag v0.3.0

# The entry point; everything else is what it loads.
C_ABI = {
    'linux': 'libnemo_speech_asr_c.so.1',
    'windows': 'nemo_speech_asr_c.dll',
    'macos': 'libnemo_speech_asr_c.1.dylib',
}

# NVIDIA's release archives. `files` is the ASR C ABI's dependency closure,
# worked out from the archives' import tables (PE imports, LC_LOAD_DYLIB):
# both link the speech-LLM library and llama.cpp, which the Linux build
# leaves out. The UCRT (api-ms-win-crt-*) ships with Windows 10 and later.
PREBUILT = {
    'windows': {
        'archive': f'nemo-speech-{VERSION}-windows-x86_64-cpu.zip',
        'sha256': '4ed9740e361e4653f56c3dce7d1e81883a1c8b831c1bd7d01567fff1130b5b05',
        'libdir': 'bin',
        'files': [
            'nemo_speech_asr_c.dll', 'nemo_speech_asr.dll', 'nemo_speech_speechlm.dll',
            'llama.dll', 'llama-common.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll',
            'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll',
        ],
    },
    'macos': {
        # Apple Silicon with the Metal backend; needs macOS 13.3 (LC_BUILD_VERSION).
        'archive': f'nemo-speech-{VERSION}-macos-aarch64-metal.tar.gz',
        'sha256': '2b972ed8b5dba07492212acf8cb75cbf34ecb141198ffa1997febe84b7c4d9ed',
        'libdir': 'lib',
        'files': [
            'libnemo_speech_asr_c.1.dylib', 'libnemo_speech_asr.dylib',
            'libnemo_speech_speechlm.dylib', 'libnemo_speech_text_normalization.dylib',
            'libllama.0.dylib', 'libllama-common.0.dylib', 'libggml.0.dylib',
            'libggml-base.0.dylib', 'libggml-cpu.0.dylib', 'libggml-blas.0.dylib',
            'libggml-metal.0.dylib',
        ],
    },
}

# Linux: NVIDIA's cpu-asr preset (ASR + diarization) without the parts the
# add-on doesn't call. The flags after the first follow the release build in
# docker/Dockerfile.release-linux.
LINUX_CMAKE_ARGS = [
    '-DGGML_NATIVE=OFF',
    '-DNEMO_SPEECH_BUILD_TTS=OFF',
    '-DNEMO_SPEECH_BUILD_NMT=OFF',
    '-DNEMO_SPEECH_BUILD_SPEECHLM=OFF',
    '-DNEMO_SPEECH_BUILD_CLI=OFF',
    '-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF',
    '-DNEMO_SPEECH_WITH_NORM=OFF',
    '-DNEMO_SPEECH_VERSION_METADATA=subtitld',
    '-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags',
    '-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags',
]

# Libraries every Linux desktop has; anything else a bundled library needs
# must be bundled too. libstdc++ and libgcc_s are those of GCC 8, which
# gcc-toolset-14 builds against.
LINUX_SYSTEM = {
    'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0', 'librt.so.1',
    'ld-linux-x86-64.so.2', 'libstdc++.so.6', 'libgcc_s.so.1',
}
# Not on every desktop, so bundled from AlmaLinux 8 itself, as NVIDIA's
# archives do: ggml's threads use OpenMP (GOMP_1.0, GOMP_4.0, OMP_1.0, all
# in GCC 8's libgomp). GPL-3 with the GCC Runtime Library Exception.
LINUX_BUNDLE_FROM_SYSTEM = {'libgomp.so.1': Path('/usr/lib64/libgomp.so.1')}
# The newest symbol version each AlmaLinux 8 (glibc 2.28, GCC 8) library
# provides; a bundled library needing more would not load there.
LINUX_CEILINGS = {
    'GLIBC': (2, 28),
    'GLIBCXX': (3, 4, 25),
    'CXXABI': (1, 3, 11),
    'GCC': (8, 0, 0),
}


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def output(cmd, cwd=None):
    env = dict(os.environ, LC_ALL='C')
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fresh(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_manifest(platform, origin):
    files = {p.name: sha256(p) for p in sorted(ASSETS.iterdir()) if p.name != 'native.json'}
    record = {'runtime': 'NeMo-Speech.cpp', 'version': VERSION, 'commit': COMMIT,
              'platform': platform, 'origin': origin, 'entry': C_ABI[platform], 'files': files}
    (ASSETS / 'native.json').write_text(json.dumps(record, indent=1) + '\n')
    total = sum((ASSETS / name).stat().st_size for name in files)
    print(f'{len(files)} libraries, {total / 2**20:.1f} MiB in {ASSETS}')


# ---- Linux: build from source ---------------------------------------------

def checkout(src):
    """The pinned commit and the one submodule ASR needs (llama.cpp, which
    also carries ggml), as shallow as git allows."""
    if not (src / '.git').exists():
        src.mkdir(parents=True, exist_ok=True)
        run(['git', 'init', '-q'], cwd=src)
        run(['git', 'remote', 'add', 'origin', REPO + '.git'], cwd=src)
    run(['git', 'fetch', '-q', '--depth', '1', 'origin', COMMIT], cwd=src)
    run(['git', 'checkout', '-q', '--force', COMMIT], cwd=src)
    run(['git', 'submodule', 'update', '--init', '--depth', '1', 'llama.cpp'], cwd=src)


def needed(lib):
    return re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', output(['readelf', '-d', lib]))


def version_needs(lib):
    """{library: [version, ...]} from the .gnu.version_r section."""
    needs, current = {}, None
    for line in output(['readelf', '-V', '--wide', lib]).splitlines():
        m = re.search(r'File: (\S+)', line)
        if m:
            current = needs.setdefault(m.group(1), [])
            continue
        m = re.search(r'Name: (\S+)', line)
        if m and current is not None:
            current.append(m.group(1))
    return needs


def check_linux(bundle):
    """Fail on anything AlmaLinux 8 couldn't load."""
    names = {p.name for p in bundle}
    problems = []
    for lib in bundle:
        for dep in needed(lib):
            if dep not in names and dep not in LINUX_SYSTEM:
                problems.append(f'{lib.name} needs {dep}, which is neither bundled nor a system library')
        for dep, versions in version_needs(lib).items():
            if dep in names:
                continue
            for version in versions:
                m = re.fullmatch(r'([A-Z]+)_([0-9.]+)', version)
                if not m or m.group(1) not in LINUX_CEILINGS:
                    continue
                have = tuple(int(x) for x in m.group(2).split('.'))
                if have > LINUX_CEILINGS[m.group(1)]:
                    problems.append(f'{lib.name} needs {version} from {dep}')
    if problems:
        sys.exit('not loadable on glibc 2.28 / GCC 8:\n  ' + '\n  '.join(sorted(set(problems))))
    print('every library loads on glibc 2.28 with the GCC 8 runtime')


def build_linux(jobs):
    src = BUILD / 'nemo-speech-src'
    checkout(src)
    env = dict(os.environ, JOBS=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs))
    # SentencePiece as a static archive at the commit NVIDIA's releases use.
    if not (src / '.deps' / 'sentencepiece' / 'lib' / 'libsentencepiece.a').exists():
        run(['bash', 'scripts/build_sentencepiece_static.sh'], cwd=src, env=env)
    run(['bash', 'scripts/configure.sh', 'cpu-asr', *LINUX_CMAKE_ARGS], cwd=src, env=env)
    run(['cmake', '--build', '--preset', 'cpu-asr', '--parallel', str(jobs)], cwd=src, env=env)
    prefix = BUILD / 'nemo-speech-install'
    if prefix.exists():
        shutil.rmtree(prefix)
    run(['cmake', '--install', 'build/cpu-asr', '--prefix', prefix], cwd=src)

    # The C ABI and what it loads from the install, under the names the
    # loader asks for (SONAMEs), with $ORIGIN as their only search path.
    libdir = prefix / 'lib64' if (prefix / 'lib64').exists() else prefix / 'lib'
    fresh(ASSETS)
    todo, seen = [C_ABI['linux']], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        found = LINUX_BUNDLE_FROM_SYSTEM.get(name, libdir / name)
        if not found.exists():
            continue  # a system library; check_linux() decides whether that's fine
        target = ASSETS / name
        shutil.copyfile(found.resolve(), target)
        target.chmod(0o755)
        run(['patchelf', '--set-rpath', '$ORIGIN', target])
        todo.extend(needed(target))
    check_linux(sorted(ASSETS.iterdir()))

    fresh(LICENCES)
    shutil.copytree(prefix / 'share' / 'licenses' / 'nemo-speech', LICENCES, dirs_exist_ok=True)
    spm = src / '.deps' / 'sentencepiece' / 'share' / 'licenses' / 'nemo-speech'
    if spm.exists():
        shutil.copytree(spm, LICENCES, dirs_exist_ok=True)
    write_manifest('linux', f'{REPO} at {COMMIT}, built in AlmaLinux 8')


# ---- Windows and macOS: NVIDIA's archives ---------------------------------

def download(name, expected):
    cache = BUILD / 'downloads' / name
    if cache.exists() and sha256(cache) == expected:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = f'{REPO}/releases/download/v{VERSION}/{name}'
    print('downloading', url, flush=True)
    part = cache.with_name(cache.name + '.part')
    with urllib.request.urlopen(url, timeout=60) as response, open(part, 'wb') as f:
        shutil.copyfileobj(response, f, 1 << 20)
    got = sha256(part)
    if got != expected:
        part.unlink()
        sys.exit(f'{name}: SHA-256 {got}, expected {expected}')
    part.replace(cache)
    return cache


class Archive:
    """Read members of a zip or tar.gz by path, following tar symlinks (the
    macOS archive names libggml.0.dylib as a link to libggml.0.25.1.dylib)."""

    def __init__(self, path):
        if path.name.endswith('.zip'):
            self.zip = zipfile.ZipFile(path)
            self.names = self.zip.namelist()
        else:
            self.zip = None
            self.tar = tarfile.open(path)
            self.members = {m.name.rstrip('/'): m for m in self.tar.getmembers()}
            self.names = list(self.members)

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        member = self.members[name]
        for _ in range(8):
            if not member.issym():
                return self.tar.extractfile(member).read()
            target = os.path.normpath(os.path.join(os.path.dirname(member.name), member.linkname))
            member = self.members[target]
        raise RuntimeError(f'{name}: symlink chain too long')

    def isfile(self, name):
        if self.zip:
            return not name.endswith('/')
        return self.members[name].isfile() or self.members[name].issym()


def fetch_prebuilt(platform):
    spec = PREBUILT[platform]
    archive = Archive(download(spec['archive'], spec['sha256']))
    top = spec['archive'].removesuffix('.zip').removesuffix('.tar.gz')
    fresh(ASSETS)
    for name in spec['files']:
        (ASSETS / name).write_bytes(archive.read(f'{top}/{spec["libdir"]}/{name}'))
    fresh(LICENCES)
    prefix = f'{top}/share/licenses/nemo-speech/'
    for name in archive.names:
        if name.startswith(prefix) and archive.isfile(name):
            target = LICENCES / name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    write_manifest(platform, f'{REPO}/releases/download/v{VERSION}/{spec["archive"]}')


def main():
    host = {'linux': 'linux', 'win32': 'windows', 'darwin': 'macos'}.get(sys.platform)
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--platform', choices=sorted(C_ABI), default=host)
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', os.cpu_count() or 4)))
    args = parser.parse_args()
    if args.platform != host:
        global ASSETS, LICENCES
        ASSETS = BUILD / f'assets-{args.platform}'
        LICENCES = BUILD / f'licences-{args.platform}' / 'nemo-speech'
    if args.platform == 'linux':
        build_linux(args.jobs)
    else:
        fetch_prebuilt(args.platform)


if __name__ == '__main__':
    main()
