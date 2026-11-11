#!/usr/bin/env python3
"""Assemble THIRD_PARTY_LICENSES/ for a release archive, and refuse native code without a notice.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Besides the add-on's own code, the frozen bundle carries NVIDIA's
NeMo-Speech.cpp runtime (Apache-2.0) with ggml, and on Windows and macOS the
llama.cpp libraries NVIDIA's build links (MIT); on Linux the GNU OpenMP runtime
from GCC; CPython with the native libraries its standard modules link against
(OpenSSL among them, for the download), the PyInstaller bootloader
(GPL-2.0-or-later with the bootloader exception) and, on Windows, the C
runtime. The speech models are downloaded, not bundled; models/README.md
describes them.

The notices committed in THIRD_PARTY_LICENSES/ are copied as they are, and
NeMo-Speech.cpp's own (its licence, NOTICE, THIRD_PARTY_NOTICES.md and the
licences of what it compiles in) from build/licences/nemo-speech/, where
tools/build_native.py puts them. CPython's licence comes from the interpreter
running this script (the one that froze the bundle), and PyInstaller's from
its installed distribution.

Then every native file in the bundle (ELF, Mach-O or PE, whatever its name)
must match a rule below, and the notice that rule names must be in the output.
Anything else fails the build: a library new to the bundle has to come with its
licence before it ships. The result is indexed in THIRD_PARTY_LICENSES/README.md.

    python tools/assemble_licences.py --bundle dist/parakeet-nemotron-addon \
        --out release-staging/THIRD_PARTY_LICENSES
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEMO_SPEECH_LICENCES = REPO_ROOT / 'build' / 'licences' / 'nemo-speech'

# (pattern on the lower-cased file name, notice inside THIRD_PARTY_LICENSES/, component)
RULES = [
    (r'^parakeet-nemotron-addon(\.exe)?$', 'pyinstaller/COPYING.txt', 'PyInstaller bootloader'),
    (r'^(lib)?nemo_speech_\w+?(\.\d+)*\.(so|dylib|dll)(\.\d+)*$', 'nemo-speech/LICENSE', 'NeMo-Speech.cpp runtime'),
    (r'^(lib)?ggml(-\w+)?(\.\d+)*\.(so|dylib|dll)(\.\d+)*$', 'nemo-speech/third_party/ggml/LICENSE', 'ggml'),
    (r'^(lib)?llama(-common)?(\.\d+)*\.(so|dylib|dll)(\.\d+)*$', 'nemo-speech/third_party/llama.cpp/LICENSE',
     'llama.cpp'),
    (r'^libgomp\.so(\.\d+)*$', 'gcc-runtime/COPYING.RUNTIME', 'GNU OpenMP runtime (GCC)'),
    (r'^(libpython3[\d.]*\.(so|dylib)[\d.]*|python3\d*\.dll|python)$', 'python/LICENSE.txt', 'CPython'),
    (r'(\.cpython-[^/]+\.(so|dylib)|\.pyd)$', 'python/LICENSE.txt', 'CPython extension module'),
    (r'^libffi[-.\d]*\.(so|dylib|dll)', 'python-runtime-libs/libffi-LICENSE', 'libffi'),
    (r'^libexpat[-.\d]*\.(so|dylib|dll)', 'python-runtime-libs/expat-COPYING', 'Expat'),
    (r'^(libz\.(so|dylib)|zlib\d*\.dll)', 'python-runtime-libs/zlib-LICENSE', 'zlib'),
    # OpenSSL 1.1.1 (the Linux build's, from AlmaLinux 8) is under the OpenSSL
    # and SSLeay licences; OpenSSL 3 under Apache-2.0.
    (r'^lib(ssl|crypto)\.so\.1\.1', 'python-runtime-libs/openssl-1.1-LICENSE.txt', 'OpenSSL 1.1.1'),
    (r'^lib(ssl|crypto)[-.\d]', 'python-runtime-libs/openssl-LICENSE.txt', 'OpenSSL'),
    # Excluded from the bundle by pyinstaller.spec. Listed so that, should one
    # come back, the build names the notice to commit instead of a bare file.
    (r'^libbz2', 'python-runtime-libs/bzip2-LICENSE', 'bzip2'),
    (r'^liblzma', 'python-runtime-libs/xz-COPYING', 'XZ Utils'),
    (r'^libzstd', 'python-runtime-libs/zstd-LICENSE', 'zstd'),
    (r'^libmpdec', 'python-runtime-libs/mpdecimal-LICENSE', 'mpdecimal'),
    (r'^libuuid', 'python-runtime-libs/libuuid-COPYING', 'libuuid'),
    (r'^(vcruntime140(_1|_threads)?|msvcp140(_\w+)?|vcomp140|concrt140)\.dll$', 'msvc-runtime/NOTICE.txt',
     'Microsoft C runtime'),
    (r'^(ucrtbase|api-ms-win-[\w-]+)\.dll$', 'msvc-runtime/NOTICE.txt', 'Universal C runtime'),
]
_COMPILED = [(re.compile(pattern), notice, component) for pattern, notice, component in RULES]

_MACHO_MAGIC = {b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe',
                b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}


def is_native(path: Path) -> bool:
    with path.open('rb') as handle:
        head = handle.read(4)
    return head == b'\x7fELF' or head[:2] == b'MZ' or head in _MACHO_MAGIC


def rule_for(name: str) -> tuple[str, str] | None:
    lowered = name.lower()
    for pattern, notice, component in _COMPILED:
        if pattern.search(lowered):
            return notice, component
    return None


def cpython_licence() -> Path:
    stdlib = Path(sysconfig.get_paths()['stdlib'])
    for candidate in (Path(sys.base_prefix) / 'LICENSE.txt', stdlib / 'LICENSE.txt',
                      Path(sys.base_prefix) / 'LICENSE'):
        if candidate.is_file():
            return candidate
    raise SystemExit(f'FATAL: no CPython licence next to {sys.base_prefix} or in {stdlib}')


def pyinstaller_licence() -> tuple[Path, str]:
    try:
        dist = distribution('pyinstaller')
    except PackageNotFoundError:
        raise SystemExit('FATAL: PyInstaller is not installed in this interpreter') from None
    for entry in dist.files or []:
        if Path(str(entry)).name == 'COPYING.txt':
            path = Path(dist.locate_file(entry))
            if path.is_file():
                return path, dist.version
    raise SystemExit('FATAL: PyInstaller is installed but ships no COPYING.txt')


def assemble(bundle: Path, out: Path) -> list[tuple[str, str, str]]:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO_ROOT / 'THIRD_PARTY_LICENSES', out)
    if not (NEMO_SPEECH_LICENCES / 'LICENSE').is_file():
        raise SystemExit(f'FATAL: no {NEMO_SPEECH_LICENCES / "LICENSE"}; run tools/build_native.py')
    shutil.copytree(NEMO_SPEECH_LICENCES, out / 'nemo-speech')

    python_licence = cpython_licence()
    (out / 'python').mkdir()
    shutil.copyfile(python_licence, out / 'python' / 'LICENSE.txt')
    (out / 'python' / 'SOURCE.txt').write_text(
        f'CPython {platform.python_version()}, the interpreter that froze this add-on.\n'
        f'Licence copied from {python_licence}.\n', encoding='utf-8')
    licence, version = pyinstaller_licence()
    (out / 'pyinstaller').mkdir()
    shutil.copyfile(licence, out / 'pyinstaller' / 'COPYING.txt')
    (out / 'pyinstaller' / 'SOURCE.txt').write_text(
        f'PyInstaller {version}. Its bootloader is the add-on executable; the GPL-2.0\n'
        f'bootloader exception in COPYING.txt allows distributing it with this program.\n',
        encoding='utf-8')

    rows: list[tuple[str, str, str]] = []
    problems: list[str] = []
    for path in sorted(p for p in bundle.rglob('*') if p.is_file() and not p.is_symlink()):
        relative = 'bin/' + path.relative_to(bundle).as_posix()
        if not is_native(path):
            continue
        match = rule_for(path.name)
        if match is None:
            problems.append(f'{relative}: native file with no licence rule')
            continue
        notice, component = match
        if not (out / notice).is_file():
            problems.append(f'{relative}: {component} needs THIRD_PARTY_LICENSES/{notice}, which is not committed')
            continue
        rows.append((relative, component, notice))
    for component in ('NeMo-Speech.cpp runtime', 'ggml'):
        if not any(row[1].startswith(component) for row in rows):
            problems.append(f'the bundle has no {component}')
    required = ['nemo-speech/NOTICE', 'nemo-speech/THIRD_PARTY_NOTICES.md', 'models/README.md',
                'models/CC-BY-4.0.txt', 'models/OpenMDW-1.1.txt']
    if sys.platform.startswith('linux'):
        # libgomp ships, and libstdc++ parts are linked into the runtime.
        required += ['gcc-runtime/COPYING.RUNTIME', 'gcc-runtime/COPYING3',
                     'nemo-speech/third_party/sentencepiece/LICENSE']
    for notice in required:
        if not (out / notice).is_file():
            problems.append(f'THIRD_PARTY_LICENSES/{notice} is not committed')
    if problems:
        raise SystemExit('FATAL: licences are incomplete:\n  ' + '\n  '.join(problems))

    lines = ['# Third-party licences', '',
             'Everything frozen into this add-on, and the notice in this folder that covers it.', '',
             '| File | Component | Notice |', '|------|-----------|--------|']
    lines += [f'| `{relative}` | {component} | [{notice}]({notice}) |' for relative, component, notice in rows]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, required=True, help='the PyInstaller onedir output')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = assemble(args.bundle, args.out)
    print(f'{len(rows)} bundled file(s) covered:')
    for relative, component, notice in rows:
        print(f'  {relative}  ->  {notice}  ({component})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
