"""The ctypes binding against NeMo-Speech.cpp's own headers, and the built library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

tests/data/nemo_speech/ holds asr.h and diar.h as NeMo-Speech.cpp v0.3.0
installs them. Every struct engine.py mirrors must list the same fields, in the
same order, with the same C types: a field out of place would be read as
another one, and the runtime only checks a struct's size.
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parakeet_nemotron_addon import engine  # noqa: E402

HEADERS = REPO_ROOT / 'tests' / 'data' / 'nemo_speech'

MIRRORS = {
    'nemo_speech_asr_backend_config': engine.BackendConfig,
    'nemo_speech_asr_model_config': engine.ModelConfig,
    'nemo_speech_asr_streaming_config': engine.StreamingConfig,
    'nemo_speech_asr_endpointing_config': engine.EndpointingConfig,
    'nemo_speech_asr_recognizer_config': engine.RecognizerConfig,
    'nemo_speech_asr_recognition_options': engine.RecognitionOptions,
    'nemo_speech_diar_model_config': engine.DiarModelConfig,
    'nemo_speech_diar_segment': engine.DiarSegment,
}

_C_TYPES = {'size_t': ctypes.c_size_t, 'int32_t': ctypes.c_int32, 'int64_t': ctypes.c_int64,
            'float': ctypes.c_float, 'double': ctypes.c_double, 'bool': ctypes.c_bool}


def _structs() -> dict[str, list[tuple[str, str]]]:
    text = '\n'.join(path.read_text(encoding='utf-8') for path in sorted(HEADERS.glob('*.h')))
    text = re.sub(r'//[^\n]*', '', text)
    out = {}
    for name, body, alias in re.findall(r'typedef struct (\w+) \{(.*?)\} (\w+);', text, re.S):
        assert name == alias
        fields = []
        for declaration in body.split(';'):
            declaration = ' '.join(declaration.split())
            if declaration:
                ctype, _, field = declaration.rpartition(' ')
                stars = len(field) - len(field.lstrip('*'))
                fields.append((ctype + '*' * stars, field.lstrip('*')))
        out[name] = fields
    return out


def _matches(c_type: str, py_type) -> bool:
    c_type = c_type.replace('const ', '').replace(' ', '')
    if c_type == 'char*':
        return py_type is ctypes.c_char_p
    if c_type.endswith('*'):
        target = c_type[:-1]
        if py_type is ctypes.c_void_p:  # a subsystem config the add-on leaves at NULL
            return target.startswith('nemo_speech_')
        mirror = MIRRORS.get(target)
        return mirror is not None and py_type is ctypes.POINTER(mirror)
    return _C_TYPES.get(c_type) is py_type


@pytest.mark.parametrize('name', sorted(MIRRORS))
def test_struct_matches_the_header(name):
    header = _structs()[name]
    mirror = MIRRORS[name]._fields_
    assert [field for _, field in header] == [field for field, _ in mirror], name
    for (c_type, field), (_, py_type) in zip(header, mirror):
        assert _matches(c_type, py_type), f'{name}.{field}: {c_type} vs {py_type}'


def test_every_function_bound_is_in_the_headers():
    text = '\n'.join(path.read_text(encoding='utf-8') for path in HEADERS.glob('*.h'))
    declared = set(re.findall(r'\b(nemo_speech_(?:asr|diar)_\w+)\s*\(', text))
    bound = {name for name, _, _ in engine._SIGNATURES}
    assert bound <= declared, bound - declared


def test_the_built_library_loads_and_reports_0_3():
    if not engine.library_path().is_file():
        pytest.skip('no runtime in build/assets; run tools/build_native.py')
    assert engine.runtime_version().split()[-1].startswith('0.3.')
    lib = engine.load_library()
    options = lib.nemo_speech_asr_recognition_options_default()
    assert options.size == ctypes.sizeof(engine.RecognitionOptions)
