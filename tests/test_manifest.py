"""manifest.json, pins.json and the code agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parakeet_nemotron_addon import ADDON_ID, __version__, models  # noqa: E402
from parakeet_nemotron_addon.__main__ import DEFAULT_LIVE_DELAY, LIVE_DELAYS, _options  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_REVISION = re.compile(r'^[0-9a-f]{40}$')


def test_versions_agree():
    text = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == __version__
    assert MANIFEST['version'] == __version__
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    # The host takes the add-on for its first task; transcription comes first.
    assert MANIFEST['tasks'] == ['asr.transcribe', 'asr.stream']
    assert MANIFEST['languages'] == list(models.LANGUAGES)
    assert MANIFEST['compute'] == 'cpu' and MANIFEST['accelerators'] == ['mps']
    assert MANIFEST['executable'] == 'bin/parakeet-nemotron-addon'
    assert set(MANIFEST['platforms']) == {'linux', 'macos', 'windows'}


def test_options_match_the_code():
    fields = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert set(fields) == set(_options({}))
    assert [o['value'] for o in fields['model']['options']] == list(models.CHOICES)
    assert fields['model']['default'] == _options({})['model']
    assert [o['value'] for o in fields['live_delay']['options']] == list(LIVE_DELAYS)
    assert fields['live_delay']['default'] == DEFAULT_LIVE_DELAY
    assert fields['slice_by_phrases']['default'] is True and fields['label_speakers']['default'] is False
    for field in fields.values():
        assert field['help'].strip(), field['key']


def test_pins_match_the_manifest_models():
    data = models.pins()
    assert set(data) == {models.PARAKEET, models.NEMOTRON, models.DIARIZATION}
    by_id = {m['id']: m for m in MANIFEST['models']}
    assert set(by_id) == set(data)
    for name, pin in data.items():
        assert pin['repo'].startswith('nvidia/') and _REVISION.match(pin['revision']), name
        assert _SHA256.match(pin['sha256']) and pin['size'] > 0 and pin['path'].endswith('.gguf'), name
        assert by_id[name]['repo'] == pin['repo'] and by_id[name]['revision'] == pin['revision']
        assert by_id[name]['license'] == pin['license']
        assert abs(by_id[name]['size_mb'] - pin['size'] / 1e6) < 1, name
        assert by_id[name]['bundled'] is False
    assert [m['id'] for m in MANIFEST['models'] if m['default']] == [models.PARAKEET]


def test_the_licences_are_spelled_out():
    assert 'Apache-2.0' in MANIFEST['license']
    assert 'CC-BY-4.0' in MANIFEST['license'] and 'OpenMDW-1.1' in MANIFEST['license']
    licences = REPO_ROOT / 'THIRD_PARTY_LICENSES'
    readme = (licences / 'models' / 'README.md').read_text(encoding='utf-8')
    for pin in models.pins().values():
        assert pin['repo'] in readme and pin['revision'] in readme
    assert 'Creative Commons Attribution 4.0' in (licences / 'models' / 'CC-BY-4.0.txt').read_text(encoding='utf-8')
    assert 'OpenMDW' in (licences / 'models' / 'OpenMDW-1.1.txt').read_text(encoding='utf-8')
    for notice in ('gcc-runtime/COPYING.RUNTIME', 'gcc-runtime/COPYING3', 'msvc-runtime/NOTICE.txt'):
        assert (licences / notice).read_text(encoding='utf-8').strip(), notice
