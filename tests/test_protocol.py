"""The add-on spoken over its wire format, in a real process.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

These run `python -m parakeet_nemotron_addon` and talk JSON lines to it, the
way the host does. They cover every answer the add-on owes the host without
the native runtime, a model or the network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parakeet_nemotron_addon import models  # noqa: E402

SAMPLE = REPO_ROOT / 'tests' / 'data' / 'jfk.wav'


@pytest.fixture()
def proc(tmp_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    env['PARAKEET_NEMOTRON_ASSETS'] = str(tmp_path / 'no-such-assets')
    env['SUBTITLD_MODELS_DIR'] = str(tmp_path / 'models')
    env['HF_ENDPOINT'] = 'http://127.0.0.1:9'  # nothing listens: no test reaches the network
    # Found, so the request gets as far as the download; never run.
    env['SUBTITLD_FFMPEG_EXECUTABLE'] = sys.executable
    for name in list(env):
        if name.startswith('PARAKEET_NEMOTRON_') and name != 'PARAKEET_NEMOTRON_ASSETS':
            env.pop(name)
    process = subprocess.Popen([sys.executable, '-m', 'parakeet_nemotron_addon'], cwd=tmp_path, env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()


def _send(proc, frame: dict) -> None:
    proc.stdin.write((json.dumps(frame) + '\n').encode('utf-8'))
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, proc.stderr.read().decode('utf-8', 'replace')
    return json.loads(line)


def _terminal(proc, req_id: str) -> dict:
    while True:
        frame = _recv(proc)
        if frame.get('id') == req_id and frame['type'] in ('result', 'error'):
            return frame


def _transcribe(proc, req_id: str, **params) -> dict:
    params.setdefault('audio_path', str(SAMPLE))
    _send(proc, {'id': req_id, 'type': 'asr.transcribe', 'params': params})
    return _terminal(proc, req_id)


def test_hello_offers_files_and_live_captions(proc):
    hello = _recv(proc)
    assert hello['type'] == 'hello' and hello['addon'] == 'parakeet-nemotron' and hello['protocol'] == 1
    files, live = hello['capabilities']
    assert files['task'] == 'asr.transcribe' and files['languages'] == list(models.LANGUAGES)
    assert live['task'] == 'asr.stream' and live['languages'] == list(models.NEMOTRON_LANGUAGES)
    assert live['streaming'] is True


def test_requests_it_cannot_serve_get_clear_errors(proc, tmp_path):
    _recv(proc)
    _send(proc, {'type': 'ready', 'host': 'subtitld', 'host_version': 'test'})
    assert _transcribe(proc, 'a', language='th')['code'] == 'unsupported_language'
    forced = _transcribe(proc, 'b', language='ja', options={'model': 'parakeet'})
    assert forced['code'] == 'unsupported_language' and 'Automatic or Nemotron' in forced['message']
    assert _transcribe(proc, 'c', language='el', options={'model': 'nemotron'})['code'] == 'unsupported_language'
    assert _transcribe(proc, 'd', audio_path='')['code'] == 'bad_params'
    assert _transcribe(proc, 'e', audio_path=str(tmp_path / 'missing.wav'))['code'] == 'bad_params'
    _send(proc, {'id': 'f', 'type': 'tts.synthesize', 'params': {}})
    assert _terminal(proc, 'f')['code'] == 'bad_params'
    _send(proc, {'id': 'g', 'type': 'asr.transcribe', 'params': 'nope'})
    assert _terminal(proc, 'g')['code'] == 'bad_params'


def test_a_download_that_cannot_reach_the_server_is_retryable(proc):
    _recv(proc)
    frame = _transcribe(proc, 'h', language='pt-br')
    assert frame['code'] == 'network_unavailable' and frame['retryable'] is True
    # Nemotron for Japanese, and the speaker model with it.
    frame = _transcribe(proc, 'i', language='ja', options={'label_speakers': True})
    assert frame['code'] == 'network_unavailable'


def test_live_captions_refuse_what_nemotron_does_not_stream(proc):
    _recv(proc)
    _send(proc, {'id': 's1', 'type': 'asr.stream', 'params': {'language': 'el', 'samplerate': 16000}})
    frame = _terminal(proc, 's1')
    assert frame['code'] == 'unsupported_language' and 'Import' in frame['message']
    _send(proc, {'id': 's2', 'type': 'asr.stream', 'params': {'language': 'en', 'samplerate': 44100}})
    assert _terminal(proc, 's2')['code'] == 'bad_params'


def test_a_live_session_without_its_model_fails_after_saying_why(proc):
    _recv(proc)
    _send(proc, {'id': 's3', 'type': 'asr.stream', 'params': {'language': 'en-us', 'samplerate': 16000}})
    _send(proc, {'id': 's3', 'type': 'asr.audio', 'data': {'pcm': ''}})
    frames = []
    while True:
        frame = _recv(proc)
        if frame.get('id') == 's3':
            frames.append(frame)
            if frame['type'] in ('result', 'error'):
                break
    assert frames[-1]['code'] == 'network_unavailable'
    # The host shows interim text, not progress: the download is announced there.
    assert frames[0]['type'] == 'partial' and frames[0]['data']['final'] is False
    assert frames[0]['data']['text'].startswith('Downloading Nemotron')


def test_a_cancelled_live_session_ends(proc):
    _recv(proc)
    _send(proc, {'id': 's4', 'type': 'asr.stream', 'params': {'language': 'de', 'samplerate': 16000}})
    _send(proc, {'type': 'cancel', 'target': 's4'})
    assert _terminal(proc, 's4')['code'] in ('cancelled', 'network_unavailable')


def test_shutdown_exits_cleanly(proc):
    _recv(proc)
    _send(proc, {'type': 'shutdown'})
    assert proc.wait(timeout=10) == 0
