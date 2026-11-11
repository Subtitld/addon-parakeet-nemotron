"""The add-on as Subtitld runs it: a process spoken to over stdio, transcribing for real.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

By default the process is the source tree (`python -m parakeet_nemotron_addon`)
with the libraries from tools/build_native.py. With
PARAKEET_NEMOTRON_TEST_BINARY set it is that executable instead, which is how
the release workflow runs these tests: on the frozen add-on unpacked from the
release zip the way Subtitld installs it, with nothing but what is inside its
own bundle.

The models (1.6 GB) download into $SUBTITLD_MODELS_DIR, a temporary folder
when it is not set. ffmpeg decodes the sample, as the host's would; where
there is none (the clean containers of tools/test_archive.sh), on Linux and
macOS tests/wav_ffmpeg.py stands in for it.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parakeet_nemotron_addon import engine  # noqa: E402

BINARY = os.environ.get('PARAKEET_NEMOTRON_TEST_BINARY')
HAVE_RUNTIME = bool(BINARY) or engine.library_path().is_file()
SAMPLE = REPO_ROOT / 'tests' / 'data' / 'jfk.wav'

pytestmark = pytest.mark.skipif(not HAVE_RUNTIME, reason='no runtime; run tools/build_native.py')

_MODELS = Path(os.environ.get('SUBTITLD_MODELS_DIR') or tempfile.mkdtemp(prefix='parakeet-nemotron-models-'))


def _ffmpeg() -> str:
    found = shutil.which('ffmpeg')
    if found:
        return found
    if sys.platform == 'win32':
        pytest.fail('needs ffmpeg on the PATH')
    script = Path(tempfile.mkdtemp(prefix='wav-ffmpeg-')) / 'ffmpeg'
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{REPO_ROOT / "tests" / "wav_ffmpeg.py"}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


FFMPEG = _ffmpeg()


def _start(cwd: Path, **extra_env: str) -> subprocess.Popen:
    env = dict(os.environ)
    env['SUBTITLD_MODELS_DIR'] = str(_MODELS)
    env['SUBTITLD_FFMPEG_EXECUTABLE'] = FFMPEG
    for name in list(env):
        if name.startswith('PARAKEET_NEMOTRON_') and name != 'PARAKEET_NEMOTRON_TEST_BINARY':
            env.pop(name)
    env.update(extra_env)
    if BINARY:
        env.pop('PYTHONPATH', None)  # the bundle's own code, nothing from this checkout
        command = [BINARY]
    else:
        env['PYTHONPATH'] = str(REPO_ROOT)
        command = [sys.executable, '-m', 'parakeet_nemotron_addon']
    # The log goes to a file, not a pipe nobody reads while waiting on stdout:
    # Windows pipes fill after a few KiB and the add-on would block on its log.
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log)
    proc.log = log
    hello = json.loads(proc.stdout.readline())
    assert hello['type'] == 'hello' and hello['addon'] == 'parakeet-nemotron'
    return proc


def _log(proc) -> str:
    proc.log.seek(0)
    return proc.log.read().decode('utf-8', 'replace')[-20000:]


def _send(proc, frame: dict) -> None:
    proc.stdin.write((json.dumps(frame) + '\n').encode())
    proc.stdin.flush()


def _wait(proc, req_id: str) -> tuple[dict, list[dict]]:
    """The terminal frame for `req_id`, and the frames before it."""
    seen = []
    while True:
        line = proc.stdout.readline()
        assert line, _log(proc)
        frame = json.loads(line)
        if frame.get('id') != req_id:
            continue
        if frame['type'] in ('result', 'error'):
            return frame, seen
        seen.append(frame)


def _transcribe(proc, req_id: str, **params) -> tuple[dict, list[dict]]:
    params.setdefault('audio_path', str(SAMPLE))
    _send(proc, {'id': req_id, 'type': 'asr.transcribe', 'params': params})
    return _wait(proc, req_id)


def _stop(proc) -> None:
    proc.stdin.close()
    proc.wait(timeout=30)
    proc.stdout.close()
    # The add-on's log, so a failure in CI says what the runtime did.
    sys.stderr.write(_log(proc))
    proc.log.close()


def _text(segments: list[dict]) -> str:
    return ' '.join(s['text'] for s in segments).lower()


def _pcm(path: Path) -> bytes:
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        return wav.readframes(wav.getnframes())


def test_transcribes_a_file_with_each_model(tmp_path):
    proc = _start(tmp_path)
    try:
        frame, seen = _transcribe(proc, 'p', language='en')
        assert frame['type'] == 'result', frame
        data = frame['data']
        assert data['model'] == 'parakeet' and data['language'] == 'en'
        assert data['duration_sec'] == pytest.approx(11.0, abs=0.1)
        text = _text(data['segments'])
        assert 'fellow americans' in text and 'ask not what your country can do for you' in text
        first = data['segments'][0]
        assert first['speaker'] == 'A' and 0.0 <= first['start'] < 0.6 and first['end'] > first['start']
        progress = [f for f in seen if f['type'] == 'progress']
        assert progress[-1]['value'] == 1.0
        # Finished cues arrive as they are made, the last one with the result.
        partials = [f['data'] for f in seen if f['type'] == 'partial']
        assert partials == data['segments'][:len(partials)]

        frame, _ = _transcribe(proc, 'n', language='en-US', options={'model': 'nemotron'})
        assert frame['type'] == 'result', frame
        assert frame['data']['model'] == 'nemotron'
        text = _text(frame['data']['segments'])
        assert 'fellow americans' in text and 'your country' in text
        # Moved back onto the words: Nemotron alone stamps "And" at 0.96 s.
        assert frame['data']['segments'][0]['start'] < 0.7

        frame, seen = _transcribe(proc, 's', language='en', options={'label_speakers': True})
        assert frame['type'] == 'result', frame
        assert {s['speaker'] for s in frame['data']['segments']} <= {'A', 'B'}
        assert any('speakers' in f.get('message', '') for f in seen if f['type'] == 'progress')
    finally:
        _stop(proc)


def test_live_captions(tmp_path):
    proc = _start(tmp_path)
    try:
        pcm = _pcm(SAMPLE)
        _send(proc, {'id': 'live', 'type': 'asr.stream',
                     'params': {'language': 'en', 'options': {'live_delay': '320'}, 'samplerate': 16000}})
        # Frames of uneven size, as a recorder's callback gives them.
        sizes, position = [3200, 1000, 6402, 320], 0
        while position < len(pcm):
            size = sizes[position % len(sizes)]
            _send(proc, {'id': 'live', 'type': 'asr.audio',
                         'data': {'pcm': base64.b64encode(pcm[position:position + size]).decode()}})
            position += size
        _send(proc, {'id': 'live', 'type': 'asr.stop'})
        frame, seen = _wait(proc, 'live')
        assert frame['type'] == 'result', frame
        finals = [f['data'] for f in seen if f['type'] == 'partial' and f['data']['final']]
        assert finals and [{k: v for k, v in f.items() if k != 'final'} for f in finals] == frame['data']['segments']
        text = _text(frame['data']['segments'])
        assert 'fellow americans' in text and 'your country' in text
        starts = [s['start'] for s in frame['data']['segments']]
        assert starts == sorted(starts) and all(s['end'] > s['start'] for s in frame['data']['segments'])
        assert any(not f['data']['final'] for f in seen if f['type'] == 'partial'), 'no interim text'
    finally:
        _stop(proc)


def test_a_cancel_stops_a_long_file(tmp_path):
    long_file = tmp_path / 'long.wav'
    with wave.open(str(long_file), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(_pcm(SAMPLE) * 20)
    proc = _start(tmp_path)
    try:
        _send(proc, {'id': 'long', 'type': 'asr.transcribe',
                     'params': {'audio_path': str(long_file), 'language': 'en'}})
        while True:  # wait until it is transcribing, then cancel
            line = proc.stdout.readline()
            assert line, _log(proc)
            frame = json.loads(line)
            if frame.get('id') != 'long':
                continue
            assert frame['type'] not in ('result', 'error'), frame
            if frame['type'] == 'progress' and frame['message'] == 'Transcribing...':
                break
        _send(proc, {'type': 'cancel', 'target': 'long'})
        frame, _ = _wait(proc, 'long')
        assert frame['type'] == 'error' and frame['code'] == 'cancelled', frame
        # The runtime still transcribes afterwards.
        frame, _ = _transcribe(proc, 'after', language='en')
        assert frame['type'] == 'result', frame
    finally:
        _stop(proc)
