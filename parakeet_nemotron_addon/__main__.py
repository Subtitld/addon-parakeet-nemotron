"""Parakeet & Nemotron add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline speech-to-text with two NVIDIA models on NVIDIA's NeMo-Speech.cpp
runtime, loaded in-process (engine.py):

- `asr.transcribe`: a file, with Parakeet TDT v3 in its 25 European languages
  and Nemotron 3.5 ASR Streaming in the 8 others it covers (models.py), in
  windows of about a minute (windows.py). Speaker labels, when asked for, come
  from one pass of Nemotron-3-Diarization over the whole file (speakers.py).
- `asr.stream`: live captions from Nemotron, which streams.

Each model downloads on first use (download.py); after that nothing leaves the
machine.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:parakeet-nemotron]` and surfaces.

    host -> us  {"id","type":"asr.transcribe","params":{audio_path,language,options}}
    us -> host  {"id","type":"progress","value","message"}
    us -> host  {"id","type":"partial","data":{start,end,text,speaker}}  (each finished cue)
    us -> host  {"id","type":"result","data":{"segments":[...],...}}
    host -> us  {"id","type":"asr.stream","params":{language,options,samplerate}}
    host -> us  {"id","type":"asr.audio","data":{"pcm": base64 int16 LE, 16 kHz mono}}
    us -> host  {"id","type":"partial","data":{start,end,text,speaker,final}}
    host -> us  {"id","type":"asr.stop"}  ->  us: {"id","type":"result","data":{"segments":[finals]}}
    host -> us  {"type":"cancel","target"} | {"type":"shutdown"}

Options (the request's `options`, else PARAKEET_NEMOTRON_<KEY> from Configure)
------------------------------------------------------------------------------
model: auto | parakeet | nemotron; slice_by_phrases: one cue per sentence;
label_speakers: speaker labels; live_delay: 160 | 320 | 560 | 1120 (ms of
lookahead for live captions).

Limits
------
A cancel takes effect between windows of a file, or stops a download (which
resumes on the next request). Speaker labels are for files only.
"""

from __future__ import annotations

import base64
import errno
import json
import logging
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from array import array

from parakeet_nemotron_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from parakeet_nemotron_addon import download, engine, models, speakers, tls
from parakeet_nemotron_addon.engine import SAMPLE_RATE
from parakeet_nemotron_addon.phrases import PAUSE_SEC, words_to_segments
from parakeet_nemotron_addon.timing import Corrector
from parakeet_nemotron_addon.windows import windows

ADDON_VERSION = __version__
ENV_PREFIX = ADDON_ID.replace('-', '_').upper() + '_'  # PARAKEET_NEMOTRON_

ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_OOM = 'oom'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'

# No space left on the device, or over the user's disk quota.
_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, 'EDQUOT', 122)}

# Window length per model. Nemotron's offline path needs more memory per second.
WINDOW_SEC = {models.PARAKEET: 60.0, models.NEMOTRON: 30.0}
# Below this there is nothing to transcribe.
_MIN_SAMPLES = SAMPLE_RATE // 10
_READ_CHUNK_BYTES = 1 << 20

# Live captions' lookahead, in ms, and the right context it takes, in 80 ms frames.
LIVE_DELAYS = {'160': 1, '320': 3, '560': 6, '1120': 13}
DEFAULT_LIVE_DELAY = '560'

log = logging.getLogger('parakeet_nemotron_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`, so nothing else can write a
    stray line into the protocol: the runtime and ggml print to stdout, which
    now goes to stderr, the log. Falls back to the Python-level swap if the fd
    dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_partial(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'partial', 'data': data})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class _Cancelled(Exception):
    """The host cancelled this request."""


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _raw(options: dict, key: str) -> str:
    """The request's value for `key`, else the environment's, as a stripped string."""
    value = options.get(key) if isinstance(options, dict) else None
    if value is None or value == '':
        return _env(ENV_PREFIX + key.upper())
    return str(value).strip()


def _opt_bool(options: dict, key: str, default: bool) -> bool:
    """Request option, env, then default. An explicit False stays False."""
    if isinstance(options, dict) and isinstance(options.get(key), bool):
        return options[key]
    raw = _raw(options, key).lower()
    if not raw:
        return default
    return raw not in ('0', 'false', 'no', 'off')


def _options(sent) -> dict:
    sent = sent if isinstance(sent, dict) else {}
    model = _raw(sent, 'model').lower() or models.AUTO
    delay = _raw(sent, 'live_delay')
    delay = delay[:-2].strip() if delay.endswith('ms') else delay
    if delay.endswith('.0'):
        delay = delay[:-2]
    return {
        'model': model if model in models.CHOICES else models.AUTO,
        'slice_by_phrases': _opt_bool(sent, 'slice_by_phrases', True),
        'label_speakers': _opt_bool(sent, 'label_speakers', False),
        'live_delay': delay if delay in LIVE_DELAYS else DEFAULT_LIVE_DELAY,
    }


# ---------------------------------------------------------------------------
# ffmpeg (from the Whistle add-on)
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _decode(ffmpeg: str, src: str, cancel: threading.Event) -> array:
    """Decode anything ffmpeg reads into 16 kHz mono float samples in [-1, 1].

    ffmpeg writes float32 straight into a pipe, read in chunks into one
    array('f'): no temporary audio file, no PCM copy. stderr goes to a
    temporary file, so a chatty ffmpeg cannot fill a pipe and stall.
    """
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-i', src, '-vn', '-sn', '-dn',
           '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1']
    samples = array('f')
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=errors, env=_child_env(),
                                startupinfo=_no_window_startupinfo())
        pending = b''
        try:
            while True:
                if cancel.is_set():
                    raise _Cancelled()
                chunk = proc.stdout.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                data = pending + chunk
                usable = len(data) // 4 * 4
                samples.frombytes(memoryview(data)[:usable])
                pending = data[usable:]
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        proc.stdout.close()
        returncode = proc.wait()
        if returncode != 0 or not len(samples):
            errors.seek(0)
            tail = ' / '.join(errors.read().decode('utf-8', 'replace').strip().splitlines()[-5:])
            if 'No space left' in tail:
                raise _AddonError(ERR_DISK_FULL, f'Out of disk space while decoding the audio: {tail}')
            raise _AddonError(ERR_INTERNAL,
                              f'Could not decode the audio with ffmpeg '
                              f'(exit {returncode}): {tail or "no output"}')
    if sys.byteorder != 'little':
        samples.byteswap()  # ffmpeg wrote little-endian floats
    return samples


# ---------------------------------------------------------------------------
# Models: downloaded once, loaded once, shared by every request
# ---------------------------------------------------------------------------
_download_lock = threading.Lock()  # one download at a time: two would write the same .part
_models_lock = threading.Lock()
_recognizers: dict[str, engine.Recognizer] = {}
_streams_on: dict[str, int] = {}  # model -> live streams using its recognizer
_diarizer: engine.Diarizer | None = None


def _ensure_models(names: list[str], report, cancel: threading.Event) -> None:
    """Download the models in `names` that are missing. `report(fraction, message)`."""
    pins = [models.pin(name) for name in names]
    if not download.missing(pins):
        return
    with _download_lock:
        todo = download.missing(pins)
        if not todo:
            return
        total = sum(pin['size'] for pin in todo)
        which = ' and '.join(models.DISPLAY[name] for name, pin in zip(names, pins) if pin in todo)
        message = f'Downloading {which} ({total / 1e6:,.0f} MB)...'
        report(0.0, message)
        last = [0.0]

        def progress(done: int, total_bytes: int) -> None:
            now = time.monotonic()
            if now - last[0] >= 0.25 or done >= total_bytes:
                last[0] = now
                report(done / max(1, total_bytes), message)

        started = time.monotonic()
        download.ensure(todo, progress, cancel.is_set)
        log.info('downloaded %s, %.0f MB in %.1fs', which, total / 1e6, time.monotonic() - started)


def _recognizer(name: str, right_context: int | None = None) -> engine.Recognizer:
    """The loaded model `name`. For Nemotron, `right_context` asks for a live
    lookahead; the model is reloaded for a new one unless a stream is using it."""
    with _models_lock:
        current = _recognizers.get(name)
        if current is not None and (right_context is None or current.right_context == right_context
                                    or _streams_on.get(name)):
            return current
        if current is not None:
            current.close()
        rc = right_context if right_context is not None else LIVE_DELAYS[DEFAULT_LIVE_DELAY]
        recognizer = engine.Recognizer(download.local_path(models.pin(name)),
                                       right_context=rc if name == models.NEMOTRON else -1,
                                       endpointing=name == models.NEMOTRON)
        _recognizers[name] = recognizer
        return recognizer


def _get_diarizer() -> engine.Diarizer:
    global _diarizer
    with _models_lock:
        if _diarizer is None:
            _diarizer = engine.Diarizer(download.local_path(models.pin(models.DIARIZATION)))
        return _diarizer


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.inbox: queue.Queue | None = None  # live sessions: audio, then _STOP


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()
_STOP = object()


def _outcome_of(exc: BaseException, req_id: str, what: str) -> tuple:
    """The error frame for an exception that ended a request."""
    if isinstance(exc, (_Cancelled, download.Cancelled)):
        log.info('%s %s: cancelled', what, req_id)
        return ('error', ERR_CANCELLED, 'cancelled', False)
    if isinstance(exc, _AddonError):
        log.warning('%s %s failed: [%s] %s', what, req_id, exc.code, exc.message)
        return ('error', exc.code, exc.message, exc.retryable)
    if isinstance(exc, models.UnsupportedLanguage):
        return ('error', ERR_UNSUPPORTED_LANGUAGE, str(exc), False)
    if isinstance(exc, download.DownloadError):
        log.warning('%s %s: download failed: %s', what, req_id, exc)
        return ('error', ERR_NETWORK, f'Could not download the model: {exc}', True)
    if isinstance(exc, (MemoryError, engine.OutOfMemory)):
        return ('error', ERR_OOM, 'Out of memory while transcribing. Close other programs and try again.', False)
    if isinstance(exc, engine.EngineError):
        log.warning('%s %s failed in the runtime: %s', what, req_id, exc)
        return ('error', ERR_INTERNAL, str(exc), False)
    if isinstance(exc, OSError) and getattr(exc, 'errno', None) in _DISK_FULL_ERRNOS:
        return ('error', ERR_DISK_FULL, f'Out of disk space: {exc}', False)
    log.error('%s %s failed', what, req_id, exc_info=exc)
    return ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)


def _handle_transcribe(state: _Request, params: dict) -> None:
    req_id = state.req_id
    audio_path = params.get('audio_path')
    if not isinstance(audio_path, str) or not audio_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'audio_path is required')
        return
    if not os.path.isfile(audio_path):
        _emit_error(req_id, ERR_BAD_PARAMS, f'audio_path is not a file: {audio_path!r}')
        return
    options = _options(params.get('options'))
    try:
        model, language = models.choose_for_file(params.get('language'), options['model'])
    except models.UnsupportedLanguage as exc:
        _emit_error(req_id, ERR_UNSUPPORTED_LANGUAGE, str(exc))
        return
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _emit_error(req_id, ERR_BAD_PARAMS,
                    'ffmpeg not found. Set SUBTITLD_FFMPEG_EXECUTABLE or put ffmpeg on PATH.')
        return
    label_speakers = options['label_speakers']

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        log.info('transcribe %s: model=%s language=%s speakers=%s one_per_sentence=%s', req_id, model,
                 language or 'detect', label_speakers, options['slice_by_phrases'])
        if state.cancel.is_set():
            raise _Cancelled()

        # 1. Download: 0.00 -> 0.15
        needed = [model] + ([models.DIARIZATION] if label_speakers else [])
        _ensure_models(needed, lambda f, m: _emit_progress(req_id, 0.15 * f, m), state.cancel)

        # 2. Decode: 0.15 -> 0.18
        _emit_progress(req_id, 0.15, 'Decoding audio...')
        samples = _decode(ffmpeg, audio_path, state.cancel)
        duration = len(samples) / float(SAMPLE_RATE)
        if state.cancel.is_set():
            raise _Cancelled()

        segments: list[dict] = []
        if len(samples) >= _MIN_SAMPLES:
            # 3. Load: 0.18 -> 0.20
            _emit_progress(req_id, 0.18, f'Loading {models.DISPLAY[model]}...')
            recognizer = _recognizer(model)
            if state.cancel.is_set():
                raise _Cancelled()

            # 4. Speakers: 0.20 -> 0.30
            turns = None
            span = 0.2
            if label_speakers:
                _emit_progress(req_id, 0.2, 'Telling the speakers apart...')
                started = time.monotonic()
                turns = _get_diarizer().diarize(
                    samples, progress=lambda f: _emit_progress(req_id, 0.2 + 0.1 * f, 'Telling the speakers apart...'),
                    cancelled=state.cancel.is_set)
                if turns is None or state.cancel.is_set():
                    raise _Cancelled()
                log.info('transcribe %s: %d speaker turns, %d speakers, in %.1fs', req_id, len(turns),
                         len({t.speaker for t in turns}), time.monotonic() - started)
                span = 0.3

            # 5. Transcribe window by window: span -> 1.00
            _emit_progress(req_id, span, 'Transcribing...')
            started = time.monotonic()
            corrector = Corrector() if model == models.NEMOTRON else None
            # Nemotron may write no punctuation at all: its rows also close at pauses.
            pause = PAUSE_SEC if model == models.NEMOTRON else None
            names: dict = {}
            words: list[dict] = []
            sent = 0
            detected = ''
            ranges = windows(samples, window_sec=WINDOW_SEC[model])
            view = memoryview(samples)
            for begin, end in ranges:
                if end - begin >= _MIN_SAMPLES:
                    result = recognizer.recognize(array('f', view[begin:end]), language)
                    offset = begin / float(SAMPLE_RATE)
                    window_words = [{'word': w.text, 'start': w.start + offset, 'end': w.end + offset}
                                    for w in result.words]
                    if not window_words and result.text:
                        window_words = [{'word': result.text, 'start': offset, 'end': end / float(SAMPLE_RATE)}]
                    if corrector is not None:
                        corrector.correct(window_words)
                    if turns is not None:
                        speakers.assign(window_words, turns, names)
                    words.extend(window_words)
                    detected = detected or result.language
                if state.cancel.is_set():
                    raise _Cancelled()
                # Every cue but the last is finished: send those as they come.
                cues = words_to_segments(words, duration, by_sentence=options['slice_by_phrases'],
                                         pause_sec=pause)
                for cue in cues[sent:len(cues) - 1]:
                    _emit_partial(req_id, cue)
                sent = max(sent, len(cues) - 1)
                _emit_progress(req_id, span + (1.0 - span) * end / len(samples), 'Transcribing...')
            log.info('transcribe %s: %.1fs of audio in %d windows, %.2fs', req_id, duration,
                     len(ranges), time.monotonic() - started)
            segments = words_to_segments(words, duration, by_sentence=options['slice_by_phrases'],
                                         pause_sec=pause)
        if state.cancel.is_set():
            raise _Cancelled()
        primary, _region = models.split_tag(params.get('language'))
        outcome = ('result', {
            'segments': segments,
            'duration_sec': round(duration, 3),
            'model': model,
            'language': primary or (detected.split('-')[0].lower() if detected else ''),
        })
    except BaseException as exc:  # noqa: BLE001 — every path must end in one terminal frame
        outcome = _outcome_of(exc, req_id, 'transcribe')

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _pcm_to_floats(pcm: bytes) -> array:
    """16-bit little-endian PCM to float samples in [-1, 1)."""
    ints = array('h')
    ints.frombytes(pcm[:len(pcm) // 2 * 2])
    if sys.byteorder != 'little':
        ints.byteswap()
    return array('f', (s / 32768.0 for s in ints))


def _handle_stream(state: _Request, params: dict) -> None:
    """One live session: audio arrives in state.inbox until _STOP."""
    req_id = state.req_id
    options = _options(params.get('options'))
    try:
        locale = models.locale_for_live(params.get('language'))
    except models.UnsupportedLanguage as exc:
        _emit_error(req_id, ERR_UNSUPPORTED_LANGUAGE, str(exc))
        return
    try:
        rate = int(params.get('samplerate') or SAMPLE_RATE)
    except (TypeError, ValueError):
        rate = SAMPLE_RATE
    if rate != SAMPLE_RATE:
        _emit_error(req_id, ERR_BAD_PARAMS, f'live audio must be {SAMPLE_RATE} Hz, not {rate}')
        return
    right_context = LIVE_DELAYS[options['live_delay']]
    log.info('stream %s: locale=%s delay=%sms', req_id, locale, options['live_delay'])

    finals: list[dict] = []
    stream = None
    holding = False
    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        # The first time, Nemotron downloads (742 MB) while the audio queues
        # up. The host shows interim text, not progress, so the download says
        # how it is going there.
        def report(fraction: float, message: str) -> None:
            _emit_partial(req_id, {'start': 0.0, 'end': 0.0, 'speaker': 'A', 'final': False,
                                   'text': f'{message.rstrip(".")} {fraction:.0%}'})

        _ensure_models([models.NEMOTRON], report, state.cancel)
        recognizer = _recognizer(models.NEMOTRON, right_context)
        with _models_lock:
            _streams_on[models.NEMOTRON] = _streams_on.get(models.NEMOTRON, 0) + 1
            holding = True
        stream = recognizer.stream(locale, interim=True)
        corrector = Corrector()
        last_interim = ''

        def handle(results: list[engine.Result]) -> None:
            nonlocal last_interim
            for result in results:
                if not result.final:
                    if result.text and result.text != last_interim:
                        last_interim = result.text
                        _emit_partial(req_id, {'start': 0.0, 'end': 0.0, 'text': result.text,
                                               'speaker': 'A', 'final': False})
                    continue
                last_interim = ''
                words = corrector.correct([{'word': w.text, 'start': w.start, 'end': w.end}
                                           for w in result.words])
                if not words and result.text:
                    end = round(result.audio_processed, 3)
                    words = [{'word': result.text, 'start': max(0.0, end - 1.0), 'end': end}]
                for cue in words_to_segments(words, result.audio_processed,
                                             by_sentence=options['slice_by_phrases'], pause_sec=PAUSE_SEC):
                    finals.append(cue)
                    _emit_partial(req_id, {**cue, 'final': True})

        while True:
            item = state.inbox.get()
            if state.cancel.is_set():
                raise _Cancelled()
            if item is _STOP:
                break
            stream.push(_pcm_to_floats(item))
            handle(stream.results())
        handle(stream.finish())
        outcome = ('result', {'segments': finals})
    except BaseException as exc:  # noqa: BLE001
        outcome = _outcome_of(exc, req_id, 'stream')
    finally:
        if stream is not None:
            stream.close()
        if holding:
            with _models_lock:
                _streams_on[models.NEMOTRON] -= 1
    if outcome[0] == 'result':
        log.info('stream %s: %d finals', req_id, len(finals))
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, params: dict, live: bool = False) -> None:
    state = _Request(req_id)
    if live:
        state.inbox = queue.Queue()
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, params)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'{ADDON_ID}-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


def _cancel(state: _Request) -> None:
    state.cancel.set()
    if state.inbox is not None:
        state.inbox.put(_STOP)  # wake the session so it sees the cancel


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [
            {'task': 'asr.transcribe', 'languages': list(models.LANGUAGES), 'streaming': False,
             'speakers': True},
            {'task': 'asr.stream', 'languages': list(models.NEMOTRON_LANGUAGES), 'streaming': True},
        ],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _session(req_id) -> _Request | None:
    with _active_lock:
        state = _active.get(req_id) if isinstance(req_id, str) else None
    return state if state is not None and state.inbox is not None else None


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'asr.audio':
            state = _session(frame.get('id'))
            data = frame.get('data') if isinstance(frame.get('data'), dict) else {}
            if state is not None and data.get('pcm'):
                try:
                    state.inbox.put(base64.b64decode(data['pcm']))
                except (ValueError, TypeError):
                    log.warning('asr.audio with bad base64 for %s', frame.get('id'))
            continue

        if ftype == 'asr.stop':
            state = _session(frame.get('id'))
            if state is not None:
                state.inbox.put(_STOP)
            continue

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                _cancel(state)
            continue

        if ftype in ('asr.transcribe', 'asr.stream'):
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('%s without id', ftype)
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            if ftype == 'asr.transcribe':
                _spawn(req_id, _handle_transcribe, params)
            else:
                _spawn(req_id, _handle_stream, params, live=True)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        _cancel(state)
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Check what the frozen bundle needs, without the network or a model:
    the runtime loads and is 0.3, and the TLS and hashing modules the download
    uses are there."""
    import hashlib
    import ssl

    report: dict = {'version': ADDON_VERSION}
    try:
        report['runtime'] = engine.runtime_version()
        ok = True
    except engine.EngineError as exc:
        report['runtime'] = str(exc)
        ok = False
    report['languages'] = len(models.LANGUAGES)
    report['openssl'] = ssl.OPENSSL_VERSION
    report['sha256'] = hashlib.sha256(b'').hexdigest()[:8]
    report['ca_bundle'] = os.environ.get('SSL_CERT_FILE', '')
    report['ok'] = ok
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0 if ok else 1


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'parakeet-nemotron-addon {ADDON_VERSION}')
        return 0
    logging.basicConfig(
        level=(_env(ENV_PREFIX + 'ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('parakeet-nemotron-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())
    if not engine.library_path().is_file():
        # Not fatal: better to appear in the engine list and fail the first
        # request with a clear message than to vanish from the UI.
        log.warning('the speech runtime is missing: %s', engine.library_path())
    if _find_ffmpeg() is None:
        log.warning('ffmpeg not found; file transcription will fail until one is available')

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        stopped = _stop_workers()
    if not stopped:
        # A worker is still inside the runtime. Normal teardown would free
        # state under it, so leave without teardown.
        log.warning('a transcription is still running; exiting without cleanup')
        try:
            sys.stderr.flush()
        finally:
            os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
