"""Speech recognition and diarization through NeMo-Speech.cpp, loaded in-process with ctypes.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

NVIDIA's NeMo-Speech.cpp (Apache-2.0, on ggml) runs Parakeet, Nemotron and
the diarization model. Its C ABI (include/nemo_speech/asr.h and diar.h,
v0.3.0) is opaque handles and size-prefixed structs, small enough to bind here
with ctypes, so the add-on needs neither numpy nor a Python ML runtime.
tools/build_native.py puts the library and what it loads into build/assets/;
the frozen add-on carries them in _internal/assets/.

A Recognizer holds one model. It takes calls from several threads at once (a
file transcription while live captions stream); each Stream stays on the
thread that drives it. Word times come back in milliseconds and are returned
here in seconds.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import platform
import sys
import threading
import time
from array import array
from ctypes import POINTER, Structure, byref, c_bool, c_char_p, c_double, c_float, c_int32, c_int64, c_size_t, c_void_p
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger('parakeet_nemotron_addon.engine')

SAMPLE_RATE = 16000
VERSION_PREFIX = '0.3.'
# The speaker model's chunking for a whole file: large chunks, about 25 times
# real time on 8 CPU threads. The low-latency default runs at about 2.
PRESET = b'v3-offline'

_PACKAGE = Path(__file__).resolve().parent

LIBRARY = {
    'linux': 'libnemo_speech_asr_c.so.1',
    'win32': 'nemo_speech_asr_c.dll',
    'darwin': 'libnemo_speech_asr_c.1.dylib',
}

OK, INVALID_ARGUMENT, OUT_OF_MEMORY, RUNTIME_ERROR, CANCELLED = 0, 1, 2, 3, 4


class EngineError(Exception):
    """The runtime failed, or could not be loaded."""


class OutOfMemory(EngineError):
    """The runtime ran out of memory."""


def assets_dir() -> Path:
    """Where the native libraries are: $PARAKEET_NEMOTRON_ASSETS, the frozen bundle's, or build/assets/."""
    override = (os.environ.get('PARAKEET_NEMOTRON_ASSETS') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'assets'
    return _PACKAGE.parent / 'build' / 'assets'


def library_path() -> Path:
    return assets_dir() / LIBRARY.get(sys.platform, LIBRARY['linux'])


# ---------------------------------------------------------------------------
# The CPU: x86-64 builds need AVX2
# ---------------------------------------------------------------------------
def _is_x86_64() -> bool:
    return platform.machine().lower() in ('x86_64', 'amd64', 'x64')


def cpu_lacks_avx2() -> bool:
    """True when this is an x86-64 processor known not to have AVX2.

    The runtime calls ggml's CPU code built for x86-64-v3 (AVX2, FMA, F16C),
    and on an older processor it would die on an illegal instruction, taking
    the add-on with it. When the check cannot tell, it lets the load go ahead.
    """
    if not _is_x86_64():
        return False
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as handle:
                for line in handle:
                    if line.startswith('flags'):
                        return 'avx2' not in line.split(':', 1)[1].split()
        except OSError:
            return False
        return False
    if sys.platform == 'win32':
        # PF_AVX2_INSTRUCTIONS_AVAILABLE. A Windows too old to know the
        # feature answers False for it, so a False only counts on Windows 11.
        try:
            if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):
                return False
            return sys.getwindowsversion().build >= 22000
        except (AttributeError, OSError):
            return False
    return False


# ---------------------------------------------------------------------------
# The C ABI (include/nemo_speech/asr.h, diar.h at v0.3.0)
# ---------------------------------------------------------------------------
class BackendConfig(Structure):
    _fields_ = [('size', c_size_t), ('gpu', c_int32)]


class ModelConfig(Structure):
    _fields_ = [('size', c_size_t), ('path', c_char_p), ('name', c_char_p)]


class StreamingConfig(Structure):
    _fields_ = [('size', c_size_t), ('chunk_size', c_float), ('ctc_left_padding', c_float),
                ('ctc_right_padding', c_float), ('rnnt_right_context', c_int32)]


class EndpointingConfig(Structure):
    _fields_ = [('size', c_size_t), ('enable', c_bool), ('vad_based', c_bool),
                ('stop_history_eou_ms', c_int32)]


class RecognizerConfig(Structure):
    _fields_ = [('size', c_size_t), ('backend', POINTER(BackendConfig)),
                ('model', POINTER(ModelConfig)), ('streaming', POINTER(StreamingConfig)),
                ('decoder', c_void_p), ('vad', c_void_p), ('endpointing', POINTER(EndpointingConfig)),
                ('postproc', c_void_p), ('diar', c_void_p), ('batching', c_void_p), ('salm', c_void_p)]


class RecognitionOptions(Structure):
    _fields_ = [('size', c_size_t), ('request_id', c_char_p), ('language_code', c_char_p),
                ('interim_results', c_bool), ('enable_word_time_offsets', c_bool),
                ('enable_automatic_punctuation', c_bool), ('verbatim_transcripts', c_bool),
                ('profanity_filter', c_bool), ('stop_history_eou_ms', c_int32),
                ('speech_contexts', c_void_p), ('speech_context_count', c_size_t),
                ('max_alternatives', c_int32), ('enable_speaker_diarization', c_bool),
                ('max_speaker_count', c_int32)]


class DiarModelConfig(Structure):
    _fields_ = [('size', c_size_t), ('model_path', c_char_p), ('gpu', c_int32), ('preset', c_char_p),
                ('chunk_frames', c_int32), ('right_context_frames', c_int32),
                ('left_context_frames', c_int32), ('fifo_frames', c_int32),
                ('spkcache_frames', c_int32), ('update_period_frames', c_int32)]


class DiarSegment(Structure):
    _fields_ = [('start_time', c_double), ('end_time', c_double), ('speaker', c_int32)]


_p = c_void_p
_SIGNATURES = [
    ('nemo_speech_asr_version', [], c_char_p),
    ('nemo_speech_asr_last_error', [], c_char_p),
    ('nemo_speech_asr_recognition_options_default', [], RecognitionOptions),
    ('nemo_speech_asr_create', [POINTER(RecognizerConfig), POINTER(_p)], c_int32),
    ('nemo_speech_asr_destroy', [_p], None),
    ('nemo_speech_asr_recognize_f32',
     [_p, POINTER(RecognitionOptions), POINTER(c_float), c_size_t, c_int32, POINTER(_p)], c_int32),
    ('nemo_speech_asr_streaming_recognize', [_p, POINTER(RecognitionOptions), POINTER(_p)], c_int32),
    ('nemo_speech_asr_stream_push_f32', [_p, POINTER(c_float), c_size_t, c_int32], c_int32),
    ('nemo_speech_asr_stream_force_endpoint', [_p], c_int32),
    ('nemo_speech_asr_stream_finish', [_p], c_int32),
    ('nemo_speech_asr_stream_next', [_p, POINTER(_p)], c_int32),
    ('nemo_speech_asr_stream_close', [_p], None),
    ('nemo_speech_asr_result_is_final', [_p], c_bool),
    ('nemo_speech_asr_result_audio_processed', [_p], c_float),
    ('nemo_speech_asr_result_alternative_count', [_p], c_size_t),
    ('nemo_speech_asr_result_transcript', [_p, c_size_t], c_char_p),
    ('nemo_speech_asr_result_word_count', [_p, c_size_t], c_size_t),
    ('nemo_speech_asr_result_word_text', [_p, c_size_t, c_size_t], c_char_p),
    ('nemo_speech_asr_result_word_start_time', [_p, c_size_t, c_size_t], c_int32),
    ('nemo_speech_asr_result_word_end_time', [_p, c_size_t, c_size_t], c_int32),
    ('nemo_speech_asr_result_language_count', [_p, c_size_t], c_size_t),
    ('nemo_speech_asr_result_language_code', [_p, c_size_t, c_size_t], c_char_p),
    ('nemo_speech_asr_result_destroy', [_p], None),
    ('nemo_speech_diar_create', [POINTER(DiarModelConfig), POINTER(_p)], c_int32),
    ('nemo_speech_diar_destroy', [_p], None),
    ('nemo_speech_diar_num_speakers', [_p], c_int32),
    ('nemo_speech_diar_stream_open', [_p, POINTER(_p)], c_int32),
    ('nemo_speech_diar_stream_push_f32', [_p, POINTER(c_float), c_size_t, c_int32], c_int32),
    ('nemo_speech_diar_stream_finish', [_p], c_int32),
    ('nemo_speech_diar_stream_close', [_p], None),
    ('nemo_speech_diar_frame_count', [_p], c_int64),
    ('nemo_speech_diar_seconds_per_frame', [_p], c_double),
    ('nemo_speech_diar_segments', [_p, c_void_p, POINTER(DiarSegment), c_size_t, POINTER(c_size_t)], c_int32),
]

_lib = None
_lib_lock = threading.Lock()


def load_library():
    """Load the runtime's C ABI from assets_dir(), once."""
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        if cpu_lacks_avx2():
            raise EngineError('This processor lacks AVX2, which the speech runtime needs '
                              '(Intel since 2013, AMD since 2015).')
        path = library_path()
        if not path.is_file():
            raise EngineError(f'The speech runtime is missing from the add-on: {path}')
        try:
            if sys.platform == 'win32':
                os.add_dll_directory(str(path.parent))
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise EngineError(f'Could not load the speech runtime: {exc}') from exc
        for name, args, res in _SIGNATURES:
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
        # "nemo-speech-asr 0.3.0+subtitld" for our Linux build.
        version = lib.nemo_speech_asr_version().decode()
        if not version.split()[-1].startswith(VERSION_PREFIX):
            raise EngineError(f'{version} is not NeMo-Speech.cpp {VERSION_PREFIX}x')
        _lib = lib
        return lib


def runtime_version() -> str:
    return load_library().nemo_speech_asr_version().decode()


def _error(lib, status: int, what: str) -> EngineError:
    message = (lib.nemo_speech_asr_last_error() or b'').decode('utf-8', 'replace') or f'status {status}'
    cls = OutOfMemory if status == OUT_OF_MEMORY else EngineError
    return cls(f'{what} failed: {message}')


# ---------------------------------------------------------------------------
# Metal, on Apple Silicon: tried once in a child process
# ---------------------------------------------------------------------------
# NVIDIA's macOS library is the Metal build. Where Metal is not up to it (the
# paravirtual GPU of a macOS virtual machine, at least), the first graph run
# on it kills the process with a segmentation fault, which nothing in-process
# can catch. So before the add-on uses Metal for a kind of model, it runs one
# short recognition on it in a child process (`--probe-gpu`), and uses the CPU
# when the child does not come back with an answer. The verdict is kept in the
# model folder, per runtime and macOS version, so this happens once.
# $PARAKEET_NEMOTRON_DEVICE=cpu or gpu skips the test.
PROBE_TIMEOUT_SEC = 300  # loading on Metal compiles its shaders: 40 s on a CI machine
_probe_lock = threading.Lock()
_verdicts: dict[str, bool] = {}


def _apple_silicon() -> bool:
    return sys.platform == 'darwin' and platform.machine() == 'arm64'


def _verdict_file() -> Path:
    from parakeet_nemotron_addon import download
    return download.cache_root() / 'metal-check.json'


def _probe_command(kind: str, model_path: Path) -> list[str]:
    if getattr(sys, 'frozen', False):
        return [sys.executable, '--probe-gpu', kind, str(model_path)]
    return [sys.executable, '-m', 'parakeet_nemotron_addon', '--probe-gpu', kind, str(model_path)]


def metal_works(kind: str, model_path: Path) -> bool:
    """Whether `kind` ('asr' or 'diar') of model runs on Metal here; checked once, in a child process."""
    import json
    import subprocess
    with _probe_lock:
        if kind in _verdicts:
            return _verdicts[kind]
        key = f'{runtime_version()} on macOS {platform.mac_ver()[0]}'
        path = _verdict_file()
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            saved = {}
        if saved.get('runtime') == key and isinstance(saved.get(kind), bool):
            _verdicts[kind] = saved[kind]
            return saved[kind]
        started = time.monotonic()
        try:
            done = subprocess.run(_probe_command(kind, model_path), stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=PROBE_TIMEOUT_SEC)
            works = done.returncode == 0 and b'metal works' in done.stdout
            detail = f'exit {done.returncode}'
        except (OSError, subprocess.SubprocessError) as exc:
            works, detail = False, str(exc)
        log.info('Metal for %s: %s (%s, %.1fs)', kind, 'works' if works else 'fails; using the CPU', detail,
                 time.monotonic() - started)
        _verdicts[kind] = works
        if saved.get('runtime') != key:
            saved = {'runtime': key}
        saved[kind] = works
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved) + '\n', encoding='utf-8')
        except OSError as exc:
            log.warning('could not save the Metal check: %s', exc)
        return works


def probe_gpu(kind: str, model_path: Path) -> bool:
    """The child's side of metal_works(): one short run on the GPU only. A crash is the answer too."""
    samples = array('f', (0.3 * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE) * (i // 800 % 2)
                          for i in range(2 * SAMPLE_RATE)))
    if kind == 'diar':
        model = Diarizer(model_path, gpus=(0,))
        model.diarize(samples)
    else:
        model = Recognizer(model_path, gpus=(0,))
        model.recognize(samples)
    model.close()
    return True


def _gpu_choices(kind: str, model_path: Path) -> tuple[int, ...]:
    """Metal (device 0), then the CPU, on Apple Silicon when Metal works; the CPU elsewhere."""
    if not _apple_silicon():
        return (-1,)
    device = (os.environ.get('PARAKEET_NEMOTRON_DEVICE') or '').strip().lower()
    if device == 'cpu':
        return (-1,)
    if device in ('gpu', 'metal') or metal_works(kind, model_path):
        return (0, -1)
    return (-1,)


def _floats(samples: array):
    """A ctypes view of an array('f'), without copying it."""
    return (c_float * len(samples)).from_buffer(samples) if len(samples) else (c_float * 1)()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float


@dataclass
class Result:
    final: bool
    text: str
    words: list[Word] = field(default_factory=list)
    audio_processed: float = 0.0
    language: str = ''


def _read_result(lib, handle) -> Result:
    alternatives = lib.nemo_speech_asr_result_alternative_count(handle)
    result = Result(final=bool(lib.nemo_speech_asr_result_is_final(handle)), text='',
                    audio_processed=float(lib.nemo_speech_asr_result_audio_processed(handle)))
    if not alternatives:
        return result
    result.text = (lib.nemo_speech_asr_result_transcript(handle, 0) or b'').decode('utf-8', 'replace').strip()
    for i in range(lib.nemo_speech_asr_result_word_count(handle, 0)):
        text = (lib.nemo_speech_asr_result_word_text(handle, 0, i) or b'').decode('utf-8', 'replace').strip()
        if text:
            result.words.append(Word(text, lib.nemo_speech_asr_result_word_start_time(handle, 0, i) / 1000.0,
                                     lib.nemo_speech_asr_result_word_end_time(handle, 0, i) / 1000.0))
    if lib.nemo_speech_asr_result_language_count(handle, 0):
        result.language = (lib.nemo_speech_asr_result_language_code(handle, 0, 0) or b'').decode('ascii', 'replace')
    return result


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------
class Recognizer:
    """One speech model, loaded.

    `right_context` is the streaming lookahead of a cache-aware model such as
    Nemotron, in 80 ms encoder frames (-1 for the model's default); it sets
    how long live captions lag. `endpointing` closes an utterance after 800 ms
    without a new word, which is what makes a live caption final.
    """

    def __init__(self, model_path: Path, right_context: int = -1, endpointing: bool = False,
                 gpus: tuple[int, ...] | None = None):
        self._lib = load_library()
        self.model_path = Path(model_path)
        self.right_context = right_context
        self._handle = c_void_p()
        model = ModelConfig(ctypes.sizeof(ModelConfig), str(self.model_path).encode('utf-8'), None)
        streaming = StreamingConfig(ctypes.sizeof(StreamingConfig), 0.16, 1.92, 1.92, right_context)
        ending = EndpointingConfig(ctypes.sizeof(EndpointingConfig), endpointing, False, 0)
        started = time.monotonic()
        error = None
        for gpu in gpus if gpus is not None else _gpu_choices('asr', self.model_path):
            backend = BackendConfig(ctypes.sizeof(BackendConfig), gpu)
            config = RecognizerConfig(ctypes.sizeof(RecognizerConfig), ctypes.pointer(backend),
                                      ctypes.pointer(model), ctypes.pointer(streaming), None, None,
                                      ctypes.pointer(ending))
            status = self._lib.nemo_speech_asr_create(byref(config), byref(self._handle))
            if status == OK:
                self.gpu = gpu
                break
            error = _error(self._lib, status, f'Loading {self.model_path.name}')
            log.warning('%s%s', error, ' (trying the CPU)' if gpu >= 0 else '')
        else:
            raise error
        log.info('loaded %s on the %s in %.2fs (right context %d)', self.model_path.name,
                 'GPU' if self.gpu >= 0 else 'CPU', time.monotonic() - started, right_context)

    def _options(self, language: str, interim: bool = False) -> RecognitionOptions:
        options = self._lib.nemo_speech_asr_recognition_options_default()
        options.enable_word_time_offsets = True
        options.enable_automatic_punctuation = True
        options.interim_results = interim
        options.language_code = language.encode('ascii') if language else None
        return options

    def recognize(self, samples: array, language: str = '') -> Result:
        """One blocking call over `samples` (16 kHz mono float32). Best kept to a minute or so:
        time and memory grow with the square of the length."""
        options = self._options(language)
        handle = c_void_p()
        status = self._lib.nemo_speech_asr_recognize_f32(self._handle, byref(options), _floats(samples),
                                                         len(samples), SAMPLE_RATE, byref(handle))
        if status != OK:
            raise _error(self._lib, status, 'Recognition')
        try:
            return _read_result(self._lib, handle)
        finally:
            self._lib.nemo_speech_asr_result_destroy(handle)

    def stream(self, language: str = '', interim: bool = True) -> 'Stream':
        return Stream(self, self._options(language, interim))

    def close(self) -> None:
        if self._handle:
            self._lib.nemo_speech_asr_destroy(self._handle)
            self._handle = c_void_p()


class Stream:
    """Live recognition: push audio as it comes, take results as they are ready."""

    def __init__(self, recognizer: Recognizer, options: RecognitionOptions):
        self._lib = recognizer._lib
        self._options = options  # kept alive for the stream's life
        self._handle = c_void_p()
        status = self._lib.nemo_speech_asr_streaming_recognize(recognizer._handle, byref(options),
                                                               byref(self._handle))
        if status != OK:
            raise _error(self._lib, status, 'Starting a live stream')

    def push(self, samples: array) -> None:
        if not len(samples):
            return
        status = self._lib.nemo_speech_asr_stream_push_f32(self._handle, _floats(samples), len(samples),
                                                           SAMPLE_RATE)
        if status != OK:
            raise _error(self._lib, status, 'Streaming audio')

    def results(self) -> list[Result]:
        """Every result ready now; decoding happens here."""
        out = []
        while True:
            handle = c_void_p()
            status = self._lib.nemo_speech_asr_stream_next(self._handle, byref(handle))
            if status != OK:
                raise _error(self._lib, status, 'Live recognition')
            if not handle:
                return out
            try:
                out.append(_read_result(self._lib, handle))
            finally:
                self._lib.nemo_speech_asr_result_destroy(handle)

    def finish(self) -> list[Result]:
        """No more audio: the results left, the last one final."""
        status = self._lib.nemo_speech_asr_stream_finish(self._handle)
        if status != OK:
            raise _error(self._lib, status, 'Ending the live stream')
        return self.results()

    def close(self) -> None:
        if self._handle:
            self._lib.nemo_speech_asr_stream_close(self._handle)
            self._handle = c_void_p()


# ---------------------------------------------------------------------------
# Diarization: who spoke when
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    start: float
    end: float
    speaker: int  # 1-based


class Diarizer:
    """The speaker model, loaded. One pass over a whole file, through its
    streaming path: the offline one has a length limit, and the stream keeps a
    speaker the same speaker for the whole file."""

    def __init__(self, model_path: Path, gpus: tuple[int, ...] | None = None):
        self._lib = load_library()
        self.model_path = Path(model_path)
        self._handle = c_void_p()
        error = None
        for gpu in gpus if gpus is not None else _gpu_choices('diar', self.model_path):
            config = DiarModelConfig(ctypes.sizeof(DiarModelConfig), str(self.model_path).encode('utf-8'), gpu,
                                     PRESET, 0, 0, -1, 0, 0, 0)
            status = self._lib.nemo_speech_diar_create(byref(config), byref(self._handle))
            if status == OK:
                break
            error = _error(self._lib, status, f'Loading {self.model_path.name}')
            log.warning('%s', error)
        else:
            raise error
        self.num_speakers = self._lib.nemo_speech_diar_num_speakers(self._handle)

    def diarize(self, samples: array, progress=lambda fraction: None, cancelled=lambda: False,
                step_sec: float = 10.0) -> list[Turn]:
        """Speaker turns over `samples`, sorted by start. Returns None if cancelled."""
        stream = c_void_p()
        status = self._lib.nemo_speech_diar_stream_open(self._handle, byref(stream))
        if status != OK:
            raise _error(self._lib, status, 'Starting diarization')
        try:
            step = int(step_sec * SAMPLE_RATE)
            view = memoryview(samples)
            for start in range(0, len(samples), step):
                if cancelled():
                    return None
                chunk = array('f', view[start:start + step])
                status = self._lib.nemo_speech_diar_stream_push_f32(stream, _floats(chunk), len(chunk),
                                                                    SAMPLE_RATE)
                if status != OK:
                    raise _error(self._lib, status, 'Diarization')
                progress(min(1.0, (start + len(chunk)) / max(1, len(samples))))
            status = self._lib.nemo_speech_diar_stream_finish(stream)
            if status != OK:
                raise _error(self._lib, status, 'Diarization')
            count = c_size_t()
            status = self._lib.nemo_speech_diar_segments(stream, None, None, 0, byref(count))
            if status != OK:
                raise _error(self._lib, status, 'Diarization')
            segments = (DiarSegment * max(1, count.value))()
            status = self._lib.nemo_speech_diar_segments(stream, None, segments, count.value, byref(count))
            if status != OK:
                raise _error(self._lib, status, 'Diarization')
            return [Turn(s.start_time, s.end_time, s.speaker) for s in segments[:count.value]]
        finally:
            self._lib.nemo_speech_diar_stream_close(stream)

    def close(self) -> None:
        if self._handle:
            self._lib.nemo_speech_diar_destroy(self._handle)
            self._handle = c_void_p()
