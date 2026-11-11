"""Transcribe a file with the libraries in build/assets/, through ctypes.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

M1's check that the native libraries load and recognize speech. The add-on's
real binding (engine.py) comes later; this one stays small on purpose.

    python tools/smoke_native.py MODEL.gguf AUDIO [--language en-US]
        [--stream [--right-context N]] [--gpu N] [--assets DIR]

AUDIO is anything ffmpeg reads, or raw 16 kHz mono float32 samples in a
.f32 file (for containers without ffmpeg). Prints the transcript, every word with its
start and end in seconds, the time taken and the peak memory, and exits 1 if
the transcript comes back empty.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from array import array
from ctypes import (POINTER, Structure, byref, c_bool, c_char_p, c_float, c_int32,
                    c_size_t, c_void_p)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

C_ABI = {
    'linux': 'libnemo_speech_asr_c.so.1',
    'win32': 'nemo_speech_asr_c.dll',
    'darwin': 'libnemo_speech_asr_c.1.dylib',
}


# Mirrors of include/nemo_speech/asr.h (v0.3.0). Every struct starts with its
# own size; the library defaults whatever a shorter struct leaves out.
class BackendConfig(Structure):
    _fields_ = [('size', c_size_t), ('gpu', c_int32)]


class ModelConfig(Structure):
    _fields_ = [('size', c_size_t), ('path', c_char_p), ('name', c_char_p)]


class StreamingConfig(Structure):
    _fields_ = [('size', c_size_t), ('chunk_size', c_float), ('ctc_left_padding', c_float),
                ('ctc_right_padding', c_float), ('rnnt_right_context', c_int32)]


class RecognizerConfig(Structure):
    _fields_ = [('size', c_size_t), ('backend', POINTER(BackendConfig)),
                ('model', POINTER(ModelConfig)), ('streaming', POINTER(StreamingConfig)),
                ('decoder', c_void_p), ('vad', c_void_p), ('endpointing', c_void_p),
                ('postproc', c_void_p), ('diar', c_void_p), ('batching', c_void_p),
                ('salm', c_void_p)]


class RecognitionOptions(Structure):
    _fields_ = [('size', c_size_t), ('request_id', c_char_p), ('language_code', c_char_p),
                ('interim_results', c_bool), ('enable_word_time_offsets', c_bool),
                ('enable_automatic_punctuation', c_bool), ('verbatim_transcripts', c_bool),
                ('profanity_filter', c_bool), ('stop_history_eou_ms', c_int32),
                ('speech_contexts', c_void_p), ('speech_context_count', c_size_t),
                ('max_alternatives', c_int32), ('enable_speaker_diarization', c_bool),
                ('max_speaker_count', c_int32)]


def load(assets):
    if sys.platform == 'win32':
        os.add_dll_directory(str(assets))
    lib = ctypes.CDLL(str(assets / C_ABI[sys.platform]))
    p = c_void_p
    for name, args, res in [
        ('nemo_speech_asr_version', [], c_char_p),
        ('nemo_speech_asr_last_error', [], c_char_p),
        ('nemo_speech_asr_recognition_options_default', [], RecognitionOptions),
        ('nemo_speech_asr_create', [POINTER(RecognizerConfig), POINTER(p)], c_int32),
        ('nemo_speech_asr_destroy', [p], None),
        ('nemo_speech_asr_recognize_f32',
         [p, POINTER(RecognitionOptions), POINTER(c_float), c_size_t, c_int32, POINTER(p)], c_int32),
        ('nemo_speech_asr_streaming_recognize', [p, POINTER(RecognitionOptions), POINTER(p)], c_int32),
        ('nemo_speech_asr_stream_push_f32', [p, POINTER(c_float), c_size_t, c_int32], c_int32),
        ('nemo_speech_asr_stream_finish', [p], c_int32),
        ('nemo_speech_asr_stream_next', [p, POINTER(p)], c_int32),
        ('nemo_speech_asr_stream_close', [p], None),
        ('nemo_speech_asr_result_is_final', [p], c_bool),
        ('nemo_speech_asr_result_alternative_count', [p], c_size_t),
        ('nemo_speech_asr_result_transcript', [p, c_size_t], c_char_p),
        ('nemo_speech_asr_result_word_count', [p, c_size_t], c_size_t),
        ('nemo_speech_asr_result_word_text', [p, c_size_t, c_size_t], c_char_p),
        ('nemo_speech_asr_result_word_start_time', [p, c_size_t, c_size_t], c_int32),
        ('nemo_speech_asr_result_word_end_time', [p, c_size_t, c_size_t], c_int32),
        ('nemo_speech_asr_result_destroy', [p], None),
    ]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib


def check(lib, status, what):
    if status != 0:
        error = lib.nemo_speech_asr_last_error() or b''
        sys.exit(f'{what} failed ({status}): {error.decode(errors="replace")}')


def decode(path):
    if path.suffix == '.f32':
        raw = path.read_bytes()
    else:
        raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vn', '-ac', '1',
                              '-ar', str(SAMPLE_RATE), '-f', 'f32le', 'pipe:1'],
                             check=True, capture_output=True).stdout
    samples = array('f')
    samples.frombytes(raw)
    if sys.byteorder == 'big':
        samples.byteswap()
    return samples


def words_of(lib, result):
    if not result or lib.nemo_speech_asr_result_alternative_count(result) == 0:
        return '', []
    text = (lib.nemo_speech_asr_result_transcript(result, 0) or b'').decode()
    words = [((lib.nemo_speech_asr_result_word_text(result, 0, i) or b'').decode(),
              lib.nemo_speech_asr_result_word_start_time(result, 0, i) / 1000,
              lib.nemo_speech_asr_result_word_end_time(result, 0, i) / 1000)
             for i in range(lib.nemo_speech_asr_result_word_count(result, 0))]
    return text, words


def peak_rss_mib():
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == 'darwin' else peak / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('model', type=Path)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--language', default='')
    parser.add_argument('--stream', action='store_true', help='push 0.5 s chunks through a stream')
    parser.add_argument('--right-context', type=int, default=-1,
                        help='cache-aware RNNT lookahead in 80 ms frames (-1 = model default)')
    parser.add_argument('--gpu', type=int, default=-1, help='-1 = CPU')
    parser.add_argument('--assets', type=Path, default=ROOT / 'build' / 'assets')
    args = parser.parse_args()

    lib = load(args.assets)
    print('runtime', lib.nemo_speech_asr_version().decode())
    samples = decode(args.audio)
    seconds = len(samples) / SAMPLE_RATE

    backend = BackendConfig(ctypes.sizeof(BackendConfig), args.gpu)
    model = ModelConfig(ctypes.sizeof(ModelConfig), str(args.model).encode(), None)
    streaming = StreamingConfig(ctypes.sizeof(StreamingConfig), 0.16, 1.92, 1.92, args.right_context)
    config = RecognizerConfig(ctypes.sizeof(RecognizerConfig), ctypes.pointer(backend),
                              ctypes.pointer(model), ctypes.pointer(streaming))
    recognizer = c_void_p()
    started = time.monotonic()
    check(lib, lib.nemo_speech_asr_create(byref(config), byref(recognizer)), 'create')
    loaded = time.monotonic()

    options = lib.nemo_speech_asr_recognition_options_default()
    options.enable_word_time_offsets = True
    options.enable_automatic_punctuation = True
    options.language_code = args.language.encode() or None

    texts, words = [], []
    buffer = (c_float * len(samples)).from_buffer(samples)
    if not args.stream:
        result = c_void_p()
        check(lib, lib.nemo_speech_asr_recognize_f32(recognizer, byref(options), buffer,
                                                     len(samples), SAMPLE_RATE, byref(result)),
              'recognize')
        text, words = words_of(lib, result)
        texts.append(text)
        lib.nemo_speech_asr_result_destroy(result)
    else:
        stream = c_void_p()
        check(lib, lib.nemo_speech_asr_streaming_recognize(recognizer, byref(options), byref(stream)),
              'streaming_recognize')

        def drain():
            result = c_void_p()
            while lib.nemo_speech_asr_stream_next(stream, byref(result)) == 0 and result:
                if lib.nemo_speech_asr_result_is_final(result):
                    text, final_words = words_of(lib, result)
                    texts.append(text)
                    words.extend(final_words)
                lib.nemo_speech_asr_result_destroy(result)
                result = c_void_p()

        step = SAMPLE_RATE // 2
        for start in range(0, len(samples), step):
            n = min(step, len(samples) - start)
            chunk = (c_float * n).from_buffer(samples, start * samples.itemsize)
            check(lib, lib.nemo_speech_asr_stream_push_f32(stream, chunk, n, SAMPLE_RATE), 'push')
            drain()
        check(lib, lib.nemo_speech_asr_stream_finish(stream), 'finish')
        drain()
        lib.nemo_speech_asr_stream_close(stream)
    done = time.monotonic()
    lib.nemo_speech_asr_destroy(recognizer)

    transcript = ' '.join(t for t in texts if t)
    print('transcript:', transcript)
    for word, start, end in words:
        print(f'  {start:7.2f} {end:7.2f}  {word}')
    peak = peak_rss_mib()
    print(f'{seconds:.1f} s of audio: load {loaded - started:.2f} s, '
          f'recognize {done - loaded:.2f} s ({seconds / max(done - loaded, 1e-9):.1f}x real time)'
          + (f', peak {peak:.0f} MiB' if peak else ''))
    sys.exit(0 if transcript.strip() else 1)


if __name__ == '__main__':
    main()
