"""A stand-in for ffmpeg that decodes 16 kHz WAV files only, with the standard library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

tests/test_installed.py runs in clean containers of distributions that have no
ffmpeg package in their own repositories (AlmaLinux 8). The add-on asks ffmpeg
for 16 kHz mono float32 on stdout; this answers the same call for a 16 kHz
16-bit WAV, mono or stereo, and fails like ffmpeg for anything else.

    python wav_ffmpeg.py -nostdin ... -i INPUT ... -f f32le pipe:1
"""

import struct
import sys
import wave


def main(argv: list[str]) -> int:
    source = argv[argv.index('-i') + 1]
    try:
        with wave.open(source, 'rb') as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (OSError, wave.Error, EOFError) as exc:
        print(f'{source}: {exc}', file=sys.stderr)
        return 1
    if width != 2 or rate != 16000 or channels not in (1, 2):
        print(f'{source}: only 16 kHz 16-bit WAV, not {rate} Hz {width * 8}-bit', file=sys.stderr)
        return 1
    ints = struct.unpack(f'<{len(frames) // 2}h', frames)
    mono = ints if channels == 1 else [(a + b) / 2 for a, b in zip(ints[::2], ints[1::2])]
    sys.stdout.buffer.write(struct.pack(f'<{len(mono)}f', *(s / 32768.0 for s in mono)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
