"""Install a release zip with Subtitld's own installer and transcribe through its AddonManager.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The path a user's click takes, short of the window: Subtitld's installer
unpacks the archive, its AddonManager discovers the add-on and spawns it, and
its ASR provider sends a file transcription (with speaker labels) and a live
session, as the Import panel and Record mode do. Run it with Subtitld's own
interpreter, in scratch folders so the real installation is not touched:

    XDG_DATA_HOME=$T/data XDG_CONFIG_HOME=$T/config XDG_CACHE_HOME=$T/cache \\
    SUBTITLD_MODELS_DIR=~/.cache/subtitld/models \\
    ~/Projetos/subtitld/.venv/bin/python tools/host_e2e.py ZIP [AUDIO] [--subtitld ~/Projetos/subtitld]

AUDIO defaults to tests/data/jfk.wav. Prints what the host received.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('zip', type=Path)
    parser.add_argument('audio', type=Path, nargs='?', default=ROOT / 'tests' / 'data' / 'jfk.wav')
    parser.add_argument('--language', default='en')
    parser.add_argument('--subtitld', type=Path, default=Path.home() / 'Projetos' / 'subtitld')
    args = parser.parse_args()

    sys.path.insert(0, str(args.subtitld / 'src'))
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication([])
    from subtitld.modules.addons import installer
    from subtitld.modules.addons.manager import get_manager

    addon_id = installer.install_from_zip(args.zip)
    print('installed', addon_id)
    manager = get_manager()
    manager.discover()
    provider = manager.get(addon_id)
    assert provider is not None, 'not discovered'
    print('provider', type(provider).__name__, 'streams' if provider.supports_streaming() else 'no streaming')

    def run_until(done, timeout):
        deadline = time.monotonic() + timeout
        while not done() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert done(), 'timed out'

    # A file, as the Import panel sends it.
    outcome: dict = {'partials': 0}
    provider.partial.connect(lambda seg: outcome.__setitem__('partials', outcome['partials'] + 1))
    provider.transcript_finished.connect(lambda segments: outcome.setdefault('segments', segments))
    provider.error.connect(lambda message: outcome.setdefault('error', message))
    started = time.monotonic()
    provider.transcribe(str(args.audio), args.language, {'label_speakers': True})
    run_until(lambda: 'segments' in outcome or 'error' in outcome, 1800)
    if 'error' in outcome:
        print('transcribe: error', outcome['error'])
        return 1
    print(f'transcribe: {len(outcome["segments"])} cues ({outcome["partials"]} as partials) '
          f'in {time.monotonic() - started:.1f}s')
    for segment in outcome['segments'][:8]:
        print(f'  {segment["start"]:7.2f} {segment["end"]:7.2f} {segment["speaker"]}  {segment["text"]}')

    # Live captions, as Record mode feeds them: 100 ms of int16 at a time.
    live: dict = {'interims': 0, 'finals': []}

    def on_segment(segment, final):
        if final:
            live['finals'].append(segment)
        else:
            live['interims'] += 1

    provider.stream_segment.connect(on_segment)
    provider.stream_finished.connect(lambda segments: live.setdefault('segments', segments))
    provider.stream_error.connect(lambda message: live.setdefault('error', message))
    with wave.open(str(ROOT / 'tests' / 'data' / 'jfk.wav'), 'rb') as wav:
        pcm = wav.readframes(wav.getnframes())
    sid = provider.stream_start(args.language, {})
    assert sid, live.get('error')
    for position in range(0, len(pcm), 3200):
        provider.stream_feed(pcm[position:position + 3200])
        app.processEvents()
    provider.stream_stop()
    run_until(lambda: 'segments' in live or 'error' in live, 600)
    if 'error' in live:
        print('stream: error', live['error'])
        return 1
    print(f'stream: {len(live["segments"])} finals, {live["interims"]} interims')
    for segment in live['segments']:
        print(f'  {segment["start"]:7.2f} {segment["end"]:7.2f}  {segment["text"]}')
    manager.shutdown_all()
    return 0


if __name__ == '__main__':
    sys.exit(main())
