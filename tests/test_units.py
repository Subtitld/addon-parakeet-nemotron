"""The pure-Python parts: options, model choice, cues, timing and speakers.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import sys
from array import array
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parakeet_nemotron_addon import models, speakers  # noqa: E402
from parakeet_nemotron_addon.__main__ import _options, _pcm_to_floats  # noqa: E402
from parakeet_nemotron_addon.engine import Turn  # noqa: E402
from parakeet_nemotron_addon.phrases import join, words_to_segments  # noqa: E402
from parakeet_nemotron_addon.timing import Corrector  # noqa: E402


def _words(*items):
    return [{'word': w, 'start': s, 'end': e} for w, s, e in items]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def test_options_come_from_the_request_then_the_environment(monkeypatch):
    for name in ('MODEL', 'SLICE_BY_PHRASES', 'LABEL_SPEAKERS', 'LIVE_DELAY'):
        monkeypatch.delenv(f'PARAKEET_NEMOTRON_{name}', raising=False)
    assert _options(None) == {'model': 'auto', 'slice_by_phrases': True, 'label_speakers': False,
                              'live_delay': '560'}
    monkeypatch.setenv('PARAKEET_NEMOTRON_MODEL', 'nemotron')
    monkeypatch.setenv('PARAKEET_NEMOTRON_LABEL_SPEAKERS', 'true')
    monkeypatch.setenv('PARAKEET_NEMOTRON_SLICE_BY_PHRASES', '0')
    monkeypatch.setenv('PARAKEET_NEMOTRON_LIVE_DELAY', '1120')
    assert _options({}) == {'model': 'nemotron', 'slice_by_phrases': False, 'label_speakers': True,
                            'live_delay': '1120'}
    # The request wins, an explicit False included; nonsense falls back to the default.
    assert _options({'model': 'Parakeet', 'label_speakers': False, 'slice_by_phrases': True,
                     'live_delay': 320}) == {'model': 'parakeet', 'slice_by_phrases': True,
                                             'label_speakers': False, 'live_delay': '320'}
    assert _options({'model': 'whisper', 'live_delay': '42'})['model'] == 'auto'
    assert _options({'live_delay': '42'})['live_delay'] == '560'


def test_pcm_becomes_floats():
    pcm = array('h', [0, 16384, -32768, 32767])
    if sys.byteorder != 'little':
        pcm.byteswap()
    assert list(_pcm_to_floats(pcm.tobytes() + b'\x01')) == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


# ---------------------------------------------------------------------------
# Which model, which language
# ---------------------------------------------------------------------------
def test_the_language_lists():
    assert len(models.PARAKEET_LANGUAGES) == 25 and len(models.NEMOTRON_LANGUAGES) == 28
    assert len(models.LANGUAGES) == 33
    assert set(models.LANGUAGES) - set(models.PARAKEET_LANGUAGES) == {'ar', 'hi', 'ja', 'ko', 'nb', 'tr', 'vi', 'zh'}
    assert set(models.LANGUAGE_NAMES) == set(models.LANGUAGES)
    for language, locales in models.NEMOTRON_LOCALES.items():
        assert all(locale.split('-')[0] == language for locale in locales)


@pytest.mark.parametrize('tag, choice, expected', [
    ('pt-br', 'auto', ('parakeet', '')),
    ('en', 'auto', ('parakeet', '')),
    ('', 'auto', ('parakeet', '')),
    ('ja', 'auto', ('nemotron', 'ja-JP')),
    ('no', 'auto', ('nemotron', 'nb-NO')),
    ('zh-tw', 'auto', ('nemotron', 'zh-CN')),
    ('pt-br', 'nemotron', ('nemotron', 'pt-BR')),
    ('pt', 'nemotron', ('nemotron', 'pt-PT')),
    ('en-gb', 'nemotron', ('nemotron', 'en-GB')),
    ('es_419', 'nemotron', ('nemotron', 'es-US')),
    ('', 'nemotron', ('nemotron', 'auto')),
    ('de', 'parakeet', ('parakeet', '')),
])
def test_choose_for_file(tag, choice, expected):
    assert models.choose_for_file(tag, choice) == expected


@pytest.mark.parametrize('tag, choice', [('th', 'auto'), ('ja', 'parakeet'), ('el', 'nemotron')])
def test_choose_for_file_refuses(tag, choice):
    with pytest.raises(models.UnsupportedLanguage):
        models.choose_for_file(tag, choice)


def test_live_captions_are_nemotron_only():
    assert models.locale_for_live('fr-ca') == 'fr-CA'
    assert models.locale_for_live('') == 'auto'
    with pytest.raises(models.UnsupportedLanguage, match='Import'):
        models.locale_for_live('lv')


# ---------------------------------------------------------------------------
# Cues
# ---------------------------------------------------------------------------
def test_one_cue_per_sentence_and_the_caps():
    words = _words(('Hello', 0.0, 0.4), ('there.', 0.5, 0.9), ('How', 1.0, 1.2), ('are', 1.2, 1.4),
                   ('you?', 1.4, 1.8))
    rows = words_to_segments(words, 2.0)
    assert [r['text'] for r in rows] == ['Hello there.', 'How are you?']
    assert (rows[0]['start'], rows[0]['end'], rows[0]['speaker']) == (0.0, 0.9, 'A')
    long = _words(*[(f'w{i}', i * 0.5, i * 0.5 + 0.4) for i in range(30)])
    assert len(words_to_segments(long, 15.0)) == 1
    assert len(words_to_segments(long, 15.0, by_sentence=False)) == 3  # 7 s each


def test_a_speaker_change_or_a_pause_closes_a_cue():
    words = _words(('so', 0.0, 0.3), ('we', 0.3, 0.5), ('go', 0.5, 0.8), ('now', 2.0, 2.3))
    for word, speaker in zip(words, 'AABB'):
        word['speaker'] = speaker
    assert [(r['text'], r['speaker']) for r in words_to_segments(words, 3.0)] == [('so we', 'A'), ('go now', 'B')]
    plain = _words(('so', 0.0, 0.3), ('we', 0.3, 0.5), ('go', 1.5, 1.8))
    assert len(words_to_segments(plain, 2.0)) == 1
    assert [r['text'] for r in words_to_segments(plain, 2.0, pause_sec=0.8)] == ['so we', 'go']


def test_scripts_without_spaces_are_joined_as_they_are():
    assert join('東京', '大学') == '東京大学'
    assert join('東京', 'Tower') == '東京 Tower'
    assert words_to_segments(_words(('今日は', 0, 1), ('晴れ。', 1, 2)), 2.0)[0]['text'] == '今日は晴れ。'


def test_text_without_words_is_one_cue():
    assert words_to_segments([], 3.5, text=' Hi. ') == [{'start': 0.0, 'end': 3.5, 'text': 'Hi.', 'speaker': 'A'}]


# ---------------------------------------------------------------------------
# Nemotron's timing
# ---------------------------------------------------------------------------
def test_nemotron_times_move_back_onto_the_words():
    # JFK, as Nemotron stamps it: each word at its emission, 80 ms long.
    words = Corrector().correct(_words(('And', 0.96, 1.04), ('so', 1.12, 1.20), ('my', 1.6, 1.68),
                                       ('fellow', 1.76, 1.84), ('ask', 3.92, 4.0)))
    assert words[0]['start'] == pytest.approx(0.88 - 0.35 - 3 * 0.07)  # after silence: a guessed duration
    assert words[1]['start'] == words[0]['end']                         # in a run: back to back
    assert all(w['end'] > w['start'] for w in words)
    assert words[-1]['start'] >= words[-2]['end']
    assert words[-1]['end'] - words[-1]['start'] <= 1.2


def test_a_word_never_spans_a_long_silence():
    # The spike once saw an 82-second word.
    [word] = Corrector().correct(_words(('timety', 505.0, 505.08)))
    assert word['end'] - word['start'] <= 1.2


def test_the_corrector_carries_over_between_windows():
    corrector = Corrector()
    first = corrector.correct(_words(('one', 10.0, 10.08)))
    second = corrector.correct(_words(('two', 10.2, 10.28)))
    assert second[0]['start'] == first[0]['end']


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------
def test_words_take_the_speaker_they_overlap_most():
    turns = [Turn(0.0, 2.0, 3), Turn(2.0, 4.0, 1), Turn(6.0, 7.0, 3)]
    words = speakers.assign(_words(('a', 0.5, 1.0), ('b', 1.8, 2.6), ('c', 4.2, 4.5), ('d', 5.6, 5.9),
                                   ('e', 9.0, 9.5)), turns)
    # Labelled in order of appearance, not by the model's numbers.
    assert [w['speaker'] for w in words] == ['A', 'B', 'B', 'A', 'A']


def test_a_speaker_keeps_its_letter_across_windows():
    turns = [Turn(0.0, 1.0, 2), Turn(1.0, 2.0, 5)]
    names: dict = {}
    first = speakers.assign(_words(('x', 1.2, 1.5)), turns, names)
    second = speakers.assign(_words(('y', 0.2, 0.5), ('z', 1.6, 1.9)), turns, names)
    assert first[0]['speaker'] == 'A' and [w['speaker'] for w in second] == ['B', 'A']


def test_no_turns_is_one_speaker():
    assert speakers.assign(_words(('x', 0, 1)), [])[0]['speaker'] == 'A'
    assert [speakers.label(i) for i in (0, 25, 26)] == ['A', 'Z', 'AA']


# ---------------------------------------------------------------------------
# Metal, tried once in a child process
# ---------------------------------------------------------------------------
def test_metal_is_used_only_where_a_child_survives_it(monkeypatch, tmp_path):
    from parakeet_nemotron_addon import engine
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    monkeypatch.delenv('PARAKEET_NEMOTRON_DEVICE', raising=False)
    monkeypatch.setattr(engine, '_apple_silicon', lambda: True)
    monkeypatch.setattr(engine, 'runtime_version', lambda: 'nemo-speech-asr 0.3.0')
    monkeypatch.setattr(engine, '_verdicts', {})
    crash = [sys.executable, '-c', 'import os, signal; os.kill(os.getpid(), signal.SIGSEGV)']
    if sys.platform == 'win32':
        crash = [sys.executable, '-c', 'import sys; sys.exit(3221225477)']
    answer = [sys.executable, '-c', 'print("metal works")']
    commands = {'asr': crash, 'diar': answer}
    calls = []

    def command(kind, _path):
        calls.append(kind)
        return commands[kind]

    monkeypatch.setattr(engine, '_probe_command', command)
    assert engine._gpu_choices('asr', tmp_path / 'm.gguf') == (-1,)
    assert engine._gpu_choices('diar', tmp_path / 'd.gguf') == (0, -1)
    assert engine._gpu_choices('asr', tmp_path / 'm.gguf') == (-1,)
    assert calls == ['asr', 'diar']
    # Kept on disk for the next process, per runtime and macOS version.
    monkeypatch.setattr(engine, '_verdicts', {})
    assert engine._gpu_choices('asr', tmp_path / 'm.gguf') == (-1,) and calls == ['asr', 'diar']
    monkeypatch.setattr(engine, 'runtime_version', lambda: 'nemo-speech-asr 0.3.1')
    monkeypatch.setattr(engine, '_verdicts', {})
    engine._gpu_choices('asr', tmp_path / 'm.gguf')
    assert calls == ['asr', 'diar', 'asr']
    # The environment can decide instead.
    monkeypatch.setenv('PARAKEET_NEMOTRON_DEVICE', 'cpu')
    assert engine._gpu_choices('diar', tmp_path / 'd.gguf') == (-1,)
    monkeypatch.setattr(engine, '_apple_silicon', lambda: False)
    monkeypatch.setenv('PARAKEET_NEMOTRON_DEVICE', 'gpu')
    assert engine._gpu_choices('asr', tmp_path / 'm.gguf') == (-1,)
