"""Turn word timings into subtitle rows.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Bark 3's rules (the whisper.cpp add-on's phrase regrouping, applied to words),
with speakers. A row closes at sentence-final punctuation, and always when the
speaker changes. By default that is all, as AssemblyAI's "one cue per
sentence"; with `by_sentence=False` a row also closes early when it would pass
the readability caps.
"""

from __future__ import annotations

from collections.abc import Iterable

# Same sets as the whisper.cpp add-on. Its comments explain the choices.
PHRASE_TERMINATORS = frozenset('.?!。？！．।॥؟')
TRAILING_CLOSERS = frozenset('"\')]}»›”’') | {chr(0x300D), chr(0x300F)}  # and 」 』

# Captioning readability caps: about two lines of 42 characters, or 7 seconds.
MAX_CHARS = 84
MAX_DURATION_SEC = 7.0

# One row per sentence still closes a sentence that never ends, as speech with
# no punctuation would otherwise become a single row. The AssemblyAI add-on's
# caps: far above any real sentence.
SENTENCE_MAX_CHARS = 500
SENTENCE_MAX_DURATION_SEC = 30.0

# Nemotron often writes no punctuation at all over film sound, so its rows
# also close at a pause this long between two words.
PAUSE_SEC = 0.8

# Scripts written without spaces between words: their words are joined as they are.
# (kana, CJK ideographs and their extensions, full-width forms, Thai).
_UNSPACED = tuple((chr(low), chr(high)) for low, high in (
    (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF), (0x0E00, 0x0E7F)))


def ends_with_terminator(text: str) -> bool:
    """True if `text` ends with sentence-final punctuation.

    Closing quotes and brackets may follow it, so `She said "hello."` still counts.
    """
    t = text.rstrip()
    for _ in range(4):
        if not t:
            return False
        if t[-1] in TRAILING_CLOSERS:
            t = t[:-1].rstrip()
            continue
        break
    return bool(t) and t[-1] in PHRASE_TERMINATORS


def _unspaced(char: str) -> bool:
    return any(low <= char <= high for low, high in _UNSPACED)


def join(text: str, token: str) -> str:
    """`text` and the next word: with a space, unless both sides are a script without spaces."""
    if text and token and _unspaced(text[-1]) and _unspaced(token[0]):
        return text + token
    return f'{text} {token}' if text else token


def words_to_segments(words: Iterable[dict], duration_sec: float, text: str = '',
                      by_sentence: bool = True, pause_sec: float | None = None) -> list[dict]:
    """Group word timings into rows of {start, end, text, speaker}.

    Each word is {word, start, end} with an optional `speaker`. `by_sentence`
    gives one row per sentence; False also splits a long sentence at the
    readability caps. `pause_sec`, if given, also closes a row at a silence
    that long. With no word timings but some text, the whole text becomes one
    row over the full duration. Rows keep the order of the words.
    """
    max_chars, max_sec = ((SENTENCE_MAX_CHARS, SENTENCE_MAX_DURATION_SEC) if by_sentence
                          else (MAX_CHARS, MAX_DURATION_SEC))
    rows: list[dict] = []
    current: dict | None = None
    for word in words:
        token = str(word.get('word') or '').strip()
        if not token:
            continue
        start = float(word.get('start') or 0.0)
        end = float(word.get('end') or start)
        speaker = str(word.get('speaker') or 'A')
        if current is not None and (
                speaker != current['speaker']
                or ends_with_terminator(current['text'])
                or (pause_sec is not None and start - current['end'] >= pause_sec)
                or len(current['text']) + 1 + len(token) > max_chars
                or end - current['start'] > max_sec):
            rows.append(current)
            current = None
        if current is None:
            current = {'start': round(start, 3), 'end': round(end, 3), 'text': token, 'speaker': speaker}
        else:
            current['text'] = join(current['text'], token)
            current['end'] = round(max(end, current['end']), 3)
    if current is not None:
        rows.append(current)
    if not rows and text.strip():
        rows.append({'start': 0.0, 'end': round(duration_sec, 3), 'text': text.strip(), 'speaker': 'A'})
    return rows
