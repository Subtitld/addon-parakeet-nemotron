"""Move Nemotron's word times to where the words are spoken.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Nemotron stamps a word when it emits it, about when the word ends, and every
word "lasts" one 80 ms frame: "And" in JFK's speech is at 0.96-1.04 s, where
Parakeet hears it at 0.24-0.56. Cues built on those times start about 0.7 s
late. So, word by word:

- the end is the emission, less LAG_SEC;
- a word emitted within PAUSE_SEC of the one before starts where that one
  ended (words in a run are back to back);
- a word after a pause starts a guessed duration before its end:
  BASE_SEC plus PER_CHAR_SEC per letter, at most MAX_WORD_SEC. That also
  keeps a word from stretching over a long silence (the spike once saw an
  82-second word).

Fitted on the first six minutes of Tears of Steel against its subtitles
(rounded to 0.5 s): cue starts within 0.5 s went from 24% to 76% (Parakeet:
85%), with a median offset of +0.01 s. Parakeet's times are used as they are.
"""

from __future__ import annotations

import re

LAG_SEC = 0.08
PAUSE_SEC = 0.5
BASE_SEC = 0.35
PER_CHAR_SEC = 0.07
MAX_WORD_SEC = 1.2
_MIN_SEC = 0.02

_LETTERS = re.compile(r"[^\w']")


class Corrector:
    """Corrects words in order; keeps its place, so a stream of results can go through it piece by piece."""

    def __init__(self):
        self._prev_end: float | None = None
        self._prev_emit: float | None = None

    def correct(self, words: list[dict]) -> list[dict]:
        """Words {word, start, end, ...} in time order, with corrected times (in place, and returned)."""
        for word in words:
            emitted = float(word['start'])
            end = max(emitted - LAG_SEC, _MIN_SEC)
            if self._prev_end is not None:
                end = max(end, self._prev_end + _MIN_SEC)
            if self._prev_end is not None and emitted - self._prev_emit < PAUSE_SEC:
                start = self._prev_end
            else:
                letters = len(_LETTERS.sub('', str(word.get('word') or ''))) or 1
                start = end - min(MAX_WORD_SEC, BASE_SEC + PER_CHAR_SEC * letters)
                if self._prev_end is not None:
                    start = max(start, self._prev_end)
            word['start'] = round(max(0.0, min(start, end - _MIN_SEC)), 3)
            word['end'] = round(end, 3)
            self._prev_end, self._prev_emit = end, emitted
        return words
