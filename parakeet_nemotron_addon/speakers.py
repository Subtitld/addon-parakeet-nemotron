"""Give each word the speaker the diarization model heard over it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The speaker model numbers speakers 1, 2, 3... in the order it first hears
them. Subtitld's speakers are names; they are labelled A, B, C... here, as
the AssemblyAI add-on does, in order of first appearance in the transcript.
"""

from __future__ import annotations

import string

# A word this far from every turn keeps the speaker of the word before it.
NEAREST_SEC = 1.0


def label(index: int) -> str:
    """A, B, ... Z, AA, AB, ..."""
    letters = string.ascii_uppercase
    out = ''
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        out = letters[rest] + out
    return out


def assign(words: list[dict], turns: list, names: dict | None = None) -> list[dict]:
    """Set `speaker` on every word, from `turns` (start, end, speaker).

    `names` maps the model's speaker numbers to labels; pass the same dict for
    every window of a file, so a speaker keeps its letter. Its `None` key holds
    the speaker of the last word labelled.
    """
    names = {} if names is None else names
    if not turns:
        for word in words:
            word['speaker'] = 'A'
        return words
    turns = sorted(turns, key=lambda t: t.start)
    previous = names.get(None)
    for word in words:
        start, end = float(word['start']), float(word['end'])
        best, best_overlap, best_gap = None, 0.0, NEAREST_SEC
        for turn in turns:
            if turn.start > end + NEAREST_SEC:
                break
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > best_overlap:
                best, best_overlap = turn.speaker, overlap
            elif best_overlap <= 0.0:
                gap = max(turn.start - end, start - turn.end)
                if gap < best_gap:
                    best, best_gap = turn.speaker, gap
        speaker = best if best is not None else previous
        if speaker is None:
            speaker = turns[0].speaker
        previous = speaker
        if speaker not in names:
            names[speaker] = label(sum(1 for key in names if key is not None))
        word['speaker'] = names[speaker]
    names[None] = previous
    return words
