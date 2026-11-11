"""The two speech models, the speaker model, and which languages each covers.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

- Parakeet TDT 0.6B v3 (CC-BY-4.0): 25 European languages. The better one for
  files: fewer errors, numbers as digits, and word times that span the word.
  It detects the language by itself and has no live mode.
- Nemotron 3.5 ASR Streaming 0.6B (OpenMDW 1.1): its 32 "transcription-ready"
  and "broad-coverage" locales (28 languages), told the language by a prompt.
  It streams, so it serves live captions, and it covers the languages
  Parakeet lacks. The model card's 8 "adaptation-ready" locales need
  fine-tuning first and are left out.
- Nemotron-3-Diarization (OpenMDW 1.1): who spoke when, up to 8 speakers.

Each downloads on first use (download.py).
"""

from __future__ import annotations

import json
from pathlib import Path

PARAKEET = 'parakeet'
NEMOTRON = 'nemotron'
DIARIZATION = 'diarization'
AUTO = 'auto'
CHOICES = (AUTO, PARAKEET, NEMOTRON)

PARAKEET_LANGUAGES = ('bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'hr', 'hu', 'it',
                      'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'uk')

# Nemotron's locales, per language, the language's default first. A bare
# language tag gets the default the model itself picks for it, where it has one.
NEMOTRON_LOCALES = {
    'ar': ('ar-AR',), 'bg': ('bg-BG',), 'cs': ('cs-CZ',), 'da': ('da-DK',), 'de': ('de-DE',),
    'en': ('en-US', 'en-GB'), 'es': ('es-US', 'es-ES'), 'et': ('et-EE',), 'fi': ('fi-FI',),
    'fr': ('fr-FR', 'fr-CA'), 'hi': ('hi-IN',), 'hr': ('hr-HR',), 'hu': ('hu-HU',), 'it': ('it-IT',),
    'ja': ('ja-JP',), 'ko': ('ko-KR',), 'nb': ('nb-NO',), 'nl': ('nl-NL',), 'pl': ('pl-PL',),
    'pt': ('pt-PT', 'pt-BR'), 'ro': ('ro-RO',), 'ru': ('ru-RU',), 'sk': ('sk-SK',), 'sv': ('sv-SE',),
    'tr': ('tr-TR',), 'uk': ('uk-UA',), 'vi': ('vi-VN',), 'zh': ('zh-CN',),
}
NEMOTRON_LANGUAGES = tuple(sorted(NEMOTRON_LOCALES))

# Tags that name a covered language another way.
_ALIASES = {'no': 'nb', 'iw': 'he', 'zh-hans': 'zh', 'cmn': 'zh'}

LANGUAGES = tuple(sorted(set(PARAKEET_LANGUAGES) | set(NEMOTRON_LANGUAGES)))

LANGUAGE_NAMES = {
    'ar': 'Arabic', 'bg': 'Bulgarian', 'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French', 'hi': 'Hindi',
    'hr': 'Croatian', 'hu': 'Hungarian', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean',
    'lt': 'Lithuanian', 'lv': 'Latvian', 'mt': 'Maltese', 'nb': 'Norwegian', 'nl': 'Dutch',
    'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak',
    'sl': 'Slovenian', 'sv': 'Swedish', 'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese',
    'zh': 'Chinese',
}

DISPLAY = {PARAKEET: 'Parakeet', NEMOTRON: 'Nemotron', DIARIZATION: 'the speaker model'}

_PACKAGE = Path(__file__).resolve().parent


class UnsupportedLanguage(Exception):
    """Neither model (or not the one asked for) covers the language."""


def pins() -> dict:
    return json.loads((_PACKAGE / 'pins.json').read_text(encoding='utf-8'))


def pin(model: str) -> dict:
    return pins()[model]


def split_tag(raw) -> tuple[str, str]:
    """(primary language, region) of a host tag: `pt-br` -> ('pt', 'BR'), '' -> ('', '')."""
    tag = str(raw or '').strip().replace('_', '-').lower()
    if tag in ('auto', 'und', 'mul'):
        return '', ''
    tag = _ALIASES.get(tag, tag)
    parts = [p for p in tag.split('-') if p]
    if not parts:
        return '', ''
    primary = _ALIASES.get(parts[0], parts[0])
    region = parts[1].upper() if len(parts) > 1 and len(parts[1]) == 2 else ''
    return primary, region


def nemotron_locale(primary: str, region: str) -> str:
    """The locale Nemotron is prompted with: `auto` when no language is given."""
    if not primary:
        return 'auto'
    locales = NEMOTRON_LOCALES[primary]
    wanted = f'{primary}-{region}'
    return wanted if wanted in locales else locales[0]


def _names(codes) -> str:
    return ', '.join(LANGUAGE_NAMES[c] for c in sorted(codes, key=lambda c: LANGUAGE_NAMES[c]))


def choose_for_file(raw_language, choice: str) -> tuple[str, str]:
    """(model, language for the runtime) for transcribing a file.

    `auto` takes Parakeet where it covers the language, Nemotron elsewhere.
    With no language Parakeet detects it among its own 25 (or Nemotron among
    its 28, when chosen). Raises UnsupportedLanguage with a message for the user.
    """
    primary, region = split_tag(raw_language)
    if choice not in CHOICES:
        choice = AUTO
    if choice == AUTO:
        if not primary or primary in PARAKEET_LANGUAGES:
            choice = PARAKEET
        elif primary in NEMOTRON_LOCALES:
            choice = NEMOTRON
        else:
            raise UnsupportedLanguage(
                f'Parakeet & Nemotron do not transcribe {raw_language!r}. They transcribe '
                f'{_names(LANGUAGES)}; pick another engine for other languages.')
    if choice == PARAKEET:
        if primary and primary not in PARAKEET_LANGUAGES:
            others = ' Set the model to Automatic or Nemotron for it.' if primary in NEMOTRON_LOCALES else ''
            raise UnsupportedLanguage(
                f'Parakeet does not transcribe {raw_language!r}. It transcribes '
                f'{_names(PARAKEET_LANGUAGES)}.{others}')
        return PARAKEET, ''  # it detects the language itself
    if primary and primary not in NEMOTRON_LOCALES:
        others = ' Set the model to Automatic or Parakeet for it.' if primary in PARAKEET_LANGUAGES else ''
        raise UnsupportedLanguage(
            f'Nemotron does not transcribe {raw_language!r}. It transcribes '
            f'{_names(NEMOTRON_LANGUAGES)}.{others}')
    return NEMOTRON, nemotron_locale(primary, region)


def locale_for_live(raw_language) -> str:
    """Nemotron's locale for live captions; it is the only model that streams."""
    primary, region = split_tag(raw_language)
    if primary and primary not in NEMOTRON_LOCALES:
        raise UnsupportedLanguage(
            f'Live captions do not cover {raw_language!r}: they come from Nemotron, which transcribes '
            f'{_names(NEMOTRON_LANGUAGES)}. Parakeet transcribes it from a file (Import).'
            if primary in PARAKEET_LANGUAGES else
            f'Parakeet & Nemotron do not transcribe {raw_language!r}. Live captions cover '
            f'{_names(NEMOTRON_LANGUAGES)}.')
    return nemotron_locale(primary, region)
