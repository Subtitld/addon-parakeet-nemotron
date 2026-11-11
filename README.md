# Parakeet & Nemotron add-on for Subtitld

Offline speech-to-text for [Subtitld](https://subtitld.org) with two NVIDIA
models, on NVIDIA's [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)
runtime, with no PyTorch:

- **Subtitles from a file in 33 languages.**
  - [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
    transcribes the 25 European languages: Bulgarian, Croatian, Czech, Danish,
    Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian,
    Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian,
    Russian, Slovak, Slovenian, Spanish, Swedish and Ukrainian. It punctuates,
    writes numbers as digits, and times each word accurately.
  - [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
    transcribes Arabic, Chinese, Hindi, Japanese, Korean, Norwegian, Turkish
    and Vietnamese. Configure can also pick it for the languages both cover.
- **Speaker labels**, when asked for:
  [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
  tells up to 8 speakers apart (A, B, C…) in one pass over the file, and a
  new subtitle starts when the speaker changes.
- **Live captions while recording**, from Nemotron, in 28 languages: every
  language above except Greek, Latvian, Lithuanian, Maltese and Slovenian.

Each model downloads the first time it is needed: Parakeet 714 MB, Nemotron
742 MB, the speaker model 107 MB.

## Speed and memory

Measured on an Intel i7-13700T laptop CPU, 8 threads, with other work running:

| | Time | Peak memory |
|---|---|---|
| 12-minute film, Parakeet, with speaker labels | 87 s (8× real time) | 1.3 GB |
| 12-minute film, Nemotron | 63 s (12× real time) | 1.2 GB |
| Speaker labels alone, 3 minutes | 7 s | 0.2 GB |

Files are transcribed a minute at a time (30 s for Nemotron), so memory stays
flat however long the file is, progress is reported as it goes, and a cancel
takes effect at the next window. Finished subtitles appear on the timeline
while the rest is still being transcribed. On a Mac with Apple Silicon the
runtime uses the GPU (Metal).

x86-64 processors need AVX2 (Intel since 2013, AMD since 2015).

## Settings

| Setting | Default | |
|---|---|---|
| Model | Automatic | Parakeet where it covers the language, Nemotron elsewhere. Live captions always use Nemotron. |
| One cue per sentence | On | Off: a long sentence is split at about two lines or 7 seconds. |
| Label speakers | Off | Files only. |
| Live caption delay | 560 ms | 160, 320, 560 ms or 1.1 s of lookahead: longer is more accurate. |

## How it works

The add-on loads NeMo-Speech.cpp's C ABI (`nemo_speech/asr.h`, `diar.h`)
in-process with `ctypes`; it needs neither numpy nor a Python
machine-learning runtime. ffmpeg (the one Subtitld uses) decodes the file.

- **Nemotron's word times.** Nemotron stamps a word when it emits it, about
  when the word ends, so its subtitles would start some 0.7 s late. The
  add-on moves each word back onto where it is spoken (`timing.py`): on the
  first six minutes of Tears of Steel, subtitles starting within 0.5 s of the
  film's own went from 24% to 76% (Parakeet: 85%).
- **Nemotron and punctuation.** Over film sound Nemotron often writes no
  punctuation at all, so its subtitles also end at a pause of 0.8 s.
  Live captions end where Nemotron hears the end of an utterance (800 ms
  without a new word); they come out in lower case more often than files do.
- **Downloads** come from NVIDIA's own Hugging Face repositories, pinned to
  one revision and checked by SHA-256 (`parakeet_nemotron_addon/pins.json`),
  into Subtitld's model folder (`$SUBTITLD_MODELS_DIR/parakeet-nemotron/`).
  An interrupted download resumes on the next request.

## Licences

| Part | Licence |
|---|---|
| This add-on | Apache-2.0 |
| NeMo-Speech.cpp | Apache-2.0, with ggml and llama.cpp (MIT) and SentencePiece (Apache-2.0) |
| Parakeet TDT 0.6B v3 | CC-BY-4.0, by NVIDIA |
| Nemotron 3.5 ASR Streaming 0.6B, Nemotron-3-Diarization | OpenMDW-1.1, by NVIDIA |

The models are downloaded, not redistributed. `THIRD_PARTY_LICENSES/` in each
release archive covers everything bundled, file by file, and the models'
licences.

## Building

```bash
python -m venv .venv && .venv/bin/pip install -e '.[build,dev]'
tools/build_native_linux.sh                  # Linux: NeMo-Speech.cpp from source, in AlmaLinux 8
.venv/bin/python tools/build_native.py       # Windows, macOS: NVIDIA's release archives
.venv/bin/pytest                             # tests/test_installed.py downloads the models (1.6 GB)
```

The runtime lands in `build/assets/`, with its licence and third-party notices
in `build/licences/`.

- **Linux** is built from source at the v0.3.0 commit in an AlmaLinux 8
  container, so it runs on glibc 2.28 and newer. `tools/build_linux.sh`
  builds the whole release archive there, and `tools/check_glibc.py` refuses
  it if any file needs a newer glibc. `tools/test_archive.sh IMAGE` runs the
  installed-archive tests in a clean container of another distribution.
- **Windows and macOS** use NVIDIA's release archives, pinned by SHA-256 and
  cut down to what the ASR library loads (`tools/build_native.py`).

`.github/workflows/release.yml` builds and tests all three platforms and
publishes a release for a `v*` tag.

`tests/data/jfk.flac` is the JFK clip the other Subtitld speech add-ons test
with (a US government work, in the public domain). `tests/data/nemo_speech/`
holds NeMo-Speech.cpp v0.3.0's C headers (Apache-2.0, NVIDIA), which
`tests/test_engine.py` checks the ctypes structs against.
