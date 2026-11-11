# Parakeet & Nemotron add-on for Subtitld

Offline speech-to-text for [Subtitld](https://subtitld.org) on NVIDIA's
[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) runtime, with no
PyTorch:

- **Parakeet TDT 0.6B v3** (CC-BY-4.0) transcribes files in its 25 European
  languages, with punctuation, numbers as digits and accurate word times.
- **Nemotron 3.5 ASR Streaming 0.6B** (OpenMDW-1.1) transcribes files in
  Arabic, Hindi, Japanese, Korean, Norwegian, Turkish, Vietnamese and Chinese,
  and gives live captions while recording.
- **Nemotron 3 Diarization** (OpenMDW-1.1) labels up to 8 speakers, when
  asked.

The models download from NVIDIA's Hugging Face repositories on first use.

Work in progress: the native runtime builds (milestone 1 of
`PARAKEET_NEMOTRON_ADDON_PLAN.md` in the Subtitld repository).

## Native libraries

The add-on calls the runtime's C ABI (`nemo_speech/asr.h`) through ctypes.

```bash
tools/build_native_linux.sh               # Linux: from source, in AlmaLinux 8 (glibc 2.28)
python tools/build_native.py              # Windows, macOS: NVIDIA's release archives
python tools/smoke_native.py MODEL.gguf AUDIO --language en
```

They land in `build/assets/`, with the runtime's licence and third-party
notices in `build/licences/`. x86-64 builds need a CPU with AVX2.

## Licence

Apache-2.0. The runtime and models keep their own licences.
