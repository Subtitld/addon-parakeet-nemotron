# The speech models

None of the models is in this archive. Each downloads from NVIDIA's own
Hugging Face repository the first time it is needed, pinned to one revision
and checked by SHA-256, into Subtitld's model folder.

| Model | Repository and revision | File | Licence |
|---|---|---|---|
| Parakeet TDT 0.6B v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) at `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | `parakeet-tdt-0.6b-v3.q8_0.gguf` (714 MB) | CC-BY-4.0, [CC-BY-4.0.txt](CC-BY-4.0.txt) |
| Nemotron 3.5 ASR Streaming 0.6B | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) at `1c8deaecc64b91f034d73e08dd8b64625eb3395d` | `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` (742 MB) | OpenMDW-1.1, [OpenMDW-1.1.txt](OpenMDW-1.1.txt) |
| Nemotron-3-Diarization | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) at `f667ed73aee57d40cc39428eb768b4fd87a0a29e` | `Nemotron-3-Diarization.q8_0.gguf` (107 MB) | OpenMDW-1.1, [OpenMDW-1.1.txt](OpenMDW-1.1.txt) |

**Attribution (CC-BY-4.0).** Parakeet TDT 0.6B v3 is by NVIDIA, published at
the repository above under the Creative Commons Attribution 4.0 International
licence. The add-on uses NVIDIA's q8_0 GGUF conversion of it, unchanged; the
transcripts it makes are yours.

**OpenMDW-1.1.** The licence asks that a copy of it and the models' notices
travel with any copy of the models. The add-on does not redistribute them: it
downloads them from NVIDIA. The licence places no restriction on what is done
with the transcripts.
