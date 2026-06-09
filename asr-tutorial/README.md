# Medical ASR Finetuning Tutorial

General-purpose ASR (Automatic Speech Recognition) models struggle with medical terminology — "Dermatofibrosarkoma" or "Efalizumab" become gibberish. Finetuning fixes this, but you need a labeled dataset of medical dictation recordings, and most clinicians don't have one.

In this tutorial, you learn how to generate such a dataset synthetically and finetune a small ASR model on it. [Qwen3.5-35B](https://ollama.com/library/qwen3.5:35b) produces sentences rich in the medical terminology you care about, [Qwen3-TTS-0.6B](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16) converts them to audio (optionally in your own voice), and the result is used to finetune [Qwen3-ASR-0.6B-5bit](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - also locally 🤗. Every step runs on your Mac! So you could even base your training on real clinical data without uploading anything to a cloud provider.

The tutorial defaults to German dermatology for demonstration purposes, but you can adapt it for any language (as long as it's compatible with Qwen3-ASR-0.6B) and specialty.

## Quick Start

```bash
# Clone and install
git clone [https://github.com/isareeai/tutorials.git]
cd tutorials/asr-tutorial
uv sync

# Start Ollama (needed for text generation)
ollama serve
ollama pull qwen3.5:35b # Or any other model you prefer

# Open the tutorial
jupyter notebook tutorial.ipynb
```

## What's Inside

**`tutorial.ipynb`** — The complete tutorial. Open it and run cell by cell.

**`asr/`** — The implementation package. You want to take a look at this in case you want to customize the pipeline.

## Pipeline Overview

Every stage runs locally on macOS — no cloud dependency, no API costs, no privacy headaches.

| Stage              | What it does                                                                                      | Full run (~21k samples) |
| ------------------ | ------------------------------------------------------------------------------------------------- | ----------------------- |
| 1. Generate Text   | LLM creates medical sentences via Ollama                                                          | ~8 h                    |
| 2. Generate Audio  | TTS synthesizes speech (optionally in your cloned voice) + augmented copies (noise, speed, pitch) | ~16 h                   |
| 3. Package Dataset | Stratified 80/10/10 split by taxonomy category                                                    | seconds                 |
| 4. Finetune        | Qwen3-ASR-0.6B is finetuned on your vocabulary                                                    | ~1.5 h                  |
| 5. Evaluate        | Compare base vs finetuned WER                                                                     | ~30 min                 |
| 6. Transcribe      | Try it on your own recordings                                                                     | seconds                 |

The default tutorial runs on 50 samples only to demonstrate the pipeline. A full run takes roughly 26 hours end-to-end (~24h data generation + ~2h finetuning and evaluation) with the appropriate hardware.

## Requirements

- macOS 13.5+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com) for text generation
- ~40 GB disk space (Ollama model ~22 GB, TTS + ASR models ~5 GB, full audio dataset ~2 GB)

**Tested on:** Mac Studio M4 Max, 64 GB RAM. Timings above are from that machine.

## Adapting to Another Specialty

The pipeline structure works for any medical specialty. To adapt it to e.g. cardiology (staying in German), update these in the notebook's Stage 1 cell:

- **`taxonomy`** — replace `asr/taxonomy.json` with your conditions/topics, grouped by category
- **`vocabulary`** — replace with domain-specific terms for lexical diversity
- **`length_class_instructions`** — rewrite to match how your specialty dictates (e.g. a cardiologist describes auscultation findings, not skin morphology)
- **`specialty=`** — the name passed to the LLM prompt

You may also need to update `ABBREVIATION_PATTERNS` in `asr/text.py` — the validation filter that rejects generated text containing abbreviations. The default patterns cover common German medical abbreviations, but specialties like cardiology have many more (e.g. LVEF, PCI, CABG). Abbreviations that slip through will be mispronounced by TTS, degrading training data quality.

Audio generation, dataset packaging, finetuning, and evaluation work unchanged.

## Adapting to Another Language

Changing the language is a larger effort. Beyond the specialty changes above, you'll need to:

- Rewrite the **prompt templates** in `make_generator_prompt()` — these are German prose
- Rewrite **`length_class_instructions`** and **`contexts`** in your target language
- Replace **`ABBREVIATION_PATTERNS`** with language-appropriate patterns
- Update the **`language=`** parameter across `generate_audio()`, `finetune()`, `evaluate()`, and `transcribe()`

## Results

Evaluated on German dermatology dictation (full run, ~21k samples):

| Model                 | WER   |
| --------------------- | ----- |
| Qwen3-ASR-0.6B (base) | 11.0% |
| Finetuned             | 4.4%  |

## Acknowledgments

This tutorial is built on top of great open-source work:

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — the base speech recognition model by Alibaba's Qwen team
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — TTS and audio inference on Apple Silicon
- [mlx-tune](https://github.com/ARahim3/mlx-tune) — LoRA finetuning for audio models on Apple Silicon
- [NeurologyAI/neuro-parakeet-mlx](https://huggingface.co/NeurologyAI/neuro-parakeet-mlx) — inspiration for medical ASR finetuning

## Use in Production

Want to run your finetuned model on an iPhone in your clinical workflows — on-device, for free, and compliant?

Check out [Isaree](https://isaree.ai) and join the waitlist.

## Feedback

Suggestions for improving this pipeline are welcome — reach out at hendrik at isaree dot ai.
