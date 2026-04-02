"""Stage 5: Evaluate ASR models by comparing base vs finetuned WER."""

import json
import re
import string
from pathlib import Path

import jiwer
from num2words import num2words

from asr.dataset import QWEN3_ASR_PREFIX_TEMPLATE


def normalize(text: str, language: str = "de") -> str:
    text = text.lower()
    text = re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=language), text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_wer(references: list[str], hypotheses: list[str], language: str = "de") -> float:
    refs = [normalize(r, language) for r in references]
    hyps = [normalize(h, language) for h in hypotheses]
    return jiwer.wer(refs, hyps)


def _load_test_samples(test_file: Path) -> list[dict]:
    samples = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = entry["text"]
            # Strip Qwen3-ASR prefix (e.g. "language German<asr_text>")
            marker = "<asr_text>"
            if marker in text:
                reference = text[text.index(marker) + len(marker):]
            else:
                reference = text
            samples.append({"audio": entry["audio"], "reference": reference})
    return samples


def _transcribe_batch(model_id: str, samples: list[dict], language: str = "de") -> list[str]:
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model

    model = load_model(model_id)
    hypotheses = []
    for s in samples:
        result = generate_transcription(model=model, audio=s["audio"], language=language)
        hypotheses.append(result.text)
    return hypotheses


def _transcribe_batch_with_adapters(
    base_model_id: str, adapter_path: str, samples: list[dict], language: str = "de"
) -> list[str]:
    from mlx_tune import FastSTTModel

    model, _ = FastSTTModel.from_pretrained(model_name=base_model_id, max_seq_length=448)
    model.load_adapter(adapter_path)
    FastSTTModel.for_inference(model)

    hypotheses = []
    for s in samples:
        text = model.transcribe(s["audio"], language=language)
        hypotheses.append(text)
    return hypotheses


def evaluate(
    test_file: Path,
    finetuned_model: str | None = None,
    base_model: str = "mlx-community/Qwen3-ASR-0.6B-5bit",
    language: str = "de",
) -> dict:
    """Compare base vs finetuned WER on the held-out test set.

    Args:
        test_file: Path to test.jsonl (Qwen3-ASR format).
        finetuned_model: Path to finetuned model or adapter directory.
        base_model: Base model HuggingFace ID for comparison.
        language: Language code for transcription and WER normalization.

    Returns:
        Dict with "base_wer" and optionally "finetuned_wer" as floats.
    """
    print(f"Loading test samples from {test_file}...")
    samples = _load_test_samples(test_file)
    references = [s["reference"] for s in samples]
    print(f"Loaded {len(samples)} test samples.")

    print(f"\nTranscribing with base model: {base_model}...")
    base_hyps = _transcribe_batch(base_model, samples, language=language)
    base_wer = compute_wer(references, base_hyps, language=language)
    result = {"base_wer": base_wer}
    print(f"Base WER: {base_wer:.1%}")

    if finetuned_model:
        is_adapter = (Path(finetuned_model) / "adapters.safetensors").exists()
        if is_adapter:
            print(f"\nTranscribing with finetuned model (adapters): {finetuned_model}...")
            ft_hyps = _transcribe_batch_with_adapters(base_model, finetuned_model, samples, language=language)
        else:
            print(f"\nTranscribing with finetuned model: {finetuned_model}...")
            ft_hyps = _transcribe_batch(finetuned_model, samples, language=language)
        ft_wer = compute_wer(references, ft_hyps, language=language)
        result["finetuned_wer"] = ft_wer
        print(f"Finetuned WER: {ft_wer:.1%}")

    return result
