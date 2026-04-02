"""Stage 2: Synthesize audio from text samples using Qwen3-TTS voice cloning."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from tqdm import tqdm

DEFAULT_AUGMENTATION = Compose([
    AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.003, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
])


def _load_samples(input_file: Path) -> list[dict]:
    samples = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            samples.append(sample)
    return samples


def generate_audio(
    text_file: Path,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    augmentation: Compose | None = None,
    model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    language: str = "German",
    output_dir: Path | None = None,
) -> Path:
    """Synthesize WAV files + augmented copies via Qwen3-TTS.

    Args:
        text_file: Path to the text dataset JSONL from Stage 1.
        ref_audio: Optional reference voice WAV for cloning (15-20s of clean speech).
                   If None, uses the model's built-in voice.
        ref_text: Transcript of ``ref_audio``, used by the TTS model for voice
                  cloning. Required when ``ref_audio`` is provided.
        augmentation: An ``audiomentations.Compose`` pipeline applied to each
                      sample to produce an augmented training copy. Defaults to
                      :data:`DEFAULT_AUGMENTATION` (Gaussian noise, time stretch,
                      pitch shift).
        model: HuggingFace model ID for the TTS model.
        language: Language name for TTS generation (e.g. "German", "English").
        output_dir: Directory for WAV output files.

    Returns:
        Path to manifest.jsonl listing all generated audio files.
    """
    if output_dir is None:
        output_dir = Path("data/audio")

    if ref_audio is not None and not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found at {ref_audio}")

    if not text_file.exists():
        raise FileNotFoundError(f"Input file not found at {text_file}")

    samples = _load_samples(text_file)
    print(f"Loaded {len(samples)} samples from {text_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "augmented").mkdir(exist_ok=True)

    def needs_generation(s: dict) -> bool:
        if not (output_dir / f"{s['id']}.wav").exists():
            return True
        if not (output_dir / "augmented" / f"{s['id']}_aug.wav").exists():
            return True
        return False

    to_generate = [s for s in samples if needs_generation(s)]
    print(f"Skipping {len(samples) - len(to_generate)} existing, generating {len(to_generate)}")

    if not to_generate:
        print("Nothing to generate.")
        return output_dir / "manifest.jsonl"

    from mlx_audio.tts.utils import load_model

    print(f"Loading TTS model: {model}...")
    tts_model = load_model(model)
    print("Model loaded. Generating audio...\n")

    ref_audio_str = str(ref_audio) if ref_audio else None
    if ref_audio is not None and ref_text is None:
        raise ValueError(
            "ref_text is required when ref_audio is provided. "
            "Pass the exact transcript of your reference audio recording."
        )
    if augmentation is None:
        augmentation = DEFAULT_AUGMENTATION
    errors = 0
    new_count = 0

    manifest_path = output_dir / "manifest.jsonl"

    # Load existing manifest IDs to avoid duplicate entries on re-runs
    existing_manifest_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_manifest_ids.add(json.loads(line)["id"])

    with open(manifest_path, "a", encoding="utf-8") as manifest_file:
        for sample in tqdm(to_generate, desc="Generating audio", unit="file"):
            out_path = output_dir / f"{sample['id']}.wav"
            try:
                generate_kwargs = {
                    "text": sample["text"],
                    "language": language,
                }
                if ref_audio_str:
                    generate_kwargs["ref_audio"] = ref_audio_str
                    generate_kwargs["ref_text"] = ref_text

                results = list(tts_model.generate(**generate_kwargs))
                audio = np.array(results[0].audio)
                sample_rate = results[0].sample_rate

                sf.write(out_path, audio, sample_rate)

                entry = {
                    "id": sample["id"],
                    "audio_path": str(out_path),
                    "text": sample["text"],
                    "duration_s": round(len(audio) / sample_rate, 2),
                }
                if entry["id"] not in existing_manifest_ids:
                    manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest_file.flush()
                    new_count += 1

                aug_audio = augmentation(samples=audio, sample_rate=sample_rate)
                if len(aug_audio) < len(audio):
                    aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)))
                aug_path = output_dir / "augmented" / f"{sample['id']}_aug.wav"
                sf.write(aug_path, aug_audio, sample_rate)
                aug_id = f"{sample['id']}_aug"
                if aug_id not in existing_manifest_ids:
                    aug_entry = {**entry, "id": aug_id, "audio_path": str(aug_path)}
                    manifest_file.write(json.dumps(aug_entry, ensure_ascii=False) + "\n")
                    manifest_file.flush()
                    new_count += 1

            except Exception as e:
                errors += 1
                tqdm.write(f"  Error generating {sample['id']}: {e}")
                continue

    print(f"\nDone. Added {new_count} entries to manifest, {errors} errors.")
    return manifest_path
