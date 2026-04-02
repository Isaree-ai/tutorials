"""Stage 6: Transcribe a single audio file using Qwen3-ASR."""

from pathlib import Path


def transcribe(
    audio_file: Path,
    model: str = "mlx-community/Qwen3-ASR-0.6B-5bit",
    base_model: str = "mlx-community/Qwen3-ASR-0.6B-5bit",
    language: str = "de",
) -> str:
    """Transcribe a single audio file.

    Args:
        audio_file: Path to audio file (WAV, MP3, etc.)
        model: Adapter directory, merged model path, or HuggingFace model ID.
        base_model: Base model HuggingFace ID (used when *model* is an
            adapter directory).
        language: Language code for transcription.

    Returns:
        Transcription text.
    """
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model_path = Path(model)
    is_adapter = model_path.is_dir() and (model_path / "adapters.safetensors").exists()

    if is_adapter:
        from mlx_tune import FastSTTModel

        print(f"Loading base model: {base_model}")
        stt_model, _ = FastSTTModel.from_pretrained(
            model_name=base_model, max_seq_length=448
        )
        print(f"Loading adapters from {model}")
        stt_model.load_adapter(str(model_path))
        FastSTTModel.for_inference(stt_model)

        print(f"Transcribing: {audio_file}")
        return stt_model.transcribe(str(audio_file), language=language)
    else:
        from mlx_audio.stt.utils import load_model
        from mlx_audio.stt.generate import generate_transcription

        print(f"Loading model: {model}")
        stt_model = load_model(model)

        print(f"Transcribing: {audio_file}")
        result = generate_transcription(model=stt_model, audio=str(audio_file), language=language)
        return result.text
