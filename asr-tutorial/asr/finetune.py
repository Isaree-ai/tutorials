"""Stage 4: LoRA finetune Qwen3-ASR on Apple Silicon via mlx-tune."""

import json
import shutil
from pathlib import Path

from datasets import Audio, Dataset

from asr.dataset import QWEN3_ASR_PREFIX_TEMPLATE


def _load_split_as_dataset(split_file: Path) -> Dataset:
    """Load a Qwen3-ASR format JSONL split into a HuggingFace Dataset."""
    audio_paths = []
    texts = []

    with open(split_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            audio_path = Path(entry["audio"])
            if not audio_path.exists():
                print(f"Warning: audio not found, skipping: {audio_path}")
                continue
            audio_paths.append(str(audio_path))
            text = entry["text"]
            # Strip Qwen3-ASR prefix (e.g. "language German<asr_text>")
            marker = "<asr_text>"
            if marker in text:
                text = text[text.index(marker) + len(marker):]
            texts.append(text)

    ds = Dataset.from_dict({"audio": audio_paths, "text": texts})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def finetune(
    train_split: Path,
    val_split: Path,
    output_dir: Path | None = None,
    model: str = "mlx-community/Qwen3-ASR-0.6B-5bit",
    epochs: int = 2,
    lr: float = 1e-4,
    lora_rank: int = 8,
    gradient_accumulation_steps: int = 4,
    max_steps: int | None = None,
    language: str = "de",
    logging_steps: int = 50,
    save_steps: int = 500,
) -> Path:
    """LoRA finetune Qwen3-ASR via mlx-tune.

    Args:
        train_split: Path to train.jsonl (Qwen3-ASR format).
        val_split: Path to val.jsonl (Qwen3-ASR format).
        output_dir: Directory for model output.
        model: Base model HuggingFace ID.
        epochs: Number of training epochs.
        lr: Learning rate.
        lora_rank: LoRA adapter rank.
        gradient_accumulation_steps: Number of forward passes before each
            weight update (effective batch size).
        max_steps: Override max training steps. If None, computed from epochs.
        language: Language code for the data collator (e.g. "de", "en").
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.

    Returns:
        Path to the saved adapter directory.
    """
    from mlx_tune import FastSTTModel, STTDataCollator, STTSFTConfig, STTSFTTrainer

    if output_dir is None:
        output_dir = Path("data/finetuned_model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training dataset...")
    train_ds = _load_split_as_dataset(train_split)
    print(f"Training samples: {len(train_ds)}")

    print("Loading val dataset...")
    val_ds = _load_split_as_dataset(val_split)
    print(f"Val samples: {len(val_ds)}")

    print(f"\nLoading model: {model}")
    stt_model, processor = FastSTTModel.from_pretrained(
        model_name=model,
        max_seq_length=448,
    )

    print(f"Adding LoRA adapters (rank={lora_rank})")
    stt_model = FastSTTModel.get_peft_model(
        stt_model,
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        finetune_encoder=True,
        finetune_decoder=True,
    )

    data_collator = STTDataCollator(
        model=stt_model,
        processor=processor,
        language=language,
        task="transcribe",
        audio_column="audio",
        text_column="text",
    )

    if max_steps is not None:
        computed_steps = max_steps
    else:
        computed_steps = (len(train_ds) // gradient_accumulation_steps) * epochs

    effective_warmup = min(50, max(1, computed_steps // 5))
    effective_logging = min(logging_steps, max(1, computed_steps // 5))
    effective_save = min(save_steps, computed_steps)

    config = STTSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=effective_warmup,
        learning_rate=lr,
        max_steps=computed_steps,
        logging_steps=effective_logging,
        save_steps=effective_save,
        output_dir=str(output_dir),
    )

    print(f"\nStarting training:")
    if max_steps is not None:
        print(f"  Max steps: {computed_steps} (epochs ignored)")
    else:
        print(f"  Epochs: {epochs} ({computed_steps} steps)")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Warmup steps: {effective_warmup}")
    print(f"  Output: {output_dir}")
    print()

    trainer = STTSFTTrainer(
        model=stt_model,
        processor=processor,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )
    trainer.train()

    # The trainer saves adapters internally to output_dir/adapters.
    adapter_path = output_dir / "adapters"
    if not (adapter_path / "adapters.safetensors").exists():
        print(f"\nSaving LoRA adapters to {adapter_path}")
        stt_model.save_pretrained(str(adapter_path))

    print(f"\nAdapters saved to {adapter_path}")
    return adapter_path


def export_model(
    adapter_path: str | Path,
    output_dir: str | Path | None = None,
    base_model: str = "mlx-community/Qwen3-ASR-0.6B-5bit",
) -> Path:
    """Merge LoRA adapters into the base model for standalone use or HF upload.

    Args:
        adapter_path: Path to the adapter directory (from finetune()).
        output_dir: Where to write the merged model. Defaults to
            ``<adapter_path>/../merged``.
        base_model: Base model HuggingFace ID.

    Returns:
        Path to the merged model directory.
    """
    from mlx_tune import FastSTTModel
    from mlx_audio.utils import get_model_path

    adapter_path = Path(adapter_path)
    if output_dir is None:
        output_dir = adapter_path.parent / "merged"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model}")
    stt_model, _ = FastSTTModel.from_pretrained(
        model_name=base_model, max_seq_length=448
    )
    print(f"Loading adapters from {adapter_path}")
    stt_model.load_adapter(str(adapter_path))

    print(f"Merging and saving to {output_dir}")
    stt_model.save_pretrained_merged(str(output_dir))

    # Copy config and tokenizer files from the base model so the merged
    # directory is fully self-contained (loadable without the base model).
    # Skip weight files — the merged weights are already written above.
    base_path = get_model_path(base_model)
    _SKIP_SUFFIXES = {".safetensors", ".npz", ".bin", ".pt", ".pth"}
    for f in base_path.iterdir():
        if (
            f.is_file()
            and f.suffix not in _SKIP_SUFFIXES
            and "safetensors" not in f.name
            and not (output_dir / f.name).exists()
        ):
            shutil.copy2(f, output_dir / f.name)

    print(f"Merged model ready at {output_dir}")
    return output_dir
