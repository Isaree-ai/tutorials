"""Dataset packaging for Qwen3-ASR finetuning.

Joins audio manifest with text dataset, performs stratified train/val/test
splitting, and writes output as Qwen3-ASR JSONL files — the format consumed
by the finetuning stage.

Each output line is a JSON object with "audio" (path) and "text" (transcript
prefixed with "language <LANG><asr_text>") as expected by the Qwen3-ASR model.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Structural prefix template for Qwen3-ASR. The language is filled in at
# runtime via format_qwen3_asr(). Imported by finetune.py and evaluate.py
# for prefix stripping.
QWEN3_ASR_PREFIX_TEMPLATE = "language {language}<asr_text>"


def format_qwen3_asr(audio_path: str, text: str, language: str = "German") -> str:
    """Format a single sample as a Qwen3-ASR JSONL line.

    Args:
        audio_path: Absolute path to the audio file.
        text: Transcript text.
        language: Language name for the Qwen3-ASR prefix (e.g. "German", "English").

    Returns:
        A JSON string in Qwen3-ASR format.
    """
    prefix = QWEN3_ASR_PREFIX_TEMPLATE.format(language=language)
    entry = {
        "audio": audio_path,
        "text": f"{prefix}{text}",
    }
    return json.dumps(entry, ensure_ascii=False)


def load_and_join(manifest_path: Path, dataset_path: Path) -> list[dict]:
    """Join audio manifest with text dataset by ID.

    Only includes samples that have a corresponding entry in both manifest and dataset.

    Args:
        manifest_path: Path to audio manifest JSONL.
        dataset_path: Path to text dataset JSONL.

    Returns:
        List of joined sample dicts with keys from both sources.
    """
    dataset_by_id: dict[str, dict] = {}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            dataset_by_id[record["id"]] = record

    # Load manifest and join with dataset
    joined: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            manifest_record = json.loads(line)
            sample_id = manifest_record["id"]
            # Strip _aug suffix to match augmented samples to their parent
            base_id = sample_id.removesuffix("_aug")
            if base_id in dataset_by_id:
                merged = {**dataset_by_id[base_id], **manifest_record}
                joined.append(merged)

    return joined


def stratified_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/val/test sets, stratified by category.

    Splits on base sample IDs (without _aug suffix) so that a sample and its
    augmented version always land in the same split. Augmented samples are only
    included in the training set — val and test use originals only.

    Args:
        samples: List of sample dicts, each must have a 'category' and 'id' key.
        train_ratio: Fraction for training set (of base samples).
        val_ratio: Fraction for validation set (of base samples).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) sample lists.
    """
    # We split on base IDs (originals), not on individual samples. If we split
    # on all samples including augmented ones, a speed- or pitch-shifted variant
    # of a test utterance could leak into training — the model would have heard
    # the reference audio and WER would be artificially low.
    #
    # Stratifying by category ensures all categories appear in every split,
    # so no category is absent from evaluation.
    rng = random.Random(seed)

    # Separate original and augmented samples
    originals = [s for s in samples if not s["id"].endswith("_aug")]
    augmented_by_base: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["id"].endswith("_aug"):
            base_id = s["id"].removesuffix("_aug")
            augmented_by_base[base_id].append(s)

    # Group originals by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for s in originals:
        by_category[s["category"]].append(s)

    train: list[dict] = []
    val_set: list[dict] = []
    test: list[dict] = []

    # Split each category independently to ensure representation
    for category in sorted(by_category.keys()):
        cat_samples = by_category[category]
        rng.shuffle(cat_samples)

        n = len(cat_samples)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        # Train gets whatever remains — this guarantees train is never starved
        # by the max(1, ...) guarantees on val and test.
        n_train = max(1, n - n_val - n_test)

        train_originals = cat_samples[:n_train]
        val_originals = cat_samples[n_train : n_train + n_val]
        test_originals = cat_samples[n_train + n_val :]

        # Training uses augmented versions only (speed/pitch variants) — this
        # is intentional. The originals were used solely to decide which base
        # IDs belong to which split; they are not passed to the trainer. Val
        # and test keep clean originals so WER scores reflect real-world
        # performance on unmodified audio and remain comparable across runs.
        for s in train_originals:
            train.extend(augmented_by_base.get(s["id"], []))
        for s in val_originals:
            val_set.append(s)
        for s in test_originals:
            test.append(s)

    return train, val_set, test


def write_split(
    samples: list[dict],
    output_path: Path,
    language: str = "German",
) -> None:
    """Write samples as Qwen3-ASR JSONL with absolute audio paths.

    Args:
        samples: List of sample dicts with 'audio_path' and 'text' keys.
        output_path: Path for the output JSONL file.
        language: Language name for the Qwen3-ASR prefix.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            # Resolve to absolute from CWD — manifest stores paths relative
            # to the project root (e.g. "data/audio/sample_00001.wav").
            audio_path = str(Path(s["audio_path"]).resolve())
            line = format_qwen3_asr(audio_path, s["text"], language=language)
            f.write(line + "\n")


def package_dataset(
    manifest: Path,
    text_file: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    language: str = "German",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Package synthetic dataset into train/val/test JSONL splits.

    Joins the audio manifest with the text dataset, performs stratified
    splitting by category, and writes Qwen3-ASR-formatted JSONL files.

    Args:
        manifest: Path to audio manifest JSONL.
        text_file: Path to text dataset JSONL.
        train: Fraction of base samples for training set.
        val: Fraction of base samples for evaluation set.
        test: Fraction of base samples for test set.
        language: Language name for the Qwen3-ASR prefix (e.g. "German").
        output_dir: Output directory for split files. Defaults to
            manifest.parent / "splits".

    Returns:
        Dict with keys "train", "val", "test" mapping to output file paths.

    Raises:
        ValueError: If train + val + test does not sum to 1.0 (within 1e-6).
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train}, val={val}, test={test} "
            f"(sum={train + val + test})"
        )

    if output_dir is None:
        output_dir = manifest.parent / "splits"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_and_join(manifest, text_file)

    train_samples, eval_samples, test_samples = stratified_split(
        samples, train_ratio=train, val_ratio=val
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    write_split(train_samples, train_path, language=language)
    write_split(eval_samples, val_path, language=language)
    write_split(test_samples, test_path, language=language)

    return {"train": train_path, "val": val_path, "test": test_path}
