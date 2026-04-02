"""Stage 1 of the ASR fine-tuning pipeline: generate synthetic medical text
samples using a local Ollama LLM.

Public API:
    generate_text()  — run the full generation loop, return Path to output JSONL
"""

import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from tqdm import tqdm

SAMPLES_PER_SLOT = 50
MAX_RETRIES = 3
SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Taxonomy & constants
# ---------------------------------------------------------------------------

LENGTH_CLASSES = {
    "kurz": (8, 10),
    "mittel": (15, 18),
    "lang": (25, 30),
    "sehr_lang": (38, 45),
}

ABBREVIATION_PATTERNS = [
    r"\bmg\b", r"\bml\b", r"\bcm\b", r"\bmm\b", r"\bkg\b",
    r"\bz\.B\.", r"\bd\.h\.", r"\bbzw\.", r"\bu\.a\.", r"\bs\.c\.",
    r"\bi\.v\.", r"\bp\.o\.", r"\bggf\.", r"\bevtl\.", r"\bca\.",
    r"\b\d+x\b", r"\bDr\.", r"\bProf\.", r"\bNr\.",
    r"\bEKG\b", r"\bMRT\b", r"\bCT\b",
]

ABBREVIATION_RE = re.compile("|".join(ABBREVIATION_PATTERNS))
MARKDOWN_RE = re.compile(
    r"^\s*#{1,6}\s"
    r"|^\s*[-*+]\s"
    r"|\*\*.+\*\*"
    r"|__.+__"
    r"|\[.+\]\(.+\)"
    r"|```",
    re.MULTILINE,
)

_TOPIC_KEY_TABLE = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", " ": "_"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def topic_to_key(topic: str) -> str:
    return topic.lower().translate(_TOPIC_KEY_TABLE)


def make_slot_key(topic_key: str, context: str, length_class: str) -> str:
    return f"{topic_key}_{context}_{length_class}"


def pick_vocabulary_term(topic: str, sample_index: int, vocabulary: list[str]) -> str:
    rng = random.Random(f"{topic}_{sample_index}")
    return rng.choice(vocabulary)


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------


def check_ollama(url: str, model: str) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        with httpx.Client(timeout=5.0) as client:
            client.get(url)
            resp = client.post(f"{url}/api/show", json={"model": model})
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
        return True
    except httpx.ConnectError:
        return False


def ollama_generate(
    prompt: str,
    url: str,
    model: str,
    client: httpx.Client,
    temperature: float = 0.8,
    max_tokens: int = 512,
) -> str:
    resp = client.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# What makes a good prompt for medical text generation:
#
# 1. Persona framing ("Du bist ein niedergelassener Facharzt fuer …") steers the
#    LLM toward the register we actually want — clinical German, not textbook
#    prose. Without it, models default to formal encyclopaedic writing that no
#    doctor would dictate aloud.
#
# 2. Explicit length constraints prevent both one-liners and multi-paragraph essays.
#    ASR models are evaluated on utterance-level WER, so extreme lengths distort
#    the distribution of the training corpus.
#
# 3. The abbreviation ban is critical: spoken German expands "mg" to "Milligramm",
#    but LLMs default to writing abbreviations. We need the transcription target to
#    match how the TTS model will actually pronounce the word, so we forbid
#    abbreviations in the text and let the TTS handle full words naturally.
#
# 4. Seeding a domain-specific vocabulary term per sample (e.g. "Papel",
#    "Induration") increases lexical diversity across samples for the same topic,
#    which is important for training a robust ASR model on specialist vocabulary.
def make_generator_prompt(
    slot: dict, specialty: str, length_class_instructions: dict[str, str]
) -> str:
    length_instr = f"Laenge: {slot['word_min']} bis {slot['word_max']} Woerter."
    abbrev_instr = (
        "Schreibe alle Einheiten und Abkuerzungen vollstaendig aus "
        "(zum Beispiel 'Milligramm' statt 'mg', 'Zentimeter' statt 'cm', 'zum Beispiel' statt 'z.B.')."
    )
    closing = "Antworte NUR mit dem Text, ohne Anführungszeichen, ohne Erklärungen, ohne Markdown."
    scope_instr = length_class_instructions[slot["length_class"]]
    vocab_instr = f"Verwende dabei den Begriff '{slot['vocabulary_term']}' im Text, falls fachlich passend."

    if slot["context"] == "befund_diktat":
        return (
            f"Du bist ein niedergelassener Facharzt fuer {specialty} in Deutschland. "
            f"Diktiere einen klinischen Befund zum Thema {slot['topic']}. "
            f"{length_instr} {scope_instr} {vocab_instr} {abbrev_instr} "
            f"Diktiere natuerlich, als wuerdest du in ein Diktiergeraet sprechen. "
            f"{closing}"
        )
    return (
        f"Du bist ein Facharzt fuer {specialty} im Gespraech mit einem Patienten. "
        f"Sage einen einzelnen Satz oder kurzen Absatz zum Thema {slot['topic']}. "
        f"{length_instr} {scope_instr} {vocab_instr} {abbrev_instr} "
        f"Sprich verstaendlich, keine Abkuerzungen. "
        f"{closing}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_abbreviations(text: str) -> str | None:
    match = ABBREVIATION_RE.search(text)
    if match:
        return f"abbreviation found: '{match.group()}'"
    return None


def check_word_count(text: str, word_min: int, word_max: int) -> str | None:
    count = len(text.split())
    lower = int(word_min * 0.5)
    upper = int(word_max * 1.5)
    if count < lower or count > upper:
        return f"word count {count} outside tolerance ({lower}-{upper})"
    return None


def check_formatting(text: str) -> str | None:
    if MARKDOWN_RE.search(text):
        return "markdown or formatting detected"
    return None


def check_duplicates(text: str, existing_texts: list[str]) -> str | None:
    text_len = len(text)
    for existing in existing_texts:
        if abs(len(existing) - text_len) / max(text_len, len(existing)) > 0.2:
            continue
        ratio = SequenceMatcher(None, text, existing).ratio()
        if ratio > SIMILARITY_THRESHOLD:
            return f"too similar to existing sample (similarity: {ratio:.2f})"
    return None


def rule_based_validation(
    text: str,
    word_min: int,
    word_max: int,
    existing_texts: list[str] | None = None,
) -> str | None:
    """Run all rule-based checks. Return first error message or None if all pass."""
    if existing_texts is None:
        existing_texts = []
    for check_fn, args in [
        (check_abbreviations, (text,)),
        (check_word_count, (text, word_min, word_max)),
        (check_formatting, (text,)),
        (check_duplicates, (text, existing_texts)),
    ]:
        result = check_fn(*args)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def load_all_samples(output_file: Path) -> list[dict]:
    """Load all samples as a flat list."""
    if not output_file.exists():
        return []
    with open(output_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_samples(output_file: Path) -> dict[str, list[dict]]:
    """Load existing JSONL and group by slot key."""
    samples_by_slot: dict[str, list[dict]] = {}
    for sample in load_all_samples(output_file):
        key = make_slot_key(sample["topic"], sample["context"], sample["length_class"])
        samples_by_slot.setdefault(key, []).append(sample)
    return samples_by_slot


def build_slots(taxonomy: dict[str, list[str]], contexts: list[str]) -> list[dict]:
    slots = []
    for category, topics in taxonomy.items():
        for topic in topics:
            for context in contexts:
                for length_class, (word_min, word_max) in LENGTH_CLASSES.items():
                    slots.append({
                        "category": category,
                        "topic": topic,
                        "topic_key": topic_to_key(topic),
                        "context": context,
                        "length_class": length_class,
                        "word_min": word_min,
                        "word_max": word_max,
                    })
    return slots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_text(
    taxonomy: dict[str, list[str]],
    vocabulary: list[str],
    contexts: list[str],
    length_class_instructions: dict[str, str],
    n_samples: int = 50,
    specialty: str = "Dermatologie",
    ollama_host: str = "http://localhost:11434",
    model: str = "qwen3.5:35b",
    output_dir: Path | None = None,
) -> Path:
    """Generate synthetic medical text samples using Ollama.

    Args:
        taxonomy: Dict mapping category names to lists of topic strings.
        vocabulary: Domain-specific terms seeded into prompts for lexical
            diversity.
        contexts: List of dictation context types (e.g.
            ``["befund_diktat", "arzt_patient_dialog"]``).
        length_class_instructions: Dict mapping length class names to
            scope instructions for the LLM prompt (e.g. what to include
            at each length).
        n_samples: Maximum number of new samples to generate (None = generate all).
        specialty: Medical specialty name used in the LLM prompt persona
            (e.g. ``"Dermatologie"``, ``"Kardiologie"``).
        ollama_host: Base URL for the Ollama server.
        model: Ollama model name to use for generation.
        output_dir: Directory to write the output JSONL. Defaults to ``data/``.

    Returns:
        Path to the output JSONL file.

    Raises:
        ConnectionError: If Ollama is not reachable at ``ollama_host``.
    """

    if not check_ollama(ollama_host, model):
        raise ConnectionError(
            f"Cannot connect to Ollama at {ollama_host} or model '{model}' not found. "
            f"Start Ollama with: ollama serve\n"
            f"Pull the model with: ollama pull {model}"
        )

    if output_dir is None:
        output_dir = Path("data")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset.jsonl"

    samples_by_slot = load_existing_samples(output_file)
    next_id = max(
        (int(s["id"].rsplit("_", 1)[-1]) for samples in samples_by_slot.values() for s in samples),
        default=0,
    ) + 1

    slots = build_slots(taxonomy, contexts)
    total_target = len(slots) * SAMPLES_PER_SLOT
    existing_total = sum(len(v) for v in samples_by_slot.values())
    to_generate = total_target - existing_total
    if n_samples is not None:
        to_generate = min(to_generate, n_samples)

    if n_samples is not None and n_samples < total_target:
        print(f"Generating {to_generate} samples (quick mode). Full dataset would be {total_target}.")
    else:
        print(f"Generating {to_generate} of {total_target} samples ({existing_total} already exist).")
    print(f"Specialty: {specialty} | Model: {model}")
    print()

    if to_generate <= 0:
        print("Nothing to generate.")
        return output_file

    stats = {"generated": 0, "rule_rejected": 0, "errors": 0}
    progress = tqdm(
        total=to_generate,
        desc="Generating",
        unit="sample",
        mininterval=30,
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]\n",
    )

    with httpx.Client() as client, open(output_file, "a", encoding="utf-8") as out_f:
        for slot in slots:
            sk = make_slot_key(slot["topic_key"], slot["context"], slot["length_class"])
            existing = samples_by_slot.get(sk, [])
            if len(existing) >= SAMPLES_PER_SLOT:
                continue

            existing_texts = [s["text"] for s in existing]
            needed = SAMPLES_PER_SLOT - len(existing_texts)

            for sample_idx in range(SAMPLES_PER_SLOT - needed, SAMPLES_PER_SLOT):
                vocab_term = pick_vocabulary_term(slot["topic"], sample_idx, vocabulary)
                prompt = make_generator_prompt(
                    {**slot, "vocabulary_term": vocab_term}, specialty, length_class_instructions
                )

                for _retry in range(MAX_RETRIES):
                    try:
                        text = ollama_generate(prompt, ollama_host, model, client).strip("\"'")
                    except httpx.HTTPError as e:
                        stats["errors"] += 1
                        tqdm.write(f"  API error: {e}")
                        continue

                    rule_error = rule_based_validation(
                        text, slot["word_min"], slot["word_max"], existing_texts
                    )
                    if rule_error:
                        stats["rule_rejected"] += 1
                        continue

                    sample = {
                        "id": f"sample_{next_id:05d}",
                        "text": text,
                        "category": slot["category"],
                        "topic": slot["topic_key"],
                        "context": slot["context"],
                        "length_class": slot["length_class"],
                        "word_count": len(text.split()),
                        "vocabulary_term": vocab_term,
                    }
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    out_f.flush()
                    existing_texts.append(text)
                    next_id += 1
                    stats["generated"] += 1
                    progress.update(1)
                    break
                else:
                    tqdm.write(f"  SKIP: {sk} — failed after {MAX_RETRIES} retries")

                if n_samples is not None and stats["generated"] >= n_samples:
                    progress.close()
                    print(
                        f"\nDone. Generated {stats['generated']} samples "
                        f"({stats['rule_rejected']} rejected by validation, "
                        f"{stats['errors']} errors)."
                    )
                    return output_file

    progress.close()
    print(
        f"\nGenerated: {stats['generated']}, "
        f"Rule rejected: {stats['rule_rejected']}, "
        f"Errors: {stats['errors']}"
    )
    return output_file
