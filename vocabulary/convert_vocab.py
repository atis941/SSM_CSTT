import json
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

KALDI_MAPPING_PATH = Path(r"/Users/atis/Desktop/MasterArbeit/Code/vocabulary/kaldi_mapping.txt")
ORIGINAL_VOCAB_PATH = Path(r"vocabulary/phoneme_to_idx_61.json")
OUTPUT_PATH = Path("./vocabulary/61_to_39_vocab.json")


def load_kaldi_mapping(mapping_path: Path) -> dict[str, str | None]:
    """
    Read Kaldi's 60-48-39 mapping file.

    Format:
        original_phone  phone_48  phone_39

    The phoneme 'q' has no corresponding 39-phone label.
    """

    mapping = {}

    with mapping_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            columns = line.split()

            if len(columns) == 3:
                original_phone, _, phone_39 = columns

            elif len(columns) == 1 and columns[0] == "q":
                original_phone = "q"
                phone_39 = None

            else:
                raise ValueError(
                    f"Invalid mapping at line {line_number}: {line}"
                )

            if original_phone in mapping:
                raise ValueError(
                    f"Duplicate phoneme: {original_phone}"
                )

            mapping[original_phone] = phone_39

    return mapping


def create_mapping_vocabulary(
    original_vocab: dict[str, int],
    kaldi_mapping: dict[str, str | None],
) -> dict[int, list]:

    # Check for duplicate original indices.
    if len(set(original_vocab.values())) != len(original_vocab):
        raise ValueError("Original vocabulary contains duplicate indices.")

    original_phones = set(original_vocab) - {"<blank>"}

    missing_phones = original_phones - set(kaldi_mapping)

    if missing_phones:
        raise ValueError(
            f"Missing phonemes in Kaldi mapping: {sorted(missing_phones)}"
        )

    # Generate the 39-phone vocabulary alphabetically.
    phones_39 = sorted({
        phone
        for phone in kaldi_mapping.values()
        if phone is not None
    })

    if len(phones_39) != 39:
        raise ValueError(
            f"Expected 39 phonemes, got {len(phones_39)}."
        )

    # Reserve index 0 for the CTC blank.
    vocab_39 = {"<blank>": 0}

    for index, phone in enumerate(phones_39, start=1):
        vocab_39[phone] = index

    # Original index -> [39-phone index, original string, mapped string]
    result = {}

    for original_phone, original_index in original_vocab.items():

        if original_phone == "<blank>":
            mapped_phone = "<blank>"
        else:
            mapped_phone = kaldi_mapping[original_phone]

        mapped_index = (
            vocab_39[mapped_phone]
            if mapped_phone is not None
            else None
        )

        result[original_index] = [
            mapped_index,
            original_phone,
            mapped_phone,
        ]

    return dict(sorted(result.items()))


def main() -> None:

    # Load Kaldi mapping.
    kaldi_mapping = load_kaldi_mapping(KALDI_MAPPING_PATH)

    # Load original 61-phone vocabulary.
    with ORIGINAL_VOCAB_PATH.open("r", encoding="utf-8") as file:
        original_vocab = json.load(file)

    # Generate mapping vocabulary.
    mapping_vocab = create_mapping_vocabulary(
        original_vocab,
        kaldi_mapping,
    )

    # Create output directory if necessary.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save mapping to JSON.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(mapping_vocab, file, indent=4, ensure_ascii=False)

    print(f"Mapping saved to: {OUTPUT_PATH}")
    print(f"Original vocabulary entries: {len(mapping_vocab)}")
    print("Evaluation phonemes: 39 (+ CTC blank)")


if __name__ == "__main__":
    main()