
import json
from pathlib import Path
import pandas as pd

# Set these paths for your environment.
csv_file = r"/Users/atis/Desktop/MasterArbeit/Code/test_data.csv"
root_directory = "/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data"
output_directory = "/Users/atis/Desktop/MasterArbeit/Code/validation_sample_single_testing"


def read_annotations(file_path):
    annotations = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=2)

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid annotation in {file_path}, "
                    f"line {line_number}: {line!r}"
                )

            start, end, label = parts
            annotations.append(label)

    return " ".join(annotations)


def create_validation_sample_phonemes():
    root = Path(root_directory).expanduser()
    output = Path(output_directory).expanduser()

    # Same CSV loading and filtering as TIMITDataset.
    df = pd.read_csv(csv_file)
    df = df.loc[df["test_or_train"] == "TEST"].copy()

    # Same utterance ID creation as TIMITDataset.
    df["utterance_id"] = (
        df["path_from_data_dir"].str.rsplit(".", n=1).str[0]
    )

    samples = []

    # Same grouping and ordering as TIMITDataset.
    grouped = df.groupby("utterance_id")

    for utterance_id, group in grouped:
        audio_rows = group[
            group["filename"].str.endswith(".WAV", na=False)
        ]
        phoneme_rows = group[
            group["filename"].str.endswith(".PHN", na=False)
        ]

        # Same validation as TIMITDataset.
        if len(audio_rows) != 1 or len(phoneme_rows) != 1:
            continue

        phn_path = root / phoneme_rows.iloc[0]["path_from_data_dir"]

        # WRD path shares the same utterance ID.
        wrd_path = root / f"{utterance_id}.WRD"

        if not phn_path.is_file():
            raise FileNotFoundError(
                f"Phoneme file does not exist: {phn_path}"
            )

        if not wrd_path.is_file():
            raise FileNotFoundError(
                f"Word annotation does not exist: {wrd_path}"
            )

        phn_elements = read_annotations(phn_path)
        wrd_transcript = read_annotations(wrd_path)

        samples.append({
            "sample_idx": len(samples),
            "phn_path": str(phn_path),
            "phn_length": len(phn_elements.split()),
            "phn_elements": phn_elements,
            "wrd_path": str(wrd_path),
            "wrd_transcript": wrd_transcript,
        })

    # Sort by phoneme count while preserving the original dataset indices.
    samples.sort(key=lambda sample: (sample["phn_length"], sample["sample_idx"]))

    output.mkdir(parents=True, exist_ok=True)

    output_path = output / "validation_sample_phonemes.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(samples, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    create_validation_sample_phonemes()
